"""Relays (what a child inherits from its parent besides words) and the lock-step level sampler (x0-blend between siblings).

The sampler is parameterization-agnostic: it only knows a list of noise LEVELS from the backend and three operations on them
  B.denoise(x, lvl, cond, cfg) -> (x0_hat, eps_hat)   B.mix(x0, eps, lvl) -> x at that level   B.noise_frac(lvl) in [0, 1]
so that a flow-matching model (SD3: level = sigma, mix = (1-s) x0 + s eps, noise_frac = s) and a VP epsilon/v model
(SD2: level = (t, alpha, sigma), mix = alpha x0 + sigma eps = the DDIM eta=0 update, noise_frac = sigma/(alpha+sigma))
run through the same `run_level`. Every relay threshold is stated in `noise_frac`, so "field:0.9" means the same thing on both models.

Level schedule (SD3 flow-match, shift 3, 20 steps): noise_frac >= 0.95 -> 3 steps, >= 0.9 -> 5, >= 0.8 -> 9, >= 0.6 -> 13, >= 0.3 -> 17 of 20.
So a relay threshold of 0.3 is NOT "early" (v37 run1 lesson); early-only means thresholds around 0.8-0.95.

field_edge:sigma_c:sigma_e -- "anchor the joints, free the core". The child starts from mix(field, eps, sigma(x)) with sigma(x) = sigma_e on
every overlap it shares with a sibling, ramping linearly to sigma_c over one halo width into the core (sigma_c elsewhere); the window is
sampled from the level of sigma_c. While the next level's noise_frac is still > sigma_e, the joint columns (same ramp) are re-projected
after every step onto mix(anchor, eps, current level) with the SAME eps as the init (RePaint-style), where the anchor is the node's field
with the two siblings' fields averaged over their overlap (both siblings get the same target). Below sigma_e the joints are free and only
the usual x0 blend acts. field_edge:0.9:0.6 differs from field:0.9 only on the joints. Only B.mix / B.noise_frac / B.linspace_w and the
field operations are used, so it is backend-generic (exact on flow matching; the sigma(x) interpolation is first-order on VP backends).

Solver (CFG["solver"]). "euler" (default): x' = mix(x0_hat, eps_hat, next level) after the x0 blend -- first order (DDIM on VP).
"dpmpp2m": DPM-Solver++(2M) in data-prediction form, written on the backend's (alpha, sigma) per level (`B.alpha_sigma`: flow -> (1-s, s),
VP -> (alpha, sigma)); with lambda = log(alpha/sigma), h = lambda' - lambda, r = h_prev / h:
    x' = (sigma'/sigma) x + alpha' (1 - e^{-h}) D,   D = (1 + 1/(2r)) x0_n - 1/(2r) x0_{n-1}      [alpha'(1 - e^{-h}) = alpha' - alpha sigma'/sigma]
where x0_n, x0_{n-1} are the node's BLENDED x0 of this and the previous step (per-node history). The first step of a level (also when it
starts mid-schedule at a field relay's sigma_0), any step whose previous h is infinite (sigma = 1 on flow: alpha = 0) and the final step to
sigma = 0 are first order (D = x0_n; the last step returns x0_n), and so is a step more than DPM_H_RATIO_MAX times longer in lambda than the
previous one (the SD3/FLUX schedules' last step into sigma ~ 0.003). The solver only reads the x0 history and the CURRENT x, so the relays that edit
x0 (rho, lowpass_skip, the sibling blend) or overwrite x (field_edge re-anchoring) keep working unchanged.
"unipc": UniPC (UniP predictor + UniC corrector, B(h) bh1/bh2, data prediction) = diffusers' UniPCMultistepScheduler, which is what
WanPipeline samples with (Euler / DPM++(2M) make objects pop in and out between frames on Wan 2.2 at 30 steps; the official UniPC does not).
Per step and node, in diffusers' order: model call -> relay edits -> sibling x0 blend -> UniC corrects the CURRENT x with the new blended x0
and the node's history (the last `solver_order` blended x0's and their lambdas) -> UniP advances to the next level (`unipc_step`). Options
(order, solver_type, lower_order_final, disable_corrector) are read from the backend's scheduler config when it is a UniPC one (Wan 2.2:
order 2, bh2, lower_order_final), else UNIPC_DEFAULTS (the same values), and CFG["unipc"] overrides. A level that starts mid-schedule (field
relays) warms up exactly like UniPC at its start: first step order 1 and no corrector. field_edge: the joints are re-anchored AFTER the
predictor; the next corrector is applied to that re-anchored x as an increment (x + C - P, see `unipc_step`), and above sigma_e the
re-anchoring overwrites the joints again after the next predictor, so a correction on the anchored columns does not persist until they are
freed (the first corrector after the last re-anchoring does act on them). Single window: identical to diffusers step by step. Only B.alpha_sigma is needed (flow and VP).

"consistency": the few-step sampler of consistency / DMD-distilled models (Wan2.2-TI2V-5B-Turbo's Self-Forcing inference,
pipeline/wan22_fewstep_inference.py of quanhaol/Wan2.2-TI2V-5B-Turbo): x' = mix(x0_hat, eps', next level) with FRESH Gaussian noise eps'
(not the predicted eps_hat -- that would be "euler"/DDIM). eps' is one noise canvas per level and step (`level_noise_canvas` with a
step-dependent seed), cropped per node, so siblings share the fresh noise on their overlap exactly as they share the initial noise.
The x0 blend and every relay edit act on x0_hat before the re-noising, unchanged; the final step to level 0 returns the blended x0.

anchor_dense[:sigma0][@tok|@repaint] / anchor_ends[...] -- the child as TEMPORAL INPAINTING of its stretched parent (video backends only;
the per-step equality constraint "decimated child = parent" of METHOD.md section 4, realised natively instead of through a low-passed field).
A child latent frame i sits at global time g = child.canvas_g0 + i * child.scale; it is an ANCHOR when g lies on the parent's latent-frame
lattice inside the parent's canvas, i.e. (g - parent.canvas_g0) % parent.scale == 0 and j = (g - parent.canvas_g0) // parent.scale is in
[0, parent.canvas_w); its anchor value is the parent's FINAL clean latent frame j (run_tree samples level by level, so every parent is fully
denoised before its children start). With branch 2 (parent.scale = 2 child.scale) this is every second child frame; which parity depends
on the child's canvas start (the halo offset). Worked example, core 27 + halo 2 latent frames, root_halo 1 (the v40 geometry):
  depth 1 tree: root scale 2, canvas g[-4, 58) = 31 frames (frame j at g = -4 + 2j); children scale 1:
    L1n00 canvas g[0, 29)  w29 (no left halo):   i even, i = 0..28 -> j = 2..16    (15 anchors; i = 2k -> j = k + 2)
    L1n01 canvas g[25, 54) w29 (no right halo):  i odd,  i = 1..27 -> j = 15..28   (14 anchors; i = 2k + 1 -> j = k + 15)
  depth 2 tree: root scale 4, canvas g[-8, 116) = 31 frames (j at g = -8 + 4j); level 1 scale 2, level 2 scale 1:
    L1n00 canvas g[0, 58)   w29: i = 0..28 (all) -> g = 2i -> j = i/2 + 2, anchors i even 0..28 -> j = 2..16    (15)
    L1n01 canvas g[50, 108) w29: g = 50 + 2i, anchors i odd 1..27 -> j = 15..28                                  (14)
    L2n00 canvas g[0, 29)   w29 (parent L1n00, g = 2j):       anchors i even 0..28 -> j = 0..14                  (15)
    L2n01 canvas g[25, 56)  w31 (parent L1n00):               anchors i odd 1..29  -> j = 13..27                 (15)
    L2n02 canvas g[52, 83)  w31 (parent L1n01, g = 50 + 2j):  anchors i even 0..30 -> j = 1..16                  (16)
    L2n03 canvas g[79, 108) w29 (parent L1n01):               anchors i odd 1..27  -> j = 15..28                 (14)
  (The parent's halo frames are used too: e.g. L2n01's last anchor j = 27 is L1n00's first right-halo frame; no child frame ever falls
  outside its parent's canvas because the child's halo is 1/branch of the parent's.) Siblings share their 2*halo overlap frames, which sit
  at the same g and therefore map onto the same parent-level time; at depth >= 2 neighbours may have DIFFERENT parents (L2n01 / L2n02),
  whose overlap frames are equal after the parent level's x0 blend -- the anchor values are nevertheless averaged over every node of the
  level that anchors the same g (`level_anchors`), so both siblings are constrained to identical values by construction. The x0 blend on
  the overlaps still acts on the non-anchor frames (and is a no-op on the anchored ones).
  anchor_dense: all of those frames. anchor_ends: endpoint-conditioned generation -- only the anchors in the child's two boundary regions:
  on a side with a sibling, the whole shared overlap (its own halo + the first `halo` core frames, i.e. including the first/last core frame;
  both siblings then anchor exactly the same frames), on a horizon end (no halo) the first/last core frame, extended inwards to the nearest
  anchorable frame when it is not on the parent lattice (e.g. L1n01's last frame i = 28 -> anchor i = 27). Example depth 2: L2n01 anchors
  i in {1, 3} and {27, 29}, L2n02 {0, 2} and {28, 30}.
  Start: the non-anchor frames start from pure noise at the first level (default) or, with ':sigma0' < 1, from mix(field, eps, sigma0) at the
  first level with noise_frac <= sigma0 (the field = the usual low-passed stretched parent, `child_field`).
  Mechanism '@tok' (per-token timesteps, needs B.token_anchors: Wan): the anchor frames are CLEAN in x (= anchor value) and
  B.denoise(x, lvl, cond, cfg, anchor_mask, anchor_x) gives their tokens timestep 0 and the other tokens the level's -- exactly WanPipeline's
  TI2V first-frame condition (expand_timesteps: latent_model_input = (1 - mask) * condition + mask * latents, timestep = mask * t per
  token), generalised from frame 0 to any set of frames. '@repaint' (any backend): before every model call the anchor frames are replaced
  by mix(anchor, eps0, current level) with the level's FIXED initial noise eps0 (shared between siblings). Both: after the model call
  x0_hat[anchor] = anchor (the equality constraint on the estimate, before the sibling blend and the solver), after the solver step the
  anchor frames are reset (tok: to the anchor; repaint: to mix(anchor, eps0, next level) = the anchor at level 0), so the returned latent
  equals the parent on every anchored frame exactly. Default mechanism: '@tok' when the backend has token_anchors, else '@repaint'.

The x0 blend between siblings already covers the WHOLE shared region (2 * halo_px); a wider blend therefore means a wider halo
(CFG["halo_px"] 128 / 192 -> 256 / 384 px of overlap), not a separate relay.

No-overlap (abutting) video windows -- CFG halo_px = 0 on a video backend (v43). Every window is its core (e.g. core 31 = one native
31-latent-frame clip), neighbours share NO frame, there is no x0 blend (blend_pairs skips a zero overlap) and `video.compose_level` is
plain concatenation of the cores (one VAE decode of 31 * 2^d latent frames). The anchor relays then use the BLOCK mapping instead of the
global lattice: the parent's P = parent.canvas_w frames are split into consecutive blocks, child k of b gets parent frames
[a_k, a_{k+1} - 1] with a_k = floor(k P / b + 1/2), and core frame c of the child (c = i - core_n0, core width w) sits at parent time
t(c) = a_k + c (e_k - a_k) / (w - 1), e_k = a_{k+1} - 1 (`parent_time`). Anchor pairs = the core frames with an integer t. anchor_ends
anchors exactly the first core frame to parent frame a_k and the last to e_k (';w=K': the K integer pairs nearest each end), so at every
joint the left window's LAST frame is anchored to parent frame m - 1 and the right window's FIRST frame to parent frame m (consecutive
parent frames). Mapping with core 31 (every canvas 31 frames, root = the native 31-frame clip; P = 31 -> blocks [0, 15] and [16, 30]):
  depth 1: L1n00 frames 0..30 <-> root 0..15 (t = c / 2: anchors c even, 16 pairs; ends (0, 0), (30, 15))
           L1n01 frames 0..30 <-> root 16..30 (t = 16 + 14 c / 30: integer at c = 0, 15, 30; ends (0, 16), (30, 30))
           composite 62 latent frames = 31 * 2, joint between latent frames 30 | 31 (pixel frames 120 | 121).
  depth 2: L1n00 / L1n01 as above; L2n00 <-> L1n00 0..15, L2n01 <-> L1n00 16..30, L2n02 <-> L1n01 0..15, L2n03 <-> L1n01 16..30
           (ends (0, 0)/(30, 15), (0, 16)/(30, 30), (0, 0)/(30, 15), (0, 16)/(30, 30)); across the middle joint L2n01's last frame =
           L1n00 frame 30 (= root 15) and L2n02's first = L1n01 frame 0 (= root 16). Composite 124 latent frames = 31 * 4 (493 px frames).
  The block map is not the uniform global lattice (L1n01 frame c is at global time 15.5 + c / 2 in root frames, the map says 16 + 14 c / 30):
  an odd frame count (31 = 2 * 15 + 1) cannot be split into two stretch-2 halves; the block map moves the half-frame to the joint step.

Sequential handoff composition -- CFG compose = "handoff" (v43, needs halo_px 0; K = CFG handoff_k, default 2). The windows of a level are
sampled LEFT TO RIGHT (`run_level_seq`), not in lock-step. Window k >= 1 has a canvas of K + core frames (ext_l = K, geometry.build_tree):
its first K frames sit at the same global time as the LAST K frames of window k-1 and are anchored (clean, per-token timestep 0 under
@tok) to window k-1's FINAL latent there (n.anchor['fixed']); its left end gets NO parent anchor; its right end is anchored to the parent
as in anchor_ends (last core frame <-> e_k); its interior is generated. Window 0 is anchored to the parent at both ends. The composite
keeps only the cores (the K handed-off frames of window k are exact copies of window k-1's and are dropped: overlap-by-K, no averaging).
Each window draws its noise from the same level canvas as lock-step would (`noise_nodes`).

first_slot = "image" (CFG, v43; default "latent" = unchanged). Wan's latent slot 0 is the causal IMAGE slot (it encodes ONE pixel frame; TI2V
conditions exactly this slot with the VAE encoding of an image), so anchoring a window's slot 0 with a mid-clip 4-frame latent is out of
distribution (v42: every arm starts washed out). With "image" every anchor on a window's canvas slot 0 is an image latent (B.encode_image of
one decoded pixel frame): a parent anchor (0, j > 0) -> the parent's pixel frame px_of_lat(j) (the first pixel frame of slot j, continuous
with the left sibling's last anchor j - 1) from the decode of the parent's own latent; j = 0 -> the parent's slot 0 as is (already an
image latent). A handoff (compose='handoff') -> slot 0 = the image latent of the LAST pixel frame of window k-1's slot at the same time
(K = 1: window k-1's last decoded frame, i.e. Wan's native image-to-video continuation; K = 2: slot 1 stays window k-1's last latent).
In a lock-step abutting composite the slot-0 image latent of a window that does not start the horizon is replaced by the regular latent at
that time (n.slot0_regular, the parent's slot j) before the one-pass decode, since there every slot but the first is a 4-frame slot.

Soft anchors (v44, abutting windows only; relay options, default off = unchanged). ';endsoft=K': the parent anchor on a window's RIGHT end
(its last core frame) is held only for the first K model calls of the level, then released: from call K on, that frame is an ordinary
generated frame (its mask is 0; at the release it is left at the value the step produced, i.e. the clean anchor re-noised with the step's
own noise to the next level -- the SDEdit-style hand-over, so it is never a clean frame fed with a noisy timestep). ';mid=N': N extra parent
anchors inside each window, at the core frames with an integer parent time nearest to fractions q / (N + 1) of the core (the parent's
REGULAR latent frame, as anchor_ends; a mid slot is a 4-frame slot, not Wan's image slot), released after K calls of ';endsoft=K', else
after 2 calls. The window's left end (window 0's slot 0 / the handoff frames) stays hard.

Pixel-space duplicated parent as the child's initial state (v45, relay option ';pixfield', abutting windows, needs ':sigma0' < 1; default off
= unchanged). 'anchor_ends:0.8334@tok;pixfield' ("pixfield:0.8333": the Turbo levels are 1, 0.9375, 0.8333.., 0.625, so sigma0 0.8334 starts
at 0.8333.. = 2 calls; ':0.625' = 1 call) replaces the low-passed latent field by a VALID video latent of a slow-motion copy of the parent:
(1) the parent's own clip is decoded (`node_frames`, the per-window convention: slot 0 -> 1 pixel frame, slot j >= 1 -> 4) and the child's
block [a_k, e_k] (`parent_block`) is cut out as pixel frames [px_of_lat(a_k), px_of_lat(e_k + 1)); (2) it is stretched in TIME by FRAME
DUPLICATION (nearest neighbour, no blending: output frame f <- source frame floor(f n_src / n_out), so every source frame is repeated ~2x at
branch 2; e.g. 61 -> 121 or 60 -> 121 frames) to the child's pixel length 1 + 4 (w - 1); (3) the clip is re-encoded with the backend's VAE
(`B.encode_video`, Wan: the causal 3D VAE, slot 0 a genuine image slot) into a canvas_w-frame latent = child.field_lp, and the window starts
from mix(field_lp, eps, first level <= sigma0) exactly like ':sigma0' (the anchors are then substituted as usual). Handoff windows (ext_l = K):
the clip's first 1 + 4 (K - 1) pixel frames are window k-1's decoded frames of the handed-off slots (K = 1: its last frame, the frame whose image
latent is the handoff anchor), followed by the stretched parent over the core (4 w_core frames); it is built in `run_level_seq` once window k-1
is final. Lock-step abutting windows (compose 'blend', halo 0) start from the parent's pixel frame at px_of_lat(a_k) (= the frame of the
first_slot='image' anchor).
v46: ';pixfield=blend' replaces step (2) by a pixel-space LINEAR time stretch (cross-fade of consecutive parent frames): output frame f =
(1 - alpha) src[floor(s)] + alpha src[floor(s) + 1], s = f (n_src - 1) / (n_out - 1), alpha = s - floor(s) (`blend_index`, `blend_frames`).
';pixfield' = ';pixfield=dup' (v45, byte-identical).
v47: ';rootcolor' (needs ';pixfield'; default off = unchanged) ties every level's colours to the ROOT. Each node carries the ROOT time of
each of its decoded pixel frames (`node_root_time`: the root -> its frame index; a pixfield child -> its lead frames' times from window
k-1, then the parent's root times linearly interpolated at the stretch's source positions). (1) After the stretch and before the
re-encode, every stretched frame's per-channel (RGB) mean and std over H x W are matched to the ROOT's decoded frame at the same root
time (root stats linearly interpolated between the two nearest root frames; `root_color_match`); the handed-off lead frames are left as
they are. (2) The parent's end anchor (a regular latent frame of the parent) is replaced by the same slot of the re-encoded, colour-matched
parent clip (`rootcolor_latent`: every decoded frame of the parent matched to the root, encoded with B.encode_video as one clip, so slot j
stays a regular 4-frame latent, not an image slot). A parent that is the root itself keeps its own latent (it IS the colour reference)."""
import math
from . import config, state
from .config import CFG
from .geometry import parent_span_cols

ANCHOR_KINDS = ("anchor_dense", "anchor_ends", "anchor_stride")                             # the temporal-inpainting relays (video only)
EDGE_SIGMA_DEFAULT = 0.6                                                  # field_edge: default joint level sigma_e when only sigma_c is given
def parse_relay(arm):
    """'field' | 'field:0.9' | 'field_edge:0.9:0.6' | 'rho_early:0.3' | 'lowpass_skip:0.6' | 'rho' | 'none' (+ optional '@cutoff') -> (kind, param).
    For field_edge the param is the CORE start level sigma_c; the joint level sigma_e comes from `edge_sigma(arm)`."""
    if arm is None: return None, None
    arm = arm.partition(";")[0]                                        # ';w=K;stat' options of the anchor relays: relay_opts
    arm, _, _cut = arm.partition("@")                                  # 'field:0.9@inf' -> low-pass cutoff override ('2H', '4H', 'inf' = row means only)
    kind, _, p = arm.partition(":")
    if kind == "anchor_stride": p = (p.split(":") + [""])[1]          # 'anchor_stride:N[:sigma0]': param = sigma0, N = relay_opts()['stride']
    p = p.split(":")[0]; p = float(p) if p else None
    if kind in ANCHOR_KINDS:                                           # 'anchor_dense[:sigma0][@tok|@repaint]': param = sigma0 (None = pure noise at level 1)
        assert _cut in ("", "tok", "repaint"), f"anchor relay mechanism must be @tok or @repaint: {arm}@{_cut}"
        return kind, (None if p is None or p >= 1.0 else p)
    if kind == "rho": kind, p = "rho_early", 0.0                      # full-length projection = rho_early with t_end 0
    if kind in ("rho_early", "rho_soft", "lowpass_skip") and p is None: p = 0.3
    if kind in ("field", "field_bg", "field_edge") and p is None: p = CFG["field_init_sigma"]        # field:0.9 -> start the child at noise_frac 0.9 from the stretched low-passed parent
    assert kind in ("field", "field_bg", "field_edge", "rho_early", "rho_soft", "lowpass_skip", "none"), arm
    return kind, p
def edge_sigma(arm):
    """field_edge:sigma_c:sigma_e -> sigma_e (the joint columns' start level and the level until which they are re-anchored)."""
    parts = (arm or "").partition("@")[0].split(":")
    return float(parts[2]) if len(parts) > 2 and parts[2] else EDGE_SIGMA_DEFAULT

def relay_cutoff(arm):
    """Low-pass cutoff of a field arm: 'H' (default), '2H', '4H' (multiples of the window height) or 'inf' (row means only = rho as init)."""
    base, _, cut = (arm or "").partition(";")[0].partition("@")
    if base.partition(":")[0] in ANCHOR_KINDS: return CFG["lowpass"]  # '@' names the anchor mechanism there, not a cutoff
    return cut or CFG["lowpass"]
def lowpass_sigma_lat(cutoff):
    H, VS = config.height(), config.vae_stride()
    lam = H if cutoff == "H" else (H * float(cutoff[:-1]) if str(cutoff).endswith("H") else float(cutoff)); return math.sqrt(-(lam / VS) ** 2 * math.log(0.1) / (2 * math.pi ** 2))
def child_field(child, source="latent", cutoff=None):
    """Low-passed, stretched parent latent over the child's span (the injective relay). source 'bg_latent': the parent's object-free twin.
    cutoff 'inf': the parent's row means over the span, broadcast across the child's width (the S_b-invariant rho given as the init)."""
    B = state.B; p = child.parent; n0, n1 = parent_span_cols(child); cutoff = cutoff or CFG["lowpass"]; w = child.canvas_w // config.vae_stride()
    if cutoff == "inf": return B.repeat_cols(B.row_mean(getattr(p, source)[..., n0:n1]), w)
    return B.stretch_w(B.lowpass_x(getattr(p, source), lowpass_sigma_lat(cutoff))[..., n0:n1], w)
def prepare_child(child, relay):
    kind, _ = parse_relay(relay); n0, n1 = parent_span_cols(child); seg = child.parent.latent[..., n0:n1]
    child.rho = state.B.row_mean(seg) if kind in ("rho_early", "rho_soft") else None
    cut = relay_cutoff(relay)
    child.field_lp = child_field(child, cutoff=cut) if kind in ("field", "field_edge", "lowpass_skip") else (child_field(child, "bg_latent", cut) if kind == "field_bg" else None)
    child.anchor = None
    if kind in ANCHOR_KINDS:
        _check_anchor_backend(state.B)
        opts = relay_opts(relay); pairs = anchor_select(child, anchor_pairs(child), kind, opts)
        child.anchor = dict(pairs=pairs, mech=anchor_mech(relay), stat=opts["stat"])
        if opts["stat"] is not None: child.anchor["stat_target"] = stat_target(child)
        so = soft_anchor_idx(child, pairs, opts)
        if so is not None: child.anchor["soft"] = so
        if opts["endlow"] is not None:                                  # ';endlow[=s]' (v50): the right-end anchor becomes a low-band constraint (endlow_x0)
            if not abuts() or kind != "anchor_ends": raise NotImplementedError(f"';endlow' is defined for anchor_ends on abutting windows (halo_px 0): {relay}")
            if opts["endsoft"] or opts["first"] or opts["noparent"]: raise ValueError(f"';endlow' needs the hard end anchor (no endsoft / first / noparent): {relay}")
            idx = [i for i, _ in pairs]; K = opts.get("w") or 1; left = set(idx[:K]) if not (CFG.get("compose") == "handoff" and child.ext_l > 0) else set()
            child.anchor["endlow"] = ([i for i in idx[-K:] if i not in left], float(opts["endlow"]))
        if opts["endlowclean"] is not None:                             # ';endlowclean[=s]' (v51): clean end anchor, value refreshed per call (endlowclean_value)
            if not abuts() or kind != "anchor_ends": raise NotImplementedError(f"';endlowclean' is defined for anchor_ends on abutting windows (halo_px 0): {relay}")
            if opts["endsoft"] or opts["first"] or opts["noparent"] or opts["endlow"] is not None:
                raise ValueError(f"';endlowclean' needs the hard end anchor (no endsoft / first / noparent / endlow): {relay}")
            idx = [i for i, _ in pairs]; K = opts.get("w") or 1; left = set(idx[:K]) if not (CFG.get("compose") == "handoff" and child.ext_l > 0) else set()
            ends = [i for i in idx[-K:] if i not in left]; assert ends and min(ends) >= 1, (child.name, ends)
            child.anchor["endlowclean"] = (ends, float(opts["endlowclean"]))
        if opts["rootcolor"]:
            if not opts["pixfield"]: raise ValueError(f"';rootcolor' needs ';pixfield' (it matches the pixfield init): {relay}")
            child.anchor["rootcolor"] = True
        if opts["pixfield"]:                                            # ';pixfield' (v45): the pixel-space duplicated parent, re-encoded (module docstring)
            if prm_of(relay) is None: raise ValueError(f"';pixfield' needs ':sigma0' < 1 (the start level): {relay}")
            child.anchor["pixfield"] = opts["pixfield"]
            if opts["pixlp"]: child.anchor["pixlp"] = opts["pixlp"]                  # 'dup' (v45, default) / 'blend' (v46): the time-stretch mode
            child.field_lp = None if (CFG.get("compose") == "handoff" and child.ext_l > 0) else pix_field(child)   # handoff: built in run_level_seq
        elif prm_of(relay) is not None: child.field_lp = child_field(child, cutoff=cut)       # ':sigma0' < 1: the non-anchor frames start from the field

def level_noise_canvas(nodes, seed):
    """One noise canvas for the whole level; each node crops its own span out of it (so siblings share noise in the overlap).
    The shape is the backend's (`B.noise_shape`): 4D (1, C, H, w) for images, 5D (1, C, H, W, w) for video, always with the HORIZON LAST."""
    VS = config.vae_stride(); s = nodes[0].scale; g0 = min(n.canvas_g0 for n in nodes); g1 = max(n.canvas_g1 for n in nodes)
    return state.B.randn(state.B.noise_shape((g1 - g0) // s // VS), seed * 1000 + nodes[0].depth), g0
def crop_noise(canvas_g0, n):
    VS = config.vae_stride(); canvas, g0 = canvas_g0; c0 = (n.canvas_g0 - g0) // n.scale // VS; c1 = (n.canvas_g1 - g0) // n.scale // VS; return canvas[..., c0:c1]
def blend_pairs(nodes, d):
    OVL = config.overlap_lat(); w = state.B.linspace_w(1.0, 0.0, OVL)
    for a, b in zip(nodes[:-1], nodes[1:]):
        o, wo = OVL, w
        if getattr(a, "ovl_r", None) is not None and a.ovl_r != OVL: o = int(a.ovl_r); wo = state.B.linspace_w(1.0, 0.0, o)     # edge_native 1: wider overlap
        if o <= 0: continue                                            # abutting windows (halo 0): nothing shared, nothing to blend
        A, Bx = d[a.name], d[b.name]; m = wo * A[..., -o:] + (1 - wo) * Bx[..., :o]; A[..., -o:] = m; Bx[..., :o] = m

# ---- field_edge: "anchor the joints, free the core" ----
def edge_profile(n):
    """Per-column weight m in [0, 1] over the node's latent width (horizon last): 1 on every overlap it shares with a sibling (the first /
    last 2*halo columns when it has a halo on that side), a linear ramp 1 -> 0 over one halo width into the core, 0 elsewhere.
    m = 1 <-> sigma(x) = sigma_e, m = 0 <-> sigma(x) = sigma_c."""
    B = state.B; w = n.canvas_w // config.vae_stride(); OVL = config.overlap_lat(); h = OVL // 2
    m = B.linspace_w(0.0, 0.0, w)
    if n.halo_l: m[..., :OVL] = 1.0; m[..., OVL:OVL + h] = B.linspace_w(1.0, 0.0, h + 2)[..., 1:-1]
    if n.halo_r: m[..., w - OVL:] = 1.0; m[..., w - OVL - h:w - OVL] = B.linspace_w(0.0, 1.0, h + 2)[..., 1:-1]
    return m
def edge_anchors(nodes):
    """The joint anchor of every node: its own field, except that on each overlap the two siblings' fields are averaged with the same
    linear ramp as the x0 blend, so both siblings are re-projected onto the SAME target there (their fields can differ in the overlap when
    they come from different parents, e.g. the parents' low-pass edge effects)."""
    anchors = {n.name: n.field_lp * 1.0 for n in nodes}
    if len(nodes) > 1: blend_pairs(nodes, anchors)
    return anchors
def _check_edge_backend(B):
    need = ("mix", "noise_frac", "linspace_w", "lowpass_x", "stretch_w")
    miss = [a for a in need if not callable(getattr(B, a, None))]
    if miss: raise NotImplementedError(f"relay field_edge needs backend operations {need}; {type(B).__name__} lacks {miss}")

# ---- anchor relays: the child as temporal inpainting of its stretched parent (see the module docstring) ----
def prm_of(relay): return parse_relay(relay)[1]
def anchor_mech(relay):
    """'tok' | 'repaint' from the '@' suffix; default 'tok' when the backend conditions per token (B.token_anchors), else 'repaint'."""
    m = (relay or "").partition(";")[0].partition("@")[2]
    if m: return m
    return "tok" if getattr(state.B, "token_anchors", False) else "repaint"
def _check_anchor_backend(B):
    if len(B.noise_shape(1)) != 5 or config.vae_stride() != 1:
        raise NotImplementedError("anchor relays are defined for video backends (horizon = latent frames, VS 1); images are unchanged")
def anchor_pairs(child):
    """[(i, j)]: child latent frame i sits at the same global time as parent latent frame j (see the module docstring for the mapping and
    the core 27 + halo 2 example). Every child frame on the parent's frame lattice inside the parent's canvas, in increasing i."""
    if abuts(): return abut_pairs(child)
    p = child.parent; out = []
    for i in range(child.canvas_w):
        q, r = divmod(child.canvas_g0 + i * child.scale - p.canvas_g0, p.scale)
        if r == 0 and 0 <= q < p.canvas_w: out.append((i, q))
    return out
def abuts():
    """True when the video windows abut (halo_px 0): anchor relays use the block mapping (module docstring)."""
    return int(CFG["halo_px"]) == 0
def parent_block(child):
    """(a_k, e_k): the parent frames child k of b covers under the block mapping (a_k = floor(k P / b + 1/2), e_k = a_{k+1} - 1)."""
    p = child.parent; P = p.canvas_w; b = len(p.children); k = p.children.index(child)
    a = [int(math.floor(q * P / b + 0.5)) for q in (k, k + 1)]; return a[0], a[1] - 1
def parent_time(child, i):
    """Parent frame time (float) of child canvas frame i under the block mapping (core frames; handed-off frames map before a_k)."""
    a, e = parent_block(child); w = child.core_w; c = i - child.core_n0
    return a + c * (e - a) / (w - 1) if w > 1 else float(a)
def abut_pairs(child):
    """Block-mapping anchor pairs [(i, j)]: core frames with an integer parent time (the ends always are)."""
    out = []
    for c in range(child.core_w):
        i = child.core_n0 + c; t = parent_time(child, i); j = int(round(t))
        if abs(t - j) < 1e-9: out.append((i, j))
    return out
def relay_opts(relay):
    """Options of an anchor relay after ';': 'w=K' (anchor_ends / anchor_stride: K anchored frames on each side instead of the boundary
    regions), 'stat' (per-frame statistic matching, weight 1 -> 0 linearly over the model calls) or 'stat=K' (weight 1 for the first K
    calls, then 0), 'first' (abutting windows only: anchor ONLY the window's first core frame -- pure first-frame conditioning, no end
anchor; under compose='handoff' a window k >= 1 then has no parent anchor at all), 'endsoft=K' / 'mid=N' (v44, abutting windows: the
right-end anchor held for the first K calls only / N soft mid-window anchors; module docstring), 'pixfield' (v45: the init field is the
pixel-space duplicated parent re-encoded, `pix_field`; 'pixfield=dup' = 'pixfield', 'pixfield=blend' (v46) = linear pixel blend
stretch), 'rootcolor' (v47: root colour match of the pixfield init and the end anchor), 'noparent' (stage 3 broadcast baseline: NO parent
anchor on any window; with compose='handoff' and no ':sigma0' the level is a plain left-to-right image-handoff chain from noise that never
reads the parent), 'endlow[=s]' (v50, abutting windows: the right-end anchor carries only the parent's spatial LOW band -- see
`endlow_x0`; s = Gaussian sigma in latent px, default ENDLOW_SIGMA), 'endlowclean[=s]' (v51: the right-end anchor stays @tok clean but
its value is refreshed before every model call to LP_s(target) + HP_s(the window's own x0 next to it) -- see `endlowclean_value`).
Also 'stride' = N of 'anchor_stride:N'.
Example: 'anchor_ends:0.95@tok;w=4;stat'."""
    o = dict(w=None, stat=None, stride=None, first=False, endsoft=None, mid=0, pixfield=False, rootcolor=False, noparent=False, endlow=None, endlowclean=None, pixlp=None); parts = (relay or "").split(";"); base = parts[0].partition("@")[0]
    if base.partition(":")[0] == "anchor_stride":
        n = base.split(":")[1] if ":" in base else ""; o["stride"] = int(n) if n else 2; assert o["stride"] >= 1
    for t in parts[1:]:
        k, _, v = t.strip().partition("=")
        if k == "w": o["w"] = int(v); assert o["w"] >= 1
        elif k == "stat": o["stat"] = int(v) if v else "lin"
        elif k == "first": o["first"] = True
        elif k == "endsoft": o["endsoft"] = int(v); assert o["endsoft"] >= 1
        elif k == "mid": o["mid"] = int(v) if v else 1; assert o["mid"] >= 0
        elif k == "pixfield":
            o["pixfield"] = v or "dup"
            if o["pixfield"] not in ("dup", "blend"): raise ValueError(f"';pixfield={v}': the stretch mode is 'dup' or 'blend' ({relay!r})")
        elif k == "rootcolor": o["rootcolor"] = True
        elif k == "pixlp":                                                  # v58: coarse parent init, 'pixlp=S' or 'pixlp=S:T' (see `pix_lowpass`)
            sp, _, tp = v.partition(":"); o["pixlp"] = (float(sp or 0), float(tp or 0)); assert o["pixlp"][0] >= 0 and o["pixlp"][1] >= 0
        elif k == "noparent": o["noparent"] = True
        elif k == "endlow": o["endlow"] = float(v) if v else ENDLOW_SIGMA; assert o["endlow"] > 0
        elif k == "endlowclean": o["endlowclean"] = float(v) if v else ENDLOW_SIGMA; assert o["endlowclean"] > 0
        elif k: raise ValueError(f"unknown anchor relay option {t!r} in {relay!r} (w=K, stat, stat=K, first, endsoft=K, mid=N, pixfield[=dup|blend], rootcolor, noparent, endlow[=s], endlowclean[=s], pixlp=S[:T])")
    return o
def anchor_select(child, pairs, kind, opts=None):
    """anchor_dense: all pairs. anchor_ends: the pairs in the two boundary regions -- the whole sibling overlap on a side with a halo
    (2 * halo frames, or the wider overlap of edge_native 1), else the first / last core frame extended inwards to the nearest anchorable
    frame (edge_native 'out': every anchorable frame outside the horizon plus the first / last core frame). opts['w'] = K: instead the K
    pairs nearest each canvas end. anchor_stride:N: the anchor_ends set plus every pair whose global parent-lattice index
    g // parent.scale is a multiple of N (N = 2 -> every 4th child frame at branch 2; siblings select the same global frames)."""
    opts = opts or dict(w=None, stat=None, stride=None)
    if opts.get("noparent"): return []                                  # ';noparent': no parent anchor at all (stage 3 broadcast baseline)
    if kind == "anchor_dense" or not pairs: return list(pairs)
    from .geometry import ovl_of
    w = child.canvas_w; idx = [i for i, _ in pairs]
    if abuts():                                                        # abutting windows: the first / last core frame (';w=K': K pairs per end)
        K = opts.get("w") or 1; left = not (CFG.get("compose") == "handoff" and child.ext_l > 0)     # handoff: the left end comes from window k-1
        sel = (set(idx[:K]) if left else set()) | (set() if opts.get("first") else set(idx[-K:]))
        if kind == "anchor_stride": sel |= {i for i, j in pairs if j % (opts.get("stride") or 2) == 0}
        if opts.get("mid"): sel |= set(mid_anchor_idx(child, pairs, opts["mid"]))
        return [(i, j) for i, j in pairs if i in sel]
    if opts.get("w"):
        K = opts["w"]; sel = set(idx[:K]) | set(idx[-K:])
    else:
        lo = ovl_of(child, "l") if child.halo_l else next(i for i in idx if i >= child.ext_l) + 1          # left region [0, lo)
        hi = w - ovl_of(child, "r") if child.halo_r else max(i for i in idx if i <= w - 1 - child.ext_r)   # right region [hi, w)
        sel = {i for i in idx if i < lo or i >= hi}
    if kind == "anchor_stride":
        N = opts.get("stride") or 2; p = child.parent
        sel |= {i for i in idx if ((child.canvas_g0 + i * child.scale) // p.scale) % N == 0}
    return [(i, j) for i, j in pairs if i in sel]
def mid_anchor_idx(child, pairs, N):
    """';mid=N' (abutting windows): the canvas indices of the N pairs (integer parent time) nearest to core fractions q / (N + 1), q = 1..N."""
    if not abuts(): raise NotImplementedError("';mid' is defined for abutting windows (halo_px 0)")
    out = []
    for q in range(1, int(N) + 1):
        t = child.core_n0 + q * (child.core_w - 1) / (N + 1.0); out.append(min((i for i, _ in pairs), key=lambda i: (abs(i - t), i)))
    return sorted(set(out))
def soft_anchor_idx(child, pairs, opts):
    """(indices, K): the soft anchors of ';endsoft=K' (the right-end anchor) and ';mid=N' (released after K calls, default 2), or None."""
    if not opts.get("endsoft") and not opts.get("mid"): return None
    if not abuts(): raise NotImplementedError("';endsoft' / ';mid' are defined for abutting windows (halo_px 0)")
    idx = [i for i, _ in pairs]; soft = set(mid_anchor_idx(child, pairs, opts["mid"])) if opts.get("mid") else set()
    if opts.get("endsoft") and idx and not opts.get("first"): soft.add(idx[-1])
    return sorted(soft), int(opts.get("endsoft") or 2)
def _hw_mean(x):
    """Per-frame per-channel mean over H x W of a (1, C, H, W, T) latent (torch or numpy), kept 5D."""
    return x.mean(dim=(2, 3), keepdim=True) if hasattr(x, "dim") else x.mean(axis=(2, 3), keepdims=True)
def stat_target(child):
    """The parent's per-frame per-channel latent mean (over H x W), linearly interpolated in time at every child frame's global time
    (clamped to the parent's canvas): the S_b-invariant statistic the ';stat' option matches on the child's non-anchor frames."""
    p = child.parent; P = _hw_mean(p.latent); cols = []
    for i in range(child.canvas_w):
        if abuts():
            q = min(max(parent_time(child, i), 0.0), p.canvas_w - 1.0); a = int(math.floor(q)); f = q - a
            b = min(a + 1, p.canvas_w - 1); cols.append(P[..., a:a + 1] * (1 - f) + P[..., b:b + 1] * f if f > 0 else P[..., a:a + 1] * 1.0); continue
        q = min(max((child.canvas_g0 + i * child.scale - p.canvas_g0) / p.scale, 0.0), p.canvas_w - 1.0); a = int(math.floor(q)); f = q - a
        b = min(a + 1, p.canvas_w - 1); cols.append(P[..., a:a + 1] * (1 - f) + P[..., b:b + 1] * f if f > 0 else P[..., a:a + 1] * 1.0)
    return state.B.cat_w(cols) if callable(getattr(state.B, "cat_w", None)) else (__import__("torch").cat(cols, -1) if hasattr(cols[0], "dim") else __import__("numpy").concatenate(cols, -1))
def stat_weight(stat, k, n_calls):
    """Weight of the statistic matching at the k-th model call (0-based) of a level with n_calls calls: 'lin' 1 -> 0 linearly, int K: 1 for k < K."""
    if stat is None: return 0.0
    if stat == "lin": return 1.0 if n_calls <= 1 else 1.0 - k / (n_calls - 1)
    return 1.0 if k < int(stat) else 0.0
def level_anchors(nodes):
    """{name: (mask, anchor_x)} for one level: mask (1, 1, 1, 1, w) is 1 on the node's anchored frames; anchor_x holds the parent's clean
    frames there (0 elsewhere). Values at the same global time are averaged over every node of the level that anchors it (identical
    siblings' overlap anchors by construction, also when the two siblings have different parents)."""
    B = state.B; acc = {}
    for n in nodes:
        for i, j in n.anchor["pairs"]:
            g = n.canvas_g0 + i * n.scale
            v = rootcolor_latent(n.parent)[..., j:j + 1] if (n.anchor.get("rootcolor") and n.parent.depth > 0) else n.parent.latent[..., j:j + 1]
            s_, c_ = acc.get(g, (0, 0)); acc[g] = (s_ + v, c_ + 1)
    out = {}
    for n in nodes:
        w = n.canvas_w; m = B.linspace_w(0.0, 0.0, w); ax = B.zeros(B.noise_shape(w))
        for i, _ in n.anchor["pairs"]:
            s_, c_ = acc[n.canvas_g0 + i * n.scale]; ax[..., i:i + 1] = s_ / c_ if c_ > 1 else s_; m[..., i] = 1.0
        if CFG.get("first_slot", "latent") == "image":                # slot 0 anchored with an IMAGE latent (module docstring)
            for i, j in n.anchor["pairs"]:
                if i != 0: continue
                p = n.parent
                if j == 0: ax[..., 0:1] = p.latent[..., 0:1] * 1.0; n.slot0_regular = getattr(p, "slot0_regular", None)
                else:
                    from .video import px_of_lat
                    ax[..., 0:1] = B.encode_image(node_frames(p)[px_of_lat(j)]); n.slot0_regular = p.latent[..., j:j + 1] * 1.0
        for i, v in n.anchor.get("fixed") or ():                      # handoff: window k-1's final frames at the same global time
            ax[..., i:i + 1] = v; m[..., i] = 1.0
        if n.anchor.get("endlow"):                                     # ';endlow' (v50): the end frames leave the hard mask (sampled, not clean); ax keeps the target
            ml = B.linspace_w(0.0, 0.0, w)
            for i in n.anchor["endlow"][0]: ml[..., i] = 1.0; m[..., i] = 0.0
            n.anchor["endlow_mask"] = ml
        if n.anchor.get("soft"):                                       # ';endsoft' / ';mid': the mask after the release (module docstring)
            ml = m * 1.0; fx = {i for i, _ in (n.anchor.get("fixed") or ())}
            for i in n.anchor["soft"][0]:
                if i not in fx: ml[..., i] = 0.0
            n.anchor["m_late"] = ml
        out[n.name] = (m, ax)
    return out
def node_frames(n):
    """The decoded pixel frames of a node's own latent (its own clip: slot 0 -> one frame), cached on the node (the root's decode is reused)."""
    if getattr(n, "depth", None) == 0 and getattr(n, "frames", None): return n.frames
    c = getattr(n, "_px_cache", None)
    if c is None or c[0] is not n.latent: c = (n.latent, state.B.decode_frames(n.latent)); n._px_cache = c
    return c[1]
def dup_index(n_src, n_out):
    """';pixfield': nearest-neighbour time stretch by frame duplication -- output frame f takes source frame floor(f n_src / n_out)."""
    return [min(int(n_src) - 1, (f * int(n_src)) // int(n_out)) for f in range(int(n_out))]
def blend_index(n_src, n_out):
    """';pixfield=blend' (v46): linear time stretch -- output frame f takes (1 - alpha) src[i] + alpha src[i + 1] with s = f (n_src - 1) / (n_out - 1),
    i = floor(s), alpha = s - i (first and last output frames = first and last source frames). Returns [(i, alpha)]."""
    n_src, n_out = int(n_src), int(n_out); out = []
    for f in range(n_out):
        s = f * (n_src - 1) / max(n_out - 1, 1); i = min(int(math.floor(s)), n_src - 1); out.append((i, float(s - i) if i < n_src - 1 else 0.0))
    return out
def blend_frames(src, n_out):
    """';pixfield=blend': the PIL frames src stretched to n_out frames by `blend_index` (pixel-space cross-fade, rounded to uint8; alpha 0 = the
    source frame itself)."""
    import numpy as np
    from PIL import Image
    arr = {}; A = lambda i: arr.setdefault(i, np.asarray(src[i].convert("RGB"), dtype=np.float32)); out = []
    for i, al in blend_index(len(src), n_out):
        if al == 0.0: out.append(src[i]); continue
        out.append(Image.fromarray(np.clip(np.rint((1.0 - al) * A(i) + al * A(i + 1)), 0, 255).astype(np.uint8)))
    return out
def pix_field(child, lead=None):
    """';pixfield' (module docstring): the parent's decoded pixel frames over the child's block, stretched by frame duplication to the
    child's pixel length, prefixed by `lead` (handoff: window k-1's decoded frames of the K handed-off slots), re-encoded with B.encode_video
    into a (1, C, H, W, canvas_w) latent. Records child.pixfield_info (source pixel range, lengths, duplication map)."""
    if not abuts(): raise NotImplementedError("';pixfield' is defined for abutting windows (halo_px 0): it needs the block mapping")
    from .video import px_of_lat, n_px_frames
    B = state.B; a, e = parent_block(child); fr = node_frames(child.parent); p0, p1 = px_of_lat(a), min(px_of_lat(e + 1), len(fr)); src = fr[p0:p1]
    K = int(child.ext_l or 0)
    if K and lead is None: raise ValueError(f"{child.name}: a handoff window's pixfield needs the handed-off lead frames")
    lead = list(lead or []); assert len(lead) == (1 + 4 * (K - 1) if K else 0), (child.name, len(lead), K)
    n_out = n_px_frames(child.canvas_w) - len(lead); mode = (child.anchor or {}).get("pixfield") or "dup"
    if mode == "blend": bi = blend_index(len(src), n_out); idx = [i for i, _ in bi]; frames = lead + blend_frames(src, n_out)
    else: idx = dup_index(len(src), n_out); frames = lead + [src[i] for i in idx]
    if (child.anchor or {}).get("rootcolor"):                           # ';rootcolor' (v47): match the stretched frames to the root's colours
        pt = node_root_time(child.parent)[p0:p1]; pos = [i + al for i, al in bi] if mode == "blend" else [float(i) for i in idx]
        tt = [_lerp_list(pt, s_) for s_ in pos]; lt = list(getattr(child, "_lead_root_time", None) or [])
        assert len(lt) == len(lead), (child.name, len(lt), len(lead))
        frames = lead + root_color_match(frames[len(lead):], tt, root_of(child)); child.root_time = lt + tt
    assert len(frames) == n_px_frames(child.canvas_w), (child.name, len(frames), child.canvas_w)
    if (child.anchor or {}).get("pixlp"): frames = frames[:len(lead)] + pix_lowpass(frames[len(lead):], *child.anchor["pixlp"])
    z = B.encode_video(frames); assert z.shape[-1] == child.canvas_w, (child.name, z.shape, child.canvas_w)
    child.pixfield_info = dict(parent=child.parent.name, block=(a, e), src_px=(p0, p1), n_src=len(src), n_lead=len(lead), n_out=n_out, dup=idx, mode=mode,
                               **({"alpha": [round(x, 4) for _, x in bi]} if mode == "blend" else {}))
    return z
def pix_lowpass(frames, s_px, t_fr):
    """v58 ';pixlp=S[:T]': the COARSE parent -- the stretched parent frames (after the colour match, before the re-encode) blurred with a
    spatial Gaussian of sigma S px (per frame) and a temporal Gaussian of sigma T frames (reflect padding); the handed-off lead frames are
    left sharp. With ':sigma0' the child then starts from mix(coarse field, eps, sigma0): the parent fixes the layout and the colours, the
    child re-draws detail and motion under its own routed condition. Default off (no ';pixlp')."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    X = np.stack([np.asarray(f.convert("RGB"), np.float32) for f in frames])
    if s_px > 0: X = gaussian_filter(X, sigma=(0, s_px, s_px, 0), mode="reflect")
    if t_fr > 0: X = gaussian_filter1d(X, t_fr, axis=0, mode="reflect")
    return [Image.fromarray(np.clip(np.rint(x), 0, 255).astype(np.uint8)) for x in X]
def _lerp_list(v, s):
    s = min(max(float(s), 0.0), len(v) - 1.0); a = int(math.floor(s)); f = s - a
    return float(v[a]) if f <= 0 or a + 1 >= len(v) else float(v[a]) * (1 - f) + float(v[a + 1]) * f
def root_of(n):
    while n.parent is not None: n = n.parent
    return n
def node_root_time(n):
    """';rootcolor': the root time (float, in root pixel frames) of each decoded pixel frame of node n (module docstring, v47)."""
    if n.parent is None or n.depth == 0: return [float(t) for t in range(len(node_frames(n)))]
    rt = getattr(n, "root_time", None)
    if rt is None: raise ValueError(f"{n.name}: no root time (';rootcolor' needs every level built with it)")
    return rt
def root_color_stats(root):
    """Per root frame per-channel RGB mean and std over H x W (cached on the root node): two (T, 3) float64 arrays."""
    import numpy as np
    c = getattr(root, "_rc_stats", None)
    if c is None or c[0] is not root.latent:
        A = [np.asarray(f.convert("RGB"), dtype=np.float64).reshape(-1, 3) for f in node_frames(root)]
        c = (root.latent, np.stack([a.mean(0) for a in A]), np.stack([a.std(0) for a in A])); root._rc_stats = c
    return c[1], c[2]
def root_color_match(frames, times, root):
    """';rootcolor': each PIL frame's per-channel RGB mean / std over H x W set to the root's at its root time (linear in time between root
    frames); rounded to uint8."""
    import numpy as np
    from PIL import Image
    mu, sd = root_color_stats(root); out = []
    for f, t in zip(frames, times):
        x = np.asarray(f.convert("RGB"), dtype=np.float32); m = x.reshape(-1, 3).mean(0); s = x.reshape(-1, 3).std(0)
        tm = np.array([_lerp_list(mu[:, c], t) for c in range(3)], np.float32); ts = np.array([_lerp_list(sd[:, c], t) for c in range(3)], np.float32)
        out.append(Image.fromarray(np.clip(np.rint((x - m) / np.maximum(s, 1e-3) * ts + tm), 0, 255).astype(np.uint8)))
    return out
def rootcolor_latent(p):
    """';rootcolor': the parent's decoded clip colour-matched to the root at its root times and re-encoded as ONE clip (B.encode_video), so
    slot j is a regular latent frame; cached on the node."""
    c = getattr(p, "_rc_latent", None)
    if c is None or c[0] is not p.latent:
        fr = node_frames(p); rt = node_root_time(p); assert len(rt) == len(fr), (p.name, len(rt), len(fr))
        z = state.B.encode_video(root_color_match(fr, rt, root_of(p))); assert z.shape[-1] == p.canvas_w, (p.name, z.shape, p.canvas_w)
        c = (p.latent, z); p._rc_latent = c
    return c[1]
ENDLOW_SIGMA = 4.0         # ';endlow' default: spatial Gaussian sigma in LATENT pixels (Wan 2.2 VAE stride 16 -> 64 px)
def _refl_idx(n, r):
    """Indices of a length-n axis padded by r on each side with 'reflect' (no edge repeat), valid for any r."""
    if n == 1: return [0] * (n + 2 * r)
    P = 2 * (n - 1); out = []
    for t in range(-r, n + r):
        u = abs(t) % P; out.append(P - u if u > n - 1 else u)
    return out
def spatial_lowpass(x, s):
    """Separable Gaussian blur (sigma s latent px, reflect padding, truncated at 3 sigma) over the two SPATIAL axes (2, 3) of a
    (1, C, H, W, T) latent; numpy or torch. The temporal axis (last) is untouched."""
    r = int(math.ceil(3 * s)); g = [math.exp(-0.5 * (t / s) ** 2) for t in range(-r, r + 1)]; z = sum(g); g = [v / z for v in g]
    th = hasattr(x, "dim")
    for ax in (2, 3):
        y = x.movedim(ax, -1) if th else __import__("numpy").moveaxis(x, ax, -1); n = y.shape[-1]; yp = y[..., _refl_idx(n, r)]
        out = sum(gi * yp[..., i:i + n] for i, gi in enumerate(g)); x = out.movedim(-1, ax) if th else __import__("numpy").moveaxis(out, -1, ax)
    return x
def endlow_x0(n, x0, ax):
    """';endlow[=s]' (v50) -- the right-end anchor as a LOW-BAND constraint. The hard end anchor (@tok: the frame clean = the parent) copies the
    parent's CONTENT, so an event the parent lacks is erased at every window end (v49). Here the end frame(s) are NOT token-anchored: they are
    sampled like any other frame (noisy at the level's sigma, the level's timestep) and after every model call their x0 estimate is replaced by
        x0_end <- LP_s(target) + (x0_end - LP_s(x0_end))
    i.e. the window keeps its own high band (objects, edges) and takes the parent's spatial low band (colour, exposure, layout of large areas).
    target = the usual end-anchor value (the parent's latent slot, root-colour matched under ';rootcolor'); LP_s = `spatial_lowpass`, sigma s
    latent px. The '@tok' mechanism cannot express this literally (a token-anchored frame is clean and has no x0 of its own), so this is the
    closest faithful version: the anchor frames stay noisy and only their low band is replaced at every anchored step. On the last step
    (sigma 0) the output end frame = LP_s(target) + HP_s(own x0). Left / handoff anchors are unchanged (still @tok clean)."""
    el = n.anchor.get("endlow_mask") if n.anchor else None
    if el is None: return x0
    s = n.anchor["endlow"][1]; return x0 + el * (spatial_lowpass(ax, s) - spatial_lowpass(x0, s))
def endlowclean_value(ax, x0_prev, ends, s):
    """';endlowclean[=s]' (v51) -- the right-end anchor frame(s) e stay CLEAN @tok anchors (timestep 0, in the hard mask, as in FINAL), so the
    clean token keeps holding the window's exposure (v50: without it the interior drifts 4-6 grey darker), but their VALUE is refreshed
    before every model call after the first:
        ax[e] <- LP_s(target_e) + (y - LP_s(y)),   y = x0_prev[e - 1]
    target_e = the usual end-anchor value (the parent's slot, root-colour matched under ';rootcolor'); LP_s = `spatial_lowpass`, sigma s latent px.
    y is the window's own x0 estimate from the PREVIOUS call at the nearest SAMPLED frame (e - 1), not at e itself: under @tok the anchor frame
    has sigma 0, so the backend's x0 there IS the anchor value (backend_wan.denoise returns m * anchor_x; the model's velocity at a timestep-0
    token is not an estimate), and "HP of the window's own x0 at e" would be HP(target) -- a no-op that reproduces FINAL exactly. Frame e - 1 is
    the window's latest estimate of its own content at the window end (4 pixel frames earlier). First call: ax[e] = target itself (the pixfield
    init's slot e is the parent's end frame re-encoded, i.e. ~ the target, so the two choices nearly coincide; the target keeps call 1 = FINAL).
    The output end frame is the value used on the LAST call (the rest of the window was generated conditioned on it). Returns a new ax."""
    tgt = ax["target"]; out = ax["ax"] * 1.0
    for k, e in enumerate(ends):
        y = x0_prev[..., e - 1:e]; out[..., e:e + 1] = spatial_lowpass(tgt[k], s) + (y - spatial_lowpass(y, s))
    return out
def mask_at(n, m, k):
    """The anchor mask of node n at its k-th model call (0-based) of the level: m, or m_late from call soft K on (';endsoft' / ';mid')."""
    so = n.anchor.get("soft") if n.anchor else None
    return m if not so or k < so[1] else n.anchor["m_late"]
def anchor_input(mech, x, m, ax, eps0, lvl):
    """The model input with the anchors substituted: tok -> clean anchor; repaint -> mix(anchor, eps0, lvl)."""
    return (1 - m) * x + m * (ax if mech == "tok" else state.B.mix(ax, eps0, lvl))

DPM_H_RATIO_MAX = 2.0     # 2M correction only if h <= 2 h_prev: it extrapolates x0 linearly in lambda over h from a secant of length h_prev, which
                          # overshoots on a much longer step. Triggers only on the SD3/FLUX final step into sigma ~ 0.003 (h/h_prev ~ 5); never on Wan (<= 1.63).
def _lam(a, s):
    """log(alpha / sigma), +inf at the clean end (sigma 0), -inf at pure noise (alpha 0)."""
    if s <= 0: return math.inf
    if a <= 0: return -math.inf
    return math.log(a) - math.log(s)
def dpmpp2m_step(B, x, x0, lvl, lvl_n, prev):
    """One DPM-Solver++(2M) step (data prediction) from `lvl` to `lvl_n` for one node. `x0` is the node's blended x0_hat at `lvl`; `prev` is
    its history (x0 of the previous step, h of the previous step) or None. Returns (x at lvl_n, new history). See the module docstring."""
    a, s = B.alpha_sigma(lvl); a_n, s_n = B.alpha_sigma(lvl_n)
    if s_n <= 0: return x0, None                                          # final step to the clean level: first order, x' = x0
    h = _lam(a_n, s_n) - _lam(a, s); D = x0
    if prev is not None:
        x0_p, h_p = prev
        if math.isfinite(h_p) and math.isfinite(h) and h_p > 0 and 0 < h <= DPM_H_RATIO_MAX * h_p:
            c = 0.5 * h / h_p; D = (1 + c) * x0 - c * x0_p                 # 1/(2r) with r = h_prev / h
    return (s_n / s) * x + (a_n - a * s_n / s) * D, (x0, h)

# ---- UniPC (Zhao et al. 2023), B(h) variant, data prediction: the predictor-corrector of diffusers' UniPCMultistepScheduler ----
UNIPC_DEFAULTS = dict(order=2, solver_type="bh2", lower_order_final=True, disable_corrector=())   # = Wan 2.2's scheduler config
def unipc_options(B):
    """UniPC options: the backend's own scheduler config if it IS a UniPC config (Wan 2.2: solver_order 2, bh2, lower_order_final, no
    disabled corrector steps), else UNIPC_DEFAULTS; then CFG["unipc"] (optional dict with the same keys) overrides. Reading the config is a
    soft lookup (getattr), not a backend requirement. solver_type other than bh1/bh2 becomes bh2, as diffusers does. predict_x0 is always
    True here (the x0 blend needs the data-prediction form); prediction_type is irrelevant, the backend's denoise already returns x0_hat."""
    o = dict(UNIPC_DEFAULTS)
    sc = getattr(getattr(getattr(B, "pipe", None), "scheduler", None), "config", None)
    if sc is not None and "solver_order" in sc and "disable_corrector" in sc:
        o.update(order=int(sc["solver_order"]), solver_type=sc.get("solver_type", "bh2"), lower_order_final=bool(sc.get("lower_order_final", True)),
                 disable_corrector=tuple(sc.get("disable_corrector") or ()))
    o.update(CFG.get("unipc") or {})
    if o["solver_type"] not in ("bh1", "bh2"): o["solver_type"] = "bh2"
    o["disable_corrector"] = tuple(o["disable_corrector"] or ()); assert int(o["order"]) >= 1; o["order"] = int(o["order"]); return o

def _unipc_coeffs(h, rks, solver_type, corrector):
    """(h_phi_1, B_h, rhos) of diffusers' multistep_uni_{p,c}_bh_update for predict_x0 (hh = -h), in float64. `rks` are the ratios
    (lambda_si - lambda_s0) / h of the history terms (without the trailing 1). Predictor order 2 and corrector order 1 use the fixed 0.5."""
    import numpy as np
    hh = -h; h_phi_1 = math.expm1(hh); h_phi_k = h_phi_1 / hh - 1; fac = 1
    B_h = hh if solver_type == "bh1" else math.expm1(hh)
    order = len(rks) + 1; r = list(rks) + [1.0]; R, b = [], []
    for i in range(1, order + 1):
        R.append([rk ** (i - 1) for rk in r]); b.append(h_phi_k * fac / B_h); fac *= i + 1; h_phi_k = h_phi_k / hh - 1 / fac
    R, b = np.array(R, np.float64), np.array(b, np.float64)
    if corrector: rhos = [0.5] if order == 1 else list(np.linalg.solve(R, b))
    else: rhos = [] if order == 1 else ([0.5] if order == 2 else list(np.linalg.solve(R[:-1, :-1], b[:-1])))
    return h_phi_1, B_h, [float(v) for v in rhos]

class UniPCNode:
    """Per-node UniPC state: the last `order` BLENDED x0's with their lambda (the model-output history), the sample before the last
    predictor (`last`), the predictor's output (`x_pred`, to detect an x edited after the predictor) and the order of the last predictor."""
    __slots__ = ("hist", "last", "x_pred", "order")
    def __init__(self): self.hist, self.last, self.x_pred, self.order = [], None, None, 0

def unipc_step(B, st, x, x0, levels, i, opt):
    """One UniPC step of one node at level index i (levels[i] -> levels[i+1]), exactly diffusers' `UniPCMultistepScheduler.step` order:
    (1) x0 = the node's blended x0_hat at levels[i] (= convert_model_output with predict_x0); (2) UniC corrector: recompute the sample at
    levels[i] from the sample before the last predictor (`st.last`) with the NEW x0 and the history (order = the last predictor's order),
    skipped on the first step of a level and when i-1 is in disable_corrector; (3) push x0 into the history; (4) UniP predictor of order
    min(order, steps done in this level + 1), and min(order, steps left) if lower_order_final. Returns x at levels[i+1] (st updated in place).
    Deviations from diffusers, all at infinite lambda only (alpha = 0, i.e. sigma = 1 on flow schedules that start at exactly 1, like SD3's; or
    sigma = 0 at the end): a history term with an infinite lambda is dropped (diffusers gives 0 in the predictor and NaN in an order-2
    corrector), the step into sigma = 0 is order 1 even without lower_order_final (diffusers: NaN) and a bh1 corrector across an infinite h
    is skipped (B_h = -inf). If x was edited after the predictor (field_edge re-anchoring), the corrector is applied as an INCREMENT on the
    edited x: x <- x + (C - P), with C the corrected and P the predicted sample (both computed from `st.last`; UniC is linear in the sample,
    so this is "the corrector on the modified x"); on an unedited x this is exactly C."""
    lam = lambda l: _lam(*B.alpha_sigma(l)); a_t, s_t = B.alpha_sigma(levels[i]); l_t = lam(levels[i]); N = len(levels) - 1
    if st.last is not None and (i - 1) not in opt["disable_corrector"]:
        m0, l_s0, a_s0, s_s0 = st.hist[-1]; h = l_t - l_s0; terms = [e for e in st.hist[-st.order:-1][::-1] if math.isfinite(e[1])]
        if not math.isfinite(h): terms = []
        if math.isfinite(h) or opt["solver_type"] == "bh2":
            rks = [(e[1] - l_s0) / h for e in terms]; h_phi_1, B_h, rhos = _unipc_coeffs(h, rks, opt["solver_type"], True)
            corr = rhos[-1] * (x0 - m0)
            for rk, e, rho in zip(rks, terms, rhos[:-1]): corr = corr + rho * ((e[0] - m0) / rk)
            xc = (s_t / s_s0) * st.last - a_t * h_phi_1 * m0 - a_t * B_h * corr
            x = xc if x is st.x_pred else xc + (x - st.x_pred)
    st.hist = (st.hist + [(x0, l_t, a_t, s_t)])[-opt["order"]:]
    order = min(opt["order"], N - i) if opt["lower_order_final"] else opt["order"]; order = min(order, len(st.hist))
    a_n, s_n = B.alpha_sigma(levels[i + 1]); l_n = lam(levels[i + 1]); h = l_n - l_t
    terms = [e for e in st.hist[-order:-1][::-1] if math.isfinite(e[1])] if math.isfinite(h) else []
    st.order = len(terms) + 1 if math.isfinite(h) else 1
    rks = [(e[1] - l_t) / h for e in terms]; h_phi_1, B_h, rhos = _unipc_coeffs(h, rks, opt["solver_type"], False)
    xp = (s_n / s_t) * x - a_n * h_phi_1 * x0
    if terms:
        pred = 0
        for rk, e, rho in zip(rks, terms, rhos): pred = pred + rho * ((e[0] - x0) / rk)
        xp = xp - a_n * B_h * pred
    st.last, st.x_pred = x, xp; return xp

CONSISTENCY_SEED_STRIDE = 7919     # fresh-noise seed of step i = seed + stride * (i + 1): disjoint from the initial canvas (seed) of every level

def c_sched(t, t_end): return max(0.0, (t - t_end) / (1.0 - t_end)) if t_end < 1 else 0.0      # 1 at t=1, 0 at t=t_end and after

def run_level(nodes, levels, cfg, seed, relay, bg=False, noise_nodes=None):
    """All nodes of one level of the tree in lockstep, down the backend's noise levels. relay None for the root. bg=True: the object-free
    pass (uses n.bg_cond, writes n.bg_latent; same noise canvas as the object pass, so the two differ only by conditioning).
    noise_nodes: the nodes whose span defines the level's noise canvases (default `nodes`; run_level_seq passes the whole level so a window
    sampled alone draws exactly the noise it would draw in lock-step)."""
    B = state.B; kind, prm = parse_relay(relay); noise_nodes = noise_nodes or nodes; canvas = level_noise_canvas(noise_nodes, seed); cond_of = (lambda n: n.bg_cond) if bg else (lambda n: n.cond)
    edge = None
    if kind in ("field", "field_bg"):
        start = next(i for i, l in enumerate(levels) if B.noise_frac(l) <= prm); l0 = levels[start]
        xs = {n.name: B.mix(n.field_lp, crop_noise(canvas, n), l0) for n in nodes}
    elif kind == "field_edge":
        # Spatially varying start: x = mix(field, eps, sigma(x)), sigma = sigma_e on the joints, ramping to sigma_c over one halo into the core.
        # sigma(x) is realised as the column-wise interpolation of two mixes (exact for flow matching, where mix is linear in sigma; a
        # first-order approximation on VP backends). The window itself is sampled from the level of sigma_c.
        _check_edge_backend(B); s_e = edge_sigma(relay)
        start = next(i for i, l in enumerate(levels) if B.noise_frac(l) <= prm); l0 = levels[start]
        l_e = levels[next(i for i, l in enumerate(levels) if B.noise_frac(l) <= s_e)]
        anchors = edge_anchors(nodes); eps0 = {n.name: crop_noise(canvas, n) for n in nodes}; prof = {n.name: edge_profile(n) for n in nodes}
        xs = {n.name: (1 - prof[n.name]) * B.mix(n.field_lp, eps0[n.name], l0) + prof[n.name] * B.mix(anchors[n.name], eps0[n.name], l_e) for n in nodes}
        edge = (s_e, anchors, eps0, prof)
    elif kind in ANCHOR_KINDS:
        anc = level_anchors(nodes); eps0 = {n.name: crop_noise(canvas, n) for n in nodes}; mech = nodes[0].anchor["mech"]
        elc = {n.name: dict(ax=anc[n.name][1], target=[anc[n.name][1][..., e:e + 1] * 1.0 for e in n.anchor["endlowclean"][0]], prev=None)
               for n in nodes if n.anchor.get("endlowclean")}                # ';endlowclean' (v51): per-node target + the previous call's x0
        if elc and mech != "tok": raise NotImplementedError("';endlowclean' needs the @tok mechanism (a clean end token)")
        if mech == "tok" and not getattr(B, "token_anchors", False): raise NotImplementedError(f"@tok needs a backend with per-token timesteps (token_anchors); {type(B).__name__} has none -- use @repaint")
        if prm is None: start = 0; xs = {n.name: B.init_latent(eps0[n.name], levels[0]) for n in nodes}
        else:
            start = next(i for i, l in enumerate(levels) if B.noise_frac(l) <= prm)
            xs = {n.name: B.mix(n.field_lp, eps0[n.name], levels[start]) for n in nodes}
        xs = {n.name: anchor_input(mech, xs[n.name], *anc[n.name], eps0[n.name], levels[start]) for n in nodes}
        anchor = (mech, anc, eps0)
    else:
        start = 0; xs = {n.name: B.init_latent(crop_noise(canvas, n), levels[0]) for n in nodes}
    if kind not in ANCHOR_KINDS: anchor = None
    stream = None
    if CFG.get("root_noise", "harness") == "pipeline" and kind is None and not bg and len(nodes) == 1 and nodes[0].depth == 0:
        # the root drawn exactly as the stand-alone pipeline draws it (config: root_noise); children keep the level canvases
        stream = B.noise_stream(seed, nodes[0].canvas_w // config.vae_stride()); xs = {nodes[0].name: B.init_latent(stream(), levels[0])}
    solver = CFG.get("solver", "euler"); hist = {}
    if solver not in ("euler", "dpmpp2m", "unipc", "consistency"): raise ValueError(f"CFG['solver'] must be 'euler', 'dpmpp2m', 'unipc' or 'consistency', got {solver!r}")
    if solver in ("dpmpp2m", "unipc") and not callable(getattr(B, "alpha_sigma", None)): raise NotImplementedError(f"solver {solver} needs B.alpha_sigma; {type(B).__name__} lacks it")
    if solver == "unipc": uopt = unipc_options(B); hist = {n.name: UniPCNode() for n in nodes}
    sig_child = CFG["branch"] * lowpass_sigma_lat(relay_cutoff(relay) if relay_cutoff(relay) != "inf" else CFG["lowpass"]); n_nodes = len(nodes)
    for i in range(start, len(levels) - 1):
        lvl, lvl_n = levels[i], levels[i + 1]; s = B.noise_frac(lvl); x0s, epss = {}, {}
        for n in nodes:
            if anchor is None: x0, eps = B.denoise(xs[n.name], lvl, cond_of(n), cfg)
            else:
                mech, anc, eps0 = anchor; m, ax = anc[n.name]
                if n.name in elc and elc[n.name]["prev"] is not None:        # ';endlowclean' (v51): refresh the clean end value from the last call's x0
                    ax = endlowclean_value(elc[n.name], elc[n.name]["prev"], n.anchor["endlowclean"][0], n.anchor["endlowclean"][1]); anc[n.name] = (anc[n.name][0], ax)
                m = mask_at(n, m, i - start); xs[n.name] = anchor_input(mech, xs[n.name], m, ax, eps0[n.name], lvl)
                x0, eps = (B.denoise(xs[n.name], lvl, cond_of(n), cfg, anchor_mask=m, anchor_x=ax) if mech == "tok" else B.denoise(xs[n.name], lvl, cond_of(n), cfg))
                x0 = (1 - m) * x0 + m * ax                                   # the equality constraint on the estimate
                x0 = endlow_x0(n, x0, ax)                                    # ';endlow' (v50): low band of the end frames -> the anchor's
                sw = stat_weight(n.anchor.get("stat"), i - start, len(levels) - 1 - start)
                if sw > 0: x0 = x0 + sw * (1 - m) * (n.anchor["stat_target"] - _hw_mean(x0))   # ';stat': per-frame channel means -> the parent's (non-anchor frames)
                if n.name in elc: elc[n.name]["prev"] = x0
            epss[n.name] = eps
            if kind == "rho_early" and s >= prm: x0 = x0 - B.row_mean(x0) + n.rho                       # hard: row means from the parent while noise_frac >= prm
            elif kind == "rho_soft": x0 = x0 + c_sched(s, prm) * (n.rho - B.row_mean(x0))               # soft: blend toward the parent's row means, weight 1 at t=1 -> 0 at noise_frac=prm
            elif kind == "lowpass_skip": x0 = x0 + c_sched(s, prm) * (n.field_lp - B.lowpass_x(x0, sig_child))   # low band pulled to the stretched parent, decaying to 0 at t=prm
            x0s[n.name] = x0
        if CFG["sync"] == "x0" and n_nodes > 1: blend_pairs(nodes, x0s)
        if solver == "euler":
            for n in nodes: xs[n.name] = B.mix(x0s[n.name], epss[n.name], lvl_n)
        elif solver == "consistency":
            if B.noise_frac(lvl_n) <= 0:
                for n in nodes: xs[n.name] = x0s[n.name]
            elif stream is not None:
                for n in nodes: xs[n.name] = B.mix(x0s[n.name], stream(), lvl_n)
            else:
                fresh = level_noise_canvas(noise_nodes, seed + CONSISTENCY_SEED_STRIDE * (i + 1))
                for n in nodes: xs[n.name] = B.mix(x0s[n.name], crop_noise(fresh, n), lvl_n)
        elif solver == "dpmpp2m":
            for n in nodes: xs[n.name], hist[n.name] = dpmpp2m_step(B, xs[n.name], x0s[n.name], lvl, lvl_n, hist.get(n.name))
        else:
            for n in nodes: xs[n.name] = unipc_step(B, hist[n.name], xs[n.name], x0s[n.name], levels, i, uopt)
        if anchor is not None:                                            # anchors reset after the step (tok: clean; repaint: at the next level)
            mech, anc, eps0 = anchor
            for n in nodes: m, ax = anc[n.name]; xs[n.name] = anchor_input(mech, xs[n.name], mask_at(n, m, i + 1 - start), ax, eps0[n.name], lvl_n)
        if edge is not None and B.noise_frac(lvl_n) > edge[0]:            # RePaint-style: joints re-projected onto the anchor at the CURRENT level, same eps
            s_e, anchors, eps0, prof = edge
            for n in nodes: m = prof[n.name]; xs[n.name] = (1 - m) * xs[n.name] + m * B.mix(anchors[n.name], eps0[n.name], lvl_n)
    for n in nodes:
        if bg: n.bg_latent = xs[n.name]
        else: n.latent = xs[n.name]

def run_level_seq(nodes, levels, cfg, seed, relay, bg=False):
    """compose='handoff': the windows of one level sampled LEFT TO RIGHT (module docstring). Window k >= 1 gets its first K = ext_l frames
    anchored to window k-1's final latent at the same global time (n.anchor['fixed']), then is sampled alone with run_level (no sibling
    blend) on the level's shared noise canvas. Needs an anchor relay and halo_px 0."""
    kind, _ = parse_relay(relay)
    if kind not in ANCHOR_KINDS or not abuts() or bg: raise NotImplementedError("compose='handoff' needs an anchor relay, halo_px 0 and no bg pass")
    prev = None
    for n in nodes:
        K = int(n.ext_l)
        if prev is not None and K > 0:
            fixed = []
            for i in range(K):
                ip = prev.canvas_w - K + i
                assert n.canvas_g0 + i * n.scale == prev.canvas_g0 + ip * prev.scale, (n.name, prev.name, i)
                fixed.append((i, prev.latent[..., ip:ip + 1] * 1.0))
            if CFG.get("first_slot", "latent") == "image":             # slot 0 = image latent of the last pixel frame of window k-1's slot ip = w - K
                fixed[0] = (0, state.B.encode_image(node_frames(prev)[4 * (prev.canvas_w - K)]))
            n.anchor["fixed"] = fixed
            if n.anchor.get("pixfield"):                               # ';pixfield': lead = window k-1's decoded frames of the handed-off slots
                if n.anchor.get("rootcolor"): n._lead_root_time = node_root_time(prev)[4 * (prev.canvas_w - K):]
                n.field_lp = pix_field(n, lead=node_frames(prev)[4 * (prev.canvas_w - K):])
        run_level([n], levels, cfg, seed, relay, noise_nodes=nodes); prev = n
