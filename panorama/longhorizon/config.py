import os, json, hashlib

CFG = dict(
    version="v37",
    # ---- base model ----
    backend="sd3",                                  # "sd3" (flow matching) | "sd2" (VP / epsilon UNet) | "cogvideo"/"video" (VP video transformer, horizon = time) | "wan" (flow-matching video transformer, horizon = time)
    # ---- geometry / sampler ----
    geometry="legacy",                              # "legacy" = the v37/v38 768 x 384 window (old runs stay reproducible) | "native" = NATIVE_IMG, the model's own
                                                      # resolution: root 1280 x 704 (the whole horizon in one window), child canvases 1664 x 704 (applied by runners.setup)
    core_px=768, halo_px=64, height=384, branch=2,
    root_halo=0,                                    # 1: the root gets a halo on both sides too, so its canvas is core_px + 2 halo_px (video: 13 latent frames = the model's native clip)
    steps=20, cfg=7.0, sync="x0",
    solver="euler",                                 # level-to-level update: "euler" (first order: x' = mix(x0_hat, eps_hat, next), = DDIM on VP) | "dpmpp2m" | "consistency" (x0 then re-noise with fresh noise: distilled few-step models, WAN_TURBO_OVERRIDES) | "unipc" (UniPC predictor-corrector, Wan's scheduler; options from the model's UniPC config or CFG["unipc"] = dict(order, solver_type, lower_order_final, disable_corrector))
                                                      # (DPM-Solver++(2M), data prediction, on the per-node history of the BLENDED x0_hat; sampling.dpmpp2m_step)
    # ---- stages & grid ----
    stages=["route_tune", "routing", "relay", "length"],
    scene_ids=[0, 1, 2],
    seeds_tune=[11, 12, 13], seeds_tree=[11, 12], seeds_length=[11],
    depth_main=3,                                   # 1:16
    depths_length=[1, 2, 3, 4, 5],                  # 1:4 ... 1:64
    t5_arms=[False, True],                          # T5-XXL off / on (an arm, not a method: the root must read "far left")
    routing_arms=["oracle_span", "exact_hard", "exact_hard_group"],
    relay_arms=["field", "rho_early:0.3", "lowpass_skip:0.3", "lowpass_skip:0.6", "none"],
    relay_main="field", routing_main="exact_hard",  # the fixed arm of the other axis
    length_routing="exact_hard", length_route_levels=["once", "hier"], length_relays=["field", "rho_early:0.3"], length_t5=None,  # None -> best of route_tune
    # ---- conditioning: how (I - pi_a) removes one item from the condition (see longhorizon/textcond.py, METHOD.md section 3) ----
    atom_mode="delete",                             # "delete" = rebuild the prompt without the item and re-encode (no hooks, any encoder) | "zero" = mask its token embeddings in place
    item_order="context_first",                     # instantiation of the condition as an UNORDERED list of items (scenes.prompt_parts/prompt_of): "context_first"
                                                      # (default) = base, entity phrases..., style | "entity_first" = entity phrases..., base, style
    entity_gain=1.0,                                # instantiation of item WEIGHTS: != 1 amplifies the kept entity atom(s)' coefficient under the hook path,
                                                      # on top of "delete" (textcond.encode_weighted / scenes.respec); phrase-level atoms only
    # ---- routing (ablation probe on word atoms) ----
    probe_ts=[0.5, 0.75], probe_eps=1, probe_batch=4,
    region_norm="row_mean", beta=None,              # beta None -> 3 / percentile98(contrast)
    loc_rmax=0.5,                                   # loc_rule "rmax": localized iff max_k r >= this (trivial for 2 regions!)
    loc_rule="rmax", loc_z=3.0,                     # loc_rule "ztest": localized iff the peak region's effect exceeds the others' mean by loc_z noise sigmas (needs probe_eps >= 2)
    entity_rule="argmax",                           # "argmax" = every entity atom goes to its argmax-responsibility window (typed condition: an entity item is placed exactly once);
                                                      # "span" = it goes to the whole INTERVAL of windows the probe gives it (routing.span_windows) -- a video event persists over a
                                                      # stretch of the horizon; "test" = the old behaviour (loc_rule rmax/ztest classification + head_fallback).
                                                      # Context atoms are never routed under any rule.
    group_rmin=0.3,                                 # exact_hard_group: co-localize a word if its peak sits in a localized window with r >= this
    head_fallback="argmax",                         # head word not localized -> "argmax" (model's best window) | "stated" (v36, uses the prompt = oracle leak) | "none"
    route_tune_grid=[dict(probe_ts=[0.5, 0.75], probe_eps=1, region_norm="row_mean"),
                     dict(probe_ts=[0.5, 0.75], probe_eps=2, region_norm="row_mean"),
                     dict(probe_ts=[0.5, 0.75], probe_eps=1, region_norm="none"),
                     dict(probe_ts=[0.3, 0.5, 0.7, 0.9], probe_eps=1, region_norm="none")],
    # ---- attention: DIAGNOSTIC ONLY (never an arm) ----
    attn_diag=True, attn_ts=[0.5, 0.75], attn_blocks="all", attn_tau_loc=2.0,
    # ---- relays ----
    field_init_sigma=0.80, lowpass="H",
    # ---- root selection: rejection sampling of the plan ----
    root_candidates=6, root_score="min",
    # ---- evaluation / display ----
    fidelity=True, det_thr=0.12, show_sheets=True,
    sheet_chunk_px=6144, sheet_w=3072,              # every strip row shows 6144 px of image at half scale (1:16 -> 2 rows)
    # ---- video only ----
    video_fps=16, video_det_every=4, video_strip_every=10, video_iqa_every=16,  # matches CDGS's fps; detector on every 4th decoded frame, filmstrip every 10th, CLIP-IQA every 16th
    # ---- models ----
    model_id="stabilityai/stable-diffusion-3-medium-diffusers", owl_id="google/owlvit-base-patch32", t5_max_len=256,
    transformer_id=None,                            # Wan: load the transformer from this repo instead of model_id (Wan 2.2 TI2V-5B-Turbo: only the transformer differs); None = model_id's own
    child_levels=None,                              # video anchor-relay tuning (v42): noise levels of the CHILD windows only (the root keeps `levels`, so a cached root is reused); None = same as the root
    compose="blend",                                # video: "blend" = siblings sampled in lock-step, x0-blended on their overlap (halo_px 0: abutting windows, concatenated) |
                                                      # "handoff" (v43, halo_px 0) = siblings sampled LEFT TO RIGHT, window k's first handoff_k frames anchored to window k-1's last
                                                      # handoff_k final frames (sampling.run_level_seq, geometry.apply_handoff)
    handoff_k=2,
    video_decode="composite",                       # "composite" = the composed latent decoded in one pass | "windows" (v43) = every leaf decoded as its own clip, frames concatenated (video.decode_windows)
    first_slot="latent",                            # video anchor relays: "latent" = a window's slot 0 may be anchored with a regular 4-frame latent (v41/v42) | "image" (v43) =
                                                      # slot-0 anchors are image latents (B.encode_image of one decoded pixel frame: Wan TI2V's native first-frame condition)
    edge_native=0,                                  # 0 | 1 ("in": the two horizon-end leaves extended inward to the native canvas) | "out" (end windows of every level extended past the horizon, cropped); geometry.apply_edge_native
    levels=None,                                    # explicit noise-level list (flow: sigmas, descending, WITHOUT the final 0) replacing the scheduler's `steps` schedule
                                                      # (a distilled model's fixed timesteps, e.g. WAN_TURBO_OVERRIDES); None = read off the scheduler
    timestep_int=False,                             # Wan: pass the transformer the INTEGER timestep int(sigma * 1000) (the Turbo repo's few-step inference casts
                                                      # the timestep to long: 937 / 833 at sigma 0.9375 / 0.8333); False = the exact float, as WanPipeline does
    root_cache="results/roots",                     # a sampled root's clean latent is saved here (fp16 .pt + key .json; runners.get_root) and reused whenever the
                                                      # key (prompt, seed, backend, model, levels, sampler, resolution) is identical; None = always sample. Not in tag().
    root_noise="harness",                           # "harness" = the root's noise from the level canvas (seed * 1000 + depth, fresh noise seed + 7919 (i + 1));
                                                      # "pipeline" = ONE torch.Generator(cuda).manual_seed(seed) stream in the model's (1, C, T, H, W) layout, drawn
                                                      # init first then one fresh draw per consistency step -- exactly WanPipeline.prepare_latents + the Turbo
                                                      # repo's loop, so a root seed found by a stand-alone search reproduces in the harness (B.noise_stream)
    dtype=None,                                     # "fp16" | "bf16" | None -> fp16 for SD3-medium, bf16 for SD3.5 / FLUX (fp16 overflows on SD3.5-large); see dtype_name()
    # ---- resource guard ----
    guard_rss_gb=120, threads=32, cpus=None, gpu="0",
    out_dir=os.environ.get("OUT_DIR", "./results/v37/dev"),
    # ---- v38: root check + attention comparison arms (routing.route_attn; attention is a COMPARISON arm, never the method) ----
    seeds_v38=[11, 12, 13, 14],                     # seeds of the v38 stages (root_check, routing_v38)
    attn_tau_cut=0.5, attn_w_max=2.0,               # attn_scale_phrase: w = clip(A / mean_k A, 0, w_max), 0 below tau_cut (v36 values)
)
DUMMY_OVERRIDES = dict(steps=6, scene_ids=[0], seeds_tune=[11], seeds_tree=[11], seeds_length=[11], depth_main=2, depths_length=[1, 2],
                       root_candidates=2, t5_arms=[False, True], route_tune_grid=[dict(probe_ts=[0.5, 0.75], probe_eps=1, region_norm="row_mean")],
                       attn_diag=False, out_dir="./out_dummy", root_cache=None)
# Native image geometry: the root is ONE native 1280 x 704 window covering the whole horizon; every child canvas is
# core 1280 + halo 192 on each interior side = 1664 x 704 (edge windows 1472), so siblings share 2 * 192 = 384 px (23 % of a window). Depth 3 ->
# 8 leaves -> 10240 x 704. Every width/height is a multiple of 16 (VAE 8 x patch 2). The low-pass cutoff "H" stays relative (= 704 px now).
# sheet_chunk_px 5120: a depth-3 composite shows as 2 rows. Applied by runners.setup when CFG["geometry"] == "native", BEFORE user overrides.
NATIVE_IMG = dict(core_px=1280, halo_px=192, height=704, sheet_chunk_px=5120)
# Stable Diffusion 3.5-large: same StableDiffusion3Pipeline / SD3Transformer2DModel as SD3-medium (backend "sd3"); bf16 (fp16 overflows).
# The scheduler shift (3.0) and joint_attention_dim are READ from the loaded model, not set here. Plain CFG (no skip-layer guidance).
SD35L_OVERRIDES = dict(model_id="stabilityai/stable-diffusion-3.5-large", steps=28, cfg=3.5, dtype="bf16")
# FLUX.1-dev (backend_flux): guidance-distilled flow matching -- `cfg` is the GUIDANCE EMBEDDING (no uncond batch); CLIP-L pooled + T5-XXL
# (the pipeline's default max_sequence_length 512); attention arms are unavailable (attn_mass raises), so attn_diag is off.
FLUX_OVERRIDES = dict(backend="flux", model_id="black-forest-labs/FLUX.1-dev", steps=28, cfg=3.5, dtype="bf16", t5_arms=[True], t5_max_len=512, attn_diag=False)
# Stable Diffusion 2.0-base: 512 px window, one CLIP encoder, epsilon prediction, DDIM. Applied by runners.setup BEFORE the user's overrides.
SD2_OVERRIDES = dict(backend="sd2", core_px=512, height=512, model_id="Manojb/stable-diffusion-2-base", t5_arms=[False], steps=50, cfg=7.5)
# CogVideoX-2b: the HORIZON IS TIME and its unit is one LATENT FRAME (VS = 1). The model's native clip is 49 pixel frames = 13 latent frames,
# so core_px = 11 latent frames + halo_px = 1 on each interior side gives a 13-frame canvas; the root gets the same halo (root_halo=1) so it is
# a native 49-frame clip covering the whole story. A depth-d tree composes 11 * 2^d latent frames (d=3 -> 88 -> 349 pixel frames, CDGS's ~350).
# `height` is the LATENT height (60 = 480 px); the latent width is the backend's LAT_W (90 = 720 px). `lowpass` is stated directly in latent
# frames ("6" ~ half a window, the same cutoff/window ratio the image arms use: 384 px cutoff for a 768 px window).
VIDEO_OVERRIDES = dict(backend="cogvideo", core_px=11, halo_px=1, root_halo=1, height=60, steps=50, cfg=6.0,
                       model_id="THUDM/CogVideoX-2b", t5_arms=[True], det_thr=0.12, lowpass="6", attn_diag=False,
                       sheet_chunk_px=1536, sheet_w=1536)
# Wan 2.2 TI2V-5B: the same horizon (TIME, unit = one latent frame) as CogVideoX, but flow matching and a 4 x 16 x 16 VAE with 48 latent
# channels. The model's default clip is 121 pixel frames = (121 - 1) / 4 + 1 = 31 latent frames, so core_px = 29 latent frames + halo_px = 1
# on each interior side gives a 31-frame canvas, and root_halo=1 makes the root a native 121-frame clip; a depth-d tree composes 29 * 2^d
# latent frames (d=3 -> 232 -> 925 pixel frames at 24 fps ~ 39 s). `height` is the LATENT height (30 = 480 px at stride 16) and the latent
# width is the backend's LAT_W (52 = 832 px): the 480p-class point, not the model's advertised 720P (44 x 80 = 704 x 1280), which costs
# ~2.3x the tokens per window -- any multiple of 32 px is a legal resolution (VAE stride 16 x transformer patch 2). `lowpass` is stated
# directly in latent frames ("15" ~ half a 31-frame window, the same cutoff/window ratio the image and CogVideoX arms use). cfg 5.0 and
# 24 fps are WanPipeline's / the repo's own defaults for TI2V-5B.
WAN_OVERRIDES = dict(solver="unipc", backend="wan", core_px=29, halo_px=1, root_halo=1, height=30, steps=30, cfg=5.0,
                     model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", t5_arms=[True], det_thr=0.12, lowpass="15", attn_diag=False,
                     video_fps=24, sheet_chunk_px=1536, sheet_w=1536)
# Wan 2.2 TI2V-5B-Turbo (quanhaol/Wan2.2-TI2V-5B-Turbo, Self-Forcing DMD step + CFG distillation; Diffusers transformer from
# yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers -- the umT5 and the VAE are the base repo's). Backend "wan" with these as (part of) the caller's
# overrides, like SD35L_OVERRIDES: 4 FIXED levels = the training config's denoising_step_list [1000, 750, 500, 250] warped at shift 5
# (sigma = 5u / (1 + 4u), u = t/1000), no CFG (cfg 1.0 -> backend_wan runs the conditional branch only), and the repo's own few-step
# sampler: x0 prediction then re-noise with FRESH noise (sampling "consistency"). UniPC does not apply: on these 4 levels the Diffusers
# model-card recipe (UniPC flow_shift 5, guidance 1) leaves a translucent ghost of the early x0 over the clip (results/root_study/
# turbo_search/probe). Relays: field:s0 starts at the first level <= s0 -> 0.95: 3 steps, 0.9: 2, 0.8: 1, <= 0.6: none (backend_wan docstring).
WAN_TURBO_OVERRIDES = dict(transformer_id="yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers", steps=4, cfg=1.0, solver="consistency",
                           levels=[1.0, 0.9375, 0.8333333333333334, 0.625], timestep_int=True)

H, VS, LC = CFG["height"], 8, 16                    # defaults: image height, VAE stride, latent channels (the backend may override VS/LC)
OV = 2 * CFG["halo_px"]; OVL = OV // VS             # overlap between siblings in px / latent columns
TREE = dict(depth=CFG["depth_main"])

# ---- derived quantities, read live (the backend decides the latent geometry; CFG is the fallback when no backend is loaded) ----
def _bk(attr, default):
    from . import state
    return default if state.B is None else getattr(state.B, attr, default)
def height(): return int(CFG["height"])
def vae_stride(): return int(_bk("VS", VS))
def latent_ch(): return int(_bk("LC", LC))
def overlap(): return 2 * int(CFG["halo_px"])
def child_canvas_px():
    """Width of an interior child canvas (core + a halo on both sides): 1664 px native, 896 px legacy. Backends whose noise schedule depends
    on the token count (FLUX's dynamic shift) compute ONE schedule for the whole tree from this size."""
    return int(CFG["core_px"]) + 2 * int(CFG["halo_px"])
def overlap_lat(): return overlap() // vae_stride()

VIDEO_BACKEND_NAMES = ("cogvideo", "video", "wan")  # horizon = time (backend_cogvideo, backend_wan); `runners.VIDEO_BACKENDS` is the same set
def dtype_name():
    """"fp16" | "bf16": CFG["dtype"] if set, else bf16 for SD3.5 and FLUX (fp16 is known to overflow on SD3.5-large), fp16 otherwise."""
    if CFG.get("dtype"): return str(CFG["dtype"])
    mid = str(CFG.get("model_id", "")).lower()
    return "bf16" if (CFG.get("backend") == "flux" or "3.5" in mid or "flux" in mid) else "fp16"
def is_video(): return bool(_bk("is_video", CFG["backend"] in VIDEO_BACKEND_NAMES))
def set_depth(d): TREE["depth"] = int(d)
def n_leaves(): return CFG["branch"] ** TREE["depth"]
def final_w(): return CFG["core_px"] * n_leaves()
def aspect():
    """The headline length ratio of a tree. Images: composite width / window height (1:16 at depth 3). Video: the number of windows
    (the horizon unit is a latent frame, so `height` is not a length along the horizon)."""
    return int(n_leaves()) if is_video() else int(final_w() // height())
def tag():
    """Hash of the config. `solver` enters only when it is not the default "euler" (and `transformer_id` / `levels` only when set,
    `timestep_int` / `root_noise` only when not False / "harness"), so tags of
    runs made before these options existed are unchanged."""
    return hashlib.md5(json.dumps({k: v for k, v in CFG.items() if k not in ("out_dir", "show_sheets", "root_cache") and not (k == "solver" and v == "euler")
                                   and not (k in ("transformer_id", "levels") and v is None)
                                   and not (k == "timestep_int" and v is False)
                                   and not (k in ("child_levels", "edge_native") and not v)
                                   and not (k in ("compose", "handoff_k") and CFG.get("compose", "blend") == "blend") and not (k == "first_slot" and v == "latent") and not (k == "video_decode" and v == "composite") and not (k == "root_noise" and v == "harness")}, sort_keys=True).encode()).hexdigest()[:9]
def as_dict(overrides):
    """Overrides given as a dict or a JSON string -> dict."""
    if overrides is None: return {}
    if isinstance(overrides, str): return json.loads(overrides) if overrides.strip() else {}
    return dict(overrides)
def update(overrides):
    """Apply overrides (dict or JSON string) and re-derive globals that depend on them."""
    global H, OV, OVL
    overrides = as_dict(overrides)
    unknown = [k for k in overrides if k not in CFG]
    if unknown: raise KeyError(f"unknown CFG keys: {unknown}")
    CFG.update(overrides); H = CFG["height"]; OV = 2 * CFG["halo_px"]; OVL = OV // VS; TREE["depth"] = CFG["depth_main"]
    return CFG
