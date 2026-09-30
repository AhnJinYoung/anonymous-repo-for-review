"""Roots, trees, stages. Every finished tree is appended to <out_dir>/logs.jsonl immediately (resumable; nothing is lost on interruption)."""
import os, sys, json, time, subprocess, numpy as np
from . import config, state
from .config import CFG, TREE, set_depth, n_leaves, final_w, tag
from .scenes import SCENES, VIDEO_SCENES, full_prompt
from .geometry import build_tree, core_regions, child_regions
from .routing import responsibilities, classify_ztest, route_exact, route_oracle, route_attn_diag, route_attn, route_attn_time, frames_to_regions, weights_to_nodes, route_accuracy, plan_positions, head_fallback, to_np, stated_region, stated_windows
from .sampling import run_level, run_level_seq, prepare_child
from .metrics import leaf_metrics, root_metrics, compose_level, core_crop
from .display import sheet, show, banner

VIDEO_BACKENDS = ("cogvideo", "video", "wan")       # every backend whose horizon is TIME; "wan" has its own geometry/parameterization (config.WAN_OVERRIDES, backend_wan)
def is_video(): return bool(getattr(state.B, "is_video", False))
def scenes_of(sid):
    """The benchmark the active backend runs on: panorama scenes for images, the CDGS-comparison event clips for video."""
    return (VIDEO_SCENES if is_video() else SCENES)[sid]

# ---------------- setup ----------------
def setup(dummy=False, overrides=None, out_dir=None):
    """Guard -> config -> backend. `CFG["backend"]` ('sd3' | 'sd2' | 'flux' | 'cogvideo'/'video' | 'wan') picks the base model; its geometry defaults
    (SD2_OVERRIDES: 512 px window and height; VIDEO_OVERRIDES: horizon = time, 11+1 latent frames, latent height 60; WAN_OVERRIDES: horizon
    = time, 29+1 latent frames, latent height 30, flow matching; FLUX_OVERRIDES: FLUX.1-dev, guidance embedding) are applied BEFORE the caller's
    overrides, so the caller can still change them; then, if geometry == "native", NATIVE_IMG (root 1280 x 704, child canvases 1664 x 704).
    SD3.5 is backend "sd3" with config.SD35L_OVERRIDES passed as (part of) the overrides. Returns the backend."""
    from . import config, guard
    overrides = config.as_dict(overrides); bk = overrides.get("backend", CFG["backend"])
    if bk == "sd2": config.update(config.SD2_OVERRIDES)
    elif bk == "flux": config.update(config.FLUX_OVERRIDES)
    elif bk == "wan": config.update(config.WAN_OVERRIDES)
    elif bk in VIDEO_BACKENDS: config.update(config.VIDEO_OVERRIDES)
    if overrides.get("geometry", CFG["geometry"]) == "native":          # native image geometry (config.NATIVE_IMG), still before the caller's overrides
        if bk in VIDEO_BACKENDS: raise ValueError("geometry='native' is the image geometry (1280/192/704); video backends have their own")
        config.update(dict(config.NATIVE_IMG, geometry="native"))
    if overrides: config.update(overrides)
    if out_dir: CFG["out_dir"] = out_dir
    if dummy: config.update(config.DUMMY_OVERRIDES | ({"out_dir": out_dir} if out_dir else {}))
    guard.install(CFG["guard_rss_gb"], CFG["threads"], CFG["cpus"], CFG["gpu"])
    os.makedirs(CFG["out_dir"], exist_ok=True)
    if dummy:
        from .dummy import Backend
        B = state.set_backend(Backend(load_t5=any(CFG["t5_arms"]), kind=CFG["backend"]))
    else:
        if CFG["backend"] == "sd2": from .backend_sd2 import Backend
        elif CFG["backend"] == "flux": from .backend_flux import Backend
        elif CFG["backend"] == "wan": from .backend_wan import Backend
        elif CFG["backend"] in VIDEO_BACKENDS: from .backend_cogvideo import Backend
        else: from .backend import Backend
        B = state.set_backend(Backend(load_t5=any(CFG["t5_arms"])))
    json.dump(CFG, open(f"{CFG['out_dir']}/config.json", "w"), indent=1)
    print(f"setup: tag {tag()} | out {CFG['out_dir']} | git {git_rev()}"); return B
def git_rev():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()
    except Exception: return "?"

# ---------------- incremental log ----------------
def log_path(): return f"{CFG['out_dir']}/logs.jsonl"
def _key(d): return (d.get("stage"), d["scene"], d["seed"], d["depth"], d["routing"], d["relay"], d["route_level"], bool(d["t5"]))
def load_logs(path=None):
    path = path or log_path()
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]
def append_log(rec):
    with open(log_path(), "a") as f: f.write(json.dumps(rec, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)) + "\n")

# ---------------- roots ----------------
def decode_root(root, spec):
    """The root as something the metrics and the sheets can read. Video: the decoded clip's frames, its filmstrip, and detections PER FRAME
    (the horizon is time, so 'where is the lighthouse' becomes 'in which frames is the balloon on the ground')."""
    B = state.B; qs = [o["query"] for o in spec["objects"]]
    if is_video():
        from . import video as V
        root.frames = V.decode_frames(root.latent); root.image = V.filmstrip(root.frames)
        root.frame_dets = V.detect_frames(root.frames, qs); root.dets = [d for _, dd in root.frame_dets for d in dd]
    else:
        root.image = B.decode(root.latent); root.dets = B.detect(root.image, qs, CFG["det_thr"])
    return root

def root_cond(scene, spec):
    """The root's condition: the full typed condition, or the scene's own `root_prompt` text (scenes.root_text) when it has one."""
    B = state.B
    if scene.get("root_prompt"): return B.text_cond(scene["root_prompt"])
    return B.make_cond(spec, np.ones(spec["K"]))

def sample_root(scene, seed, spec):
    B = state.B; levels = build_tree(TREE["depth"]); root = levels[0][0]
    root.w = np.ones(spec["K"]); root.cond = root_cond(scene, spec)
    run_level([root], B.levels(CFG["steps"]), CFG["cfg"], seed, None)
    decode_root(root, spec)
    return levels, root

def root_metrics_of(root, spec):
    if is_video():
        from . import video as V
        return V.root_metrics_video(root, spec)
    return root_metrics(root, spec)

def root_compliance(spec, root, seed):
    """Per object: responsibility of its head word in the STATED span (the best window of `stated_windows`; for an image object the span is
    one window, so this is the old number). Aggregate = min (worst object)."""
    B = state.B; atoms = list(range(spec["K"])); n_reg = n_leaves(); regions = core_regions(root, n_reg)
    if not spec["objects"]: return 0.0, {}                 # all-context prompt: there is no plan to reject-sample, every root scores the same
    D = to_np(B.delta_exact(root.latent, spec, atoms, regions, CFG["probe_ts"], seed, n_eps=CFG["probe_eps"]))
    r, _, _ = responsibilities(D, CFG["region_norm"], CFG["beta"])
    per = {o["query"]: max(float(r[k, spec["heads"][o["query"]]]) for k in stated_windows(o, n_reg)) for o in spec["objects"]}   # best window of the stated span
    agg = min(per.values()) if CFG["root_score"] == "min" else float(np.mean(list(per.values())))
    return agg, per

# ---------------- root cache ----------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def root_cache_key(scene, seed, t5):
    """Everything that decides the root's clean latent: prompt, seed, backend, model, level list, sampler, resolution, root canvas, and the
    candidate search when there is one. A cached root is used only when this key is IDENTICAL."""
    from .scenes import root_text, root_name
    B = state.B; lv = B.levels(CFG["steps"]); root = build_tree(CFG["depth_main"])[0][0]
    k = dict(scene=root_name(scene), prompt=root_text(scene), seed=int(seed), t5=bool(t5), backend=CFG["backend"], model_id=CFG.get("model_id"),
             transformer_id=CFG.get("transformer_id"), levels=[round(float(l), 8) if isinstance(l, (int, float)) else str(l) for l in lv],
             steps=CFG["steps"], cfg=CFG["cfg"], solver=CFG.get("solver", "euler"), root_noise=CFG.get("root_noise", "harness"),
             timestep_int=bool(CFG.get("timestep_int", False)), height_lat=config.height() // config.vae_stride() if not is_video() else CFG["height"],
             width_lat=int(getattr(B, "LAT_W", 0)) if is_video() else None, root_canvas=int(root.canvas_w), dtype=CFG.get("dtype"),
             atom_mode=CFG.get("atom_mode"), root_candidates=int(CFG["root_candidates"]))
    if int(CFG["root_candidates"]) > 1: k.update({x: CFG.get(x) for x in ("probe_ts", "probe_eps", "beta", "region_norm", "root_score", "root_compliance")})
    return k
def root_cache_paths(scene, seed, key):
    """results/roots/<scene>_s<seed>_<backend>.pt (+ .json); a second root with the same name but another key goes to ..._<hash8>.pt."""
    import hashlib
    d = CFG.get("root_cache"); d = d if os.path.isabs(d) else os.path.join(REPO, d)
    from .scenes import root_name
    stem = f"{root_name(scene)}_s{seed}_{CFG['backend']}"; h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return [os.path.join(d, stem), os.path.join(d, f"{stem}_{h}")]
def load_root_cache(scene, seed, key):
    import torch
    for stem in root_cache_paths(scene, seed, key):
        if os.path.exists(stem + ".json") and os.path.exists(stem + ".pt"):
            meta = json.load(open(stem + ".json"))
            if meta.get("key") == json.loads(json.dumps(key)):
                z = torch.load(stem + ".pt", map_location="cpu", weights_only=True)["latent"].float()
                lat = z.numpy() if getattr(state.B, "dummy", False) else z.to(state.B.device)
                print(f"  root loaded from cache: {os.path.relpath(stem, REPO)}.pt (chosen seed {meta['search']['chosen_seed']})", flush=True)
                return lat, meta
    return None, None
def save_root_cache(scene, seed, key, latent, search):
    """Save the chosen root's clean latent in fp16 + a json with the key. Returns the fp16-rounded latent (what every later run will load),
    so the run that SAMPLES the root continues on exactly the same values as the runs that load it."""
    import torch
    paths = root_cache_paths(scene, seed, key); stem = paths[0]
    if os.path.exists(stem + ".json") and json.load(open(stem + ".json")).get("key") != json.loads(json.dumps(key)): stem = paths[1]
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    z = torch.as_tensor(np.asarray(latent) if getattr(state.B, "dummy", False) else latent).detach().to("cpu", torch.float16).contiguous()
    torch.save({"latent": z}, stem + ".pt")
    json.dump(dict(key=key, search=search, shape=list(z.shape), dtype="float16", layout="(1, C, H, W, T) harness layout, horizon last" if is_video() else "(1, C, H, W)",
                   git=git_rev(), saved=time.strftime("%Y-%m-%d %H:%M:%S")), open(stem + ".json", "w"), indent=1, default=str)
    print(f"  root cached -> {os.path.relpath(stem, REPO)}.pt", flush=True)
    return z.float().numpy() if getattr(state.B, "dummy", False) else z.float().to(state.B.device)

_ROOTS = {}
def get_root(scene, seed, t5, on_candidate=None):
    """Root shared across arms/stages/depths for the same (scene, seed, t5): best of `root_candidates` by the worst object's compliance
    (rejection sampling of the plan). The root image is depth-independent; the tree around it is rebuilt for the current depth.
    `on_candidate(c, seed_c, root_c, spec, score, per, root_metrics)` is called for every candidate while the search runs (v38
    root_check: save and evaluate every candidate); it never changes which candidate is picked."""
    B = state.B; key = (scene["name"], seed, bool(t5))
    if key not in _ROOTS and CFG.get("root_cache") and on_candidate is None:
        d_now = TREE["depth"]; set_depth(CFG["depth_main"]); ck = root_cache_key(scene, seed, t5); lat, meta = load_root_cache(scene, seed, ck)
        if lat is not None:
            spec = B.encode_text(scene, t5=t5); root = build_tree(TREE["depth"])[0][0]; root.latent = lat; decode_root(root, spec)
            root.search = meta["search"]
            _ROOTS[key] = dict(latent=root.latent, image=root.image, dets=root.dets, frames=root.frames, frame_dets=root.frame_dets,
                               search=root.search, spec=spec)
        set_depth(d_now)
    if key not in _ROOTS:
        d_now = TREE["depth"]; set_depth(CFG["depth_main"]); spec = B.encode_text(scene, t5=t5); best, cands = None, []; t0 = time.time()
        for c in range(CFG["root_candidates"]):
            s = seed + 1000 * c; levels, root = sample_root(scene, s, spec); score, per = root_compliance(spec, root, s)
            rm = root_metrics_of(root, spec)
            if on_candidate is not None: on_candidate(c, s, root, spec, score, per, rm)
            cands.append(dict(seed=s, score=round(score, 3), per={k: round(v, 3) for k, v in per.items()}, root=rm))
            if best is None or score > best[0]: best = (score, root, s)
        _, root, s_best = best; root.search = dict(candidates=cands, chosen_seed=s_best, t5=bool(t5), time_s=round(time.time() - t0, 1))
        if CFG.get("root_cache"):                                    # cache the chosen root; continue on the cached (fp16) values, re-decoded
            root.latent = save_root_cache(scene, seed, root_cache_key(scene, seed, t5), root.latent, root.search); decode_root(root, spec)
        root.image.save(f"{CFG['out_dir']}/root_{scene['name']}_s{seed}_t5{int(bool(t5))}.jpg", quality=92)
        if is_video() and root.frames:
            from . import video as V
            V.save_video(root.frames, f"{CFG['out_dir']}/root_{scene['name']}_s{seed}_t5{int(bool(t5))}.mp4")
        print(f"  root {scene['name']} s{seed} t5={int(bool(t5))}: chosen {s_best} | scores {[c['score'] for c in cands]} | "
              f"root pos err vs prompt {[(q[2:10], round(v, 2)) for q, v in cands[[c['seed'] for c in cands].index(s_best)]['root']['pos_err_prompt'].items()]} | {root.search['time_s']}s")
        _ROOTS[key] = dict(latent=root.latent, image=root.image, dets=root.dets, frames=root.frames, frame_dets=root.frame_dets,
                           search=root.search, spec=spec); set_depth(d_now)
    R = _ROOTS[key]; levels = build_tree(TREE["depth"]); root = levels[0][0]
    root.latent, root.image, root.dets, root.search = R["latent"], R["image"], R["dets"], R["search"]; root.bg_latent = R.get("bg_latent")
    root.frames, root.frame_dets = R.get("frames"), R.get("frame_dets")
    root.w = np.ones(R["spec"]["K"]); root.cond = root_cond(scene, R["spec"])
    return levels, root, R["spec"]

def bg_weights(spec, w):
    """The node's weights with every object phrase removed (all encoders): the object-free conditioning used by the field_bg relay."""
    wb = np.asarray(w, np.float64).copy()
    for ph in spec["phrase_atoms"].values():
        for j in ph: wb[j] = 0.0
    return wb

def ensure_bg_root(root, spec, seed_root):
    """Object-free twin of the chosen root: same seed and noise, object phrases removed from the conditioning. Cached with the root."""
    B = state.B; key = (spec["scene"]["name"], seed_root, bool(spec["t5"]))   # seed_root = the (scene, seed) key seed, not the chosen candidate seed
    R = _ROOTS.get(key)
    if R is not None and "bg_latent" in R: root.bg_latent = R["bg_latent"]; return
    d_now = TREE["depth"]; set_depth(CFG["depth_main"]); levels = build_tree(TREE["depth"]); twin = levels[0][0]
    twin.w = bg_weights(spec, np.ones(spec["K"])); twin.cond = B.make_cond(spec, twin.w)
    run_level([twin], B.levels(CFG["steps"]), CFG["cfg"], root.search["chosen_seed"], None); set_depth(d_now)
    root.bg_latent = twin.latent
    if R is not None: R["bg_latent"] = twin.latent
    B.decode(twin.latent).save(f"{CFG['out_dir']}/rootbg_{spec['scene']['name']}_s{seed_root}_t5{int(bool(spec['t5']))}.jpg", quality=90)

# ---------------- routing at the root / per level ----------------
def route_root(root, spec, routing, n_reg):
    B = state.B; atoms = list(range(spec["K"])); regions = core_regions(root, n_reg); info = {}
    if routing in ("broadcast", "broadcast_root"): W = np.ones((n_reg, spec["K"]))
    elif routing == "oracle_span": W = route_oracle(spec, n_reg)
    elif routing == "exact_hard_time":                                  # v54: black-box TIME routing from deletion responsibility (route_del_time)
        W, info = route_del_time(root, spec, regions, n_reg)
    elif routing.startswith("exact"):
        Ds = B.delta_exact(root.latent, spec, atoms, regions, CFG["probe_ts"], root.search["chosen_seed"], n_eps=CFG["probe_eps"], return_all=True)
        W, info = route_exact(Ds.mean(0), routing.split("_", 1)[1], spec, n_reg, Ds=Ds)
    elif routing in ("attn_hard", "attn_scale", "attn_thr"):                     # v38 comparison arms: attention mass at the root (not the method)
        A = B.attn_mass(root.latent, spec, atoms, regions, CFG["attn_ts"], root.search["chosen_seed"], CFG["attn_blocks"])
        W, info = route_attn(A, routing.split("_", 1)[1], spec, n_reg)
    elif routing in ("attn_hard_time", "attn_scale_time"):           # v52: video TIME routing from the root's own cross-attention (backend_wan.attn_time_profile)
        W, info = route_time(root, spec, routing.split("_")[1], regions, n_reg)
    else: raise ValueError(routing)
    return W, info

def attn_time_atoms(P, spec, blocks=None, norm=None, contrast=None):
    """(n_ts, n_blocks, T, n_txt) cross-attention profile -> (T, K) mass per latent frame and atom: mean over the steps and the selected
    blocks (CFG["attn_time_blocks"]: "all" or a list of block indices), then summed over each atom's tokens. norm (CFG["attn_time_norm"]):
    "atom" = raw token mass (the per-atom normalisation over windows happens in routing, as for images); "token" = every text token's
    column is first divided by its mean over frames (a token that is attended everywhere does not dominate its atom's profile)."""
    B = state.B; blocks = CFG.get("attn_time_blocks", "all") if blocks is None else blocks; norm = norm or CFG.get("attn_time_norm", "atom")
    sel = list(range(P.shape[1])) if blocks == "all" else [int(b) for b in blocks]; prof = np.asarray(P, np.float64)[:, sel].mean(axis=(0, 1))
    if norm == "token": prof = prof / (prof.mean(0, keepdims=True) + 1e-12)
    elif norm != "atom": raise ValueError(f"attn_time_norm {norm!r}")
    tok = B.atom_token_idx(spec, range(spec["K"])); ents = list(spec.get("entity_atoms", []))
    if CFG.get("attn_time_tokens", "all") == "distinct" and ents:       # v53: an event's score from its DISTINGUISHING tokens only (routing.distinct_token_idx)
        from .routing import distinct_token_idx
        pieces = B.tok3.convert_ids_to_tokens(B.tok3(spec["prompt"])["input_ids"]); dt = distinct_token_idx(pieces, tok, ents)
        tok = [dt.get(j, t) for j, t in enumerate(tok)]
    a = np.stack([prof[:, idx].sum(1) for idx in tok], 1)
    if (CFG.get("attn_time_contrast", False) if contrast is None else contrast) and len(ents) > 1:          # v53: contrastive event score At_j - mean_{k != j} At_k (+1, so its mean over frames is 1)
        At = a[:, ents] / (a[:, ents].mean(0, keepdims=True) + 1e-12); c = At - (At.sum(1, keepdims=True) - At) / (len(ents) - 1)
        a = a.copy(); a[:, ents] = np.clip(c + 1.0, 1e-6, None)
    return a

def route_time(root, spec, mode, regions, n_reg):
    """v52: the root's per-latent-frame cross-attention profile a_j(tau) (logged as `attn_profile`), aggregated to the leaf regions and
    routed by routing.route_attn_time. Steps CFG["attn_time_ts"], blocks CFG["attn_time_blocks"], norm CFG["attn_time_norm"]."""
    B = state.B; ts = CFG.get("attn_time_ts") or CFG["attn_ts"]
    P = B.attn_time_profile(root.latent, spec, ts, root.search["chosen_seed"], CFG.get("attn_time_blocks", "all"))
    a = attn_time_atoms(P, spec); A = frames_to_regions(a, regions); A_thr, extra = None, {}
    if CFG.get("time_hard_rule", "dominant") == "threshold":     # v56: log the non-contrastive score too; threshold on it with time_thresh_score "plain"
        ap = attn_time_atoms(P, spec, contrast=False); extra = dict(attn_profile_plain=np.round(ap, 6).tolist())
        if CFG.get("time_thresh_score", "same") == "plain": A_thr = frames_to_regions(ap, regions)
    if mode == "hard" and CFG.get("time_hard_rule", "dominant") == "support":   # v58: per-frame interval support (routing.route_time_support)
        from .routing import route_time_support
        W, info = route_time_support(a, regions, spec, n_reg, float(CFG.get("time_support_margin", 0.02)), int(CFG.get("time_support_run", 2)))
    else: W, info = route_attn_time(A, mode, spec, n_reg, A_thr=A_thr)
    info.update(extra)
    info.update(attn_profile=np.round(a, 6).tolist(), attn_time_cfg=dict(ts=list(ts), blocks=CFG.get("attn_time_blocks", "all"), norm=CFG.get("attn_time_norm", "atom"), tokens=CFG.get("attn_time_tokens", "all"), contrast=bool(CFG.get("attn_time_contrast", False))),
                regions=[list(map(int, r)) for r in regions])
    return W, info

def route_del_time(root, spec, regions, n_reg):
    """v54: the black-box analogue of route_time. D[k, a] = increase of the denoising loss on window k's ROOT frames when atom a is deleted
    from the condition (backend.delta_exact: paired ablation, the noise shared between the full and the ablated condition), averaged over
    CFG["del_time_ts"] (default [0.5, 0.625, 0.8333, 0.9375], the v52/v53 deletion levels) and CFG["del_time_eps"] (default 2) noises.
    D -> per-atom responsibilities over the windows (routing.responsibilities, as the images' exact routing) -> route_attn_time "hard"
    (each window gets the event that dominates it relative to its own mean, every event keeps its argmax window). Context atoms stay 1."""
    from .routing import responsibilities
    B = state.B; ts = CFG.get("del_time_ts") or [0.5, 0.625, 0.8333333333333334, 0.9375]; ne = int(CFG.get("del_time_eps", 2))
    Ds = B.delta_exact(root.latent, spec, list(range(spec["K"])), regions, ts, root.search["chosen_seed"], n_eps=ne, return_all=True)
    D = to_np(Ds.mean(0)); r, _, beta = responsibilities(D, CFG["region_norm"], CFG["beta"])
    W, info = route_attn_time(r, "hard", spec, n_reg)
    info.update(D=D, del_r=np.round(r, 4).tolist(), del_time_cfg=dict(ts=list(ts), n_eps=ne, beta=beta), regions=[list(map(int, rg)) for rg in regions])
    return W, info

def factorize_node_hard(node, spec, seed, localized=None, W_root=None):
    """Per-level routing: exact ablation on the node's own latent over its child regions, hard assignment, on top of the node's weights.
    `localized`: the set of atoms the root plan localized; with b=2 children r_max >= 0.5 is trivially true for every atom, so per-level
    routing only REFINES where an already-localized atom goes and never localizes new atoms (the scene/style phrases stay everywhere).

    MULTI-WINDOW ATOMS (a video event is an interval, not an instant). `W_root` is the root plan at LEAF resolution (n_leaves x K). An atom
    that the root plan gave more than one of this node's leaves is NOT refined -- refinement assigns an atom to exactly one child, which would
    collapse the interval to a point. Instead it stays active in every child whose region OVERLAPS the atom's current windows, i.e. the root
    plan is simply restricted to the child's leaves. Atoms that occupy a single leaf of the node (every image atom, and a video atom whose
    interval has already been narrowed to one window) are refined exactly as before, so image behaviour is unchanged. Under route_level
    "once" nothing here runs at all: `weights_to_nodes` takes the max over each node's leaves and already handles intervals."""
    B = state.B; b = len(node.children); regions = child_regions(node)
    active = [j for j in range(spec["K"]) if node.w[j] > 0.5 and (localized is None or j in localized)]
    W = np.repeat(node.w[None], b, 0).astype(np.float64)
    if W_root is not None and active:
        multi = [j for j in active if float(np.asarray(W_root)[node.leaf_ids, j].sum()) > 1.5]
        for j in multi:
            for k, c in enumerate(node.children): W[k, j] = node.w[j] * float(np.asarray(W_root)[c.leaf_ids, j].max() >= 0.5)
        active = [j for j in active if j not in multi]
    if not active: return W
    Ds = B.delta_exact(node.latent, spec, active, regions, CFG["probe_ts"], seed, w_base=node.w, n_eps=CFG["probe_eps"], return_all=True); D = to_np(Ds.mean(0))
    r, cls, _ = responsibilities(D, CFG["region_norm"], CFG["beta"])
    if CFG["loc_rule"] == "ztest": cls, _ = classify_ztest(Ds)
    if CFG["entity_rule"] == "argmax":                       # typed condition: every active (entity) atom is refined to its argmax child, no classification
        for i, j in enumerate(active): W[:, j] = 0.0; W[int(np.argmax(r[:, i])), j] = node.w[j]
    else:
        for i, j in enumerate(active):
            if cls[i] == "localized": W[:, j] = 0.0; W[int(np.argmax(r[:, i])), j] = node.w[j]
    return W

def routed_words(W, spec):
    """Human-readable plan: window -> words routed there (only words that are not everywhere). An atom routed to SEVERAL windows (a video
    event holds over an interval) is listed once under all of them: the key is the window index for a single window, "first-last" for a
    contiguous range ("0-2") and a comma list otherwise. Words sharing the same window set are grouped, as before."""
    n_reg = W.shape[0]; groups = {}
    for j in range(spec["K"]):
        if W[:, j].min() >= 0.5: continue
        ks = tuple(k for k in range(n_reg) if W[k, j] >= 0.5) or (int(np.argmax(W[:, j])),)
        groups.setdefault(ks, []).append(spec["words"][j])
    out = {}
    for ks in sorted(groups):
        key = int(ks[0]) if len(ks) == 1 else (f"{ks[0]}-{ks[-1]}" if list(ks) == list(range(ks[0], ks[-1] + 1)) else ",".join(map(str, ks)))
        out[key] = " ".join(groups[ks])
    return out

def save_wide(img, path, quality=90):
    """JPEG cannot exceed 65500 px; beyond that the composite is stored at half resolution (metrics never read this file)."""
    while img.width > 65000: img = img.resize((img.width // 2, img.height // 2))
    img.save(path, quality=quality)

# ---------------- one tree ----------------
def run_tree(scene, seed, depth, routing, relay, route_level="once", t5=False, stage="", resume=True):
    set_depth(depth); B = state.B; levels, root, spec = get_root(scene, seed, t5); n_reg = n_leaves(); t_all = time.time(); out_dir = CFG["out_dir"]
    atom_level = "phrase" if routing.endswith("_phrase") else "word"; routing_base = routing[:-7] if atom_level == "phrase" else routing
    if atom_level == "phrase": spec = B.encode_text(scene, t5=t5, atom_level="phrase"); root.w = np.ones(spec["K"])   # root already sampled; only its cond bookkeeping changes
    rec_key = dict(stage=stage, scene=scene["name"], seed=seed, depth=depth, routing=routing, relay=relay, route_level=route_level, t5=bool(t5))
    stem = f"{out_dir}/final_{stage}_{scene['name']}_s{seed}_d{depth}_{routing}_{relay.replace(':', '').replace('@', '-')}_{route_level}_t5{int(bool(t5))}"
    fname = (stem + "_strip.jpg") if is_video() else (stem + ".jpg")     # video: the filmstrip is the composite the sheets and the resume check read
    if resume:
        prev = [l for l in load_logs() if _key(l) == _key(rec_key)]
        if prev and os.path.exists(fname):
            from PIL import Image; print(f"  [{stage}] resume: {os.path.basename(fname)}"); return prev[-1], Image.open(fname).convert("RGB"), root
    for lvl in levels[1:]:
        for n in lvl: n.latent = n.image = None; n.dets = []
    info = {}
    W, info = route_root(root, spec, routing_base, n_reg)
    localized = set(spec["entity_atoms"]) & set(int(j) for j in np.where(W.min(0) < 0.5)[0])   # per-level refinement only ever moves entity atoms
    if route_level == "once": weights_to_nodes(levels, W)
    else: root.w = np.ones(spec["K"]); info = dict(info, root_localized=sorted(localized))
    lvls = B.levels(CFG["steps"]); qs = [o["query"] for o in spec["objects"]]
    if CFG.get("child_levels"): lvls = [float(v) for v in CFG["child_levels"]] + [0.0]   # the children's own level list (v42); the root keeps CFG["levels"]
    for d in range(1, depth + 1):
        parents, nodes = levels[d - 1], levels[d]
        if route_level == "hier":
            for p in parents:
                Wc = factorize_node_hard(p, spec, seed, localized, W_root=W)
                for k, c in enumerate(p.children): c.w = Wc[k]
            if d == depth:
                W = np.stack([n.w for n in nodes]); info["n_head_fallback"] = head_fallback(W, None, spec, n_reg)
                for k, n in enumerate(nodes): n.w = W[k]
        if relay.startswith("field_bg"):
            if d == 1: ensure_bg_root(root, spec, seed)
            for n in nodes: n.bg_cond = B.make_cond(spec, bg_weights(spec, n.w))
        for n in nodes: n.cond = ((B.text_cond(scene["broadcast_prompt"]) if scene.get("broadcast_prompt") else root_cond(scene, spec)) if routing_base == "broadcast_root" else B.make_cond(spec, n.w)); prepare_child(n, relay)   # broadcast_root: every window gets the ROOT's own text (stage 3 baseline), or the scene's `broadcast_prompt` (v49)
        if relay.startswith("field_bg"): run_level(nodes, lvls, CFG["cfg"], seed, relay, bg=True)     # object-free pass first: it is the next level's field source
        (run_level_seq if CFG.get("compose", "blend") == "handoff" else run_level)(nodes, lvls, CFG["cfg"], seed, relay)
        if not is_video():                                                                           # video decodes the COMPOSITE once, not each window (the VAE is causal in time)
            for n in nodes: n.image = B.decode(n.latent)
    leaves = levels[-1]
    W_leaf = np.stack([n.w for n in leaves]); plan = plan_positions(W_leaf, spec)
    if is_video():
        from . import video as V
        comp_lat = V.compose_level(leaves); frames = V.decode_windows(leaves) if CFG.get("video_decode", "composite") == "windows" else V.decode_frames(comp_lat)
        lm = V.leaf_metrics_video(leaves, spec, root, plan, frames); comp = V.filmstrip(frames)
        vid_path = V.save_video(frames, stem + ".mp4"); lm["video_file"] = os.path.basename(vid_path); lm["n_latent_frames"] = int(comp_lat.shape[-1])
    else:
        for n in leaves: n.dets = B.detect(n.image, qs, CFG["det_thr"])
        lm = leaf_metrics(leaves, spec, root, plan); comp = compose_level(leaves)
    log = dict(**rec_key, atom_level=atom_level, atom_mode=CFG["atom_mode"], aspect=config.aspect(), tag=tag(), git=git_rev(), plan={q: round(v, 3) for q, v in plan.items()}, routed=routed_words(W_leaf, spec),
               n_head_fallback=int(info.get("n_head_fallback", 0)), n_localized=int((W_leaf.min(0) < 0.5).sum()), loc_rule=CFG["loc_rule"],
               route_D=(np.round(info["D"], 6).tolist() if info.get("D") is not None else None), route_z=(np.round(info["z"], 2).tolist() if info.get("z") is not None else None),
               route_classes=(list(map(str, info["classes"])) if info.get("classes") is not None else None),
               route_classes_probe=(list(map(str, info["classes_probe"])) if info.get("classes_probe") is not None else None),
               context_would_localize=int(info.get("context_would_localize", 0)), atoms=spec["words"] if spec["K"] <= 12 else None,
               route_A=(np.round(info["A"], 4).tolist() if info.get("A") is not None else None),
               **({k: info[k] for k in ("attn_profile", "attn_time_cfg", "regions", "rule", "del_r", "del_time_cfg", "attn_profile_plain") if k in info}),
               **({"route_A_thr": np.round(info["A_thr"], 4).tolist()} if info.get("A_thr") is not None else {}),
               w_leaf=(None if bool(np.all((np.abs(W_leaf) < 1e-6) | (np.abs(W_leaf - 1) < 1e-6))) else np.round(W_leaf, 3).tolist()),
               root_seed=root.search["chosen_seed"], root_score=root.search["candidates"][[c["seed"] for c in root.search["candidates"]].index(root.search["chosen_seed"])]["score"],
               root=root_metrics_of(root, spec), leaf_metrics=lm, time_s=round(time.time() - t_all, 1), vram_gb=round(B.vram_gb(), 2))
    if is_video():
        if CFG["fidelity"]:
            rf = root.frames[::max(1, CFG["video_iqa_every"])] if root.frames else []
            log["fidelity"] = dict(root=B.fidelity(rf), leaves=lm.get("clipiqa_frames", {}), intra_lpips=lm.get("joint_lpips"))
        log["artifacts"] = None                                       # the image seam/streak/saturation scores are spatial; the video seams are in TIME (leaf_metrics.joint_lpips)
    else:
        if CFG["fidelity"]: log["fidelity"] = dict(root=B.fidelity([root.image]), leaves=B.fidelity([core_crop(n) for n in leaves]), intra_lpips=B.intra_lpips(comp))
        from .artifacts import artifact_scores; log["artifacts"] = artifact_scores(comp, root.image, CFG["core_px"])
    save_wide(comp, fname); append_log(log)
    print(f"  [{stage}] {scene['name']} s{seed} 1:{log['aspect']} t5={int(bool(t5))} {routing}/{relay}/{route_level}: counts {lm['count_sum']} | off-owner {lm['off_owner_count']} | "
          f"owner h {{{', '.join(f'{k[2:9]}:{v:.2f}' for k, v in lm['owner_box_h_frac'].items())}}} | pos err prompt {{{', '.join(f'{k[2:9]}:{v:.2f}' for k, v in lm['pos_err_prompt'].items())}}} | disp {lm['row_profile_dispersion']:.1f} | {log['time_s']}s")
    return log, comp, root

def label_of(log):
    lm = log["leaf_metrics"]; fd = log.get("fidelity", {}); fl, fr = fd.get("leaves", {}), fd.get("root", {})
    sh = lambda d: "{" + ", ".join(f"{k[2:9]}:{v:.2f}" if isinstance(v, float) else f"{k[2:9]}:{v}" for k, v in d.items()) + "}"
    return (f"{log['routing']} / {log['relay']} / {log['route_level']} / T5 {'on' if log['t5'] else 'off'} | 1:{log['aspect']} | counts {sh(lm['count_sum'])} | off-owner {sh(lm['off_owner_count'])} | "
            f"owner h {sh(lm['owner_box_h_frac'])} | pos err vs prompt {sh(lm['pos_err_prompt'])} vs plan {sh(lm['pos_err_plan'])} | disp {lm['row_profile_dispersion']:.1f} | "
            f"NIQE {fl.get('niqe', float('nan')):.2f}/{fr.get('niqe', float('nan')):.2f} | CLIP-IQA {fl.get('clipiqa', float('nan')):.2f}/{fr.get('clipiqa', float('nan')):.2f} | routed {log['routed']}")

# ---------------- stages ----------------
def stage_route_tune():
    """Stage 0: on the roots only. For each (scene, seed, t5) and each probe config: routing accuracy + false positives. Attention is logged
    as a diagnostic column only. Picks the best exact config for the later stages; returns rows."""
    banner("Stage 0 — route_tune (roots only): which probe setting sends the head words to their stated windows, with T5 off vs on?",
           "Score = routing accuracy (head one-hot in the stated window), false positives (non-object words localized). Attention: diagnostic column.")
    B = state.B; set_depth(CFG["depth_main"]); n_reg = n_leaves(); rows = []; path = f"{CFG['out_dir']}/route_tune.json"
    for sid in CFG["scene_ids"]:
        scene = scenes_of(sid)
        for t5 in CFG["t5_arms"]:
            for seed in CFG["seeds_tune"]:
                levels, root, spec = get_root(scene, seed, t5); atoms = list(range(spec["K"])); regions = core_regions(root, n_reg); s_root = root.search["chosen_seed"]
                for cfg in CFG["route_tune_grid"]:
                    t0 = time.time(); Ds = B.delta_exact(root.latent, spec, atoms, regions, cfg["probe_ts"], s_root, n_eps=cfg["probe_eps"], return_all=True); D = Ds.mean(0)
                    for mode in ("hard", "hard_group"):
                        W, info = route_exact(D, mode, spec, n_reg, region_norm=cfg["region_norm"], Ds=Ds if cfg["probe_eps"] >= 2 else None); acc, fp = route_accuracy(W, spec, n_reg, info["classes"])
                        rows.append(dict(kind=f"exact_{mode}", cfg=json.dumps(cfg), scene=scene["name"], seed=seed, t5=bool(t5), acc=acc, fp=fp, n_fb=info["n_head_fallback"],
                                         root_score=root.search["candidates"][[c["seed"] for c in root.search["candidates"]].index(s_root)]["score"], routed=routed_words(W, spec), s=round(time.time() - t0, 1)))
                if CFG["attn_diag"]:
                    try:
                        t0 = time.time(); A = B.attn_mass(root.latent, spec, atoms, regions, CFG["attn_ts"], s_root, CFG["attn_blocks"])
                        W, info = route_attn_diag(A, spec, n_reg); acc, fp = route_accuracy(W, spec, n_reg, info["classes"])
                        rows.append(dict(kind="attn_DIAG", cfg=json.dumps(dict(attn_ts=CFG["attn_ts"], tau=CFG["attn_tau_loc"])), scene=scene["name"], seed=seed, t5=bool(t5), acc=acc, fp=fp, n_fb=0, routed=routed_words(W, spec), s=round(time.time() - t0, 1)))
                    except Exception as e: print("  attn diag failed:", repr(e)[:100])
                json.dump(rows, open(path, "w"), indent=1)
    agg = {}
    for r in rows: agg.setdefault((r["kind"], r["t5"], r["cfg"]), []).append(r)
    print("\n| kind | T5 | config | routing acc | false pos | head fallback | s/root |"); print("|---|---|---|---|---|---|---|"); best = None
    for (kind, t5, cfg), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        acc = float(np.mean([r["acc"] for r in rs])); fp = float(np.mean([r["fp"] for r in rs])); fb = float(np.mean([r["n_fb"] for r in rs]))
        print(f"| {kind} | {'on' if t5 else 'off'} | {cfg} | {acc:.2f} | {fp:.1f} | {fb:.1f} | {np.mean([r['s'] for r in rs]):.1f} |")
        if kind == "exact_hard" and (best is None or (acc, -fp) > best[0]): best = ((acc, -fp), json.loads(cfg), t5)
    if best:
        CFG.update(best[1]); print(f"  -> exact config for later stages: {best[1]} (acc {best[0][0]:.2f}, fp {-best[0][1]:.1f}; best T5 arm: {'on' if best[2] else 'off'})")
        if CFG["length_t5"] is None: CFG["length_t5"] = bool(best[2])
    return rows

def stage_grid(name, seeds, depth, combos, show_sheets=None):
    """combos: list of (routing, relay, route_level, t5). One sheet per (scene, seed)."""
    logs = []; show_sheets = CFG["show_sheets"] if show_sheets is None else show_sheets
    for sid in CFG["scene_ids"]:
        scene = scenes_of(sid)
        for seed in seeds:
            rows, root = [], None
            for routing, relay, rl, t5 in combos:
                log, comp, root = run_tree(scene, seed, depth, routing, relay, rl, t5, stage=name); logs.append(log); rows.append((label_of(log), comp))
            if show_sheets:
                banner(f"{name}: {scene['name']} seed {seed} — root (last arm's T5 setting) and one chunked strip per arm", f"1:{config.aspect()}; ideal: counts 1/1/1, off-owner 0, owner h ≥ root h, low pos err, low dispersion", color="#2da44e")
                show(sheet(f"{scene['name']} s{seed} | {full_prompt(scene)}", root, rows, f"{CFG['out_dir']}/sheet_{name}_{scene['name']}_s{seed}.jpg"), 1536, q=70)
    return logs

def stage_routing():
    banner("Stage 1 — routing × T5 at 1:16, relay = " + CFG["relay_main"], "Same root per (scene, seed, T5); only the word split differs.", color="#bf8700")
    combos = [(r, CFG["relay_main"], "once", t5) for t5 in CFG["t5_arms"] for r in CFG["routing_arms"]]
    logs = stage_grid("routing", CFG["seeds_tree"], CFG["depth_main"], combos); summarize(logs, ["t5", "routing"]); return logs
def stage_relay():
    t5 = CFG["length_t5"] if CFG["length_t5"] is not None else CFG["t5_arms"][-1]
    banner(f"Stage 2 — relay arms at 1:16, routing = {CFG['routing_main']}, T5 {'on' if t5 else 'off'}", "Same root and word split; only what the parent hands down differs.", color="#bf8700")
    logs = stage_grid("relay", CFG["seeds_tree"], CFG["depth_main"], [(CFG["routing_main"], r, "once", t5) for r in CFG["relay_arms"]]); summarize(logs, ["relay"]); return logs
def stage_length():
    t5 = CFG["length_t5"] if CFG["length_t5"] is not None else CFG["t5_arms"][-1]
    banner("Stage 3 — length: 1:4 → 1:64, route once vs per level, relays " + str(CFG["length_relays"]), "Route-once regions shrink with depth (24 → 1.5 tokens); per-level routing keeps 24 tokens at every level.", color="#8250df")
    logs = []
    for sid in CFG["scene_ids"]:
        scene = scenes_of(sid)
        for seed in CFG["seeds_length"]:
            for depth in CFG["depths_length"]:
                rows, root = [], None
                for rl in CFG["length_route_levels"]:
                    for relay in CFG["length_relays"]:
                        log, comp, root = run_tree(scene, seed, depth, CFG["length_routing"], relay, rl, t5, stage="length"); logs.append(log); rows.append((label_of(log), comp))
                if CFG["show_sheets"]:
                    banner(f"length: {scene['name']} seed {seed} depth {depth} (1:{config.aspect()})", color="#8250df")
                    show(sheet(f"{scene['name']} s{seed} d{depth}", root, rows, f"{CFG['out_dir']}/sheet_length_{scene['name']}_s{seed}_d{depth}.jpg"), 1536, q=65)
    summarize(logs, ["aspect", "route_level", "relay"]); return logs

def summarize(logs, keys, file=None):
    """Mean over scenes/seeds of the main metrics, grouped by `keys`. Prints (and optionally writes) a markdown table."""
    groups = {}
    for l in logs: groups.setdefault(tuple(l[k] for k in keys), []).append(l)
    hdr = ("| " + " | ".join(keys) + " | n | count err | off-owner | pos err prompt | pos err plan | plan err prompt | owner h / root h | disp | NIQE leaf/root | CLIP-IQA leaf/root | intra-LPIPS | joint row / horizon | s/tree |")
    lines = [hdr, "|" + "---|" * (len(keys) + 13)]
    for g, ls in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        lm = [l["leaf_metrics"] for l in ls]
        ce = np.mean([np.mean([abs(v - 1) for v in m["count_sum"].values()]) for m in lm]); off = np.mean([sum(m["off_owner_count"].values()) for m in lm])
        pe = [np.mean(list(m["pos_err_prompt"].values())) for m in lm if m["pos_err_prompt"]]; pp = [np.mean(list(m["pos_err_plan"].values())) for m in lm if m["pos_err_plan"]]
        pl = [np.mean(list(m["plan_err_prompt"].values())) for m in lm if m["plan_err_prompt"]]
        oh = [np.mean(list(m["owner_box_h_frac"].values())) for m in lm if m["owner_box_h_frac"]]; rh = [np.mean(list(m["root_box_h_frac"].values())) for m in lm if m["root_box_h_frac"]]
        disp = np.mean([m["row_profile_dispersion"] for m in lm]); fl = [l.get("fidelity", {}).get("leaves", {}) for l in ls]; fr = [l.get("fidelity", {}).get("root", {}) for l in ls]
        f = lambda xs: f"{np.mean(xs):.2f}" if xs else "—"
        nq = f"{f([x['niqe'] for x in fl if 'niqe' in x])}/{f([x['niqe'] for x in fr if 'niqe' in x])}"; cq = f"{f([x['clipiqa'] for x in fl if 'clipiqa' in x])}/{f([x['clipiqa'] for x in fr if 'clipiqa' in x])}"
        il = [l.get("fidelity", {}).get("intra_lpips") for l in ls]; il = [v for v in il if v is not None]
        ar = [l.get("artifacts") or {} for l in ls]; jr = [x["joint_row_jump"] for x in ar if "joint_row_jump" in x]; hj = [x["horizon_jump"] for x in ar if "horizon_jump" in x]
        lines.append(f"| {' | '.join(str(x) for x in g)} | {len(ls)} | {ce:.2f} | {off:.1f} | {f(pe)} | {f(pp)} | {f(pl)} | {f(oh)} / {f(rh)} | {disp:.1f} | {nq} | {cq} | {f(il)} | {f(jr)} / {f(hj)} | {np.mean([l['time_s'] for l in ls]):.0f} |")
    txt = "\n".join(lines); print("\n" + txt)
    if file: open(file, "a").write(f"\n### by {', '.join(keys)}\n\n{txt}\n")
    return txt

# ---------------- v38: root check, phrase-level attention arms, prompt + root + final sheets ----------------
V38_STAGE = "routing_v38"
def _v38_t5(): return bool(CFG["t5_arms"][-1])                        # v38 runs one T5 setting (on); the dummy's t5_arms end with True as well

def root_position_check(root, spec):
    """EVALUATION ONLY (never used for selection): does each entity appear in its stated third of the root (left / middle / right, from
    `nominal`)? OWL-ViT detections of the entity's query at det_thr; pos_ok = at least one detection whose box centre lies in the stated
    third; only_ok = pos_ok and no detection in another third; `thirds` = detections per third. all_ok = every entity pos_ok."""
    from .scenes import third_of
    qs = [o["query"] for o in spec["objects"]]; W = float(root.image.width); out = {}
    for i, o in enumerate(spec["objects"]):
        th = [0, 0, 0]; best = None
        for d in root.dets:
            if d["label"] != i: continue
            k = third_of((d["box"][0] + d["box"][2]) / 2 / W); th[k] += 1
            if best is None or d.get("score", 0) > best[0]: best = (d.get("score", 0), k)
        st = third_of(o["nominal"]); ok = th[st] > 0
        out[o["query"]] = dict(stated=st, thirds=th, pos_ok=bool(ok), only_ok=bool(ok and sum(th) == th[st]), best_third=(best[1] if best else None))
    return dict(per=out, pos_ok={q: v["pos_ok"] for q, v in out.items()}, all_ok=bool(all(v["pos_ok"] for v in out.values())) if out else False)

def _pos_str(pos_ok, spec_objects):
    from .scenes import THIRDS, third_of
    return " ".join(f"{THIRDS[third_of(o['nominal'])][0].upper()}:{o['head']}{'+' if pos_ok.get(o['query']) else '-'}" for o in spec_objects)

def _pbis(scores, oks):
    """Point-biserial correlation of the probe score with all_ok (None if one class is empty or the scores are constant)."""
    x = np.asarray(scores, np.float64); y = np.asarray(oks, np.float64)
    if len(x) < 3 or y.min() == y.max() or x.std() == 0: return None
    return float(np.corrcoef(x, y)[0, 1])

def stage_root_check():
    """v38 stage 1. For each (scene, seed): sample `root_candidates` roots (the same search `get_root` always runs; its pick is unchanged and
    becomes the shared root of the routing stage), save EVERY candidate as roots/<scene>_s<seed>_c<i>.jpg, and record per candidate the probe
    compliance score used for selection and the evaluation-only OWL-ViT third check. Writes root_check.json, root_check.md and one
    root_sheet_<scene>.jpg per scene. Answers: does root selection pick roots that follow the prompt?"""
    banner("v38 stage 1 — root_check: does the probe's root selection pick roots that follow the prompt?",
           "Every candidate saved; selection = probe compliance (worst entity); evaluation = OWL-ViT: is each entity in its stated third?")
    B = state.B; set_depth(CFG["depth_main"]); out = CFG["out_dir"]; os.makedirs(f"{out}/roots", exist_ok=True); t5 = _v38_t5(); rows = []
    for sid in CFG["scene_ids"]:
        scene = scenes_of(sid)
        for seed in CFG["seeds_v38"]:
            cands = []
            def cb(c, s, root, spec, score, per, rm, scene=scene, seed=seed, cands=cands):
                path = f"{out}/roots/{scene['name']}_s{seed}_c{c}.jpg"; root.image.save(path, quality=92); pc = root_position_check(root, spec)
                cands.append(dict(c=c, seed=s, score=round(float(score), 3), per={k: round(float(v), 3) for k, v in per.items()}, pos_ok=pc["pos_ok"], all_ok=pc["all_ok"],
                                  thirds={q: v["thirds"] for q, v in pc["per"].items()}, only_ok={q: v["only_ok"] for q, v in pc["per"].items()},
                                  dets=[dict(box=[round(float(x), 1) for x in d["box"]], score=round(float(d.get("score", 0)), 3), label=int(d["label"])) for d in root.dets],
                                  root_pos_err_prompt=rm.get("pos_err_prompt"), file=os.path.relpath(path, out)))
            _ROOTS.pop((scene["name"], seed, bool(t5)), None)          # force the search so every candidate passes through `cb`
            levels, root, spec = get_root(scene, seed, t5, on_candidate=cb)
            picked = next(c["c"] for c in cands if c["seed"] == root.search["chosen_seed"])
            rows.append(dict(scene=scene["name"], seed=seed, t5=bool(t5), prompt=full_prompt(scene), queries=[o["query"] for o in spec["objects"]],
                             heads=[o["head"] for o in spec["objects"]], stated={o["query"]: o["nominal"] for o in spec["objects"]},
                             candidates=cands, picked=picked, picked_all_ok=cands[picked]["all_ok"], picked_score=cands[picked]["score"]))
            json.dump(rows, open(f"{out}/root_check.json", "w"), indent=1)
            print(f"  root_check {scene['name']} s{seed}: scores {[c['score'] for c in cands]} | all_ok {[int(c['all_ok']) for c in cands]} | picked c{picked} (all_ok {int(cands[picked]['all_ok'])})")
    txt = root_check_report(rows, f"{out}/root_check.md")
    for sid in CFG["scene_ids"]: root_sheet(scenes_of(sid), [r for r in rows if r["scene"] == scenes_of(sid)["name"]], out)
    print(txt); return rows

def root_check_stats(rows):
    cands = [c for r in rows for c in r["candidates"]]
    if not cands: return {}
    ok = [c for c in cands if c["all_ok"]]; nok = [c for c in cands if not c["all_ok"]]
    chance = [np.mean([c["all_ok"] for c in r["candidates"]]) for r in rows]
    qs = list(dict.fromkeys(q for r in rows for q in r["queries"]))
    return dict(n_groups=len(rows), n_candidates=len(cands), frac_candidates_all_ok=float(np.mean([c["all_ok"] for c in cands])),
                frac_picks_all_ok=float(np.mean([r["picked_all_ok"] for r in rows])), chance_pick_all_ok=float(np.mean(chance)),
                frac_groups_any_ok=float(np.mean([any(c["all_ok"] for c in r["candidates"]) for r in rows])),
                mean_score_ok=(float(np.mean([c["score"] for c in ok])) if ok else None), mean_score_not_ok=(float(np.mean([c["score"] for c in nok])) if nok else None),
                pbis_score_all_ok=_pbis([c["score"] for c in cands], [c["all_ok"] for c in cands]),
                entity_pos_ok_rate={q: float(np.mean([c["pos_ok"][q] for c in cands if q in c["pos_ok"]])) for q in qs},
                entity_pos_ok_rate_picked={q: float(np.mean([r["candidates"][r["picked"]]["pos_ok"][q] for r in rows if q in r["queries"]])) for q in qs})

def root_check_report(rows, path=None):
    """Markdown: one row per (scene, seed) with every candidate's score / pos_ok, the pick and whether it is all_ok; then the overall numbers."""
    L = ["| scene | seed | candidates: score [pos L/M/R] (picked = *) | picked | picked all_ok | n all_ok |", "|---|---|---|---|---|---|"]
    for r in rows:
        cs = []
        for c in r["candidates"]:
            po = "".join("+" if c["pos_ok"][q] else "-" for q in r["queries"])
            cs.append(f"{'*' if c['c'] == r['picked'] else ''}c{c['c']} {c['score']:.2f} [{po}]{' ok' if c['all_ok'] else ''}")
        L.append(f"| {r['scene']} | {r['seed']} | {'; '.join(cs)} | c{r['picked']} | {'yes' if r['picked_all_ok'] else 'no'} | {sum(c['all_ok'] for c in r['candidates'])}/{len(r['candidates'])} |")
    st = root_check_stats(rows); f = lambda v: "—" if v is None else f"{v:.2f}"
    if st:
        L += ["", f"**Overall** ({st['n_groups']} (scene, seed) groups, {st['n_candidates']} candidates; pos_ok = OWL-ViT detection of the entity in its stated third, evaluation only):", "",
              f"* fraction of candidates all_ok: **{f(st['frac_candidates_all_ok'])}**",
              f"* fraction of picks all_ok: **{f(st['frac_picks_all_ok'])}** (chance = mean all_ok rate within a group: {f(st['chance_pick_all_ok'])}; groups with ≥ 1 all_ok candidate: {f(st['frac_groups_any_ok'])})",
              f"* probe score, mean over all_ok candidates {f(st['mean_score_ok'])} vs not all_ok {f(st['mean_score_not_ok'])}; point-biserial r(score, all_ok) = {f(st['pbis_score_all_ok'])}",
              "* per-entity pos_ok rate, all candidates / picks: " + ", ".join(f"{q}: {f(v)} / {f(st['entity_pos_ok_rate_picked'].get(q))}" for q, v in st["entity_pos_ok_rate"].items())]
    txt = "\n".join(L)
    if path: open(path, "w").write(f"# root_check — {CFG['out_dir']}\n\n{txt}\n")
    return txt

def root_sheet(scene, rows, out):
    """root_sheet_<scene>.jpg: the prompt, then every candidate (rows = seeds, columns = candidates) with its detector boxes, labelled
    seed / candidate / probe score / pos_ok; the probe's pick is outlined in yellow."""
    from PIL import Image
    from .display import text_block, outline, draw_dets, vstack, grid
    if not rows: return None
    cells, ncol = [], max(len(r["candidates"]) for r in rows); heads = rows[0]["heads"]; objs = scene["objects"]
    for r in rows:
        for c in r["candidates"]:
            im = draw_dets(Image.open(f"{out}/{c['file']}").convert("RGB"), c["dets"], heads)
            if c["c"] == r["picked"]: im = outline(im)
            lab = f"s{r['seed']} c{c['c']} (seed {c['seed']}) | score {c['score']:.2f} | {_pos_str(c['pos_ok'], objs)} | {'ALL OK' if c['all_ok'] else 'not ok'}{' | PICKED' if c['c'] == r['picked'] else ''}"
            cells.append(vstack([text_block(im.width, [lab], sz=14, fg=(255, 214, 0) if c["c"] == r["picked"] else (235, 235, 235)), im], gap=0))
        cells += [Image.new("RGB", cells[-1].size, (60, 60, 60))] * (ncol - len(r["candidates"]))
    g = grid(cells, ncol)
    hdr = text_block(g.width, [f"{scene['name']} - root candidates (rows = seeds {[r['seed'] for r in rows]}, columns = candidates; yellow = the probe's pick)",
                               f"PROMPT: {rows[0]['prompt']}",
                               "score = probe compliance used for selection (worst entity's responsibility in its stated window). L/M/R:head+/- = OWL-ViT finds the entity in its stated third "
                               "(evaluation only; white lines = thirds). A missing box is not proof of absence."], sz=20)
    s = vstack([hdr, g]); path = f"{out}/root_sheet_{scene['name']}.jpg"; s.save(path, quality=85); show(s, 1536, q=70); return path

def stage_routing_v38():
    """v38 stage 2: routing arms x relays at depth_main (1:16), T5 on, on the probe-picked root of stage 1 (shared across arms)."""
    t5 = _v38_t5(); depth = CFG["depth_main"]
    banner(f"v38 stage 2 — routing {CFG['routing_arms']} × relay {CFG['relay_arms']} at depth {depth}, T5 {'on' if t5 else 'off'}",
           "Same root per (scene, seed) = the probe's pick from root_check. attn_* arms are comparison arms (attention mass at the root), not the method.", color="#bf8700")
    logs = []
    for sid in CFG["scene_ids"]:
        scene = scenes_of(sid)
        for seed in CFG["seeds_v38"]:
            for routing in CFG["routing_arms"]:
                for relay in CFG["relay_arms"]:
                    log, comp, root = run_tree(scene, seed, depth, routing, relay, "once", t5, stage=V38_STAGE); logs.append(log)
    return logs

def _final_path(out, l):
    return f"{out}/final_{l['stage']}_{l['scene']}_s{l['seed']}_d{l['depth']}_{l['routing']}_{l['relay'].replace(':', '').replace('@', '-')}_{l['route_level']}_t5{int(bool(l['t5']))}.jpg"

def plan_text(l):
    """The routed plan as text: window -> phrase for 0/1 plans; for soft plans (attn_scale) every entity atom with its non-zero windows."""
    if l.get("w_leaf") is not None and l.get("atoms"):
        W = np.asarray(l["w_leaf"]); parts = []
        for j, a in enumerate(l["atoms"]):
            col = W[:, j]
            if np.allclose(col, 1.0): continue
            parts.append(f"'{a}': " + ", ".join(f"w{k} {v:.2f}" for k, v in enumerate(col) if v > 1e-6))
        return " | ".join(parts)
    return "; ".join(f"w{k}: {v}" for k, v in (l.get("routed") or {}).items()) or "(nothing routed: broadcast)"

def _sh(d, nd=2): return "{" + ", ".join(f"{k.replace('a ', '', 1)}: {round(v, nd) if isinstance(v, float) else v}" for k, v in d.items()) + "}"

def make_v38_sheets(out_dir=None, stage=V38_STAGE):
    """Per (scene, seed): v38_<scene>_s<seed>.jpg = full prompt, the picked root at 2x (score, pos_ok), then one row per (routing, relay)
    arm: the composite cut into 3072-px chunks, each resized to 1536 px wide, captioned with the arm, the routed plan, counts, off-owner
    and position error vs prompt. Also v38_index.md. Reads only files (root_check.json, logs.jsonl, roots/, final_*.jpg), so it also
    runs post hoc on a finished run."""
    from PIL import Image; Image.MAX_IMAGE_PIXELS = None
    from .display import text_block, vstack, chunked_strip, caption
    out = out_dir or CFG["out_dir"]; rc = json.load(open(f"{out}/root_check.json")) if os.path.exists(f"{out}/root_check.json") else []
    logs = [l for l in load_logs(f"{out}/logs.jsonl") if l.get("stage") == stage]
    cfg = json.load(open(f"{out}/config.json")) if os.path.exists(f"{out}/config.json") else CFG; core = cfg.get("core_px", 768)
    arms_order = [(r, y) for r in cfg.get("routing_arms", []) for y in cfg.get("relay_arms", [])]
    groups = {}
    for l in logs: groups.setdefault((l["scene"], l["seed"]), {})[(l["routing"], l["relay"])] = l    # last log per arm wins
    index = [f"# v38 sheets — {out}", "", "One sheet per (scene, seed): prompt, the probe-picked root (2x), then every (routing, relay) arm at 1:16 as 2 chunks. "
             "Line = root score / all_ok, then per arm: counts (lighthouse/boat/house order), off-owner total, mean pos err vs prompt.", ""]
    paths = []
    for (scene, seed), arms in sorted(groups.items(), key=lambda kv: ([r["scene"] for r in rc].index(kv[0][0]) if kv[0][0] in [r["scene"] for r in rc] else 99, kv[0][1])):
        r = next((x for x in rc if x["scene"] == scene and x["seed"] == seed), None); W = 1536; ims = []
        prompt = r["prompt"] if r else next(iter(arms.values())).get("prompt", "")
        if not prompt:
            from .scenes import SCENES; sc = next((s for s in SCENES if s["name"] == scene), None); prompt = full_prompt(sc) if sc else ""
        ims.append(text_block(W, [f"{scene} seed {seed} - v38 routing x relay at 1:{next(iter(arms.values()))['aspect']}", f"PROMPT: {prompt}"], sz=18))
        if r:
            c = r["candidates"][r["picked"]]; im = Image.open(f"{out}/{c['file']}").convert("RGB"); im = im.resize((min(W, im.width * 2), im.height * 2), Image.LANCZOS)
            pad = Image.new("RGB", (W, im.height), (40, 40, 40)); pad.paste(im, (0, 0))
            objs = [dict(query=q, head=h, nominal=r["stated"][q]) for q, h in zip(r["queries"], r["heads"])]
            ims.append(caption(pad, [f"ROOT (2x) = probe pick c{c['c']} (seed {c['seed']}) | probe score {c['score']:.2f} | pos_ok {_pos_str(c['pos_ok'], objs)} | "
                                     f"{'ALL OK' if c['all_ok'] else 'NOT all ok'} | candidates all_ok {sum(x['all_ok'] for x in r['candidates'])}/{len(r['candidates'])}"], sz=15))
        order = [a for a in arms_order if a in arms] + [a for a in arms if a not in arms_order]; summ = []
        for a in order:
            l = arms[a]; f = _final_path(out, l)
            if not os.path.exists(f): continue
            comp = Image.open(f).convert("RGB"); scale = comp.width / (core * 2 ** l["depth"]); lm = l["leaf_metrics"]
            strip = chunked_strip(comp, chunk_px=max(1, int(round(3072 * scale))), sheet_w=W)
            pe = list(lm["pos_err_prompt"].values()); art = l.get("artifacts") or {}
            lab = [f"{l['routing']} / {l['relay']} | counts {_sh(lm['count_sum'])} | off-owner {_sh(lm['off_owner_count'])} | pos err vs prompt {_sh(lm['pos_err_prompt'])}"
                   + (f" | seam {art['seam_ratio']:.2f}" if "seam_ratio" in art else "") + (f" | joint row {art['joint_row_jump']:.2f}" if "joint_row_jump" in art else ""),
                   f"plan: {plan_text(l)}"]
            ims.append(caption(strip, lab, sz=16))
            summ.append(f"{l['routing']}/{l['relay']}: counts {list(lm['count_sum'].values())} off {sum(lm['off_owner_count'].values())} pe {np.mean(pe):.2f}" if pe else
                        f"{l['routing']}/{l['relay']}: counts {list(lm['count_sum'].values())} off {sum(lm['off_owner_count'].values())} pe —")
        s = vstack(ims)
        while s.height > 65000: s = s.resize((s.width // 2, s.height // 2))
        p = f"{out}/v38_{scene}_s{seed}.jpg"; s.save(p, quality=85); paths.append(p)
        rl = (f"root c{r['picked']} score {r['picked_score']:.2f} {'all_ok' if r['picked_all_ok'] else 'NOT all_ok'}" if r else "root ?")
        index.append(f"* [{os.path.basename(p)}]({os.path.basename(p)}) — {scene} s{seed}: {rl} | " + " · ".join(summ))
    open(f"{out}/v38_index.md", "w").write("\n".join(index) + "\n"); print(f"v38 sheets: {len(paths)} -> {out}/v38_index.md")
    return paths

def summarize_v38(logs, rc_rows=None, file=None, keys=("routing", "relay")):
    """summary.md tables for v38: grouped by `keys` (default (routing, relay)), then by scene as well. Artifact columns = every numeric
    field of `artifacts` present in the logs (seam_ratio, color_jump, ... plus any newer seam metrics), read with .get."""
    art_keys = []
    for l in logs:
        for k, v in (l.get("artifacts") or {}).items():
            if isinstance(v, (int, float)) and k not in art_keys: art_keys.append(k)
    f = lambda xs: f"{np.mean(xs):.2f}" if xs else "—"
    def table(kk):
        groups = {}
        for l in logs: groups.setdefault(tuple(l[k] for k in kk), []).append(l)
        hdr = "| " + " | ".join(kk) + " | n | count err | off-owner | pos err prompt | pos err plan | owner h / root h | disp | CLIP-IQA leaf/root |" + "".join(f" {k} |" for k in art_keys)
        L = [hdr, "|" + "---|" * (len(kk) + 8 + len(art_keys))]
        for g, ls in groups.items():
            lm = [l["leaf_metrics"] for l in ls]
            ce = [np.mean([abs(v - 1) for v in m["count_sum"].values()]) for m in lm if m["count_sum"]]; off = [sum(m["off_owner_count"].values()) for m in lm]
            pe = [np.mean(list(m["pos_err_prompt"].values())) for m in lm if m["pos_err_prompt"]]; pp = [np.mean(list(m["pos_err_plan"].values())) for m in lm if m["pos_err_plan"]]
            oh = [np.mean(list(m["owner_box_h_frac"].values())) for m in lm if m["owner_box_h_frac"]]; rh = [np.mean(list(m["root_box_h_frac"].values())) for m in lm if m["root_box_h_frac"]]
            fl = [(l.get("fidelity") or {}).get("leaves", {}) for l in ls]; fr = [(l.get("fidelity") or {}).get("root", {}) for l in ls]
            cq = f"{f([x['clipiqa'] for x in fl if 'clipiqa' in x])}/{f([x['clipiqa'] for x in fr if 'clipiqa' in x])}"
            ac = [f([(l.get("artifacts") or {}).get(k) for l in ls if isinstance((l.get("artifacts") or {}).get(k), (int, float))]) for k in art_keys]
            L.append(f"| {' | '.join(str(x) for x in g)} | {len(ls)} | {f(ce)} | {f(off)} | {f(pe)} | {f(pp)} | {f(oh)} / {f(rh)} | {f([m['row_profile_dispersion'] for m in lm])} | {cq} |" + "".join(f" {x} |" for x in ac))
        return "\n".join(L)
    txt = [f"# {CFG['version']} — {CFG['out_dir']} (tag {tag()}, git {git_rev()})", ""]
    if rc_rows:
        txt += ["## root_check (does the probe's pick follow the prompt?)", "", root_check_report(rc_rows), ""]
    if logs:
        txt += [f"## routing × relay (mean over scenes and seeds; count err = mean |count - 1|, off-owner = detections outside the planned window)", "", table(list(keys)), "",
                "## by scene", "", table(["scene"] + list(keys)), ""]
    out = "\n".join(txt); print(out)
    if file: open(file, "w").write(out + "\n")
    return out
