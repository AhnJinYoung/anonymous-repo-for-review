"""Video modality: composition, decoding, metrics and displays for a horizon that is TIME.

Everything before this module is axis-agnostic: the backend keeps latents as (1, C, H, W, T) with the horizon LAST, so `sampling.run_level`,
the relays, the routing probe and the growth tree are the same code as for panoramas. What is genuinely different is what you do with the
result, and that is here:

  compose_level(nodes)          overlapping windows -> ONE composite latent, blended over the halo with a linear ramp along TIME (latent space:
                                the VAE is temporally causal, so pixel-space blending of separately decoded windows would fight the decoder)
  decode_frames(latent)         composite latent -> PIL frames (1 + 4 (T_lat - 1) of them), frame-batched by the CogVideoX VAE
  leaf_metrics_video(...)       per event: how many windows show it, when it first/last appears (as a fraction of the horizon), detections
                                outside its planned window SET (an event is an INTERVAL of the horizon, so the plan may give it several
                                windows), position error vs the prompt's span and vs the plan's windows; plus joint LPIPS and CLIP-IQA
  filmstrip / save_video        the audit artifacts (one frame per 10 for the sheets, an mp4 when a writer is available)

Horizon arithmetic: latent frame j covers pixel frames [1 + 4(j-1), 1 + 4j) for j >= 1, and latent frame 0 covers pixel frame 0 alone
(CogVideoX's VAE compresses time by 4 with a causal first frame). `px_of_lat` / `lat_of_px` are the only places that know this."""
import os, shutil, subprocess, numpy as np
from PIL import Image
from . import config, state
from .config import CFG

TEMPORAL_RATIO = 4

# ---------------- horizon arithmetic ----------------
def n_px_frames(t_lat): return 1 + TEMPORAL_RATIO * (int(t_lat) - 1)
def px_of_lat(j): return 0 if int(j) <= 0 else 1 + TEMPORAL_RATIO * (int(j) - 1)
def lat_of_px(i): return 0 if int(i) <= 0 else (int(i) - 1) // TEMPORAL_RATIO + 1

# ---------------- composition along time (latent space) ----------------
def compose_level(nodes):
    """The composite latent of one level: every node is pasted at its global position and the halo overlaps are averaged with a linear
    ramp (the same 1 -> 0 weighting `metrics.compose_level` applies to image columns, here along the last axis = latent frames).
    halo_px 0: the windows abut and this is plain concatenation; compose='handoff': only the cores are pasted (the handed-off frames dropped)."""
    B = state.B; d = nodes[0].depth; Tl = CFG["core_px"] * CFG["branch"] ** d; OVL = config.overlap_lat()
    nd = len(B.noise_shape(1))                                       # 5 for video (1, C, H, W, T); the ramp broadcasts over every axis but the last
    acc = B.zeros(B.noise_shape(Tl)); wsum = B.zeros(tuple([1] * (nd - 1)) + (Tl,))
    from .geometry import ovl_of
    handoff = CFG.get("compose", "blend") == "handoff"
    for n in nodes:
        x = n.latent; t = x.shape[-1]; w = B.ones(tuple([1] * (nd - 1)) + (t,))
        if n.halo_l: o = ovl_of(n, "l"); w[..., :o] = B.linspace_w(0.0, 1.0, o) if o > 0 else w[..., :0]
        if n.halo_r:
            o = ovl_of(n, "r")
            if o > 0: w[..., -o:] = B.linspace_w(1.0, 0.0, o)         # halo_px 0 (abutting windows): no ramp, plain concatenation
        t0 = n.canvas_g0 // n.scale
        if handoff and n.ext_l:                                       # compose='handoff': drop the K handed-off copies of window k-1's frames
            x, w, t0, t = x[..., n.ext_l:], w[..., n.ext_l:], t0 + n.ext_l, t - n.ext_l
        elif getattr(n, "slot0_regular", None) is not None:           # first_slot "image": slot 0 holds an image latent -> the regular latent at that time
            x = x * 1.0; x[..., 0:1] = n.slot0_regular
        if t0 < 0 or t0 + t > Tl:                                     # edge_native "out": frames past the horizon ends are cropped
            a, b = max(0, -t0), min(t, Tl - t0); x, w, t0, t = x[..., a:b], w[..., a:b], t0 + a, b - a
        acc[..., t0:t0 + t] = acc[..., t0:t0 + t] + x * w; wsum[..., t0:t0 + t] = wsum[..., t0:t0 + t] + w
    return acc / _clamp_min(wsum, 1e-6)
def _clamp_min(x, v):
    return x.clamp(min=v) if hasattr(x, "clamp") else np.maximum(x, v)

def decode_windows(nodes):
    """CFG video_decode="windows" (v43): every leaf decoded as its OWN clip (its slot 0 -> one frame, as it was generated) and the pixel
    frames concatenated; a window with ext_l handed-off slots (compose='handoff') drops their frames (slot 0 = 1 frame, the others 4 each),
    which are window k-1's frames. For handoff K=1 this is exactly the I2V-chain decode: window k's first decoded frame (the re-encoded last
    frame of window k-1) is dropped, and its slot 1.. are decoded after the image slot they were generated after (the one-pass composite
    decodes them after window k-1's 4-frame latent instead -- a causal-decoder context the model never saw). Frame count = the composite's
    when every window but the first has a handed-off slot."""
    out = []
    for n in nodes:
        fr = state.B.decode_frames(n.latent); e = int(n.ext_l or 0)
        out.extend(fr[(1 + TEMPORAL_RATIO * (e - 1)) if e else 0:])
    return out

def decode_frames(latent, chunk_lat=None):
    """Composite latent -> list of PIL frames. Delegates to the backend (the CogVideoX VAE decodes with its own exact frame batching)."""
    return state.B.decode_frames(latent, chunk_lat=chunk_lat)

# ---------------- detections over frames ----------------
def detect_frames(frames, queries, every=None, thr=None):
    """OWL-ViT on every `every`-th frame. Returns [(frame_index, [det, ...]), ...] — detections are never aggregated before the metrics,
    because 'when does this event happen' is exactly the frame index."""
    B = state.B; every = int(every or CFG["video_det_every"]); thr = CFG["det_thr"] if thr is None else thr
    return [(i, B.detect(frames[i], queries, thr)) for i in range(0, len(frames), every)]

def _med(d): return {q: float(np.median(v)) for q, v in d.items()}
def _win_of(frac, n_win): return min(n_win - 1, max(0, int(frac * n_win)))

def _dist_to_span(x, a, b): return 0.0 if a <= x <= b else float(a - x if x < a else x - b)

def _prompt_spans(spec):
    """The interval of the horizon each event's instruction states: `span` when the scene gives one (a video event is an INTERVAL), the
    point `nominal` otherwise (then the distance below is just |x - nominal|, the old number)."""
    return {o["query"]: (tuple(float(v) for v in o["span"]) if o.get("span") else (float(o["nominal"]), float(o["nominal"]))) for o in spec["objects"]}

def _owner_windows(leaves, spec, plan):
    """The window SET the plan gave each event, read off the leaves' own routing weights (an event routed to an interval owns SEVERAL
    windows). Falls back to the single window containing `plan[q]` if the leaves carry no weights."""
    n_win = len(leaves); out = {}
    try: W = np.stack([np.asarray(n.w, np.float64) for n in leaves])
    except Exception: return {q: [_win_of(p, n_win)] for q, p in plan.items()}
    for q, j in spec["heads"].items():
        col = W[:, j]
        out[q] = sorted(int(k) for k in np.where(col >= 0.5)[0]) if col.min() < 0.5 else [int(np.argmax(col))]
    return out

def _dist_to_windows(frac, ks, n_win):
    """Distance from `frac` to the nearest of the windows `ks`, each window k spanning [k / n_win, (k + 1) / n_win]; 0 inside any of them."""
    return min(_dist_to_span(frac, k / n_win, (k + 1) / n_win) for k in ks) if ks else 0.0

def _profiles(frames, n_win, size=32):
    """Per-window mean appearance (a `size` x `size` x 3 thumbnail averaged over the window's frames) — the video analogue of the image
    row profile: it answers 'do the windows look like the same scene?' without looking at the horizon axis inside a window."""
    n = len(frames); out = []
    for k in range(n_win):
        i0, i1 = int(round(k * n / n_win)), int(round((k + 1) * n / n_win))
        sel = range(i0, max(i1, i0 + 1))
        out.append(np.mean([np.asarray(frames[min(i, n - 1)].resize((size, size), Image.BILINEAR), np.float32) for i in sel], axis=0))
    return np.stack(out)

def joint_indices(n_win, t_lat_total=None):
    """The pixel-frame index of the first frame after each window joint (the boundary between leaf k-1 and leaf k). Per-window decode of
    lock-step abutting windows (video_decode 'windows', compose 'blend', v45): every window is its own n_px_frames(core) clip."""
    if CFG.get("video_decode", "composite") == "windows" and CFG.get("compose", "blend") != "handoff" and int(CFG["halo_px"]) == 0:
        return [k * n_px_frames(CFG["core_px"]) for k in range(1, n_win)]
    return [px_of_lat(k * CFG["core_px"]) for k in range(1, n_win)]

def joint_lpips(frames, n_win):
    """Temporal coherence at the seams: LPIPS between the frames on either side of each window joint, averaged. A joint that is no worse
    than an ordinary frame-to-frame step is invisible; compare against `step_lpips`."""
    B = state.B
    if not hasattr(B, "lpips_pair"): return None, None
    js = [j for j in joint_indices(n_win) if 0 < j < len(frames)]
    if not js: return None, None
    at_joint = [B.lpips_pair(frames[j - 1], frames[j]) for j in js]
    cand = [int(v) for v in np.linspace(1, len(frames) - 1, min(4 * len(js), max(len(frames) - 2, 1)))]
    ref = [j for j in dict.fromkeys(cand) if j not in js and j >= 1]
    step = [B.lpips_pair(frames[j - 1], frames[j]) for j in ref] if ref else []
    at_joint = [v for v in at_joint if v is not None]; step = [v for v in step if v is not None]
    return (float(np.mean(at_joint)) if at_joint else None), (float(np.mean(step)) if step else None)

# ---------------- metrics ----------------
def root_metrics_video(root, spec):
    """Root-only, on the root clip's own frames: in how many frames is each event visible, when does it first appear (as a distance from
    the event's stated SPAN -- 0 if it first appears inside it), how big is it."""
    qs = [o["query"] for o in spec["objects"]]; spans = _prompt_spans(spec)
    fd = root.frame_dets or []; n = max(len(fd), 1); H = root.frames[0].height if root.frames else 1
    counts, first, hs = {}, {}, {}
    for k, (i, dd) in enumerate(fd):
        for d in dd:
            q = qs[d["label"]]; counts[q] = counts.get(q, 0) + 1; first.setdefault(q, k / max(n - 1, 1))
            hs.setdefault(q, []).append((d["box"][3] - d["box"][1]) / H)
    return dict(count={q: int(counts.get(q, 0)) for q in qs}, frames_sampled=len(fd),
                pos_err_prompt={q: round(_dist_to_span(first[q], *spans[q]), 3) for q in first}, box_h_frac=_med(hs))

def leaf_metrics_video(leaves, spec, root, plan, frames=None, every=None):
    """The per-tree numbers for a video composite. `frames` = the decoded composite (decoded here if not given).

    Per event (OWL-ViT query): `count_sum` = in how many WINDOWS the event is visible at all (ideal: the number of windows its span covers,
    because an event PERSISTS over an interval of the horizon), `det_frames` = how many sampled frames show it, `first_frac` / `last_frac` =
    when it appears and disappears as a fraction of the horizon, `off_owner_count` = sampled-frame detections outside the window SET the
    plan gave it (an event routed to an interval owns every window of that interval), `pos_err_prompt` = the distance from `first_frac` to
    the event's stated span, 0 if it first appears inside it (did the result follow the instruction?), `pos_err_plan` = the distance from
    `first_frac` to the nearest owner window's own span, 0 if inside (did it follow its own plan?). `off_prompt_window_count` is the same
    count as `off_owner_count` against the STATED window set. Plus coherence (joint LPIPS, window-profile dispersion) and fidelity
    (CLIP-IQA on a subsample of frames)."""
    from .routing import stated_windows
    B = state.B; qs = [o["query"] for o in spec["objects"]]; nominal = {o["query"]: o["nominal"] for o in spec["objects"]}; spans = _prompt_spans(spec)
    if frames is None: frames = decode_frames(compose_level(leaves))
    every = int(every or CFG["video_det_every"]); n_win = len(leaves); m = {}
    fd = detect_frames(frames, qs, every); n_s = max(len(fd), 1); H = frames[0].height
    owner = _owner_windows(leaves, spec, plan)                                        # the window SET the plan gave each event
    owner_prompt = {o["query"]: stated_windows(o, n_win) for o in spec["objects"]}    # the window SET the prompt states
    wins, fracs, off, off_prompt, own_h, det_n = {}, {}, {}, {}, {}, {}
    for k, (i, dd) in enumerate(fd):
        frac = k / max(n_s - 1, 1); w = _win_of(frac, n_win)
        for d in dd:
            q = qs[d["label"]]; wins.setdefault(q, set()).add(w); fracs.setdefault(q, []).append(frac); det_n[q] = det_n.get(q, 0) + 1
            if w not in owner_prompt[q]: off_prompt[q] = off_prompt.get(q, 0) + 1
            if q in owner:
                if w not in owner[q]: off[q] = off.get(q, 0) + 1
                else: own_h.setdefault(q, []).append((d["box"][3] - d["box"][1]) / H)
    m["count_sum"] = {q: int(len(wins.get(q, ()))) for q in qs}                     # windows in which the event occurs (ideal 1)
    m["det_frame_count"] = {q: int(det_n.get(q, 0)) for q in qs}
    m["first_frac"] = {q: round(float(min(v)), 3) for q, v in fracs.items()}; m["last_frac"] = {q: round(float(max(v)), 3) for q, v in fracs.items()}
    m["pos_err_prompt"] = {q: round(_dist_to_span(min(v), *spans[q]), 3) for q, v in fracs.items()}
    m["pos_err_plan"] = {q: round(_dist_to_windows(min(v), owner.get(q, []), n_win), 3) for q, v in fracs.items() if q in owner}
    m["plan_err_prompt"] = {q: round(abs(plan[q] - nominal[q]), 3) for q in plan}
    m["off_owner_count"] = {q: int(off.get(q, 0)) for q in owner}; m["off_prompt_window_count"] = {q: int(off_prompt.get(q, 0)) for q in qs}
    m["owner_box_h_frac"] = _med(own_h)
    rmv = root_metrics_video(root, spec); m["root_box_h_frac"] = {q: round(v, 3) for q, v in rmv["box_h_frac"].items()}
    jl, sl = joint_lpips(frames, n_win); m["joint_lpips"] = jl; m["step_lpips"] = sl
    prof = _profiles(frames, n_win); m["row_profile_dispersion"] = float(prof.std(axis=0).mean())
    if root.frames:
        rp = np.mean([np.asarray(f.resize((32, 32), Image.BILINEAR), np.float32) for f in root.frames[::max(1, len(root.frames) // 12)]], axis=0)
        m["row_profile_dist_to_root"] = float(np.abs(prof - rp[None]).mean())
    m["color_mean_std_across_leaves"] = float(prof.reshape(n_win, -1, 3).mean(1).std(0).mean())
    m["n_frames"] = len(frames); m["n_frames_detected"] = len(fd)
    m["clipiqa_frames"] = B.fidelity(frames[::max(1, CFG["video_iqa_every"])]) if CFG["fidelity"] else {}
    return m

# ---------------- display / files ----------------
def filmstrip(frames, every=None, w=None, cols=8, gap=2):
    """One frame per `every` laid out in a grid `w` px wide — the audit artifact that stands in for the panorama composite."""
    every = int(every or CFG["video_strip_every"]); w = int(w or 1536)
    sel = frames[::max(1, every)] or frames[:1]
    cols = max(1, min(cols, len(sel))); tw = max(16, (w - gap * (cols - 1)) // cols)
    th = max(8, int(round(tw * sel[0].height / sel[0].width))); rows = -(-len(sel) // cols)
    out = Image.new("RGB", (cols * tw + gap * (cols - 1), rows * th + gap * (rows - 1)), (40, 40, 40))
    for i, f in enumerate(sel):
        out.paste(f.convert("RGB").resize((tw, th), Image.LANCZOS), ((i % cols) * (tw + gap), (i // cols) * (th + gap)))
    return out

_STATIC_FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")

def _ffmpeg_bin():
    """Find an ffmpeg binary without a pip install: PATH, then imageio_ffmpeg's bundled one (if that package happens to be importable),
    then the static binary the cdgs conda env ships (this env has neither ffmpeg on PATH nor imageio-ffmpeg installed)."""
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe): return exe
    except Exception: pass
    return _STATIC_FFMPEG if os.path.exists(_STATIC_FFMPEG) else None

def save_video(frames, path, fps=None):
    """mp4 via ffmpeg (H.264/yuv420p, the same codec CDGS writes) if a binary can be found; falls back to OpenCV's mp4v (MPEG-4 part 2,
    which many players/browsers cannot open — a warning is printed) only when no ffmpeg is available. fps defaults to CFG['video_fps']
    (16, matching CDGS). Returns the path actually written."""
    fps = int(fps or CFG["video_fps"])
    ffmpeg = _ffmpeg_bin()
    if ffmpeg:
        a0 = np.asarray(frames[0].convert("RGB")); h, w = a0.shape[0], a0.shape[1]
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for f in frames: p.stdin.write(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes())
            p.stdin.close(); err = p.stderr.read(); p.wait()
            if p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0: return path
            print(f"[video] ffmpeg failed (code {p.returncode}): {err.decode(errors='replace')[-500:]}")
        except Exception as e:
            print(f"[video] ffmpeg invocation failed: {e}")
    print("[video] WARNING: no working ffmpeg found -- falling back to OpenCV's mp4v codec, which many players/browsers cannot decode.")
    try:
        import cv2
        a0 = np.asarray(frames[0].convert("RGB")); vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (a0.shape[1], a0.shape[0]))
        if vw.isOpened():
            for f in frames: vw.write(np.asarray(f.convert("RGB"))[:, :, ::-1])
            vw.release()
            if os.path.exists(path) and os.path.getsize(path) > 0: return path
    except Exception: pass
    npz = os.path.splitext(path)[0] + "_frames.npz"
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    np.savez_compressed(npz, frames=arr, fps=fps); return npz
