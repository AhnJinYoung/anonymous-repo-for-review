"""Composition and the per-tree metrics. Position errors come in two flavours: vs the PROMPT (did the result follow the instruction?)
and vs the PLAN (did the tree follow its own routing?). They answer different questions and are logged separately."""
import numpy as np
from PIL import Image
from . import config, state
from .config import CFG, final_w

def core_crop(n): return n.image.crop((n.core_n0, 0, n.core_n0 + n.core_w, config.height()))

def compose_level(nodes):
    H, OV = config.height(), config.overlap()
    d = nodes[0].depth; Wl = CFG["core_px"] * CFG["branch"] ** d
    acc = np.zeros((H, Wl, 3), np.float32); wsum = np.zeros((1, Wl, 1), np.float32)
    for n in nodes:
        a = np.asarray(n.image, np.float32); w = np.ones(a.shape[1], np.float32)
        if n.halo_l: w[:OV] = np.linspace(0, 1, OV)
        if n.halo_r: w[-OV:] = np.linspace(1, 0, OV)
        x0 = n.canvas_g0 // n.scale; acc[:, x0:x0 + a.shape[1]] += a * w.reshape(1, -1, 1); wsum[:, x0:x0 + a.shape[1]] += w.reshape(1, -1, 1)
    return Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8))

def _med(d): return {q: float(np.median(v)) for q, v in d.items()}

def root_metrics(root, spec):
    """Root-only: detections, box heights and position error vs the prompt (is the plan's premise even right?)."""
    qs = [o["query"] for o in spec["objects"]]; nominal = {o["query"]: o["nominal"] for o in spec["objects"]}; W = root.image.width; H = config.height()
    counts = {q: int(sum(d["label"] == i for d in root.dets)) for i, q in enumerate(qs)}; pos, hs = {}, {}
    for d in root.dets:
        q = qs[d["label"]]; pos.setdefault(q, []).append(abs((d["box"][0] + d["box"][2]) / 2 / W - nominal[q])); hs.setdefault(q, []).append((d["box"][3] - d["box"][1]) / H)
    return dict(count=counts, pos_err_prompt=_med(pos), box_h_frac=_med(hs))

def leaf_metrics(leaves, spec, root, plan):
    B = state.B; qs = [o["query"] for o in spec["objects"]]; nominal = {o["query"]: o["nominal"] for o in spec["objects"]}; m = {}; H = config.height()
    dets = [B.detect(core_crop(n), qs, CFG["det_thr"]) for n in leaves]
    counts = [{qs[i]: sum(d["label"] == i for d in dd) for i in range(len(qs))} for dd in dets]
    m["count_sum"] = {q: int(sum(c[q] for c in counts)) for q in qs}
    W = final_w(); n_l = len(leaves); owner = {q: int(p * n_l) for q, p in plan.items()}; owner_prompt = {q: min(n_l - 1, int(v * n_l)) for q, v in nominal.items()}
    err_plan, err_prompt, off, off_prompt, own_h = {}, {}, {}, {}, {}
    for li, (n, dd) in enumerate(zip(leaves, dets)):
        for d in dd:
            q = qs[d["label"]]; gx = n.n2g(n.core_n0 + (d["box"][0] + d["box"][2]) / 2) / W
            err_prompt.setdefault(q, []).append(abs(gx - nominal[q]))
            if q in plan: err_plan.setdefault(q, []).append(abs(gx - plan[q]))
            if li != owner_prompt[q]: off_prompt[q] = off_prompt.get(q, 0) + 1
            if q in owner:
                if li != owner[q]: off[q] = off.get(q, 0) + 1
                else: own_h.setdefault(q, []).append((d["box"][3] - d["box"][1]) / H)
    m["pos_err_plan"] = _med(err_plan); m["pos_err_prompt"] = _med(err_prompt)
    m["plan_err_prompt"] = {q: round(abs(plan[q] - nominal[q]), 3) for q in plan}                  # did the plan itself follow the instruction?
    m["off_owner_count"] = {q: int(off.get(q, 0)) for q in owner}; m["off_prompt_window_count"] = {q: int(off_prompt.get(q, 0)) for q in qs}
    m["owner_box_h_frac"] = _med(own_h); m["root_box_h_frac"] = {qs[d["label"]]: round((d["box"][3] - d["box"][1]) / H, 3) for d in root.dets}
    prof = np.stack([np.asarray(core_crop(n), np.float32).mean(axis=1) for n in leaves]); root_prof = np.asarray(root.image, np.float32).mean(axis=1)
    m["row_profile_dispersion"] = float(prof.std(axis=0).mean()); m["row_profile_dist_to_root"] = float(np.abs(prof - root_prof[None]).mean())
    cores = [np.asarray(core_crop(n), np.float32) for n in leaves]; m["color_mean_std_across_leaves"] = float(np.stack([c.reshape(-1, 3).mean(0) for c in cores]).std(0).mean())
    return m
