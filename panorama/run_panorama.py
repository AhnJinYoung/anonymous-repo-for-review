#!/usr/bin/env python
"""HLC panoramas with SD3-medium (the paper's Table-1 HLC arm).

Setting (base_config.json + the overrides below): SD3-medium, native geometry (window core 1280 px, halo 192 px, height 704 px),
28 steps, CFG 7.0, T5 on, atom_mode delete, relay field:0.9, routing decided once at the root, depth 2 (4 leaves, 5120 x 704),
a single root sample (seed 0, root_candidates = 1, no candidate selection). Seed 0 for every prompt.

Routing arms (same tree, same root; only the per-window condition changes):
  attn_hard_phrase   HLC default: routing from the model's own cross-attention on the root   (paper: "ours")
  exact_hard_phrase  HLC with the black-box deletion probe                                   (ablation)
  broadcast          full prompt in every window (MultiDiffusion-style replication)          (baseline)

Out: <out>/<routing>/<id>/final.jpg + root.jpg + record.json; raw harness logs in <out>/_tree/. Resumable (existing final.jpg is skipped).

usage:
  python run_panorama.py                               # all 25 prompts of prompts.json, HLC arm
  python run_panorama.py --ids p01_meadow              # one prompt
  python run_panorama.py --routing broadcast           # a baseline arm
  python run_panorama.py --dummy --ids p01_meadow      # CPU smoke test with a stand-in backend (no model download)
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
from longhorizon import runners
from longhorizon.scenes import full_prompt

BASE_CFG = os.path.join(HERE, "base_config.json")
ROUTINGS = ("attn_hard_phrase", "exact_hard_phrase", "broadcast")
RELAY, SEED = "field:0.9", 0


def load_prompts(path, ids=None):
    P = json.load(open(path))
    if ids: keep = set(ids); P = [p for p in P if p["id"] in keep]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.json"))
    ap.add_argument("--ids", default="", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--routing", default="attn_hard_phrase", choices=ROUTINGS)
    ap.add_argument("--depth", type=int, default=2, help="tree depth: 2 -> 4 leaves (5120 x 704)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dummy", action="store_true", help="CPU stand-in backend (smoke test only)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs")); a = ap.parse_args()
    P = load_prompts(a.prompts, [i for i in a.ids.split(",") if i])
    arm_dir = os.path.join(a.out, a.routing + ("" if a.depth == 2 else f"_d{a.depth}"))
    todo = [p for p in P if not os.path.exists(os.path.join(arm_dir, p["id"], "final.jpg"))]
    print(f"[hlc] routing {a.routing} | depth {a.depth} | {len(P)} prompts | {len(todo)} to run", flush=True)
    if not todo: return
    ov = json.load(open(BASE_CFG))
    ov.update(root_candidates=1, depth_main=a.depth, show_sheets=False, fidelity=False, root_cache=os.path.join(a.out, "roots"),
              gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
              guard_rss_gb=float(os.environ.get("LH_RSS_GB", 55)), threads=int(os.environ.get("LH_THREADS", 24)))
    tree_out = os.path.join(a.out, "_tree" + ("" if a.depth == 2 else f"_d{a.depth}")); os.makedirs(tree_out, exist_ok=True)
    runners.setup(dummy=a.dummy, overrides=ov, out_dir=tree_out); t5 = bool(ov["t5_arms"][-1])
    for p in todo:
        scene = {k: p[k] for k in ("name", "base", "objects")}; prompt = full_prompt(scene)
        d = os.path.join(arm_dir, p["id"]); os.makedirs(d, exist_ok=True); t0 = time.time()
        log, comp, root = runners.run_tree(scene, a.seed, a.depth, a.routing, RELAY, "once", t5, stage="hlc")
        root.image.save(os.path.join(d, "root.jpg"), quality=92)
        rec = dict(id=p["id"], prompt=prompt, seed=a.seed, depth=a.depth, routing=a.routing, relay=RELAY, routed=log.get("routed"),
                   plan=log.get("plan"), root_seed=log.get("root_seed"), time_s=log.get("time_s"), vram_gb=log.get("vram_gb"),
                   width_px=comp.width, height_px=comp.height, dummy=bool(a.dummy), wall_s=round(time.time() - t0, 1))
        runners.save_wide(comp, os.path.join(d, "final.jpg"), quality=95)
        json.dump(rec, open(os.path.join(d, "record.json"), "w"), indent=1, default=str)
        print(f"[hlc] {a.routing} d{a.depth} {p['id']}: {comp.width}x{comp.height} {rec['wall_s']}s routed {rec['routed']}", flush=True)
    print("[hlc] DONE", flush=True)


if __name__ == "__main__":
    main()
