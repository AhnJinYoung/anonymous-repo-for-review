# HLC: code for generating the paper's outputs (anonymous supplementary)

This archive contains the code needed to **generate** the outputs of the HLC panorama and video experiments in the paper. Evaluation code (object detectors used as metrics, VLM judges, VBench-style metrics,
summary/table scripts) is not included; see "Notes" for the few logging hooks that remain inside the generation code.

```
hlc_code/
  panorama/     HLC panoramas with SD3-medium (Table 1, arm "ours"; single root, no candidate selection)
  video/        HLC video with LTX-2.5 distilled (root -> attention routing probe -> keyframe-cascade windows -> assembly)
  requirements_panorama.txt
  requirements_video.txt
```

## Environments

* **panorama/**: Python 3.10/3.11, `torch 2.4.1+cu121`, `diffusers 0.35.2`, `transformers 4.56.2`. See `requirements_panorama.txt`.
* **video/**: a separate environment: Python 3.11, `torch 2.8.0+cu128`, `diffusers` from source at commit `e0abab83`
  (0.41.0.dev0), `transformers 5.17.0`. Model `Lightricks/LTX-2.5-Diffusers` (distilled transformer). See `requirements_video.txt`.

All GPU runs used one A100-80GB per job. Every entry script calls `longhorizon.guard.install()`, which caps BLAS/torch threads,
sets `CUDA_VISIBLE_DEVICES` and starts an RSS watchdog. Optional environment variables: `LH_THREADS` (thread cap, default 24),
`LH_RSS_GB` (RSS limit in GB, default 55), `LH_CPUS` (CPU range `a-b` to pin; unset = no pinning).
Gated Hugging Face models (SD3-medium) need `HF_TOKEN` in the environment; `HF_HOME` selects the model cache.

## 1. Panorama (SD3-medium)

`panorama/run_panorama.py` is the runner for the Table-1 HLC arm: SD3-medium, native geometry (window core 1280 px,
halo 192 px, height 704 px), 28 steps, CFG 7.0, T5 on, routing `attn_hard_phrase` (the model's own cross-attention on the root),
relay `field:0.9`, routing decided once at the root, depth 2 (4 leaves, 5120 x 704), one root sample (seed 0,
`root_candidates = 1`), seed 0 for every prompt. The fixed settings are in `panorama/base_config.json`, the 25 prompts in
`panorama/prompts.json`. The package `panorama/longhorizon/` holds the sampler (config, scenes, geometry, SD3 backend, routing,
relays/lock-step sampling, runners).

```bash
cd panorama
python run_panorama.py                                   # all 25 prompts  -> outputs/attn_hard_phrase/<id>/final.jpg
python run_panorama.py --ids p01_meadow                  # one prompt
python run_panorama.py --routing exact_hard_phrase       # ablation: black-box deletion probe instead of attention
python run_panorama.py --routing broadcast               # baseline: full prompt in every window (same tree and root)
python run_panorama.py --dummy --ids p01_meadow          # CPU smoke test with a stand-in backend (no model download)
```
Outputs: `outputs/<routing>/<id>/{final.jpg, root.jpg, record.json}`; roots are cached in `outputs/roots/` and reused by the
other routing arms, so every arm of a prompt shares the same root.

## 2. Video (LTX-2.5 distilled)

`video/run_v61.py` runs the video instantiation with one model for every node (LTX-2.5 distilled, 8 sigmas, CFG 1, 1280 x 704,
121 frames at 24 fps). Stages: `root` (whole story prompt, 121 frames), `probe` (the model's own video-to-text cross-attention on the
re-noised root, per latent frame; routing of events to 4 windows), `windows` (keyframe cascade: each window is a fresh 121-frame sample
conditioned on the root keyframes at its two boundaries and on its routed text), `assemble` (concatenate windows into one video).
Arm `K0` = routed text (the paper's setting); `KB` = broadcast reference. The script also contains the two single-pass baselines
(`--stage onepass`, `--stage glv`). `video/run_ltx.sh` runs it in the LTX environment (`LTX_PYTHON=/path/to/python`).
The model snapshot is taken from the local Hugging Face cache, or downloaded (only the components the distilled path needs).

Paper figure setting (scene `park_typed`, seed 2, arm K0):
```bash
cd video
export LTX_PYTHON=/path/to/ltx-env/bin/python
bash run_ltx.sh run_v61.py --stage root     --scene park_typed --seed 2
bash run_ltx.sh run_v61.py --stage probe    --scene park_typed --seed 2
bash run_ltx.sh run_v61.py --stage windows  --scene park_typed --seed 2 --arm K0     # all 4 windows (or --windows 0,1 / 2,3 on two GPUs)
bash run_ltx.sh run_v61.py --stage assemble --scene park_typed --seed 2 --arm K0     # -> outputs/park_typed/ltx_K0_s2/video.mp4
```

## Notes / limitations

* The panorama harness logs per-window detections (OWL-ViT, `google/owlvit-base-patch32`) inside `runners.run_tree`; this is logging
  only and is interleaved with the sampler, so it was left in place. With `root_candidates = 1` the detector does not select anything
  and does not influence the generated image. The IQA metrics are switched off (`fidelity = False`); the OWL-ViT weights are still
  downloaded when the SD3 backend is constructed.
* Some comments still refer to internal experiment identifiers (e.g. "v53"); they carry no information beyond the code.
