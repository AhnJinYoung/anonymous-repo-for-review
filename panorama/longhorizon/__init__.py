"""Long-horizon generation from a single-window diffusion model, by sampling only.

Package layout (each module is one section of the old v36 notebook):
  config    - CFG dict + derived constants (H, VS, LC, OV, OVL), TREE depth, tag()
  guard     - resource guard: thread caps, CPU affinity, RSS watchdog, GPU 0 only
  scenes    - the three benchmark scenes and the word-level prompt split
  geometry  - Node + growth tree
  backend   - SD3-medium / SD3.5-medium / SD3.5-large (+ optional T5-XXL) with word-atom weighted encoding, probes, OWL-ViT, IQA
  backend_flux - FLUX.1-dev: guidance embedding (no CFG batch), packed 2x2 tokens inside `denoise` only, one dynamic-shift schedule per tree
  backend_sd2 - Stable Diffusion 2.0-base (VP / epsilon UNet, DDIM) behind the same interface
  backend_cogvideo - CogVideoX-2b with TIME as the horizon: latents (1, C, H, W, T), horizon unit = one latent frame, same interface
  backend_wan - Wan 2.2 TI2V-5B, same time horizon and 5D layout as backend_cogvideo but FLOW MATCHING and a 4x16x16 / 48-channel VAE
  schedule  - VP noise levels (t, alpha, sigma) + DDIM level list, shared by backend_sd2, backend_cogvideo and dummy
  evalmix   - backend parts that do not depend on the parameterization: array helpers, OWL-ViT, CLIP-IQA / NIQE / LPIPS
  dummy     - CPU stand-in with the same interface, flavours "sd3", "sd2", "flux", "video"/"cogvideo" and "wan" (smoke tests)
  video     - the video-only half of metrics/display: temporal composition, VAE decode to frames, per-event frame metrics, filmstrips, mp4
  routing   - (region x atom) scores -> per-window 0/1 weights
  sampling  - relays (field / rho_early / lowpass_skip / none) and the lock-step level sampler (parameterization-agnostic:
              it only uses B.levels / B.denoise / B.mix / B.noise_frac, so SD3 and SD2 run through the same code)
  metrics   - compose, detections, counts, position errors (vs prompt and vs plan), fidelity
  display   - captions, chunked strips, sheets, notebook banners
  runners   - roots, trees, stages, incremental logging, summaries
"""
__version__ = "0.37.0"
