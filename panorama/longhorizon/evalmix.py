"""Backend parts that have nothing to do with the diffusion parameterization, shared by every real backend (SD3, SD2, ...).

  TorchOpsMixin — the array helpers the sampler and the relays call (noise, low-pass, stretch, row means, concat).
  EvalMixin     — OWL-ViT detection, CLIP-IQA / NIQE / LPIPS loading and the fidelity + intra-LPIPS metrics.

Both expect the backend to have set `self.torch`, `self.F` and `self.device` before calling `_load_eval()`."""
import itertools, numpy as np
from . import config
from .config import CFG


class TorchOpsMixin:
    """Array helpers for a 4D image latent (1, C, H, w) whose LAST axis is the horizon. `backend_cogvideo.VideoOpsMixin` is the 5D
    (1, C, H, W, t) version: same contract, horizon still last, so `sampling.py` never has to know which one it is talking to."""
    canvas_multiple = 16                                             # every node canvas width must be a multiple of this (latent stride x patch)
    is_video = False
    def noise_shape(self, w): return (1, int(self.LC), config.height() // int(self.VS), int(w))
    def zeros(self, shape): return self.torch.zeros(shape, device=self.device, dtype=self.torch.float32)
    def ones(self, shape): return self.torch.ones(shape, device=self.device, dtype=self.torch.float32)
    def randn(self, shape, seed):
        g = self.torch.Generator(device=self.device).manual_seed(int(seed)); return self.torch.randn(shape, generator=g, device=self.device, dtype=self.torch.float32)
    def linspace_w(self, a, b, n): return self.torch.linspace(a, b, n, device=self.device).view(1, 1, 1, n)
    def stretch_w(self, x, new_w): return self.F.interpolate(x, size=(x.shape[-2], new_w), mode="bilinear", align_corners=False)
    def lowpass_x(self, x, sigma_lat):
        torch, F = self.torch, self.F; k = int(6 * sigma_lat) | 1; r = k // 2
        t = torch.arange(k, device=x.device, dtype=x.dtype) - r; g = torch.exp(-0.5 * (t / sigma_lat) ** 2); g = g / g.sum()
        xp = F.pad(x, (r, r, 0, 0), mode="reflect" if r < x.shape[-1] else "replicate")
        return F.conv2d(xp, g.view(1, 1, 1, k).repeat(x.shape[1], 1, 1, 1), groups=x.shape[1])
    def row_mean(self, x): return x.mean(dim=-1, keepdim=True)
    def repeat_cols(self, x, n): return x.expand(*x.shape[:-1], n).contiguous()
    def cat_w(self, xs): return self.torch.cat(xs, dim=-1)
    def vram_gb(self): return self.torch.cuda.max_memory_allocated() / 1e9


class EvalMixin:
    def _load_eval(self):
        """OWL-ViT + (optionally) the IQA metrics. Identical for every backend: evaluation never touches the sampler."""
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.owl_proc = OwlViTProcessor.from_pretrained(CFG["owl_id"]); self.owl = OwlViTForObjectDetection.from_pretrained(CFG["owl_id"]).to(self.device).eval()
        self.IQA = {}
        if CFG["fidelity"]:
            try:
                import pyiqa; self.IQA["clipiqa"] = pyiqa.create_metric("clipiqa", device=self.device); self.IQA["niqe"] = pyiqa.create_metric("niqe", device=self.device)
            except Exception as e: print("pyiqa unavailable:", repr(e)[:120])
            try:
                import lpips; self.IQA["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            except Exception as e: print("lpips unavailable:", repr(e)[:120])
        return self.IQA

    def detect(self, img, queries, thr):
        torch = self.torch
        if not queries: return []                     # all-context prompt (no entity items): nothing to detect
        with torch.no_grad():
            inputs = self.owl_proc(text=[queries], images=img, return_tensors="pt").to(self.device); out = self.owl(**inputs)
            ts = torch.tensor([[img.height, img.width]], device=self.device)
            try: r = self.owl_proc.post_process_object_detection(outputs=out, threshold=thr, target_sizes=ts)[0]
            except AttributeError: r = self.owl_proc.post_process_grounded_object_detection(outputs=out, threshold=thr, target_sizes=ts)[0]
        return [dict(box=b, score=float(s), label=int(l)) for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())]
    def _t01(self, img): return self.torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
    def fidelity(self, images):
        out = {}
        with self.torch.no_grad():
            for name in ("clipiqa", "niqe"):
                if name in self.IQA:
                    vals = []
                    for im in images:
                        try: vals.append(float(self.IQA[name](self._t01(im))))
                        except Exception: pass
                    if vals: out[name] = float(np.mean(vals))
        return out
    def lpips_pair(self, a, b):
        """LPIPS between two PIL images (video: the two frames on either side of a window joint)."""
        if "lpips" not in self.IQA: return None
        with self.torch.no_grad(): return float(self.IQA["lpips"](self._t01(a) * 2 - 1, self._t01(b) * 2 - 1))
    def intra_lpips(self, comp, crop=None, max_pairs=64):
        if "lpips" not in self.IQA: return None
        crop = int(crop or config.height())                   # one window height: square crops along the horizon
        with self.torch.no_grad():
            crops = [self._t01(comp.crop((i, 0, i + crop, crop))) * 2 - 1 for i in range(0, comp.width - crop + 1, crop)]
            pairs = list(itertools.combinations(range(len(crops)), 2))
            if len(pairs) > max_pairs: pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, max_pairs).astype(int)]
            return float(np.mean([float(self.IQA["lpips"](crops[i], crops[j])) for i, j in pairs])) if pairs else None
