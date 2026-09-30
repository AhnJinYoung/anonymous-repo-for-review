"""CPU stand-in for the real backends: same interface, synthetic tokens/probes/images. For smoke-testing the harness only.

`kind="sd3"` mirrors backend.Backend (flow matching, a level is its sigma, LC 16); `kind="sd2"` mirrors backend_sd2.Backend
(VP schedule, a level is (t, alpha, sigma), DDIM eta=0 update, LC 4); `kind="video"` (alias "cogvideo") mirrors
backend_cogvideo.Backend (VP schedule, 5D latents (1, LC, H, W, T) with TIME LAST, VS 1, canvas_multiple 1, frames instead of
an image); `kind="wan"` mirrors backend_wan.Backend (the SAME 5D video layout, but FLOW MATCHING and LC 48), so all
parameterizations and both horizon axes are exercised without a GPU. `kind="flux"` mirrors backend_flux.Backend: the sd3 dummy (flow matching,
LC 16, VS 8) but with FLUX's text side (CLIP-L + T5 only, no negative/uncond batch: `cfg` is a guidance value, not a CFG weight), FLUX's
dynamic-shift schedule computed from the child canvas (backend_flux.flux_sigmas / tree_mu, FluxPipeline's default scheduler config), the
velocity evaluated through pack -> unpack (so the packing code runs) and attn_mass raising NotImplementedError."""
import hashlib, numpy as np
from PIL import Image
from . import config
from .config import CFG
from .schedule import VPLevel, scaled_linear_alphas_cumprod, ddim_levels
from .scenes import word_split, build_spec, prompt_of
from . import textcond

HEADS = {"lighthouse": 0.12, "barn": 0.12, "truck": 0.12, "boat": 0.5, "tree": 0.5, "cactus": 0.5, "house": 0.88, "bridge": 0.88, "station": 0.88}
TIME_WORDS = {"beginning": 0.12, "middle": 0.5, "end": 0.88}        # video: the event phrases carry their own position word

class Backend:
    dummy = True
    def __init__(self, load_t5=True, kind="sd3"):
        self.flavour = "cogvideo" if kind == "video" else kind       # which REAL backend is being mirrored ("wan" and "cogvideo" share the video layout)
        self.is_flux = (kind == "flux")
        if kind == "flux": kind = "sd3"                              # same latent layout and parameterization as the sd3 dummy
        if kind in ("cogvideo", "wan"): kind = "video"
        self.kind = kind; self.is_video = (kind == "video")
        self.vp = kind in ("sd2", "video") and self.flavour != "wan" # VP (alpha, sigma) levels; False -> flow matching, a level IS its sigma (sd3, wan)
        self.LC, self.VS = (4, 8) if kind == "sd2" else ((48 if self.flavour == "wan" else 16, 1) if self.is_video else (16, 8))
        self.canvas_multiple = 1 if self.is_video else 16
        self.LAT_H, self.LAT_W = 6, 9                                # video: a tiny (H, W) latent frame; the horizon is the 5th axis
        self.device = "cpu"; self.N_CLIP = 77; self.N_BLOCKS = 24; self.t5 = load_t5 and kind != "sd2"
        self.toks = {"t5": None} if self.is_video else ({"clip": None} if kind == "sd2" else ({"clip": None, "clip2": None} | ({"t5": None} if load_t5 else {})))
        if self.is_flux: self.toks = {"clip": None, "t5": None}     # FLUX: CLIP-L (pooled) + T5-XXL, always
        self.acp = scaled_linear_alphas_cumprod() if self.vp else None
        print(f"backend: DUMMY (cpu, {self.flavour})")
    def encode_text(self, scene, t5=False, atom_level="word"):
        prompt, units = word_split(scene); spec = build_spec(scene, units, atom_level)
        spans = [(i + 1, i + 2) for i in range(len(units))]
        spec.update(prompt=prompt, t5=(True if self.is_video else (bool(t5) and self.kind != "sd2")), spans={k: spans for k in self.toks},
                    units=units, n_real={k: len(units) + 1 for k in self.toks}, E=None, P=None); return spec
    def make_cond(self, spec, w):
        """The synthetic condition. Under `atom_mode="delete"` the condition really IS a different prompt, so the rebuilt prompt text
        is carried along and hashed by `_target`; under "zero" the condition is the weight vector alone, exactly as before (so the
        v37 dummy reference outputs are unchanged by this option)."""
        w = np.asarray(w, np.float64); c = dict(w=w, t5=spec["t5"])
        if textcond.delete_mode(w): c["text"] = prompt_of(spec, w)
        return c

    def text_cond(self, prompt): return dict(w=np.ones(1), t5=True, text=prompt)
    def noise_stream(self, seed, t_lat):
        rng = np.random.default_rng(int(seed)); shp = self.noise_shape(int(t_lat))
        return lambda: rng.standard_normal(shp).astype(np.float32)

    # ---------------- sampler primitives ----------------
    FLUX_SCHED = dict(base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5, max_shift=1.15, use_dynamic_shifting=True, shift=3.0)   # FluxPipeline defaults
    def get_sigmas(self, steps):
        if self.is_flux:
            from .backend_flux import flux_sigmas, tree_mu
            return flux_sigmas(steps, tree_mu(self.FLUX_SCHED, self.VS)[0])
        u = np.linspace(1.0, 1e-3, steps); s = 3.0 * u / (1 + 2.0 * u); return [float(v) for v in s] + [0.0]
    def levels(self, steps):
        return ddim_levels(self.acp, steps) if self.vp else self.get_sigmas(steps)
    def noise_frac(self, lvl): return lvl.noise_frac if self.vp else float(lvl)
    def init_latent(self, noise, lvl): return noise if self.vp else float(lvl) * noise
    def mix(self, x0, eps, lvl): return (lvl.alpha * x0 + lvl.sigma * eps) if self.vp else (1 - float(lvl)) * x0 + float(lvl) * eps
    def alpha_sigma(self, lvl): return (float(lvl.alpha), float(lvl.sigma)) if self.vp else (1.0 - float(lvl), float(lvl))
    @property
    def token_anchors(self): return self.flavour == "wan"                # mirrors backend_wan: per-token timesteps (the @tok anchor mechanism)
    def denoise(self, x, lvl, cond, cfg, anchor_mask=None, anchor_x=None):
        if anchor_mask is not None:                                    # @tok: clean anchors at timestep 0 -> the model's x0 there IS the anchor
            assert self.token_anchors, "per-token anchors are a wan-backend feature"
            m = anchor_mask; x = (1 - m) * x + m * anchor_x; x0, eps = self.denoise(x, lvl, cond, cfg); return (1 - m) * x0 + m * anchor_x, eps
        if self.vp:
            x0 = self._target(cond, x.shape); return x0, (x - lvl.alpha * x0) / max(lvl.sigma, 1e-6)
        s = float(lvl); v = self.predict_v(x, s, cond, cfg); return x - s * v, x + (1 - s) * v
    def randn(self, shape, seed): return np.random.default_rng(int(seed)).standard_normal(shape).astype(np.float32)
    def noise_shape(self, w):
        return (1, self.LC, self.LAT_H, self.LAT_W, int(w)) if self.is_video else (1, self.LC, config.height() // self.VS, int(w))
    def zeros(self, shape): return np.zeros(shape, np.float32)
    def ones(self, shape): return np.ones(shape, np.float32)
    def _target(self, cond, shape):
        h = int(hashlib.md5(cond["w"].round(1).tobytes() + bytes([cond["t5"]]) + cond.get("text", "").encode()).hexdigest()[:6], 16)
        c = np.array([(h & 255), ((h >> 8) & 255), ((h >> 16) & 255)]) / 255.0 * 2 - 1
        yy = np.linspace(-1, 1, shape[2]).reshape([1, 1, -1] + [1] * (len(shape) - 3)); t = np.zeros(shape, np.float32)
        t[:, :3] = c.reshape([1, 3] + [1] * (len(shape) - 2)) * (1 - 0.5 * yy); return t
    def predict_v(self, x, sigma, cond, cfg):
        x0 = self._target(cond, x.shape); v = (x - (1 - sigma) * x0) / sigma - x0
        if self.is_flux:                                             # FLUX: the transformer sees packed 2x2 tokens; the harness only ever sees the unpacked latent
            from .backend_flux import pack, unpack
            v = unpack(pack(v), v.shape[-2], v.shape[-1])
        return v
    def linspace_w(self, a, b, n):
        v = np.linspace(a, b, n, dtype=np.float32); return v.reshape([1] * (4 if self.is_video else 3) + [n])
    def lowpass_x(self, x, sigma_lat):
        k = int(6 * sigma_lat) | 1; r = k // 2; t = np.arange(k) - r; g = np.exp(-0.5 * (t / sigma_lat) ** 2); g = (g / g.sum()).astype(np.float32)
        pad = [(0, 0)] * (x.ndim - 1) + [(r, r)]
        xp = np.pad(x, pad, mode="reflect" if r < x.shape[-1] else "edge"); out = np.zeros_like(x)
        for i, gi in enumerate(g): out += gi * xp[..., i:i + x.shape[-1]]
        return out
    def stretch_w(self, x, new_w):
        xs = np.linspace(0, x.shape[-1] - 1, new_w); i0 = np.floor(xs).astype(int); i1 = np.minimum(i0 + 1, x.shape[-1] - 1)
        w = (xs - i0).astype(np.float32); return x[..., i0] * (1 - w) + x[..., i1] * w
    def row_mean(self, x): return x.mean(axis=-1, keepdims=True)
    def repeat_cols(self, x, n): return np.repeat(x, n, axis=-1)
    def cat_w(self, xs): return np.concatenate(xs, axis=-1)
    def decode(self, lat):
        """Images: the latent as a picture. Video: the filmstrip of the decoded frames (what the audit sheets show for a node)."""
        if self.is_video:
            from . import video
            return video.filmstrip(self.decode_frames(lat))
        a = np.clip((lat[0, :3].transpose(1, 2, 0) + 1) / 2, 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8)).resize((lat.shape[-1] * self.VS, lat.shape[-2] * self.VS), Image.BILINEAR)
    def decode_frames(self, lat, chunk_lat=None):
        """Video: (1, C, H, W, T) latent -> 1 + 4 (T-1) small PIL frames (the same temporal compression ratio as CogVideoX's VAE)."""
        T = lat.shape[-1]; n_px = 1 + 4 * (T - 1); out = []
        idx = np.linspace(0, T - 1, n_px)
        for u in idx:
            i0 = int(np.floor(u)); i1 = min(i0 + 1, T - 1); f = float(u - i0)
            a = lat[0, :3, :, :, i0] * (1 - f) + lat[0, :3, :, :, i1] * f
            a = np.clip((a.transpose(1, 2, 0) + 1) / 2, 0, 1)
            out.append(Image.fromarray((a * 255).astype(np.uint8)).resize((lat.shape[3] * 8, lat.shape[2] * 8), Image.BILINEAR))
        return out

    def encode_image(self, img):
        """Video: one PIL frame -> a (1, C, H, W, 1) latent (the inverse of decode_frames on the first 3 channels, 0 elsewhere)."""
        a = np.asarray(img.convert("RGB").resize((self.LAT_W, self.LAT_H), Image.BILINEAR), np.float32) / 255.0 * 2 - 1
        z = np.zeros((1, self.LC, self.LAT_H, self.LAT_W, 1), np.float32); z[0, :3, :, :, 0] = a.transpose(2, 0, 1); return z

    def encode_video(self, frames):
        """Video: 1 + 4 k PIL frames -> a (1, C, H, W, 1 + k) latent; slot j = encode_image of frame 4 j (the frame decode_frames puts
        exactly on latent j), so encode_video(decode_frames(z)) recovers z's first 3 channels up to 8-bit rounding."""
        n = len(frames); assert n >= 1 and (n - 1) % 4 == 0, n
        return np.concatenate([self.encode_image(frames[4 * j]) for j in range(1 + (n - 1) // 4)], axis=-1)

    # ---------------- evaluation ----------------
    def detect(self, img, queries, thr):
        if not queries: return []
        rng = np.random.default_rng(img.width * 7 + len(queries)); out = []
        for i, q in enumerate(queries):
            if rng.random() < 0.5: x = rng.uniform(0, img.width - 60); out.append(dict(box=[x, 100.0, x + 60, 220.0], score=0.3, label=i))
        return out
    def vram_gb(self): return 0.0
    def fidelity(self, images): return dict(clipiqa=0.5, niqe=5.0)
    def intra_lpips(self, comp, crop=None, max_pairs=64): return 0.4
    def lpips_pair(self, a, b): return 0.25

    # ---------------- probes ----------------
    def _synthetic(self, spec, atoms, regions, seed, noise=0.2):
        rng = np.random.default_rng(seed); n = len(regions); D = np.full((n, len(atoms)), 3e-3) + rng.normal(0, 1e-4 * (1 + noise), (n, len(atoms)))
        table = TIME_WORDS if self.is_video else HEADS                 # video: an event phrase is localized by its position word, not by a head noun
        for j, a in enumerate(atoms):
            hit = [h for h in table if h in spec["words"][a].lower().split()]
            if hit: D[:, j] = 3e-4; D[min(n - 1, int(table[hit[0]] * n)), j] = 8e-3
        return D
    def delta_exact(self, y0, spec, atoms, regions, ts, seed, w_base=None, n_eps=1, return_all=False):
        Ds = np.stack([self._synthetic(spec, atoms, regions, seed + 17 * e) for e in range(n_eps)]); return Ds if return_all else Ds.mean(0)
    def attn_mass(self, y0, spec, atoms, regions, ts, seed, blocks="all"):
        if self.is_flux: raise NotImplementedError("attention arms/diagnostic are not available on the FLUX backend")
        D = self._synthetic(spec, atoms, regions, seed + 2, noise=2.0); return D / D.mean(0, keepdims=True) * 0.05
