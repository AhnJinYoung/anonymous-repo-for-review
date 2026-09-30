"""SD3-family backend (SD3-medium, SD3.5-medium, SD3.5-large: the same StableDiffusion3Pipeline / SD3Transformer2DModel classes) with
word-atom weighted encoding on all text encoders (CLIP-L, CLIP-G, optional T5-XXL). Model-agnostic: joint_attention_dim, the CLIP widths
and the scheduler's shift (and dynamic shifting, if a config enables it) are READ from the loaded model; the T5 length is CFG["t5_max_len"];
the dtype is config.dtype_name() (fp16 for SD3-medium, bf16 for SD3.5 -- fp16 overflows on SD3.5-large). Plain CFG only.
Everything the sampler, probes and evaluators need is a method here so that `dummy.Backend` can mirror it on CPU.

Noise parameterization (see sampling.py): flow matching, so a level IS its sigma, `mix` is the rectified-flow interpolation
(1-s) x0 + s eps and `noise_frac` is the sigma itself. The evaluation and array helpers come from `evalmix`."""
import os, math, numpy as np
from PIL import Image
from .config import CFG
from .evalmix import EvalMixin, TorchOpsMixin
from .scenes import NEG, word_split, build_spec
from . import textcond

class Backend(TorchOpsMixin, EvalMixin):
    dummy = False
    LC, VS = 16, 8                                                   # latent channels / VAE stride
    canvas_multiple = 16                                             # node canvases must be a multiple of 16 px (VAE stride 8 x patch 2)
    is_video = False
    def __init__(self, load_t5=True):
        import torch, torch.nn.functional as F
        from diffusers import StableDiffusion3Pipeline
        self.torch, self.F = torch, F; torch.set_num_threads(CFG["threads"]); self.device = "cuda"; tok = os.environ.get("HF_TOKEN")
        from . import config
        self.dtype = dict(fp16=torch.float16, bf16=torch.bfloat16)[config.dtype_name()]
        self.pipe = StableDiffusion3Pipeline.from_pretrained(CFG["model_id"], text_encoder_3=None, tokenizer_3=None, torch_dtype=self.dtype, token=tok).to(self.device)
        self.pipe.set_progress_bar_config(disable=True); self.pipe.vae.enable_tiling(); self.pipe.transformer.requires_grad_(False)
        for te in (self.pipe.text_encoder, self.pipe.text_encoder_2): te.requires_grad_(False)
        self.t5 = self.tok3 = None
        if load_t5:
            from transformers import T5EncoderModel, T5TokenizerFast
            self.tok3 = T5TokenizerFast.from_pretrained(CFG["model_id"], subfolder="tokenizer_3", token=tok)
            try: self.t5 = T5EncoderModel.from_pretrained(CFG["model_id"], subfolder="text_encoder_3", torch_dtype=self.dtype, variant="fp16", token=tok)
            except (OSError, ValueError): self.t5 = T5EncoderModel.from_pretrained(CFG["model_id"], subfolder="text_encoder_3", torch_dtype=self.dtype, token=tok)   # repos without an fp16 variant (SD3.5)
            self.t5 = self.t5.to(self.device).eval()
            self.t5.requires_grad_(False)
        self._load_eval()
        self.N_CLIP = self.pipe.tokenizer.model_max_length; self.JOINT_DIM = self.pipe.transformer.config.joint_attention_dim; self.T5_LEN = CFG["t5_max_len"]
        self.N_BLOCKS = len(self.pipe.transformer.transformer_blocks)
        self.CLIP_DIM = int(self.pipe.text_encoder.config.hidden_size + self.pipe.text_encoder_2.config.hidden_size)   # 768 + 1280, zero-padded to JOINT_DIM
        self.toks = dict(clip=(self.pipe.tokenizer, self.N_CLIP), clip2=(self.pipe.tokenizer_2, self.N_CLIP)) | (dict(t5=(self.tok3, self.T5_LEN)) if self.tok3 else {})
        self._enc, self._cond, self._neg = {}, {}, {}
        sc = self.pipe.scheduler.config
        print(f"backend: SD3 loaded | {CFG['model_id']} | {config.dtype_name()} | T5 {'loaded' if self.t5 else 'absent'} (len {self.T5_LEN}) | blocks {self.N_BLOCKS} | "
              f"joint dim {self.JOINT_DIM} | shift {sc.get('shift')} dyn {bool(sc.get('use_dynamic_shifting', False))} | vram {self.vram_gb():.1f} GB")

    # ---------------- text: word atoms -> token spans -> weighted encodings ----------------
    def _spans(self, name, prompt, units):
        tk, L = self.toks[name]; spans, prev = [], (1 if name != "t5" else 0)             # CLIP has BOS at 0, T5 has none
        for u in units:
            n = len(tk(prompt[:u["end"]], truncation=True, max_length=L)["input_ids"]) - 1  # both append EOS
            assert n >= prev, (name, u); spans.append((prev, n)); prev = n
        assert prev < L, f"prompt too long for {name}: {prev} >= {L}"; return spans
    def encode_text(self, scene, t5=False, atom_level="word"):
        """Encoding of the full prompt + the atom bookkeeping. `t5` False -> T5 slots are zeros (what the pipeline does without text_encoder_3)."""
        if t5 and self.t5 is None: raise RuntimeError("T5 requested but not loaded")
        prompt, units = word_split(scene); spec = build_spec(scene, units, atom_level)
        spec.update(prompt=prompt, t5=bool(t5), spans={k: self._spans(k, prompt, units) for k in self.toks}, units=units)
        spec["n_real"] = {k: spec["spans"][k][-1][1] for k in self.toks}
        spec["E"], spec["P"] = self._encode(spec, None); return spec
    def _tok_weights(self, spec, name, w):
        tk, L = self.toks[name]; wt = np.ones(L, np.float32)
        for j, ua in enumerate(spec["unit_of_atom"]):
            for i in ua: s, e = spec["spans"][name][i]; wt[s:e] = w[j]
        return self.torch.as_tensor(wt, device=self.device).view(1, -1, 1)
    def _encode(self, spec, w):
        torch, p = self.torch, self.pipe; hooks = []
        if w is not None:
            layers = [("clip", p.text_encoder.text_model.embeddings.token_embedding), ("clip2", p.text_encoder_2.text_model.embeddings.token_embedding)]
            if spec["t5"]: layers.append(("t5", self.t5.get_input_embeddings()))
            for name, emb in layers:
                wt = self._tok_weights(spec, name, w); hooks.append(emb.register_forward_hook(lambda m, i, o, wt=wt: o * wt.to(o.dtype)))
        try:
            with torch.no_grad():
                embs, pooled = [], []
                for tk, te in ((p.tokenizer, p.text_encoder), (p.tokenizer_2, p.text_encoder_2)):
                    ids = tk(spec["prompt"], padding="max_length", max_length=self.N_CLIP, truncation=True, return_tensors="pt").input_ids.to(self.device)
                    out = te(ids, output_hidden_states=True); pooled.append(out[0]); embs.append(out.hidden_states[-2])
                clip = self.F.pad(torch.cat(embs, dim=-1), (0, self.JOINT_DIM - self.CLIP_DIM))
                if spec["t5"]:
                    ids3 = self.tok3(spec["prompt"], padding="max_length", max_length=self.T5_LEN, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)
                    t5e = self.t5(ids3)[0].to(clip.dtype)
                else: t5e = torch.zeros((1, self.T5_LEN, self.JOINT_DIM), device=self.device, dtype=clip.dtype)
        finally:
            for h in hooks: h.remove()
        return torch.cat([clip, t5e], dim=1), torch.cat(pooled, dim=-1)
    def _wkey(self, spec, w): return textcond.wkey(spec, w)
    def encode_weighted(self, spec, w):
        """Cached (E, P) for atom weights w (K,). All-ones -> the stored full encoding. `CFG["atom_mode"]` decides how an atom is
        removed: "delete" re-encodes the prompt without the item (no hooks), "zero" masks its token embeddings (see textcond.py)."""
        return textcond.encode_weighted(self, spec, w)
    def neg_cond(self, t5):
        if t5 not in self._neg:
            spec = dict(prompt=NEG, t5=bool(t5), unit_of_atom=[], spans={}); self._neg[t5] = self._encode(spec, None)
        return self._neg[t5]
    def text_cond(self, prompt, t5=None):
        """Condition from a FREE text (no atom bookkeeping), e.g. a per-window prompt written by an LLM planner (longhorizon.flat) or a
        scene's `root_prompt`. T5 is used iff loaded (t5=None) -- the v38 setting is T5 on. Same encoder path as make_cond with all-ones
        weights, so text_cond(full_prompt(scene)) == make_cond(spec, ones) for a T5-on spec."""
        t5 = (self.t5 is not None) if t5 is None else bool(t5); key = ("text", prompt, t5)
        if key not in self._cond:
            if len(self._cond) > 600: self._cond.pop(next(iter(self._cond)))
            E, P = self._encode(dict(prompt=prompt, t5=t5, unit_of_atom=[], spans={}), None); NE, NP = self.neg_cond(t5)
            self._cond[key] = dict(pe=self.torch.cat([NE, E]), ppe=self.torch.cat([NP, P]))
        return self._cond[key]
    def make_cond(self, spec, w):
        key = self._wkey(spec, w)
        if key not in self._cond:
            if len(self._cond) > 600: self._cond.pop(next(iter(self._cond)))
            E, P = self.encode_weighted(spec, w); NE, NP = self.neg_cond(spec["t5"])
            self._cond[key] = dict(pe=self.torch.cat([NE, E]), ppe=self.torch.cat([NP, P]))
        return self._cond[key]

    # ---------------- sampler primitives (flow matching: a level is its sigma) ----------------
    def levels(self, steps): return self.get_sigmas(steps)
    def get_sigmas(self, steps):
        """The loaded scheduler's sigmas (its own shift). If its config enables dynamic shifting, mu is computed from the token count of an
        interior CHILD canvas (config.child_canvas_px() x height), one schedule for the whole tree, exactly as the pipeline would for that size."""
        sch = self.pipe.scheduler
        if sch.config.get("use_dynamic_shifting", False):
            from . import config
            from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
            n_tok = (config.height() // (self.VS * 2)) * (config.child_canvas_px() // (self.VS * 2))
            mu = calculate_shift(n_tok, sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096), sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.16))
            sch.set_timesteps(steps, device=self.device, mu=mu)
        else: sch.set_timesteps(steps, device=self.device)
        return [float(s) for s in sch.sigmas]
    def noise_frac(self, lvl): return float(lvl)
    def init_latent(self, noise, lvl): return float(lvl) * noise
    def mix(self, x0, eps, lvl): s = float(lvl); return (1 - s) * x0 + s * eps
    def alpha_sigma(self, lvl): s = float(lvl); return 1.0 - s, s                                  # flow matching: x = (1 - s) x0 + s eps (for sampling.dpmpp2m_step)
    def denoise(self, x, lvl, cond, cfg):
        """(x0_hat, eps_hat) from the velocity field: x0 = x - s v, eps = x + (1-s) v."""
        s = float(lvl); v = self.predict_v(x, s, cond, cfg); return x - s * v, x + (1 - s) * v
    def predict_v(self, x, sigma, cond, cfg):
        torch = self.torch
        with torch.no_grad():
            v = self.pipe.transformer(hidden_states=torch.cat([x, x]).to(self.dtype), timestep=torch.full((2,), sigma * 1000.0, device=self.device),
                                      encoder_hidden_states=cond["pe"], pooled_projections=cond["ppe"], return_dict=False)[0].float()
        vu, vc = v.chunk(2); return vu + cfg * (vc - vu)
    def decode(self, lat):
        torch = self.torch
        with torch.no_grad():
            z = lat.to(self.dtype) / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
            return self.pipe.image_processor.postprocess(self.pipe.vae.decode(z, return_dict=False)[0], output_type="pil")[0]

    # ---------------- probes ----------------
    def region_losses(self, err, regions): return self.torch.stack([err[..., c0:c1].mean(dim=(-2, -1)) for c0, c1 in regions], dim=-1)
    def delta_exact(self, y0, spec, atoms, regions, ts, seed, w_base=None, n_eps=1, return_all=False):
        """Paired ablation at the encoder input (all encoders at once, per atom), on top of the node's own weights. Returns (n_reg, len(atoms));
        with return_all=True the per-noise estimates (n_eps, n_reg, K), from which the measurement noise of each atom's profile is estimated."""
        torch = self.torch; n_reg, K = len(regions), len(atoms); D = torch.zeros(n_reg, K); Ds = []
        wb = np.ones(spec["K"]) if w_base is None else np.asarray(w_base, dtype=np.float64)
        E0, P0 = self.encode_weighted(spec, wb); abl = []
        for a in atoms:
            w = wb.copy(); w[a] = 0.0; abl.append(self.encode_weighted(spec, w))
        cnt = 0
        with torch.no_grad():
            for ei in range(n_eps):
                De = torch.zeros(n_reg, K)
                for ti, t in enumerate(ts):
                    eps = self.randn(y0.shape, seed * 7919 + 101 * ei + ti); y_t = (1 - t) * y0 + t * eps; v_star = eps - y0
                    def losses(Eb, Pb):
                        B = Eb.shape[0]
                        v = self.pipe.transformer(hidden_states=y_t.to(self.dtype).repeat(B, 1, 1, 1), timestep=torch.full((B,), t * 1000.0, device=self.device),
                                                  encoder_hidden_states=Eb, pooled_projections=Pb, return_dict=False)[0].float()
                        return self.region_losses(((v - v_star) ** 2).sum(1), regions)
                    base = losses(E0, P0)[0]
                    for i in range(0, K, CFG["probe_batch"]):
                        chunk = abl[i:i + CFG["probe_batch"]]
                        De[:, i:i + len(chunk)] += (losses(torch.cat([e for e, p in chunk]), torch.cat([p for e, p in chunk])) - base).T.cpu()
                Ds.append(De / len(ts)); D += De; cnt += len(ts)
        return torch.stack(Ds) if return_all else D / cnt
    def attn_mass(self, y0, spec, atoms, regions, ts, seed, blocks="all"):
        """Mean image->text attention mass per (region, atom), averaged over the atom's word tokens (CLIP + T5 if on; a phrase atom's
        separator comma is excluded). A diagnostic, and since v38 the input of the attention COMPARISON arms (routing.route_attn) --
        never part of the method."""
        torch = self.torch; import torch.nn.functional as F_
        lat_w = y0.shape[-1]; n_img = (y0.shape[-2] // 2) * (lat_w // 2); wp = lat_w // 2; n_txt = spec["E"].shape[1]
        col_of_tok = (torch.arange(n_img, device=self.device) % wp) * 2
        acc = torch.zeros(n_img, n_txt, device=self.device); orig = F_.scaled_dot_product_attention; state = dict(call=0)
        keep = set(range(self.N_BLOCKS)) if blocks == "all" else set(range(6, 18))
        def patched(q, k, v, *args, **kwargs):
            out = orig(q, k, v, *args, **kwargs)
            if k.shape[2] == n_img + n_txt and q.shape[0] == 1:
                b = state["call"]; state["call"] += 1
                if b in keep:
                    scale = kwargs.get("scale", None) or (1.0 / math.sqrt(q.shape[-1])); qi = q[0, :, :n_img].float(); kk = k[0].float()
                    for hs in range(0, qi.shape[0], 4):
                        lg = torch.einsum("hnd,hmd->hnm", qi[hs:hs + 4], kk[hs:hs + 4]) * scale
                        acc.add_(torch.softmax(lg, dim=-1)[..., n_img:].sum(0))
            return out
        try:
            F_.scaled_dot_product_attention = patched
            with torch.no_grad():
                for ti, t in enumerate(ts):
                    state["call"] = 0; eps = self.randn(y0.shape, seed * 7919 + ti); y_t = (1 - t) * y0 + t * eps
                    self.pipe.transformer(hidden_states=y_t.to(self.dtype), timestep=torch.full((1,), t * 1000.0, device=self.device),
                                          encoder_hidden_states=spec["E"], pooled_projections=spec["P"], return_dict=False)
        finally:
            F_.scaled_dot_product_attention = orig
        if float(acc.sum()) == 0.0: raise RuntimeError("SDPA not intercepted by this diffusers version")
        A_tok = torch.stack([acc[(col_of_tok >= c0) & (col_of_tok < c1)].mean(0) for c0, c1 in regions]) / len(ts)   # (n_reg, n_txt)
        cols = []; units = spec.get("units") or []; keep_maps = bool(getattr(self, "keep_attn_maps", False)); maps = []
        for a in atoms:
            ua = [i for i in spec["unit_of_atom"][a] if not (i < len(units) and units[i]["punct"])] or list(spec["unit_of_atom"][a])   # phrase atoms: the words, not the separator comma
            idx = [t for i in ua for t in range(*spec["spans"]["clip"][i])]
            if spec["t5"]: idx += [self.N_CLIP + t for i in ua for t in range(*spec["spans"]["t5"][i])]
            cols.append(A_tok[:, idx].mean(1) if idx else torch.zeros(len(regions), device=self.device))
            if keep_maps: maps.append((acc[:, idx].mean(1) / len(ts)).view(y0.shape[-2] // 2, wp).cpu().numpy() if idx else None)   # figure hook (off by default): per-atom SPATIAL map, same tokens
        if keep_maps: self.attn_maps = maps
        return torch.stack(cols, 1).cpu()
