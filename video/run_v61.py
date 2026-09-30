#!/usr/bin/env python
"""v61: HLC's video instantiation on ONE model for every node: LTX-2 distilled (8 sigmas, CFG 1), 1280x704, 121 frames at 24 fps.

Abstract structure (as v58 B, one model instead of Wan-5B root + FLF2V-14B children):
 (1) root   : the whole story prompt, 121 frames (5 s) -> <out>/<scene>/root/
 (2) routing: the model's OWN video->text cross-attention (every block's attn2) on the root, re-noised at sigmas ATTN_TS; per latent frame
              (T = 16), distinguishing tokens + per-token normalisation + contrastive event score (the v53/v54 A1 score, ported from
              longhorizon.backend_wan.attn_time_profile / runners.attn_time_atoms); aggregated over 4 windows (= root quarters) and routed
              by the settled A1 rule (dominant event per window if At > 1 + 0.02, every event keeps its argmax window; context always).
 (3) leaves : keyframe cascade. Window k covers root frames [30k, 30k + 30]; it is a fresh 121-frame LTX sample conditioned on the parent's
              frames at its two boundaries (root f30k as the first latent frame, strength 1; root f30(k+1) as an appended keyframe token
              at pixel frame 120, strength 1) plus the window's routed text. Optional `--guides` = interior parent frames at low strength.
              Joins: window k+1 starts at the SAME root keyframe window k ends on; the assembled video keeps w_k[0:120] + ... + w_3[0:121].
 (4) `--present` (arm K1): an event whose per-frame routing score at the window's FIRST latent frame(s) exceeds 1 + margin is already
              in the parent keyframe: its text becomes a state ("<entity> is already there") instead of the action phrase.

Stages: --stage root | probe | windows [--windows 0,1] | assemble.  Run via run_ltx.sh (see README).
"""
import argparse, json, os, sys, time, math
ROOT = os.path.dirname(os.path.abspath(__file__))          # the video/ directory (holds the small `longhorizon` guard package)
sys.path.insert(0, ROOT)
from longhorizon import guard
guard.install()
import numpy as np
from PIL import Image

MODELS = {"ltx25": "Lightricks/LTX-2.5-Diffusers", "ltx2": "rootonchair/LTX-2-19b-distilled"}   # LTX-2.5 distilled = main; LTX-2 19B distilled = comparison
MODEL = MODELS["ltx25"]
ATTN_TS = [0.25, 0.5, 0.625, 0.75, 0.8333333333333334, 0.9375]
W, H, NF, FPS = 1280, 704, 121, 24.0
SCENES = {"park_typed": dict(base="A static shot of a wooden park bench on a green lawn", style="the camera never moves",
                             objects=[dict(phrase="a grey pigeon lands on the bench", entity="a grey pigeon", head="pigeon"),
                                      dict(phrase="a brown dog walks past in front of the bench", entity="a brown dog", head="dog"),
                                      dict(phrase="a child holding a red balloon walks by", entity="a child holding a red balloon", head="balloon")])}
SCENES["street_typed"] = dict(base="A static shot of a quiet street corner with a crosswalk", style="the camera never moves",   # = longhorizon.scenes street_typed
                              objects=[dict(phrase="a red car drives past", head="car"), dict(phrase="a cyclist rides by on a bicycle", head="cyclist"),
                                       dict(phrase="a yellow bus pulls up at the curb", head="bus")])
SCENES["beach_typed"] = dict(base="A static shot of an empty sandy beach with a calm blue sea", style="the camera never moves",   # = longhorizon.scenes beach_typed
                             objects=[dict(phrase="a white seagull flies past", head="seagull"), dict(phrase="a black dog runs along the shore", head="dog"),
                                      dict(phrase="a person in a red swimsuit jogs by", head="person")])
SCENES["snow_typed"] = dict(base="A static shot of a snowy field with a single pine tree", style="the camera never moves",   # = longhorizon.scenes snow_typed
                            objects=[dict(phrase="a red fox trots across the snow", head="fox"), dict(phrase="a brown deer walks past", head="deer"),
                                     dict(phrase="a skier in a blue jacket glides by", head="skier")])
STOP = frozenset("a an the of on in at by to with and or into from for over is are it its his her their this that".split())


def full_prompt(sc): return ", ".join([sc["base"]] + [o["phrase"] for o in sc["objects"]] + [sc["style"]])


def entity_of(o):
    """The event's subject = its phrase up to and including the head noun (typed-list data: `head`), e.g. 'a brown dog'."""
    ws = o["phrase"].split(); i = next(i for i, w in enumerate(ws) if w.strip(",").lower() == o["head"]); return " ".join(ws[:i + 1])


def lat_of_px(i): return 0 if i <= 0 else (int(i) - 1) // 8 + 1


# ------------------------------------------------------------------ model
def load(dev="cuda"):
    import torch, transformers
    from diffusers import (LTX2ConditionPipeline, LTX2VideoTransformer3DModel, FlowMatchEulerDiscreteScheduler, AutoencoderKLLTX2Video,
                           AutoencoderKLLTX2Audio)
    from diffusers.pipelines.ltx2 import connectors as C_, vocoder as V_
    t0 = time.time(); bf = torch.bfloat16
    # components built one by one from the LOCAL snapshot (only the parts the distilled path needs are downloaded; LTX-2.5's
    # prompt_enhancer / diffusion_decoder / transformer_full are not); every big component straight to the GPU (RSS guard 55 GB).
    # LTX-2: Gemma3ForConditionalGeneration + GemmaTokenizerFast; LTX-2.5: Gemma4UnifiedForConditionalGeneration + GemmaTokenizer.
    import glob; hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")   # partial (allow_patterns) snapshot: use its dir directly
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + MODEL.replace("/", "--"), "snapshots", "*")))
    if snaps: src = snaps[-1]
    else:                                                          # not cached yet: download only the components the distilled path needs
        from huggingface_hub import snapshot_download
        src = snapshot_download(MODEL, allow_patterns=[f"{c}/*" for c in ("audio_vae", "connectors", "scheduler", "text_encoder", "tokenizer",
                                                                            "transformer", "vae", "vocoder", "processor", "duration_head")] + ["model_index.json"])
    mi = json.load(open(os.path.join(src, "model_index.json")))
    te = getattr(transformers, mi["text_encoder"][1]).from_pretrained(src, subfolder="text_encoder", dtype=bf, device_map=dev)
    tk = transformers.AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    tr = LTX2VideoTransformer3DModel.from_pretrained(src, subfolder="transformer", torch_dtype=bf, device_map=dev)
    con = getattr(C_, mi["connectors"][1]).from_pretrained(src, subfolder="connectors", torch_dtype=bf).to(dev)
    voc = getattr(V_, mi["vocoder"][1]).from_pretrained(src, subfolder="vocoder", torch_dtype=bf).to(dev)
    vae = AutoencoderKLLTX2Video.from_pretrained(src, subfolder="vae", torch_dtype=bf).to(dev)
    avae = AutoencoderKLLTX2Audio.from_pretrained(src, subfolder="audio_vae", torch_dtype=bf).to(dev)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(src, subfolder="scheduler")
    pipe = LTX2ConditionPipeline(scheduler=sch, vae=vae, audio_vae=avae, text_encoder=te, tokenizer=tk, connectors=con, transformer=tr, vocoder=voc)
    # distilled recipe: explicit sigma schedule used as given (no dynamic / terminal shift)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None)
    if getattr(pipe, "audio_scheduler", None) is not None:
        pipe.audio_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.audio_scheduler.config, use_dynamic_shifting=False, shift_terminal=None)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    print(f"[v61] loaded {MODEL} in {time.time() - t0:.0f}s; vram {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    return pipe


def remap_keyframe_pixels(pipe, remap):
    """LTX2ConditionPipeline places a condition at latent index li > 0 on pixel frame (li - 1) * 8 + 1. The reference LTX-2 keyframe
    conditioning (VideoConditionByKeyframeIndex) takes a PIXEL frame; `remap` {(li-1)*8+1: pixel} moves the appended keyframe tokens'
    temporal RoPE coordinate to the exact pixel frame (e.g. the last frame 120)."""
    orig = type(pipe)._prepare_keyframe_coords
    def f(self, keyframe_latent_num_frames, keyframe_latent_height, keyframe_latent_width, pixel_frame_idx, num_pixel_frames, fps, device):
        return orig(self, keyframe_latent_num_frames, keyframe_latent_height, keyframe_latent_width, remap.get(pixel_frame_idx, pixel_frame_idx),
                    num_pixel_frames, fps, device)
    pipe._prepare_keyframe_coords = f.__get__(pipe)


GEN = dict(guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, audio_guidance_scale=1.0, audio_stg_scale=0.0, audio_modality_scale=1.0)


def sigmas():
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    return list(DISTILLED_SIGMA_VALUES)


def decode(pipe, lat):
    import torch
    with torch.no_grad():
        v = pipe.vae.decode(lat.to(pipe.vae.dtype), None, return_dict=False)[0]
    v = pipe.video_processor.postprocess_video(v, output_type="np")[0]
    return (np.clip(v, 0, 1) * 255).round().astype(np.uint8)


def save_mp4(frames, path, fps=FPS):
    import imageio.v2 as iio
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=None, ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"])
    for f in frames: w.append_data(np.asarray(f))
    w.close()


def grid(frames, path, every=16, cols=8, w=320, off=0):
    from PIL import ImageDraw
    fr = [(i, Image.fromarray(f)) for i, f in enumerate(frames) if i % every == 0 or i == len(frames) - 1]
    h = int(fr[0][1].height * w / fr[0][1].width); rows = (len(fr) + cols - 1) // cols
    g = Image.new("RGB", (cols * (w + 2), rows * (h + 2))); d = ImageDraw.Draw(g)
    for k, (i, im) in enumerate(fr):
        x, y = (k % cols) * (w + 2), (k // cols) * (h + 2); g.paste(im.resize((w, h)), (x, y)); d.rectangle([x, y, x + 44, y + 14], fill=(0, 0, 0))
        d.text((x + 3, y + 2), f"f{i + off}", fill=(255, 255, 0))
    g.save(path, quality=85)


# ------------------------------------------------------------------ routing probe
def token_spans(pipe, prompt, phrases):
    """Gemma token indices (positions in the connector output: the valid tokens are moved to the front, BOS = 0) of each phrase."""
    enc = pipe.tokenizer(prompt.strip(), add_special_tokens=True, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]; out = []
    for p in phrases:
        c0 = prompt.index(p); c1 = c0 + len(p)
        out.append([i for i, (a, b) in enumerate(offs) if b > a and a < c1 and b > c0])
    return ids, pipe.tokenizer.convert_ids_to_tokens(ids), out


def distinct_token_idx(pieces, tok_idx, ents):
    """= longhorizon.routing.distinct_token_idx (v53): an event's content-word tokens that no other event shares (Gemma pieces, U+2581)."""
    word_of, w = {}, -1
    for i, p in enumerate(pieces):
        if p.startswith("▁") or w < 0 or not p.replace("▁", "").isalnum(): w += 1
        word_of[i] = w
    text = {}
    for i, p in enumerate(pieces): text[word_of[i]] = text.get(word_of[i], "") + p.replace("▁", "")
    content = lambda i: text[word_of[i]].isalpha() and text[word_of[i]].lower() not in STOP
    C = {j: {pieces[i] for i in tok_idx[j] if content(i)} for j in ents}; out = {}
    for j in ents:
        others = set().union(*[C[k] for k in ents if k != j]) if len(ents) > 1 else set()
        out[j] = [i for i in tok_idx[j] if content(i) and pieces[i] not in others] or list(tok_idx[j])
    return out


def attn_time_profile(pipe, lat, alat, prompt, seed, n_keep):
    """Per latent frame, the mean text cross-attention distribution of that frame's video tokens (every block's attn2, all heads, softmax
    over all 1024 connector positions exactly as the model attends; only the first n_keep = prompt-token positions are kept).
    The clean root latent is re-noised at each sigma of ATTN_TS by the pipeline itself (latents + noise_scale = sigma, one step, the
    root's audio latents alongside) -> (len(ts), n_blocks, T, n_keep) float32."""
    import torch
    tr = pipe.transformer; blks = tr.transformer_blocks; T, Hl, Wl = lat.shape[2], lat.shape[3], lat.shape[4]; n_sp = Hl * Wl
    acc = torch.zeros(len(ATTN_TS), len(blks), T, n_keep, device="cuda", dtype=torch.float32); cur = dict(ti=0)
    class Cap:
        def __init__(s_, bi, orig): s_.bi, s_.orig = bi, orig
        def __call__(s_, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query_rotary_emb=None, key_rotary_emb=None, **kw):
            hs = hidden_states[0, :T * n_sp]; eh = encoder_hidden_states[0]
            q = attn.norm_q(attn.to_q(hs)).unflatten(1, (attn.heads, -1)).float(); k = attn.norm_k(attn.to_k(eh)).unflatten(1, (attn.heads, -1)).float()
            bias = None
            if attention_mask is not None: bias = attention_mask.reshape(-1)[-k.shape[0]:].float()
            sc = 1.0 / math.sqrt(q.shape[-1]); out = acc[cur["ti"], s_.bi]
            for a in range(0, q.shape[0], 2048):
                b = min(q.shape[0], a + 2048); lg = torch.einsum("nhd,mhd->hnm", q[a:b], k) * sc
                if bias is not None: lg = lg + bias
                p = torch.softmax(lg, dim=-1)[..., :n_keep].sum(0)
                out.index_add_(0, torch.arange(a, b, device=q.device) // n_sp, p); del lg
            return s_.orig(attn, hidden_states, encoder_hidden_states, attention_mask, query_rotary_emb, key_rotary_emb, **kw)
    origs = {bi: b.attn2.processor for bi, b in enumerate(blks)}
    try:
        for bi, b in enumerate(blks): b.attn2.processor = Cap(bi, origs[bi])
        for ti, s in enumerate(ATTN_TS):
            cur["ti"] = ti; g = torch.Generator("cuda").manual_seed(seed * 7919 + ti)
            with torch.no_grad():
                pipe(prompt=prompt, latents=lat, audio_latents=alat, sigmas=[float(s)], num_inference_steps=1, noise_scale=float(s),
                     height=H, width=W, num_frames=NF, frame_rate=FPS, generator=g, output_type="latent", return_dict=False, **GEN)
    finally:
        for bi, b in enumerate(blks): b.attn2.processor = origs[bi]
    heads = blks[0].attn2.heads; acc /= float(heads * n_sp)
    return acc.cpu().numpy()


def atoms_score(P, tok, ents, contrast=True):
    """= runners.attn_time_atoms with norm 'token', distinct tokens, contrast: (T, K) per latent frame and atom."""
    prof = np.asarray(P, np.float64).mean(axis=(0, 1)); prof = prof / (prof.mean(0, keepdims=True) + 1e-12)
    a = np.stack([prof[:, idx].sum(1) for idx in tok], 1)
    if contrast and len(ents) > 1:
        At = a[:, ents] / (a[:, ents].mean(0, keepdims=True) + 1e-12); c = At - (At.sum(1, keepdims=True) - At) / (len(ents) - 1)
        a = a.copy(); a[:, ents] = np.clip(c + 1.0, 1e-6, None)
    return a


def route_dominant(a, regions, ents, margin=0.02):
    """Settled A1 rule (routing.route_attn_time 'hard', time_hard_margin): At = region means / mean over regions; window k gets its dominant
    event if At > 1 + margin; every event keeps its argmax window."""
    A = np.stack([a[c0:c1].mean(0) for c0, c1 in regions]); At = A / (A.mean(0, keepdims=True) + 1e-12)
    Wp = np.zeros((len(regions), len(ents)), int)
    for k in range(len(regions)):
        jd = int(np.argmax(At[k, ents]))
        if At[k, ents[jd]] > 1.0 + margin: Wp[k, jd] = 1
    for j in range(len(ents)): Wp[int(np.argmax(At[:, ents[j]])), j] = 1
    return Wp, At


# ------------------------------------------------------------------ stages
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="park_typed"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage", required=True, choices=["root", "probe", "windows", "assemble", "onepass", "glv"])
    ap.add_argument("--arm", default="K0", help="K0 = routed text; K1 = + 'already present' rule; KG = K1 + interior guides; KB = broadcast reference")
    ap.add_argument("--windows", default=None); ap.add_argument("--n-win", type=int, default=4)
    ap.add_argument("--present-margin", type=float, default=0.02)
    ap.add_argument("--guides", default="", help="comma list of child pixel frames that get the parent frame at low strength, e.g. 40,80")
    ap.add_argument("--guide-strength", type=float, default=0.3); ap.add_argument("--frames", type=int, default=481)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs")); ap.add_argument("--model", default="ltx25", choices=list(MODELS))
    a = ap.parse_args(); global MODEL; MODEL = MODELS[a.model]; sc = SCENES[a.scene]
    sd = os.path.join(a.out, a.scene) if a.model == "ltx25" else os.path.join(a.out, a.scene, a.model); rd = os.path.join(sd, "root" if a.seed == 1 else f"root_s{a.seed}"); os.makedirs(rd, exist_ok=True)
    prompt = full_prompt(sc)
    import torch
    torch.set_num_threads(int(os.environ.get("LH_THREADS", "24")))

    if a.stage in ("onepass", "glv"):
        run_baseline(a, sc, sd, prompt); return

    if a.stage == "root":
        pipe = load(); t0 = time.time(); g = torch.Generator("cuda").manual_seed(a.seed)
        with torch.no_grad():
            lat, alat = pipe(prompt=prompt, height=H, width=W, num_frames=NF, frame_rate=FPS, sigmas=sigmas(), num_inference_steps=len(sigmas()),
                             generator=g, output_type="latent", return_dict=False, **GEN)
        t_s = time.time() - t0; fr = decode(pipe, lat); t_d = time.time() - t0 - t_s
        torch.save(dict(lat=lat.cpu(), alat=alat.cpu()), os.path.join(rd, "root_latent.pt")); np.save(os.path.join(rd, "frames.npy"), fr)
        save_mp4(fr, os.path.join(rd, "root.mp4")); grid(fr, os.path.join(rd, "grid_root.jpg"), every=8)
        json.dump(dict(model=MODEL, prompt=prompt, seed=a.seed, size=[W, H], frames=NF, fps=FPS, sigmas=sigmas(), sample_s=round(t_s, 1),
                       decode_s=round(t_d, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), latent_shape=list(lat.shape)),
                  open(os.path.join(rd, "root_time.json"), "w"), indent=1)
        print(f"[v61] root {fr.shape} sample {t_s:.0f}s decode {t_d:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        a.stage = "probe"                                              # continue with the probe in the same process
    else: pipe = None

    if a.stage == "probe":
        pipe = pipe or load(); d = torch.load(os.path.join(rd, "root_latent.pt")); lat, alat = d["lat"].cuda(), d["alat"].cuda()
        phrases = [sc["base"]] + [o["phrase"] for o in sc["objects"]] + [sc["style"]]; ents = list(range(1, 1 + len(sc["objects"])))
        ids, pieces, tok = token_spans(pipe, prompt, phrases); dt = distinct_token_idx(pieces, tok, ents)
        tok_d = [dt.get(j, t) for j, t in enumerate(tok)]
        t0 = time.time(); P = attn_time_profile(pipe, lat, alat, prompt, a.seed, len(ids)); t_p = time.time() - t0
        np.save(os.path.join(rd, "attn_P.npy"), P.astype(np.float32))
        prof_c = atoms_score(P, tok_d, ents, True); prof_p = atoms_score(P, tok_d, ents, False)
        T = lat.shape[2]; kidx = [int(round(k * (NF - 1) / a.n_win)) for k in range(a.n_win + 1)]
        b = [lat_of_px(i) for i in kidx]; b[-1] = T; regions = [(b[k], max(b[k + 1], b[k] + 1)) for k in range(a.n_win)]
        Wp, At = route_dominant(prof_c, regions, ents)
        rec = dict(prompt=prompt, pieces=pieces, atom_tokens=tok, distinct_tokens={str(j): [pieces[i] for i in dt[j]] for j in dt},
                   attn_ts=ATTN_TS, blocks="all", n_blocks=int(P.shape[1]), probe_s=round(t_p, 1), keyframes_root=kidx, regions=regions,
                   attn_profile=np.round(prof_c, 4).tolist(), attn_profile_plain=np.round(prof_p, 4).tolist(), route_A=np.round(At, 4).tolist(),
                   plan=Wp.tolist(), rule="dominant(At>1.02)+argmax on the contrastive distinct-token score")
        json.dump(rec, open(os.path.join(rd, "route.json"), "w"), indent=1)
        print(f"[v61] probe {t_p:.0f}s; distinct {rec['distinct_tokens']}\n contrastive profile (T x events):\n{np.round(prof_c[:, ents], 2)}\n At {np.round(At[:, ents], 3)}\n plan {Wp.tolist()}", flush=True)
        return

    if a.stage == "windows":
        R = json.load(open(os.path.join(rd, "route.json"))); fr = np.load(os.path.join(rd, "frames.npy")); kidx = R["keyframes_root"]
        ents = list(range(1, 1 + len(sc["objects"]))); Wp = np.array(R["plan"]); prof = np.array(R["attn_profile"]); regions = R["regions"]
        od = os.path.join(sd, f"ltx_{a.arm}" + ("" if a.seed == 1 else f"_s{a.seed}")); os.makedirs(od, exist_ok=True)
        prompts, present = [], []
        for k in range(a.n_win):
            t_first = regions[k][0]; pres = []
            if a.arm in ("K1", "KG") and k > 0:                        # the window's first latent frame(s): the parent keyframe it starts on
                sc_first = prof[max(0, t_first - 1):t_first + 1][:, ents].mean(0)
                pres = [j for j in range(len(ents)) if sc_first[j] > 1.0 + a.present_margin]
            items = [f"{entity_of(o)} is already there" if j in pres else o["phrase"] for j, o in enumerate(sc["objects"]) if Wp[k, j] or j in pres]
            if a.arm == "KB": items = [o["phrase"] for o in sc["objects"]]     # reference: broadcast (the full list in every window, no routing)
            prompts.append(", ".join([sc["base"]] + items + [sc["style"]])); present.append(pres)
        print("[v61] " + a.arm + "\n" + "\n".join(f"  w{k}: {p}" for k, p in enumerate(prompts)), flush=True)
        json.dump(dict(prompts=prompts, present=present), open(os.path.join(od, "prompts.json"), "w"), indent=1)
        todo = [int(x) for x in a.windows.split(",")] if a.windows else list(range(a.n_win))
        todo = [k for k in todo if not os.path.exists(os.path.join(od, f"win{k}.npy"))]
        if not todo: print("[v61] nothing to do"); return
        pipe = load()
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        guides = [int(x) for x in a.guides.split(",") if x.strip()]
        remap = {(NF - 1 - 1) // 8 * 8 + 1: NF - 1}                    # last latent index 15 -> pixel 113 -> moved to pixel 120
        for p in guides: remap[((p - 1) // 8) * 8 + 1] = p
        remap_keyframe_pixels(pipe, remap)
        for k in todo:
            t0 = time.time(); k0, k1 = kidx[k], kidx[k + 1]
            conds = [LTX2VideoCondition(frames=fr[k0], index=0, strength=1.0), LTX2VideoCondition(frames=fr[k1], index=(NF - 1 - 1) // 8 + 1, strength=1.0)]
            for p in guides:                                             # interior parent frame at the same relative time, low strength
                rf = int(round(k0 + (k1 - k0) * p / (NF - 1))); conds.append(LTX2VideoCondition(frames=fr[rf], index=(p - 1) // 8 + 1, strength=a.guide_strength))
            g = torch.Generator("cuda").manual_seed(a.seed * 100 + k)
            with torch.no_grad():
                v, _ = pipe(conditions=conds, prompt=prompts[k], height=H, width=W, num_frames=NF, frame_rate=FPS, sigmas=sigmas(),
                            num_inference_steps=len(sigmas()), generator=g, output_type="np", return_dict=False, **GEN)
            v = (np.clip(v[0], 0, 1) * 255).round().astype(np.uint8); np.save(os.path.join(od, f"win{k}.npy"), v)
            save_mp4(v, os.path.join(od, f"win{k}.mp4")); dt_ = time.time() - t0
            json.dump(dict(window_s=round(dt_, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), gpu=os.environ.get("LH_GPU"),
                           keyframes_root=[k0, k1], guides=guides, guide_strength=a.guide_strength, prompt=prompts[k]),
                      open(os.path.join(od, f"win{k}_time.json"), "w"), indent=1)
            print(f"[v61] {a.arm} w{k} done in {dt_:.0f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        a.stage = "assemble"

    if a.stage == "assemble":
        od = os.path.join(sd, f"ltx_{a.arm}" + ("" if a.seed == 1 else f"_s{a.seed}"))
        if not all(os.path.exists(os.path.join(od, f"win{k}.npy")) for k in range(a.n_win)): print("[v61] waiting for the other windows"); return
        R = json.load(open(os.path.join(rd, "route.json"))); P = json.load(open(os.path.join(od, "prompts.json")))
        frames, times, jf = [], [], []
        for k in range(a.n_win):
            v = np.load(os.path.join(od, f"win{k}.npy")); jf.append(len(frames)); frames.extend(list(v if k == a.n_win - 1 else v[:-1]))
            tp = os.path.join(od, f"win{k}_time.json"); times.append(json.load(open(tp))["window_s"] if os.path.exists(tp) else None)
        save_mp4(frames, os.path.join(od, "video.mp4")); grid(frames, os.path.join(od, "grid.jpg"), every=16)
        # keyframe fidelity: child's frame 0 / 120 vs the root keyframe (mean abs diff, 0-255)
        fr = np.load(os.path.join(rd, "frames.npy")); kidx = R["keyframes_root"]; kf_err = []
        for k in range(a.n_win):
            v = np.load(os.path.join(od, f"win{k}.npy"), mmap_mode="r")
            kf_err.append([round(float(np.abs(v[0].astype(float) - fr[kidx[k]]).mean()), 2), round(float(np.abs(v[-1].astype(float) - fr[kidx[k + 1]]).mean()), 2)])
        sc_ = SCENES[a.scene]; Wp = np.array(R["plan"])
        log = dict(scene=a.scene, id=a.scene, seed=a.seed, model=MODEL, variant=f"v61 {a.arm}: LTX keyframe cascade", prompts=P["prompts"], present=P["present"],
                   routed={str(k): " ".join(o["phrase"] for j, o in enumerate(sc_["objects"]) if Wp[k, j]) for k in range(a.n_win)}, plan=Wp.tolist(),
                   keyframes_root=kidx, geometry=dict(joints_px=jf[1:], total_px_frames=len(frames), fps=FPS), window_s=times,
                   keyframe_abs_err=kf_err, n_frames=len(frames), size=[W, H])
        json.dump(log, open(os.path.join(od, "log.json"), "w"), indent=1)
        print(f"[v61] assembled {len(frames)} frames, joints {jf[1:]}, windows {times} s, keyframe err {kf_err}", flush=True)


def encode_once(pipe, prompt):
    """Prompt embeddings once, then the 12B text encoder leaves the GPU (frees ~24 GB for long single calls)."""
    import torch
    with torch.no_grad():
        pe, pm, _, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, device="cuda")
    pipe.text_encoder.to("cpu"); torch.cuda.empty_cache(); return pe, pm


def run_baseline(a, sc, sd, prompt):
    """Baselines on the SAME model (LTX-2.5 distilled, 8 sigmas, CFG 1, 1280x704, 24 fps), full prompt, no routing:
      onepass: one call of --frames frames (481 = the cascade's length; falls back 481 -> 361 -> 241 on OOM).
      glv    : Gen-L-Video-style: overlapping 121-frame windows (16 latent frames, stride 8 latent = 64 px frames; last window flush with
               the end) over 481 frames (61 latent frames), denoised JOINTLY: at every sigma each window's x0 prediction (one pipeline
               call from the shared noisy latent, noise_scale 0 so the given latent is used as is) is averaged on the overlaps, and the
               shared latent takes the Euler step z <- z + (s' - s) (z - x0_avg) / s. Audio branch: re-drawn from noise in every call."""
    import torch
    od = os.path.join(sd, f"{a.stage}_s{a.seed}"); os.makedirs(od, exist_ok=True); pipe = load(); t0 = time.time()
    pe, pm = encode_once(pipe, prompt); kw = dict(prompt_embeds=pe, prompt_attention_mask=pm, height=H, width=W, frame_rate=FPS, output_type="latent", return_dict=False, **GEN)
    info = dict(model=MODEL, prompt=prompt, seed=a.seed, stage=a.stage, size=[W, H], fps=FPS)
    if a.stage == "onepass":
        lat = None
        for nf in [a.frames, 361, 241, 121]:
            if nf > a.frames: continue
            try:
                torch.cuda.reset_peak_memory_stats(); g = torch.Generator("cuda").manual_seed(a.seed)
                with torch.no_grad(): lat, _ = pipe(num_frames=nf, sigmas=sigmas(), num_inference_steps=len(sigmas()), generator=g, **kw)
                info.update(frames=nf); break
            except torch.cuda.OutOfMemoryError as e:
                print(f"[v61] onepass {nf} frames: OOM", flush=True); info.setdefault("oom", []).append(nf); lat = None; torch.cuda.empty_cache()
        if lat is None: json.dump(info, open(os.path.join(od, "log.json"), "w"), indent=1); raise SystemExit("onepass: OOM at every length")
    else:
        T = (a.frames - 1) // 8 + 1; Wn = 16; starts = list(range(0, T - Wn + 1, 8)); starts += [T - Wn] if starts[-1] != T - Wn else []
        info.update(frames=a.frames, windows_latent=[[s0, s0 + Wn] for s0 in starts]); sg = sigmas() + [0.0]
        C, Hl, Wl = pipe.transformer.config.in_channels, H // 32, W // 32
        g = torch.Generator("cuda").manual_seed(a.seed); z = torch.randn((1, C, T, Hl, Wl), generator=g, device="cuda", dtype=torch.float32)  # normalised space
        mean, std, sf = pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
        for i in range(len(sg) - 1):
            s, s2 = sg[i], sg[i + 1]; acc = torch.zeros_like(z); cnt = torch.zeros((1, 1, T, 1, 1), device="cuda")
            for w0 in starts:
                zw = pipe._denormalize_latents(z[:, :, w0:w0 + Wn].clone(), mean, std, sf)
                gw = torch.Generator("cuda").manual_seed(a.seed * 1000 + i * 10 + w0)
                with torch.no_grad(): x0, _ = pipe(latents=zw, num_frames=121, sigmas=[float(s)], num_inference_steps=1, noise_scale=0.0, generator=gw, **kw)
                acc[:, :, w0:w0 + Wn] += pipe._normalize_latents(x0.float(), mean, std, sf); cnt[:, :, w0:w0 + Wn] += 1
            x0a = acc / cnt; z = x0a if s2 == 0 else z + (s2 - s) * (z - x0a) / s
            print(f"[v61] glv step {i} sigma {s:.4f} done ({time.time() - t0:.0f}s)", flush=True)
        lat = pipe._denormalize_latents(z, mean, std, sf)
    t_s = time.time() - t0; fr = decode(pipe, lat); t_d = time.time() - t0 - t_s
    save_mp4(fr, os.path.join(od, "video.mp4")); grid(fr, os.path.join(od, "grid.jpg"), every=16)
    info.update(sample_s=round(t_s, 1), decode_s=round(t_d, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), n_frames=len(fr),
                geometry=dict(joints_px=[], total_px_frames=len(fr), fps=FPS), routed={}, relay=a.stage)
    json.dump(info, open(os.path.join(od, "log.json"), "w"), indent=1)
    print(f"[v61] {a.stage} {len(fr)} frames sample {t_s:.0f}s decode {t_d:.0f}s peak {info['peak_vram_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
