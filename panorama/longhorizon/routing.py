"""From a (region x atom) score matrix to per-window atom weights. Atoms are words; D columns index atoms 0..K-1 directly."""
import numpy as np
from .config import CFG

def to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
def stated_region(o, n_reg): return min(n_reg - 1, int(o["nominal"] * n_reg))

def stated_windows(o, n_reg):
    """The window SET the prompt gives an object -- the one place both routing and the metrics read the instruction from.

    An object with a `span` (a, b) (video: an event is an INTERVAL of the horizon, not an instant) occupies every window whose CENTRE
    (k + 0.5) / n_reg lies in [a, b); the window containing the span's centre is always included, so the set is never empty even when
    the horizon is coarser than the span. An object with only `nominal` (image scenes: a panorama object is a point) occupies the single
    window containing `nominal`, i.e. `[stated_region(o, n_reg)]`."""
    sp = o.get("span")
    if not sp: return [stated_region(o, n_reg)]
    a, b = float(sp[0]), float(sp[1])
    ks = [k for k in range(n_reg) if a <= (k + 0.5) / n_reg < b]
    return ks or [min(n_reg - 1, max(0, int(0.5 * (a + b) * n_reg)))]

def span_windows(rj, n_reg):
    """The model-derived interval of one entity atom (CFG["entity_rule"] == "span"), from its responsibility column r[:, j]: every window
    whose responsibility reaches max(1 / n_reg, 0.5 * max_k r), made CONTIGUOUS by filling the gaps between the first and the last window
    selected, with the argmax window as the guaranteed minimum. The video analogue of "argmax" (one window), which stays the image default:
    an event holds over a stretch of the horizon, and the probe says over which stretch."""
    rj = np.asarray(rj, np.float64); thr = max(1.0 / n_reg, 0.5 * float(rj.max()))
    ks = [k for k in range(n_reg) if rj[k] >= thr] + [int(np.argmax(rj))]
    return list(range(min(ks), max(ks) + 1))

def responsibilities(D, region_norm, beta=None):
    D = to_np(D).astype(np.float64); n, K = D.shape
    if region_norm == "row_mean": D = D / (np.abs(D).mean(axis=1, keepdims=True) + 1e-12)
    contrast = D.max(0) - np.median(D, axis=0)
    beta = beta if beta is not None else 3.0 / max(np.percentile(contrast, 98), 1e-9)
    z = beta * (D - D.max(0, keepdims=True)); r = np.exp(z); r = r / r.sum(0, keepdims=True)
    peak = D.max(0); floor = np.percentile(peak, 25)
    classes = np.where(peak < floor, "floor", np.where(r.max(0) >= CFG["loc_rmax"], "localized", "broadcast"))
    return r, classes, float(beta)

def classify_ztest(Ds, z_thr=None):
    """Ds: (n_eps, n_reg, K) per-noise ablation profiles. An atom is localized iff its peak region's mean effect exceeds the mean of the other
    regions by z_thr measurement sigmas (sigma from the spread across noise draws). Threshold-light, level-independent, valid for 2 regions."""
    Ds = to_np(Ds).astype(np.float64); n_eps, n_reg, K = Ds.shape; assert n_eps >= 2, "ztest needs probe_eps >= 2"
    D = Ds.mean(0); sig = np.sqrt(Ds.var(0, ddof=1).mean(0) / n_eps) + 1e-12                        # (K,) sigma of the mean profile
    peak = D.argmax(0); rest = np.array([np.delete(D[:, j], peak[j]).mean() for j in range(K)])
    z = (D[peak, np.arange(K)] - rest) / (sig * np.sqrt(1 + 1 / max(n_reg - 1, 1)))
    cls = np.where(z >= (z_thr or CFG["loc_z"]), "localized", "broadcast"); return cls, z

def head_fallback(W, r, spec, n_reg, mode=None):
    """An object head word that stayed broadcast goes to ONE window: the model's argmax ('argmax'), the prompt's stated window ('stated', v36
    behaviour = oracle leak) or nowhere ('none'). Returns the number of heads that needed the fallback."""
    mode = mode or CFG["head_fallback"]; n_fb = 0
    if mode == "none": return 0
    for o in spec["objects"]:
        j = spec["heads"][o["query"]]
        if W[:, j].min() >= 0.999:
            k = stated_region(o, n_reg) if mode == "stated" else (int(np.argmax(r[:, j])) if r is not None else stated_region(o, n_reg))
            W[:, j] = 0.0; W[k, j] = 1.0; n_fb += 1
    return n_fb

def route_exact(D, mode, spec, n_reg, region_norm=None, Ds=None):
    """A condition is a context (scene phrase = part 0, style phrase = last part) plus entity items (object phrases). Only entity atoms
    (spec['entity_atoms']) are ever routed; context atoms (spec['context_atoms']) always get weight 1. entity_rule 'argmax': every entity
    atom is placed exactly once, at its argmax-responsibility window (typed condition; no localized/broadcast test). entity_rule 'span':
    the same, but an entity atom is placed in the whole INTERVAL of windows the probe gives it (`span_windows`) -- the model-derived
    analogue of a video event's stated span; 'argmax' is the special case of one window and stays the default for images. entity_rule 'test': the
    old classification behaviour (loc_rule rmax/ztest + head_fallback), restricted post hoc to entity atoms. mode: 'hard' (localized atoms
    one-hot at argmax r), 'hard_group' (plus co-localization: a non-localized atom whose peak window is a localized window and r >= group_rmin
    moves with it) — both only meaningful under entity_rule 'test'; 'soft' (min(1, n r), reference only). Ds (per-noise profiles) enables
    loc_rule 'ztest'. The probe (r, cls) is always computed over ALL atoms, including context atoms, purely as a diagnostic: `classes_probe`
    in the returned info is the classification the probe alone would give (ztest if available, else rmax), for every atom."""
    r, cls, beta = responsibilities(D, region_norm or CFG["region_norm"], CFG["beta"]); K = r.shape[1]; z = None
    if CFG["loc_rule"] == "ztest":
        if Ds is None: raise ValueError("loc_rule ztest needs the per-noise profiles (delta_exact(..., return_all=True))")
        cls, z = classify_ztest(Ds)
    classes_probe = np.array(cls, dtype=object)                 # diagnostic only: probe's own classification of every atom
    entity_atoms = list(spec.get("entity_atoms", range(K))); context_atoms = list(spec.get("context_atoms", [])); peak = r.argmax(0); n_fb = 0
    if mode == "soft":
        W = np.minimum(1.0, n_reg * r); W[:, cls == "floor"] = 1.0; classes = np.array(cls, dtype=object)
    elif CFG["entity_rule"] in ("argmax", "span"):
        W = np.ones((n_reg, K)); span = CFG["entity_rule"] == "span"
        for j in entity_atoms:
            ks = span_windows(r[:, j], n_reg) if span else [int(peak[j])]
            W[:, j] = 0.0
            for k in ks: W[k, j] = 1.0
        classes = np.array(["context"] * K, dtype=object)
        for j in entity_atoms: classes[j] = "entity"
    else:                                                        # entity_rule "test": old classification-based routing, entity atoms only
        W = np.ones((n_reg, K)); loc = np.where(cls == "localized")[0]
        for j in loc: W[:, j] = 0.0; W[peak[j], j] = 1.0
        if mode == "hard_group":
            loc_regions = set(int(peak[j]) for j in loc)
            for j in range(K):
                if cls[j] != "localized" and int(peak[j]) in loc_regions and r[peak[j], j] >= CFG["group_rmin"]: W[:, j] = 0.0; W[peak[j], j] = 1.0; cls[j] = "grouped"
        n_fb = head_fallback(W, r, spec, n_reg); classes = np.array(cls, dtype=object)
    context_would_localize = int(sum(1 for j in context_atoms if classes_probe[j] == "localized"))
    for j in context_atoms: W[:, j] = 1.0; classes[j] = "context"    # context atoms are never routed, under either entity_rule
    return W, dict(r=r, classes=classes, classes_probe=classes_probe, beta=beta, n_head_fallback=n_fb, z=z, D=to_np(D), context_would_localize=context_would_localize)

def route_attn_diag(A, spec, n_reg, tau_loc=None):
    """Diagnostic only: what attention would have routed. Never used to build a tree."""
    A = to_np(A).astype(np.float64); At = A / (A.mean(axis=0, keepdims=True) + 1e-12); tau = tau_loc or CFG["attn_tau_loc"]
    W = np.ones_like(At); loc = np.where(At.max(0) >= tau)[0]; peak = At.argmax(0)
    for j in loc: W[:, j] = 0.0; W[peak[j], j] = 1.0
    cls = np.where(At.max(0) >= tau, "localized", "broadcast"); return W, dict(A=At, classes=cls)

def route_attn(A, mode, spec, n_reg, tau_cut=None, w_max=None):
    """v38 COMPARISON arm (not the method): routing from image->text attention mass at the root, at the ATOM level (phrase atoms under
    the typed condition). A (n_reg, K): mean attention mass from the image tokens of window k to atom a's tokens (`backend.attn_mass`);
    normalized per atom, At = A / mean_k A (1 = uniform). Only ENTITY atoms are routed; context atoms always keep weight 1.
      mode "hard":  each entity atom one-hot at argmax_k At (the attention analogue of entity_rule "argmax").
      mode "thr" (v59): 0/1, the item goes to every window with At_{k,a} >= CFG["attn_thr"] (default 1 = its uniform share), plus argmax,
                    so an item the root spreads over the whole width reaches every window it is used in (spanning objects, gradients).
      mode "scale": w_{k,a} = clip(At_{k,a}, 0, w_max), set to 0 below tau_cut, the argmax window forced to >= 1. Non-binary weights
                    go through the encoder hook path (textcond.encode_weighted), i.e. the item is scaled, not deleted."""
    A = to_np(A).astype(np.float64); At = A / (A.mean(axis=0, keepdims=True) + 1e-12); K = At.shape[1]
    tau_cut = CFG.get("attn_tau_cut", 0.5) if tau_cut is None else tau_cut; w_max = CFG.get("attn_w_max", 2.0) if w_max is None else w_max
    W = np.ones((n_reg, K)); ents = list(spec.get("entity_atoms", range(K))); peak = At.argmax(0)
    classes = np.array(["context"] * K, dtype=object)
    for j in ents:
        classes[j] = "entity"
        if mode == "hard": W[:, j] = 0.0; W[int(peak[j]), j] = 1.0
        elif mode == "scale":
            w = np.clip(At[:, j], 0.0, w_max); w[w < tau_cut] = 0.0; w[int(peak[j])] = max(float(w[int(peak[j])]), 1.0); W[:, j] = w
        elif mode == "thr":                                      # v59: every window whose attention reaches the item's uniform share
            W[:, j] = (At[:, j] >= CFG.get("attn_thr", 1.0)).astype(np.float64); W[int(peak[j]), j] = 1.0   # (At >= 1) + argmax
        else: raise ValueError(f"route_attn mode {mode!r}")
    return W, dict(A=At, classes=classes, tau_cut=float(tau_cut), w_max=float(w_max))

def frames_to_regions(a, regions):
    """(T, K) per-latent-frame profile -> (n_reg, K): the mean over each region's frames (a region = a range of latent frames)."""
    a = np.asarray(a, np.float64); return np.stack([a[c0:c1].mean(0) for c0, c1 in regions])

STOPWORDS = frozenset("a an the of on in at by to with and or into from for over is are it its his her their this that".split())

def distinct_token_idx(pieces, tok_idx, ents):
    """v53 distinguishing-token rule (domain-agnostic): an event atom's score should come from the part of the atom NOT shared with the
    other event atoms. `pieces` = the tokenizer's token strings of the prompt (sentencepiece, a word starts with U+2581), `tok_idx[j]` =
    atom j's token indices (backend.atom_token_idx), `ents` = the event (entity) atom indices.
      1. words = runs of tokens starting at a U+2581 piece; a CONTENT word is one whose lower-cased text is alphabetic and not in STOPWORDS.
      2. C_j = the token ids (piece strings) of atom j's content words.
      3. distinct_j = atom j's content-word tokens whose piece is in no C_k of another EVENT atom k != j (set difference; context atoms are
         not subtracted -- they are in every window anyway). If distinct_j is empty, fall back to all of atom j's tokens.
    Returns {j: [token indices]} for j in ents."""
    word_of, w = {}, -1
    for i, p in enumerate(pieces):
        if p.startswith("▁") or w < 0 or not p.replace("▁", "").isalnum(): w += 1     # punctuation pieces are their own "word"
        word_of[i] = w
    text = {}
    for i, p in enumerate(pieces): text[word_of[i]] = text.get(word_of[i], "") + p.replace("▁", "")
    content = lambda i: text[word_of[i]].isalpha() and text[word_of[i]].lower() not in STOPWORDS
    C = {j: {pieces[i] for i in tok_idx[j] if i < len(pieces) and content(i)} for j in ents}
    out = {}
    for j in ents:
        others = set().union(*[C[k] for k in ents if k != j]) if len(ents) > 1 else set()
        d = [i for i in tok_idx[j] if i < len(pieces) and content(i) and pieces[i] not in others]
        out[j] = d or list(tok_idx[j])
    return out

def route_attn_time(A, mode, spec, n_reg, tau_cut=None, w_max=None, A_thr=None):
    """v52: TIME routing of a video's typed events from the root's own video->text cross-attention (backend_wan.attn_time_profile), the
    temporal analogue of `route_attn` (images: attn_hard / attn_scale). A (n_reg, K): attention mass of each window's root frames on atom
    a's tokens; normalised per atom, At = A / mean_k A (1 = uniform, the images' normalisation). Context atoms always keep weight 1.
      mode "hard":  0/1. A video event is an INTERVAL, and every window of a story shows one of its events, so the images' "argmax window
                    per entity" is extended to a partition: window k gets the entity atom that DOMINATES it (argmax_j At[k, j] over the
                    entity atoms = the event with the largest relative mass there), and every entity also keeps its own argmax window
                    (never unplaced). With one event per window this is exactly argmax routing.
      mode "scale": the images' soft formula, unchanged: w = clip(At, 0, w_max), 0 below tau_cut, the argmax window forced to >= 1
                    (non-0/1 weights go through the encoder-hook path, textcond.encode_weighted)."""
    A = to_np(A).astype(np.float64); At = A / (A.mean(axis=0, keepdims=True) + 1e-12); K = At.shape[1]
    ents = list(spec.get("entity_atoms", range(K))); classes = np.array(["context"] * K, dtype=object)
    for j in ents: classes[j] = "entity"
    if mode == "scale":
        W, info = route_attn(A, "scale", spec, n_reg, tau_cut, w_max); return W, dict(info, rule="scale")
    if mode != "hard": raise ValueError(f"route_attn_time mode {mode!r}")
    W = np.ones((n_reg, K)); W[:, ents] = 0.0
    if CFG.get("time_hard_rule", "dominant") == "threshold":
        # v56 option: a window receives EVERY event whose normalised score exceeds 1 + margin (not only the dominant one), each event also
        # keeps its argmax window, context always. `A_thr` (n_reg, K) = the score the threshold is read from (default: the routing score A
        # itself; runners.route_time passes the non-contrastive score when CFG["time_thresh_score"] == "plain"). Default rule unchanged.
        tm = float(CFG.get("time_thresh_margin", 0.02)); Ath = At if A_thr is None else to_np(A_thr).astype(np.float64)
        if A_thr is not None: Ath = Ath / (Ath.mean(axis=0, keepdims=True) + 1e-12)
        for k in range(n_reg):
            for j in ents:
                if Ath[k, j] > 1.0 + tm: W[k, j] = 1.0
        for j in ents: W[int(np.argmax(At[:, j])), j] = 1.0
        return W, dict(A=At, A_thr=Ath, classes=classes, rule=f"threshold(At>1+{tm}{',plain' if A_thr is not None else ''})+argmax")
    margin = CFG.get("time_hard_margin")    # v54 'no event' option (default None = unchanged)
    for k in range(n_reg):
        jd = ents[int(np.argmax(At[k, ents]))]
        if margin is None or At[k, jd] > 1.0 + float(margin): W[k, jd] = 1.0   # a window whose dominant event is not above its own mean by `margin` gets context only
    for j in ents: W[int(np.argmax(At[:, j])), j] = 1.0
    return W, dict(A=At, classes=classes, rule="dominant_per_window+argmax" if margin is None else f"dominant_per_window(At>1+{margin})+argmax")

def route_time_support(a, regions, spec, n_reg, margin=0.02, min_run=2):
    """v58 option (CFG time_hard_rule "support"): an event is an INTERVAL of the root, and a window must carry every event its root frames
    show -- in particular the frames at its boundaries, which the parent anchors / hands down as keyframes. a (T, K) = the root's per-
    latent-frame score (runners.attn_time_atoms: the contrastive event score, mean ~1 over frames); per atom normalised by its mean over
    frames. Window k gets event j iff a_j(t) > 1 + margin on a run of >= `min_run` consecutive latent frames t inside region k (the run may
    extend past the region; its frames inside the region count, at least min(min_run, region length) of them). Every event keeps its
    argmax window (the window-mean argmax); context atoms always 1. The window-mean rules ("dominant", "threshold") miss an event that
    occupies only the last frames of a window (v58 park: the dog enters in w1's last 2 latent frames -> w1 is told "pigeon" only while
    its end anchor shows the dog -> a pigeon/dog hybrid)."""
    a = np.asarray(to_np(a), np.float64); a = a / (a.mean(axis=0, keepdims=True) + 1e-12); T, K = a.shape
    ents = list(spec.get("entity_atoms", range(K))); W = np.ones((n_reg, K)); W[:, ents] = 0.0
    runs = {}
    for j in ents:
        on = a[:, j] > 1.0 + margin; keep = np.zeros(T, bool); t = 0
        while t < T:
            if on[t]:
                u = t
                while u < T and on[u]: u += 1
                if u - t >= min_run: keep[t:u] = True
                t = u
            else: t += 1
        runs[j] = keep
        for k, (r0, r1) in enumerate(regions):
            if keep[r0:r1].sum() >= min(min_run, r1 - r0): W[k, j] = 1.0
    A = np.stack([a[r0:r1].mean(0) for r0, r1 in regions])
    for j in ents: W[int(np.argmax(A[:, j])), j] = 1.0
    return W, dict(A=A, rule=f"support(run>={min_run} frames at a>1+{margin})+argmax",
                   support_frames={int(j): [int(t) for t in np.where(runs[j])[0]] for j in ents})

def route_oracle(spec, n_reg):
    """Every word of each object phrase goes to the window SET stated in the prompt (`stated_windows`) -- one window for an image object,
    the windows covering the event's span for a video event. The instruction itself; an upper reference, not a method. Phi_k still uses
    0/1 weights: a multi-window atom is simply 1 in several consecutive windows."""
    W = np.ones((n_reg, spec["K"]))
    for oi, o in enumerate(spec["objects"]):
        ks = stated_windows(o, n_reg)
        for j in spec["phrase_atoms"][oi]:
            W[:, j] = 0.0
            for k in ks: W[k, j] = 1.0
    return W

def weights_to_nodes(levels, w_leaf):
    """route_level "once": the leaf plan `w_leaf` (n_leaves x K) is pushed up the tree, each node taking the MAX over its own leaves. A
    multi-window atom needs no special case here -- it is 1 in several leaves, so every ancestor covering any of them is 1 as well."""
    for lvl in levels:
        for n in lvl: n.w = np.ones(w_leaf.shape[1]) if n.depth == 0 else w_leaf[n.leaf_ids].max(0)

def route_accuracy(W, spec, n_reg, classes):
    """Did each head land (one-hot) in its stated window? And how many non-object words were localized (false positives)?"""
    hit, hs = [], set()
    for o in spec["objects"]:
        j = spec["heads"][o["query"]]; hs.add(j)
        sel = [k for k in range(n_reg) if W[k, j] >= 0.5]                      # the windows the head was routed to (one for an image object)
        hit.append(bool(W[:, j].min() < 0.5 and sel and set(sel) <= set(stated_windows(o, n_reg))))
    obj_atoms = set(j for ph in spec["phrase_atoms"].values() for j in ph)
    fp = int(sum(1 for j in range(spec["K"]) if j not in obj_atoms and classes[j] == "localized"))
    return float(np.mean(hit)) if hit else 0.0, fp

def plan_positions(W_leaf, spec):
    """The plan: where each object head was routed, as a fraction of the horizon. One window -> its centre; several windows (a video event
    routed to an interval) -> the MEAN of their centres; not routed anywhere (an all-ones column) -> argmax, as before; a SOFT (non-0/1)
    column (v38 attn_scale arm) -> its heaviest window."""
    n_reg = W_leaf.shape[0]; out = {}
    for q, j in spec["heads"].items():
        col = W_leaf[:, j]; binary = bool(np.all((np.abs(col) < 1e-6) | (np.abs(col - 1) < 1e-6)))
        if not binary: ks = np.array([int(np.argmax(col))])          # soft weights (attn_scale): the plan is the heaviest window
        else: ks = np.where(col >= 0.5)[0] if col.min() < 0.5 else np.array([int(np.argmax(col))])
        out[q] = float(np.mean((ks + 0.5) / n_reg))
    return out
