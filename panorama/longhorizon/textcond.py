"""How one item is removed from the condition — shared by every backend (SD3, SD2, CogVideoX) and mirrored by the dummy.

METHOD.md section 3 asks for a projection (I - pi_a) applied at the encoder INPUT. There are two ways to instantiate it, and
`CFG["atom_mode"]` picks between them:

  "delete" (default)  rebuild the prompt WITHOUT the item (`scenes.prompt_of`) and encode that shorter list — no hooks at all,
                      so it works for any encoder and any tokenizer. The condition is a list of items; removing an item is a
                      shorter list. Needs 0/1 weights (a shorter list has no notion of "half an item").
  "zero"              multiply the item's token embeddings by w in place, positions kept (the v37 behaviour, a forward hook on
                      the token-embedding layer). Kept for word-level and SOFT weights, and used automatically whenever `w` is
                      not 0/1 — but it is a poor projection for a CLIP encoder: on SD2 a window conditioned on the zeroed
                      routed condition does not draw the entity at all, while the deleted-item prompt draws it large
                      (results/cdgs_compare/image/diag_sd2, 2026-09-22).

The cache key carries the mode, so the two instantiations never share an encoding.
"""
import numpy as np
from .config import CFG
from .scenes import prompt_of, respec


def mode(): return CFG.get("atom_mode", "delete")

def is_binary(w, tol=1e-6):
    """True iff every weight is 0 or 1 (within `tol`) — the only case "delete" can express."""
    w = np.asarray(w, np.float64); return bool(np.all((np.abs(w) <= tol) | (np.abs(w - 1.0) <= tol)))

def delete_mode(w): return mode() == "delete" and is_binary(w)

def wkey(spec, w):
    """Cache key for an encoding: prompt, T5 flag, atom level, ATOM MODE, rounded weights."""
    return (spec["prompt"], bool(spec.get("t5", False)), spec.get("atom_level", "word"), mode(),
            None if w is None else tuple(np.round(np.asarray(w, np.float64), 3)))

def encode_weighted(bk, spec, w, cache_max=600):
    """Cached encoding of the condition restricted to the atoms with weight ~1. All-ones -> the stored full encoding.

    CFG["entity_gain"] != 1 (and 0/1 weights, phrase-level atoms) takes priority over `atom_mode`: DELETE the dropped items
    (`scenes.respec`, same rebuilt-prompt convention as "delete") then AMPLIFY the surviving entity atom(s) by the gain under
    the hook path on that shorter prompt -- an item-WEIGHT instantiation on top of the item-LIST instantiation, not an
    alternative to it. Otherwise: under "delete" (and 0/1 weights) this encodes the rebuilt prompt with NO hooks; under "zero"
    (or non-binary weights) the backend's hook path on the FULL prompt, as before."""
    w = np.asarray(w, np.float64)
    if w.min() >= 0.999 and w.max() <= 1.001: return spec["E"], spec["P"]
    gain = CFG.get("entity_gain", 1.0)
    if gain != 1.0 and is_binary(w) and spec.get("atom_level") == "phrase":
        key = wkey(spec, w) + ("gain", round(float(gain), 3))
        if key not in bk._enc:
            if len(bk._enc) > cache_max: bk._enc.pop(next(iter(bk._enc)))
            sub, w_new = respec(spec, w, gain)
            sub["spans"] = {name: bk._spans(name, sub["prompt"], sub["units"]) for name in bk.toks}
            bk._enc[key] = bk._encode(sub, w_new)
        return bk._enc[key]
    key = wkey(spec, w)
    if key not in bk._enc:
        if len(bk._enc) > cache_max: bk._enc.pop(next(iter(bk._enc)))
        bk._enc[key] = bk._encode(dict(spec, prompt=prompt_of(spec, w)), None) if delete_mode(w) else bk._encode(spec, w)
    return bk._enc[key]
