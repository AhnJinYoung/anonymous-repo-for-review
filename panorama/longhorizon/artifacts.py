"""Cheap image-level artifact scores that the semantic metrics (counts, positions) and IQA (CLIP-IQA, NIQE) do not capture.
All are ratios against the root of the same tree, so 1.0 = 'no worse than a single-window sample'.
  seam_ratio   : horizontal-gradient energy in +-6 px bands around window boundaries / elsewhere            (>1.3 = visible seams)
  streak_ratio : (|d/dy| / |d/dx|) of the composite relative to the same ratio on the root                   (>1.5 = horizontal streaks/bands)
  sat_extreme  : fraction of pixels with HSV saturation > 0.85 or any channel clipped, composite minus root  (>0.05 = posterized/oversaturated)
  color_jump   : mean |difference of per-window mean colour| between neighbouring windows (0-255)             (>12 = windows do not share a palette)
Joint scores (v37 run12; seam_ratio stays ~1.0 on joints where the horizon/colour JUMPS but no sharp vertical edge is drawn). Both are
ratios against the same statistic at 20 random non-boundary positions (fixed RNG), so 1.0 = the joints look like anywhere else:
  joint_row_jump  : per-row mean colour of the 32-px strips just left / right of a boundary, mean |L - R| over rows and channels, averaged
                    over boundaries / max(0.5, median of the same at random positions)                     (1 = no seam)
  horizon_jump    : dominant horizontal edge per column (row of max vertical gradient of the Gaussian-smoothed grey image, upper 70 %),
                    median over the 64-px strips either side; mean |dRow| at boundaries (= horizon_jump_px) / max(1 px, median at random)
"""
import numpy as np
from PIL import Image

def _gray(a): return a.mean(-1)
def _grads(g): return np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))

def _rand_positions(W, bounds, excl, margin, n=20, seed=0):
    cand = np.array([x for x in range(margin, W - margin) if all(abs(x - b) > excl for b in bounds)])
    if len(cand) == 0: return []
    return [int(v) for v in np.random.default_rng(seed).choice(cand, size=min(n, len(cand)), replace=False)]

def joint_scores(a, core_px=768, strip=32, hstrip=64, excl=96, n_rand=20, top=0.7, smooth=4.0):
    """joint_row_jump, horizon_jump, horizon_jump_px of an RGB float array a (H, W, 3). Boundaries at k * core_px."""
    from scipy.ndimage import gaussian_filter
    H, W = a.shape[:2]; bounds = [k * core_px for k in range(1, W // core_px) if strip <= k * core_px <= W - strip]
    if not bounds: return dict(joint_row_jump=1.0, horizon_jump=1.0, horizon_jump_px=0.0)
    def rj(x): return float(np.abs(a[:, x - strip:x].mean(1) - a[:, x:x + strip].mean(1)).mean())
    rp = _rand_positions(W, bounds, excl, strip); jb = float(np.mean([rj(x) for x in bounds])); jr = float(np.median([rj(x) for x in rp])) if rp else jb
    g = gaussian_filter(a.mean(-1), smooth * H / 384.0); gy = np.abs(np.diff(g, axis=0)); row = gy[:max(2, int(top * gy.shape[0]))].argmax(0).astype(np.float32)
    def hj(x): return abs(float(np.median(row[max(0, x - hstrip):x])) - float(np.median(row[x:x + hstrip])))
    hb = [x for x in bounds if hstrip <= x <= W - hstrip] or bounds; rph = _rand_positions(W, bounds, excl, hstrip)
    hjb = float(np.mean([hj(x) for x in hb])); hjr = float(np.median([hj(x) for x in rph])) if rph else hjb
    return dict(joint_row_jump=round(jb / max(jr, 0.5), 3), horizon_jump=round(hjb / max(hjr, 1.0), 3), horizon_jump_px=round(hjb, 2))

def artifact_scores(comp, root, core_px=768, band=6):
    a = np.asarray(comp.convert("RGB"), np.float32); r = np.asarray(root.convert("RGB"), np.float32)
    gx, gy = _grads(_gray(a)); rx, ry = _grads(_gray(r))
    n_win = a.shape[1] // core_px; bounds = [k * core_px for k in range(1, n_win)]
    mask = np.zeros(gx.shape[1], bool)
    for b in bounds: mask[max(0, b - band):b + band] = True
    seam = float(gx[:, mask].mean() / max(gx[:, ~mask].mean(), 1e-6)) if mask.any() else 1.0
    streak = float((gy.mean() / max(gx.mean(), 1e-6)) / (ry.mean() / max(rx.mean(), 1e-6)))
    def sat_ext(x):
        mx, mn = x.max(-1), x.min(-1); s = (mx - mn) / np.maximum(mx, 1e-6)
        return float(((s > 0.85) | (mx >= 254) | (mn <= 1)).mean())
    sat = sat_ext(a) - sat_ext(r)
    means = np.stack([a[:, k * core_px:(k + 1) * core_px].reshape(-1, 3).mean(0) for k in range(n_win)])
    jump = float(np.abs(np.diff(means, axis=0)).mean()) if n_win > 1 else 0.0
    return dict(seam_ratio=round(seam, 3), streak_ratio=round(streak, 3), sat_extreme=round(sat, 4), color_jump=round(jump, 2), **joint_scores(a, core_px))
