from dataclasses import dataclass, field
from . import config, state
from .config import CFG

@dataclass
class Node:
    depth: int; idx: int
    core_g0: int; core_g1: int
    scale: int
    halo_l: bool; halo_r: bool
    parent: object = None
    children: list = field(default_factory=list)
    w: object = None; cond: object = None            # w: weight per atom (K,), 1 = word present in this window
    latent: object = None; image: object = None
    frames: object = None                            # video: the decoded PIL frames of this node (root only)
    frame_dets: object = None                        # video: detections per decoded frame
    dets: list = field(default_factory=list)
    leaf_ids: list = field(default_factory=list)
    rho: object = None; field_lp: object = None
    anchor: object = None                            # anchor relays: dict(pairs=[(child frame i, parent frame j)], mech='tok'|'repaint')
    bg_cond: object = None; bg_latent: object = None            # object-free twin (field_bg relay)
    search: object = None
    ext_l: int = 0; ext_r: int = 0                   # edge_native (video): extra canvas frames beyond core+halo on the left / right (node scale)
    ovl_l: object = None; ovl_r: object = None       # frames shared with the left / right sibling when they differ from 2*halo (edge_native 1)
    slot0_regular: object = None                     # first_slot "image": the regular (4-frame) latent at slot 0's time, used in the composite (sampling.level_anchors)
    @property
    def canvas_g0(self): return self.core_g0 - (CFG["halo_px"] * self.scale if self.halo_l else 0) - self.ext_l * self.scale
    @property
    def canvas_g1(self): return self.core_g1 + (CFG["halo_px"] * self.scale if self.halo_r else 0) + self.ext_r * self.scale
    @property
    def canvas_w(self): return (self.canvas_g1 - self.canvas_g0) // self.scale
    @property
    def core_w(self): return (self.core_g1 - self.core_g0) // self.scale
    @property
    def core_n0(self): return (CFG["halo_px"] if self.halo_l else 0) + self.ext_l
    def g2n(self, g): return (g - self.canvas_g0) / self.scale
    def n2g(self, n): return self.canvas_g0 + n * self.scale
    @property
    def name(self): return f"L{self.depth}n{self.idx:02d}"

def canvas_multiple():
    """Latent-geometry constraint of the active backend: every canvas width must be a multiple of this (SD3/SD2 16 px, video 1 latent frame)."""
    return int(getattr(state.B, "canvas_multiple", 16) if state.B is not None else 16)

def build_tree(depth, core_px=None, branch=None):
    core_px = core_px or CFG["core_px"]; branch = branch or CFG["branch"]
    W = core_px * branch ** depth; rh = bool(CFG.get("root_halo", 0)); cm = canvas_multiple()
    root = Node(0, 0, 0, W, branch ** depth, rh, rh); levels = [[root]]
    for d in range(1, depth + 1):
        s, lvl = branch ** (depth - d), []
        for p in levels[-1]:
            step = (p.core_g1 - p.core_g0) // branch
            for k in range(branch):
                g0, g1 = p.core_g0 + k * step, p.core_g0 + (k + 1) * step
                n = Node(d, len(lvl), g0, g1, s, halo_l=(g0 > 0), halo_r=(g1 < W), parent=p); p.children.append(n); lvl.append(n)
        levels.append(lvl)
    if CFG.get("edge_native") and depth > 0: apply_edge_native(levels, CFG["edge_native"])
    if CFG.get("compose", "blend") == "handoff" and depth > 0: apply_handoff(levels, int(CFG.get("handoff_k", 2)))
    leaf_w = W // branch ** depth
    for lvl in levels:
        for n in lvl:
            n.leaf_ids = list(range(n.core_g0 // leaf_w, n.core_g1 // leaf_w)); assert n.core_w == core_px and n.canvas_w % cm == 0
    return levels

def apply_handoff(levels, K):
    """compose='handoff' (video, halo_px 0): every window but the first of each level gets ext_l = K extra canvas frames on the left, at the
    same global time as the last K frames of its left neighbour (sampling.run_level_seq anchors them to that neighbour's final latent; the
    composition keeps only the cores). Canvas = K + core frames."""
    assert int(CFG["halo_px"]) == 0 and K >= 1, "compose='handoff' needs halo_px 0 and handoff_k >= 1"
    for lvl in levels[1:]:
        for n in lvl[1:]: n.ext_l = K

def apply_edge_native(levels, mode):
    """Video only: make the horizon-end windows (no halo on the outer side, canvas core + halo < the root's native canvas) native length.
    mode 1 / "in"  -- the two end LEAVES are extended INWARD (towards their sibling) by e = root.canvas_w - canvas_w frames: their overlap with
                      the sibling grows from 2*halo to 2*halo + e (ovl_l / ovl_r, used by the x0 blend, the composition ramp and anchor_ends).
    mode "out"     -- the two end windows of EVERY level are extended OUTWARD past the horizon ends, into the root's canvas (the root's halo),
                      by e frames; those frames are sampled (and anchored on the parent, so the horizon's first / last frames become interior
                      frames of the window) and cropped at composition. Asserted to stay inside the parent's canvas."""
    root = levels[0][0]; native = root.canvas_w
    lvls = levels[-1:] if mode in (1, "1", "in", True) else levels[1:]
    for d, lvl in enumerate(lvls):
        first, last = lvl[0], lvl[-1]
        if mode in (1, "1", "in", True):
            if len(lvl) < 2: continue
            for n, sib, side in ((first, lvl[1], "r"), (last, lvl[-2], "l")):
                e = native - n.canvas_w
                if e <= 0 or (side == "r" and n.halo_l) or (side == "l" and n.halo_r): continue
                ov = (n.ovl_r if side == "r" else n.ovl_l) or 2 * CFG["halo_px"]
                if side == "r": n.ext_r = e; n.ovl_r = ov + e; sib.ovl_l = ov + e
                else: n.ext_l = e; n.ovl_l = ov + e; sib.ovl_r = ov + e
        elif mode == "out":
            for n in {id(first): first, id(last): last}.values():
                e = native - n.canvas_w
                if e <= 0: continue
                if not n.halo_l: n.ext_l = e
                elif not n.halo_r: n.ext_r = e
                p = n.parent
                assert p.canvas_g0 <= n.canvas_g0 and n.canvas_g1 <= p.canvas_g1, (n.name, n.canvas_g0, n.canvas_g1, p.canvas_g0, p.canvas_g1)
        else: raise ValueError(f"edge_native must be 0, 1/'in' or 'out', got {mode!r}")

def ovl_of(n, side):
    """Frames (latent columns) node n shares with its left ('l') / right ('r') sibling: 2*halo, or ovl_l / ovl_r after edge_native 1."""
    v = n.ovl_l if side == "l" else n.ovl_r
    return int(v) if v is not None else config.overlap_lat()

def leaf_regions(lat_w, n_reg):
    """n_reg equal spans of the latent width with float boundaries (the whole width is covered for any n_reg). When n_reg > lat_w,
    neighbouring regions share a column: the probe's resolution is one latent column, and an atom whose peak sits in a shared column
    goes to the first (leftmost) of those regions, a bias of at most one column (1/lat_w of the horizon)."""
    b = [int(round(k * lat_w / n_reg)) for k in range(n_reg + 1)]
    return [(min(b[k], lat_w - 1), max(min(b[k], lat_w - 1) + 1, b[k + 1])) for k in range(n_reg)]
def core_regions(node, n_reg):
    """`n_reg` equal spans of the node's CORE inside its own latent (the probe must not spend regions on a halo that is not part of the
    horizon). With no halo (every image root) this is exactly `leaf_regions(node.latent.shape[-1], n_reg)`."""
    VS = config.vae_stride(); off = node.core_n0 // VS; w = node.core_w // VS
    return [(off + a, off + b) for a, b in leaf_regions(w, n_reg)]
def child_regions(node):
    VS = config.vae_stride(); return [(int(round(node.g2n(c.core_g0))) // VS, int(round(node.g2n(c.core_g1))) // VS) for c in node.children]
def parent_span_cols(child):
    p = child.parent; VS = config.vae_stride(); return int(round(p.g2n(child.canvas_g0) / VS)), int(round(p.g2n(child.canvas_g1) / VS))
