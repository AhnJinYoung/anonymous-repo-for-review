import os, io, textwrap
from PIL import Image, ImageDraw, ImageFont
from .config import CFG

def _font(sz=13):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
def caption(img, lines, sz=14):
    font = _font(sz); chars = max(40, int(img.width / (sz * 0.55))); wrapped = [w for ln in lines for w in (textwrap.wrap(ln, chars) or [""])]
    lh = sz + 5; strip = lh * len(wrapped) + 8; out = Image.new("RGB", (img.width, img.height + strip), (18, 18, 18)); dr = ImageDraw.Draw(out)
    for i, ln in enumerate(wrapped): dr.text((6, 4 + i * lh), ln, fill=(235, 235, 235), font=font)
    out.paste(img, (0, strip)); return out
def show(img, max_w=1536, q=72):
    try: from IPython.display import display, Image as IPImage
    except Exception: return
    im = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS) if img.width > max_w else img
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True); display(IPImage(data=buf.getvalue(), format="jpeg"))
def banner(title, sub="", color="#1f6feb"):
    try:
        from IPython.display import display, HTML
        display(HTML(f'<div style="background:{color};color:white;padding:10px 14px;margin:14px 0 6px 0;border-radius:6px;font-family:sans-serif"><div style="font-size:18px;font-weight:700">{title}</div>' + (f'<div style="font-size:13px;opacity:.95;margin-top:3px">{sub}</div>' if sub else "") + '</div>'))
    except Exception: print("\n### " + title + ("\n    " + sub if sub else ""))

def chunked_strip(comp, chunk_px=None, sheet_w=None, gap=4):
    """A wide composite cut into rows of `chunk_px` (each row = one contiguous stretch of the horizon), every row resized to `sheet_w`.
    Legacy 1:16 (12288 px) at 6144 px chunks -> 2 rows at half scale; native depth 3 (10240 x 704) at 5120 px chunks (NATIVE_IMG) -> 2 rows."""
    chunk_px = chunk_px or CFG["sheet_chunk_px"]; sheet_w = sheet_w or CFG["sheet_w"]; n = max(1, -(-comp.width // chunk_px))
    rows = []
    for i in range(n):
        c = comp.crop((i * chunk_px, 0, min((i + 1) * chunk_px, comp.width), comp.height))
        if c.width < chunk_px:
            pad = Image.new("RGB", (chunk_px, comp.height), (40, 40, 40)); pad.paste(c, (0, 0)); c = pad
        rows.append(c.resize((sheet_w, int(comp.height * sheet_w / chunk_px)), Image.LANCZOS))
    out = Image.new("RGB", (sheet_w, sum(r.height for r in rows) + gap * (n - 1)), (40, 40, 40)); y = 0
    for r in rows: out.paste(r, (0, y)); y += r.height + gap
    return out

def sheet(title, root, rows, out_path, chunk_px=None, sheet_w=None):
    """rows: list of (label, composite). Root first at 2x, then one chunked strip per arm."""
    sheet_w = sheet_w or CFG["sheet_w"]
    r = root.image.resize((min(sheet_w, root.image.width * 2), root.image.height * 2), Image.LANCZOS)
    rp = Image.new("RGB", (sheet_w, r.height), (40, 40, 40)); rp.paste(r, (0, 0)); ims = [caption(rp, [f"ROOT (2x) | {title}"])]
    for lab, comp in rows: ims.append(caption(chunked_strip(comp, chunk_px, sheet_w), [lab]))
    s = Image.new("RGB", (sheet_w, sum(i.height + 8 for i in ims)), (60, 60, 60)); y = 0
    for i in ims: s.paste(i, (0, y)); y += i.height + 8
    while s.height > 65000: s = s.resize((s.width // 2, s.height // 2))
    s.save(out_path, quality=85); return s

# ---------------- v38 sheet helpers ----------------
def text_block(width, lines, sz=16, fg=(235, 235, 235), bg=(18, 18, 18), pad=8):
    """Wrapped text lines as an image of the given width (prompt headers, legends)."""
    font = _font(sz); chars = max(40, int((width - 2 * pad) / (sz * 0.55))); wrapped = [w for ln in lines for w in (textwrap.wrap(ln, chars) or [""])]
    lh = sz + 6; out = Image.new("RGB", (width, lh * len(wrapped) + 2 * pad), bg); dr = ImageDraw.Draw(out)
    for i, ln in enumerate(wrapped): dr.text((pad, pad + i * lh), ln, fill=fg, font=font)
    return out
def outline(img, color=(255, 214, 0), px=8):
    """The image with a thick border drawn inside it (the picked root candidate)."""
    out = img.copy(); dr = ImageDraw.Draw(out)
    for i in range(px): dr.rectangle((i, i, out.width - 1 - i, out.height - 1 - i), outline=color)
    return out
DET_COLORS = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 170, 60), (220, 120, 255)]
def draw_dets(img, dets, labels, sz=13):
    """Detector boxes (evaluation only) drawn on a copy of the image, coloured per query, with the query label."""
    out = img.copy().convert("RGB"); dr = ImageDraw.Draw(out); font = _font(sz)
    for d in dets:
        c = DET_COLORS[d["label"] % len(DET_COLORS)]; x0, y0, x1, y1 = [float(v) for v in d["box"]]
        dr.rectangle((x0, y0, x1, y1), outline=c, width=2); dr.text((x0 + 2, max(0, y0 - sz - 2)), f"{labels[d['label']]} {d.get('score', 0):.2f}", fill=c, font=font)
    for x in (out.width / 3, 2 * out.width / 3): dr.line((x, 0, x, out.height), fill=(255, 255, 255), width=1)   # the thirds the check uses
    return out
def vstack(ims, gap=8, bg=(60, 60, 60), width=None):
    width = width or max(i.width for i in ims); s = Image.new("RGB", (width, sum(i.height for i in ims) + gap * (len(ims) - 1)), bg); y = 0
    for i in ims: s.paste(i, (0, y)); y += i.height + gap
    return s
def grid(cells, ncol, gap=6, bg=(60, 60, 60)):
    """cells: list of equally sized images, row-major."""
    w, h = cells[0].width, max(c.height for c in cells); nrow = -(-len(cells) // ncol)
    out = Image.new("RGB", (ncol * w + gap * (ncol - 1), nrow * h + gap * (nrow - 1)), bg)
    for i, c in enumerate(cells): out.paste(c, ((i % ncol) * (w + gap), (i // ncol) * (h + gap)))
    return out
