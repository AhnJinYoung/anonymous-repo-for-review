import re
import numpy as np
from .config import CFG

SCENES = [
    dict(name="beach",
         base="wide panoramic photo of a quiet sandy beach at golden hour, calm sea, clear sky",
         objects=[dict(phrase="a white lighthouse on the far left", query="a lighthouse", head="lighthouse", nominal=0.12),
                  dict(phrase="a small blue wooden fishing boat in the middle", query="a wooden boat", head="boat", nominal=0.50),
                  dict(phrase="a small wooden beach house on the far right", query="a wooden house", head="house", nominal=0.88)]),
    dict(name="valley",
         base="wide panoramic photo of a green alpine valley in late afternoon, meadows, distant mountains",
         objects=[dict(phrase="a red wooden barn on the far left", query="a red barn", head="barn", nominal=0.12),
                  dict(phrase="a single old oak tree in the middle", query="a tree", head="tree", nominal=0.50),
                  dict(phrase="a small stone bridge on the far right", query="a stone bridge", head="bridge", nominal=0.88)]),
    dict(name="desert",
         base="wide panoramic photo of a flat desert at dusk, dry sand, distant mesas, clear sky",
         objects=[dict(phrase="a rusty red pickup truck on the far left", query="a pickup truck", head="truck", nominal=0.12),
                  dict(phrase="a tall saguaro cactus in the middle", query="a cactus", head="cactus", nominal=0.50),
                  dict(phrase="a small white gas station on the far right", query="a gas station", head="station", nominal=0.88)]),
]
STYLE = "photorealistic, 35mm DSLR photo, natural light, sharp focus"
NEG = "painting, illustration, drawing, cartoon, anime, 3d render, oversaturated, blurry, distorted, watermark, text"
STYLE_VIDEO = "cinematic, natural light, sharp"
NEG_VIDEO = ""                                       # CogVideoX's own convention: the unconditional branch is the empty prompt

# ---- video benchmark (an internal script): context = the scene, entities = three events at beginning / middle / end. ----
# The horizon is TIME, so `nominal` is the fraction of the clip at which the event should happen. Every event carries its OWN OWL-ViT query
# (the harness keys detections and metrics by query, so the three events of a scene must be distinguishable); the queries are the frame-level
# checks of PLAN.md. Pronoun subjects of the PLAN wording are spelled out, because a routed phrase is conditioned on alone in its window.
#
# `span` = (a, b), the event's INTERVAL in horizon fractions. In a panorama an entity occupies one window; in a video an entity PERSISTS and
# its state changes, so "stands on the grass at the beginning" holds over the whole first third of the clip, not at one instant. `nominal`
# stays the point the event is centred on (root metrics, plan_err_prompt); `span` is what routing and the placement metrics use
# (`routing.stated_windows`: every window whose centre lies in [a, b)). The three spans are the thirds of the horizon with slight overlaps;
# the boundaries are nudged so that the 8 windows of the depth-3 comparison tree split 0-2 / 3-5 / 6-7 (window centres (k + 0.5) / 8).
# Image scenes have no `span` -- a panorama object is a point, and `stated_windows` falls back to the single `nominal` window.
SPAN_BEGIN, SPAN_MID, SPAN_END = (0.0, 0.34), (0.33, 0.70), (0.69, 1.0)
VIDEO_SCENES = [
    dict(name="balloon", style=STYLE_VIDEO,
         base="a wide green meadow under a clear sky, a single red hot-air balloon, static camera",
         objects=[dict(phrase="the balloon stands on the grass at the beginning", query="a hot-air balloon on the ground", head="balloon", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the balloon lifts off and rises in the middle", query="a hot-air balloon in the air", head="balloon", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the balloon is a tiny dot high in the sky at the end", query="a tiny balloon high in the sky", head="balloon", nominal=0.88, span=SPAN_END)]),
    dict(name="sunrise", style=STYLE_VIDEO,
         base="the ocean horizon seen from a beach, static camera",
         objects=[dict(phrase="a dark blue pre-dawn sky at the beginning", query="a dark pre-dawn sky", head="sky", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the sun rises above the horizon once in the middle", query="the sun", head="sun", nominal=0.50, span=SPAN_MID),
                  dict(phrase="bright daylight at the end", query="a bright daylight sky", head="daylight", nominal=0.88, span=SPAN_END)]),
    dict(name="sailboat", style=STYLE_VIDEO,
         base="a calm mountain lake, a single white sailboat, static camera",
         objects=[dict(phrase="the sailboat enters from the left edge at the beginning", query="a sailboat at the left edge", head="sailboat", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the sailboat crosses the lake in the middle", query="a sailboat on the lake", head="sailboat", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the sailboat exits at the right edge at the end", query="a sailboat at the right edge", head="sailboat", nominal=0.88, span=SPAN_END)]),
    dict(name="apple", style=STYLE_VIDEO,
         base="an empty wooden table in a kitchen, static camera",
         objects=[dict(phrase="nothing on the table at the beginning", query="an empty wooden table", head="table", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="a hand places one red apple on the table in the middle", query="a hand placing an apple", head="hand", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the red apple sits alone on the table at the end", query="a red apple", head="apple", nominal=0.88, span=SPAN_END)]),
    dict(name="candle", style=STYLE_VIDEO,
         base="a dark room with one candle on a table, static camera",
         objects=[dict(phrase="the candle burns steadily at the beginning", query="a burning candle flame", head="candle", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the flame flickers and goes out once in the middle", query="a flickering candle flame", head="flame", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the room stays dark at the end", query="a dark room", head="room", nominal=0.88, span=SPAN_END)]),
    dict(name="snowman", style=STYLE_VIDEO,
         base="a snowy garden at noon, a snowman with a carrot nose, static camera",
         objects=[dict(phrase="the snowman is complete at the beginning", query="a snowman", head="snowman", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the snowman melts and sinks in the middle", query="a melting snowman", head="snowman", nominal=0.50, span=SPAN_MID),
                  dict(phrase="only a small pile of snow is left at the end", query="a pile of snow", head="pile", nominal=0.88, span=SPAN_END)]),
    # Root test (2026-09-23): position AND colour change together over the clip. The harness keys heads by `query`, so the three
    # queries are near-synonyms of "a hot-air balloon" (identical queries would collapse `build_spec`'s heads dict to one entry).
    dict(name="balloon_lr", style=STYLE_VIDEO,
         base="a wide green meadow under a clear sky, a single hot-air balloon flying low above the grass, static camera",
         objects=[dict(phrase="the balloon is red and on the left side of the frame at the beginning", query="a hot-air balloon", head="balloon", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the balloon is orange in the center of the frame in the middle", query="a hot air balloon", head="balloon", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the balloon is yellow and on the right side of the frame at the end", query="a hot-air balloon in flight", head="balloon", nominal=0.88, span=SPAN_END)]),
    # Same content, sequential connectives ("first / then / finally"): does temporal wording help the root? Events 2-3 have the
    # pronoun "it" as subject (as written), so their head word is "it" -- "balloon" does not occur in those phrases.
    dict(name="balloon_lr_seq", style=STYLE_VIDEO,
         base="a wide green meadow under a clear sky, a single hot-air balloon flying low above the grass, static camera",
         objects=[dict(phrase="first, the red balloon floats on the left", query="a hot-air balloon", head="balloon", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="then it drifts to the center and turns orange", query="a hot air balloon", head="it", nominal=0.50, span=SPAN_MID),
                  dict(phrase="finally it reaches the right side and is yellow", query="a hot-air balloon in flight", head="it", nominal=0.88, span=SPAN_END)]),
    # v39: events carry their own time (typed condition), routing = the stated spans (oracle_span); the root only
    # provides the scene and the entities' appearance. The queries only have to be DISTINCT (build_spec keys heads by query): day_night is
    # scored by per-frame luminance, apples_count by OWL-ViT "a red apple" counts (an internal script).
    dict(name="day_night", style=STYLE_VIDEO,
         base="a quiet city street with parked cars and shop fronts, static camera",
         objects=[dict(phrase="the street in bright daylight at the beginning", query="a street in daylight", head="daylight", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the street at sunset with orange light in the middle", query="a street at sunset", head="sunset", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the street at night with street lamps and lit windows at the end", query="a street at night", head="night", nominal=0.88, span=SPAN_END)]),
    dict(name="apples_count", style=STYLE_VIDEO,
         base="a plain wooden kitchen table against a white wall, static camera",
         objects=[dict(phrase="one red apple on the table at the beginning", query="a red apple", head="apple", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="two red apples on the table in the middle", query="two red apples", head="apples", nominal=0.50, span=SPAN_MID),
                  dict(phrase="three red apples on the table at the end", query="three red apples", head="apples", nominal=0.88, span=SPAN_END)]),
]

# v40 (2026-09-23): the long video built on a FOUND root. The Turbo root search (results/root_study/turbo_search) found apples_dense seed 2:
# the table starts empty, a hand brings the apples in, and exactly three apples rest side by side from ~60 % of the clip to the end. The
# root is conditioned on its OWN text (`root_prompt` = the apples_dense phrasing, verbatim, so the seed-2 root reproduces; scenes.root_text,
# backend.text_cond), and the windows on context + the typed events of the story the root actually shows, with spans read off that root
# (empty until ~20 %, apples placed until ~60 %, three apples to the end). Context = the table / wall / static-camera part of that text.
APPLES_DENSE_TEXT = ("A static shot of a plain wooden kitchen table against a white wall. At the start exactly one red apple lies alone on the "
    "table. After one second a hand enters from the right, puts a second red apple next to the first one and leaves the frame: two "
    "red apples on the table. In the middle of the video the hand comes back, puts a third red apple beside them and leaves again. "
    "For the last seconds exactly three red apples rest side by side on the table and nothing else moves. The camera never moves.")
STORY_BEGIN, STORY_MID, STORY_END = (0.0, 0.25), (0.25, 0.6), (0.6, 1.0)
VIDEO_SCENES.append(
    dict(name="apples_story", style="the camera never moves", root_prompt=APPLES_DENSE_TEXT,
         base="A static shot of a plain wooden kitchen table against a white wall",
         objects=[dict(phrase="an empty wooden table, a hand reaches in holding a red apple", query="an empty wooden table", head="table", nominal=0.12, span=STORY_BEGIN),
                  dict(phrase="a hand places red apples on the table one by one", query="a hand placing an apple", head="hand", nominal=0.42, span=STORY_MID),
                  dict(phrase="exactly three red apples rest side by side on the table, no hand", query="three red apples", head="apples", nominal=0.80, span=STORY_END)]))

# v44: apples_story with ONLY the end event re-worded positively, without the word "hand" (a negation T5 cannot use; naming the hand invites
# one). Same root text and, via `root_of`, the SAME cached root (runners.root_cache_key / root_cache_paths key the root on root_of).
VIDEO_SCENES.append(dict(VIDEO_SCENES[-1], name="apples_story_v2", root_of="apples_story",
    objects=VIDEO_SCENES[-1]["objects"][:2] + [dict(VIDEO_SCENES[-1]["objects"][2], phrase="three red apples rest side by side on the bare table, nothing else in the frame")]))

# v49: apples_story with a NEUTRAL root -- the root is conditioned on the scene WITHOUT events (an empty
# static table), so the parent no longer carries the story; the windows keep the apples_story typed events + spans (routed arm), and the
# broadcast arm gives every window `broadcast_prompt` (the full apples_story story text) instead of the root text. Own root cache key
# (root_name = "apples_neutral"); scored with the apples_story rule (video_semantics alias).
APPLES_NEUTRAL_TEXT = "A static shot of a plain wooden kitchen table against a white wall; the table is empty; the camera never moves."
VIDEO_SCENES.append(dict({s["name"]: s for s in VIDEO_SCENES}["apples_story"], name="apples_neutral", root_prompt=APPLES_NEUTRAL_TEXT, broadcast_prompt=APPLES_DENSE_TEXT))

# Stage 2 (2026-09-24, ): two more prompts with checkable global semantics, same recipe as apples_story -- the
# ROOT is conditioned on a dense-caption text (`root_prompt`, found by a 2-seed Turbo root search, an internal script), the windows on
# the context + typed events with stated spans. Scoring: an internal script rules/plans of the same name.
SUNRISE_SEA_TEXT = ("A locked-off time-lapse shot of a calm open sea and a flat horizon seen from a beach; a whole sunrise passes in five "
    "seconds. In the first second it is still night: a dark navy sky and a black sea. After one second the sky just above the horizon "
    "glows deep red and orange while the rest of the sky stays dark. In the middle of the video the round orange sun rises out of the "
    "sea at the centre of the horizon, exactly once, and keeps climbing. For the last seconds it is bright morning: a pale blue sky, the "
    "white-yellow sun high above the horizon and the sea glittering. The image keeps getting brighter from the first frame to the last; "
    "the camera never moves.")
BALL_LR_TEXT = ("A static shot of a light wooden floor in an empty room with a plain white wall behind it. At the start a single red ball "
    "enters from the left edge of the frame and rolls slowly and steadily to the right along the floor. In the middle of the video the "
    "ball passes the centre of the frame. At the end the same ball reaches the right edge of the frame. There is only one ball; it keeps "
    "rolling to the right the whole time and never comes back. The camera never moves.")
CANDLE_OUT_TEXT = ("A static close-up of a single white candle burning on a dark wooden table in a dark room. At the start the candle is "
    "tall and its bright flame lights the table warmly. The candle slowly burns down and gets shorter. In the middle of the video the "
    "flame flickers and shrinks. Two thirds of the way through the flame goes out, once, leaving a thin wisp of smoke, and for the last "
    "seconds the room stays dark. The camera never moves.")
VIDEO_SCENES += [
    dict(name="sunrise_sea", style="the camera never moves", root_prompt=SUNRISE_SEA_TEXT,
         base="A locked-off shot of a calm open sea and a flat horizon seen from a beach",
         # events and spans read off the chosen root (seed 1): dark start with a red glow, brightening, a bright golden sky at the end,
         # the sun a small bright disc on the horizon throughout (it never climbs high)
         objects=[dict(phrase="the dark sea before dawn, a deep red glow on the horizon under a dark sky", query="a dark sea at night", head="sky", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the sun rises at the horizon and the sky turns orange over the sea", query="the sun", head="sea", nominal=0.50, span=SPAN_MID),
                  dict(phrase="a bright golden sunrise sky over the glittering sea, the sun just above the horizon", query="a bright morning sky", head="horizon", nominal=0.88, span=SPAN_END)]),
    dict(name="ball_lr", style="the camera never moves", root_prompt=BALL_LR_TEXT,
         base="A static shot of a light wooden floor in an empty room with a plain white wall behind it",
         objects=[dict(phrase="a single red ball rolls on the left side of the floor", query="a red ball on the left", head="ball", nominal=0.12, span=SPAN_BEGIN),
                  dict(phrase="the red ball rolls through the centre of the floor", query="a red ball", head="ball", nominal=0.50, span=SPAN_MID),
                  dict(phrase="the red ball rolls on the right side of the floor", query="a red ball on the right", head="ball", nominal=0.88, span=SPAN_END)]),
    dict(name="candle_out", style="the camera never moves", root_prompt=CANDLE_OUT_TEXT,
         base="A static close-up of a single white candle on a dark wooden table in a dark room",
         # spans read off the chosen root (seed 1): lit until ~50 %, the flame flickers smaller, it goes out at ~84 %, dark to the end.
         # v48 fix: the spans are aligned to the depth-2 window grid (quarters) so no window gets two events -- the
         # stage-2 mid span (0.5, 0.84) also covered the last window (centre 0.875), which then asked for a flame in a nearly black window
         # (a new candle was lit there); the end event names no flame / candle. results/stage2/candle_out used the old spans.
         objects=[dict(phrase="a white candle with a bright steady flame lights the table", query="a burning candle flame", head="flame", nominal=0.25, span=(0.0, 0.5)),
                  dict(phrase="the small candle flame flickers and a thin wisp of smoke rises", query="a small candle flame", head="smoke", nominal=0.62, span=(0.5, 0.75)),
                  dict(phrase="the room is dark, only a thin wisp of smoke rises above the table", query="a wisp of smoke", head="table", nominal=0.88, span=(0.75, 1.0))]),
]

# v52: apples_story's typed condition, with the ROOT conditioned on the
# TYPED LIST itself (context + the three event phrases, no dense story caption, no time words) -- `root_text` falls back to `full_prompt`.
# Routing reads the events' time off the model's own root (routing `attn_hard_time_phrase` / `attn_scale_time_phrase`); the `span`s are kept
# only for the metrics / the oracle reference. Own root cache key (root_name = "apples_typed"). Scored with the apples_story rule.
VIDEO_SCENES.append(dict({k: v for k, v in {s["name"]: s for s in VIDEO_SCENES}["apples_story"].items() if k not in ("root_prompt", "broadcast_prompt")},
                         name="apples_typed"))

# v53 B: events that are distinct ENTITIES (the regime where attention routing works for images): a static park bench, and three different
# things enter one after another. Typed list = context + 3 event atoms with DISTINCT nouns, no time words; the ROOT is conditioned on the
# typed list (as apples_typed). The `span`s (thirds) are for the metrics / an oracle reference only. Scored by video_semantics "park_typed".
THIRD_1, THIRD_2, THIRD_3 = (0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)
VIDEO_SCENES.append(dict(name="park_typed", style="the camera never moves", base="A static shot of a wooden park bench on a green lawn",
    objects=[dict(phrase="a grey pigeon lands on the bench", query="a pigeon", head="pigeon", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a brown dog walks past in front of the bench", query="a dog", head="dog", nominal=0.5, span=THIRD_2),
             dict(phrase="a child holding a red balloon walks by", query="a red balloon", head="balloon", nominal=5 / 6, span=THIRD_3)]))

def root_name(scene):
    """The name the scene's ROOT is cached under: `root_of` for a variant that differs only in the window texts, else the scene's name."""
    return scene.get("root_of") or scene["name"]

def root_text(scene):
    """The text the ROOT is conditioned on: the scene's own `root_prompt` when it has one (a root found by a stand-alone search, kept
    verbatim so its seed reproduces), else the typed condition `full_prompt(scene)`."""
    return scene.get("root_prompt") or full_prompt(scene)

THIRDS = ("left", "middle", "right")
def third_of(x):
    """The third of the horizon (0 left, 1 middle, 2 right) a fraction x in [0, 1] falls in -- how "far left / in the middle / far right"
    is checked on a root (v38 root_check, evaluation only)."""
    return min(2, max(0, int(float(x) * 3)))

def part_roles(scene):
    """The role of each PART in the order `prompt_parts` emits them: "base", an object index 0..len(objects)-1, or "style".
    `CFG["item_order"]` picks the emission order of the (unordered) condition list, not its membership: "context_first" (default)
    = base, entity phrases..., style; "entity_first" = entity phrases..., base, style. `build_spec` maps an object index oi back
    to its actual part position via `{r: i for i, r in enumerate(part_roles(scene))}[oi]`, so phrase_atoms/heads stay correct
    under either order."""
    n = len(scene["objects"])
    if CFG.get("item_order", "context_first") == "entity_first" and n: return list(range(n)) + ["base", "style"]
    return ["base"] + list(range(n)) + ["style"]

def prompt_parts(scene):
    roles = part_roles(scene); text_of = {"base": scene["base"], "style": scene.get("style", STYLE)}
    for oi, o in enumerate(scene["objects"]): text_of[oi] = o["phrase"]
    return [text_of[r] for r in roles]
def full_prompt(scene): return ", ".join(prompt_parts(scene))

def scene_from_prompt(name, prompt, style=None):
    """A scene whose condition is ALL CONTEXT: no entity items (`objects=[]`), so nothing is ever routed and both methods reduce to
    coherence only. Used for the CDGS release prompts, which name no placed object. `prompt_parts` always appends a style segment, so
    the prompt's last comma segment becomes the style when it has one; a prompt without a comma keeps a trailing ", " (harmless, and
    identical for every method compared)."""
    if style is None:
        head, sep, tail = prompt.rpartition(", ")
        base, style = (head, tail) if sep else (prompt, "")
    else:
        base = prompt
    return dict(name=name, base=base, objects=[], style=style)

def word_split(scene):
    """The prompt as a list of units: words (atoms) and commas (never routed, always weight 1).
    Returns prompt text and list of dict(text, end, part, punct). part i: 0 = base, 1..n = object phrases, n+1 = style."""
    parts = prompt_parts(scene); prompt = ", ".join(parts); bounds, pos = [], 0
    for p in parts: bounds.append((pos, pos + len(p))); pos += len(p) + 2
    units = []
    for m in re.finditer(r"[^\s,]+|,", prompt):
        part = next((i for i, (a, b) in enumerate(bounds) if a <= m.start() < b), None)
        if part is None: part = next(i for i, (a, b) in enumerate(bounds) if m.start() < a) - 1   # the separator comma belongs to the part before it
        units.append(dict(text=m.group(0), end=m.end(), part=part, punct=(m.group(0) == ",")))
    return prompt, units

def prompt_of(spec, w):
    """The prompt text of the KEPT atoms — the "delete" instantiation of the projection (I - pi_a) (METHOD.md section 3):
    the condition is a list of items, so removing an item is a shorter list, not a masked embedding. `w` is a 0/1 weight per atom.

    Rebuilt from the spec's units (words and commas), so both atom levels go through the same rule: a word is emitted iff its atom
    is kept; a comma is emitted iff its own atom is kept AND at least one word survives in the comma-delimited part it closes, which
    is what drops a phrase atom together with its comma at phrase level and avoids a dangling ", , " when every word of a part is
    dropped at word level. Words are re-joined with single spaces, so all-ones weights return `spec["prompt"]` exactly (and the
    whitespace after the last unit — the trailing ", " of an all-context prompt with an empty style — is preserved)."""
    units = spec["units"]
    if not units: return spec["prompt"]
    w_unit = {}
    for j, ua in enumerate(spec["unit_of_atom"]):
        for i in ua: w_unit[i] = float(w[j])
    keep = [w_unit.get(i, 1.0) >= 0.5 for i in range(len(units))]
    live = False                                                     # does the part closed by the next comma still have a word?
    for i, u in enumerate(units):
        if u["punct"]: keep[i] = keep[i] and live; live = False
        elif keep[i]: live = True
    out = ""
    for i, u in enumerate(units):
        if not keep[i]: continue
        out += "," if u["punct"] else (("" if not out else " ") + u["text"])
    return out + (spec["prompt"][units[-1]["end"]:] if keep[-1] else "")

def build_spec(scene, units, atom_level="word"):
    """atom_level 'word': atoms = non-punctuation units, head of each object = last occurrence of its head word inside its phrase.
    atom_level 'phrase': atoms = comma-delimited segments of the prompt (base, one per object phrase, style); the object's atom is its segment.
    `unit_of_atom[j]` is a list of unit indices covered by atom j (one for words, several for phrases). Object index oi's part
    position is looked up via `part_roles(scene)` (not assumed to be oi + 1), so this is correct under either CFG["item_order"]."""
    role_to_part = {r: i for i, r in enumerate(part_roles(scene))}
    if atom_level == "phrase":
        n_part = max(u["part"] for u in units) + 1; unit_of_atom = [[i for i, u in enumerate(units) if u["part"] == a] for a in range(n_part)]
        words = [" ".join(units[i]["text"] for i in ua if not units[i]["punct"]) for ua in unit_of_atom]; part_of = list(range(n_part))
        phrase_atoms = {oi: [role_to_part[oi]] for oi in range(len(scene["objects"]))}
        heads = {o["query"]: role_to_part[oi] for oi, o in enumerate(scene["objects"])}
        entity_atoms = sorted(j for ph in phrase_atoms.values() for j in ph); context_atoms = [a for a in range(n_part) if a not in entity_atoms]
        return dict(scene=scene, objects=scene["objects"], words=words, part_of=part_of, K=n_part, unit_of_atom=unit_of_atom, phrase_atoms=phrase_atoms, heads=heads,
                    entity_atoms=entity_atoms, context_atoms=context_atoms, atom_level="phrase", part_roles=part_roles(scene))
    atoms = [i for i, u in enumerate(units) if not u["punct"]]; unit_of_atom = [[i] for i in atoms]; atom_of_unit = {u: j for j, u in enumerate(atoms)}
    phrase_atoms, heads = {}, {}
    for oi, o in enumerate(scene["objects"]):
        ph = [atom_of_unit[i] for i, u in enumerate(units) if u["part"] == role_to_part[oi] and not u["punct"]]; phrase_atoms[oi] = ph
        cand = [j for j in ph if units[atoms[j]]["text"].lower().strip(".") == o["head"]]; heads[o["query"]] = cand[-1] if cand else ph[-1]
    words = [units[i]["text"] for i in atoms]; part_of = [units[i]["part"] for i in atoms]
    entity_atoms = sorted(j for ph in phrase_atoms.values() for j in ph); context_atoms = [j for j in range(len(atoms)) if j not in entity_atoms]
    return dict(scene=scene, objects=scene["objects"], words=words, part_of=part_of, K=len(atoms), unit_of_atom=unit_of_atom, phrase_atoms=phrase_atoms, heads=heads,
                entity_atoms=entity_atoms, context_atoms=context_atoms, atom_level="word")

def respec(spec, w, gain):
    """The (spec, weights) pair for CFG["entity_gain"] (textcond.encode_weighted): DELETE the atoms with w < 0.5 (a shorter item
    list, same convention as `prompt_of`/"delete") then AMPLIFY the surviving entity atom(s) by `gain` under the hook path on
    that shorter prompt. Phrase-level only: one atom = one item's own source text (`spec["part_roles"]` says which -- "base",
    an object index, or "style"), so the rebuilt prompt is `", ".join` of the KEPT atoms' own text, taken directly from the
    scene (NOT by splitting the rebuilt prompt string on ", ", which would over-split whenever an atom's own text -- e.g. a
    scene's `base`, "...golden hour, calm sea, clear sky" -- contains an internal comma). The caller still owes the rebuilt
    spec its `spans` (backend/tokenizer specific token spans of `sub["prompt"]`) before calling `_encode`."""
    if spec.get("atom_level") != "phrase": raise ValueError("entity_gain / respec is phrase-level only (see CFG['entity_gain'])")
    w = np.asarray(w, np.float64); scene = spec["scene"]; roles = spec.get("part_roles") or part_roles(scene)
    text_of = {"base": scene["base"], "style": scene.get("style", STYLE)}
    for oi, o in enumerate(scene["objects"]): text_of[oi] = o["phrase"]
    kept = [j for j in range(spec["K"]) if w[j] >= 0.5]; ea = set(spec["entity_atoms"])
    parts = [text_of[roles[j]] for j in kept]
    units, pos = [], 0
    for p in parts:
        end = pos + len(p); units.append(dict(text=p, end=end, part=len(units), punct=False)); pos = end + 2   # ", " between kept parts
    prompt = ", ".join(parts)
    w_new = np.array([gain if kept[i] in ea else 1.0 for i in range(len(parts))], dtype=np.float64)
    sub = dict(prompt=prompt, units=units, unit_of_atom=[[i] for i in range(len(parts))], K=len(parts),
               atom_level="phrase", t5=spec.get("t5", False))
    return sub, w_new

# ---- appendix gallery (2026-09-24, an internal script): new panorama scenes in the SCENES format, kept OUT of SCENES so the
# benchmark indices / loops over SCENES are unchanged. Run with the v38 run2_native setting (SD3-medium native, attn_hard_phrase). ----
APPENDIX_SCENES = [
    dict(name="harbour",
         base="wide panoramic photo of a small fishing harbour on a calm summer morning, stone quay, moored boats, gentle sea",
         objects=[dict(phrase="a tall white lighthouse on the far left", query="a lighthouse", head="lighthouse", nominal=0.12),
                  dict(phrase="a red sailboat with a white sail on the far right", query="a sailboat", head="sailboat", nominal=0.88)]),
    dict(name="snow_range",
         base="wide panoramic photo of a snowy mountain range under a clear blue winter sky, deep snow, pine trees",
         objects=[dict(phrase="a red snowmobile on the far left", query="a snowmobile", head="snowmobile", nominal=0.12),
                  dict(phrase="a small wooden cabin with a smoking chimney in the middle", query="a wooden cabin", head="cabin", nominal=0.50)]),
    dict(name="night_street",
         base="wide panoramic photo of a city street at night after rain, wet asphalt reflecting street lights, old buildings",
         objects=[dict(phrase="a yellow tram on the far left", query="a tram", head="tram", nominal=0.12),
                  dict(phrase="a glowing pink neon sign on the far right", query="a neon sign", head="sign", nominal=0.88)]),
    dict(name="autumn_forest",
         base="wide panoramic photo of an autumn forest with orange and red leaves, a small stream, soft afternoon light",
         objects=[dict(phrase="a deer standing in the middle", query="a deer", head="deer", nominal=0.50),
                  dict(phrase="an old stone bridge over the stream on the far right", query="a stone bridge", head="bridge", nominal=0.88)]),
    dict(name="palm_beach",
         base="wide panoramic photo of a tropical white sand beach at noon, turquoise sea, blue sky with a few clouds",
         objects=[dict(phrase="a tall palm tree on the far left", query="a palm tree", head="palm", nominal=0.12),
                  dict(phrase="a wooden lifeguard tower on the far right", query="a lifeguard tower", head="tower", nominal=0.88)]),
    dict(name="library",
         base="wide panoramic photo of the interior of a grand old library, tall wooden bookshelves, warm lamp light, wooden floor",
         objects=[dict(phrase="a large globe on a wooden stand on the far left", query="a globe", head="globe", nominal=0.12),
                  dict(phrase="a green leather armchair in the middle", query="an armchair", head="armchair", nominal=0.50),
                  dict(phrase="a tall grandfather clock on the far right", query="a grandfather clock", head="clock", nominal=0.88)]),
    dict(name="lavender_farm",
         base="wide panoramic photo of purple lavender fields in rows at sunset, rolling hills, warm sky",
         objects=[dict(phrase="an old stone farmhouse on the far left", query="a farmhouse", head="farmhouse", nominal=0.12),
                  dict(phrase="a red vintage tractor on the far right", query="a tractor", head="tractor", nominal=0.88)]),
    dict(name="lake_dock",
         base="wide panoramic photo of a calm mountain lake at dawn, mist over the water, pine forest shore",
         objects=[dict(phrase="a wooden dock with a rowboat on the far left", query="a rowboat", head="rowboat", nominal=0.12),
                  dict(phrase="a red canoe on the water in the middle", query="a canoe", head="canoe", nominal=0.50),
                  dict(phrase="a small log cabin on the shore on the far right", query="a log cabin", head="cabin", nominal=0.88)]),
]

# v53 count story: the count rises 1 -> 2 -> 3 exactly ONCE over the whole video; every counted object
# carries a DISTINCT attribute (its colour) so per-atom scores can separate the events (v52: shared nouns co-vary). Typed list = context +
# 3 event atoms, no time words; the ROOT is conditioned on the typed list. `span`s (thirds) for the metrics / an oracle reference only.
# Scored by video_semantics "balloons_typed" / "boats_typed" (count curve + number of 1 -> 2 -> 3 cycles).
VIDEO_SCENES.append(dict(name="balloons_typed", style="the camera never moves", base="A static shot of a clear blue sky above a green meadow",
    objects=[dict(phrase="a red balloon rises into the sky", query="a red balloon", head="balloon", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a yellow balloon rises next to it", query="a yellow balloon", head="balloon", nominal=0.5, span=THIRD_2),
             dict(phrase="a blue balloon rises next to them", query="a blue balloon", head="balloon", nominal=5 / 6, span=THIRD_3)]))
VIDEO_SCENES.append(dict(name="boats_typed", style="the camera never moves", base="A static shot of a calm pond with green reeds along the bank",
    objects=[dict(phrase="a red paper boat drifts into the frame", query="a red paper boat", head="boat", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a yellow paper boat drifts in next to it", query="a yellow paper boat", head="boat", nominal=0.5, span=THIRD_2),
             dict(phrase="a blue paper boat drifts in next to them", query="a blue paper boat", head="boat", nominal=5 / 6, span=THIRD_3)]))

# v54: a second distinct-entity story in the park_typed style (three different vehicles/people pass a static street corner, distinct nouns,
# no time words; the ROOT is conditioned on the typed list). `span`s (thirds) for the metrics / the oracle reference only.
# Scored by video_semantics "street_typed".
VIDEO_SCENES.append(dict(name="street_typed", style="the camera never moves", base="A static shot of a quiet street corner with a crosswalk",
    objects=[dict(phrase="a red car drives past", query="a red car", head="car", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a cyclist rides by on a bicycle", query="a bicycle", head="cyclist", nominal=0.5, span=THIRD_2),
             dict(phrase="a yellow bus pulls up at the curb", query="a yellow bus", head="bus", nominal=5 / 6, span=THIRD_3)]))

# v55: three more distinct-entity stories (park_typed style: static camera, simple background, three DIFFERENT nouns that pass through the
# frame and leave, no time words; the ROOT is conditioned on the typed list). `span`s for the metrics / the oracle reference only.
VIDEO_SCENES.append(dict(name="counter_typed", style="the camera never moves", base="A static shot of an empty wooden kitchen counter against a plain white wall",
    objects=[dict(phrase="a grey cat walks across the counter", query="a cat", head="cat", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a red ball rolls across the counter", query="a ball", head="ball", nominal=0.5, span=THIRD_2),
             dict(phrase="a white rabbit hops across the counter", query="a rabbit", head="rabbit", nominal=5 / 6, span=THIRD_3)]))
VIDEO_SCENES.append(dict(name="beach_typed", style="the camera never moves", base="A static shot of an empty sandy beach with a calm blue sea",
    objects=[dict(phrase="a white seagull flies past", query="a bird", head="seagull", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a black dog runs along the shore", query="a dog", head="dog", nominal=0.5, span=THIRD_2),
             dict(phrase="a person in a red swimsuit jogs by", query="a person", head="person", nominal=5 / 6, span=THIRD_3)]))
VIDEO_SCENES.append(dict(name="snow_typed", style="the camera never moves", base="A static shot of a snowy field with a single pine tree",
    objects=[dict(phrase="a red fox trots across the snow", query="a fox", head="fox", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a brown deer walks past", query="a deer", head="deer", nominal=0.5, span=THIRD_2),
             dict(phrase="a skier in a blue jacket glides by", query="a person", head="skier", nominal=5 / 6, span=THIRD_3)]))
# v55 replacements (counter/snow roots kept their entities in frame): events that are transient by nature (things that pass through).
VIDEO_SCENES.append(dict(name="sky_typed", style="the camera never moves", base="A static shot of a clear blue sky above a green hill",
    objects=[dict(phrase="a white airplane flies across the sky", query="an airplane", head="airplane", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a flock of black birds flies past", query="a bird", head="birds", nominal=0.5, span=THIRD_2),
             dict(phrase="a red helicopter flies by", query="a helicopter", head="helicopter", nominal=5 / 6, span=THIRD_3)]))
VIDEO_SCENES.append(dict(name="road_typed", style="the camera never moves", base="A static shot of an empty country road through green fields",
    objects=[dict(phrase="a black motorcycle speeds past", query="a motorcycle", head="motorcycle", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a white horse gallops by", query="a horse", head="horse", nominal=0.5, span=THIRD_2),
             dict(phrase="a red tractor drives past", query="a tractor", head="tractor", nominal=5 / 6, span=THIRD_3)]))
# v56: short, non-overlapping events with RIGID, distinctive objects instead of animals (no identity to morph), a
# static simple background, no time words; the ROOT is conditioned on the typed list. `span`s for the metrics / the oracle reference only.
VIDEO_SCENES.append(dict(name="floor_typed", style="the camera never moves", base="A static shot of an empty light wooden floor in front of a plain white wall",
    objects=[dict(phrase="a white paper plane glides past", query="a paper airplane", head="plane", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a red ball rolls across", query="a red ball", head="ball", nominal=0.5, span=THIRD_2),
             dict(phrase="a yellow toy car drives by", query="a toy car", head="car", nominal=5 / 6, span=THIRD_3)]))
VIDEO_SCENES.append(dict(name="lawn_typed", style="the camera never moves", base="A static shot of a short green lawn in front of a white wooden fence",
    objects=[dict(phrase="a red balloon floats up and away", query="a balloon", head="balloon", nominal=1 / 6, span=THIRD_1),
             dict(phrase="a yellow beach ball bounces past", query="a ball", head="ball", nominal=0.5, span=THIRD_2),
             dict(phrase="a blue umbrella tumbles by", query="an umbrella", head="umbrella", nominal=5 / 6, span=THIRD_3)]))
