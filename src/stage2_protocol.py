"""Frozen paper-description implementation of the DeRecG NoRect raw branch."""

import hashlib
import json
import math
import re

from src.stage1b_fallback import RELATION_PATTERNS, extract_head

RELATIONS = frozenset({"left_of", "right_of", "above", "below", "between", "inside", "on", "next_to", "near"})
SYSTEM_PROMPT = ("You are a precise language parser for visual referring expressions. Given a referring expression that describes ONE target object, decompose it into a constraint graph in strict JSON. The ‘head’ is the target object to localize. ‘attributes’ are visual properties of that object (color, size, clothing, state). ‘relations’ are spatial relations linking the head to other referenced objects; each relation has a type in {left_of, right_of, above, below, between, inside, on, next_to, near} and a ‘target’ object name. Put all OTHER referenced objects (relation targets) in ‘others’, each with their own attributes/relations (usually empty). Output ONLY JSON.")
DEMOS = (
    ('the person wearing red on the left of the car', '{"head": {"name": "person", "attributes": ["red"], "relations": [{"type": "left_of", "target": "car"}]}, "others": [{"name": "car", "attributes": [], "relations": []}]}'),
    ('a dog sitting between the sofa and the table', '{"head": {"name": "dog", "attributes": ["sitting"], "relations": [{"type": "between", "target": "sofa and table"}]}, "others": [{"name": "sofa", "attributes": [], "relations": []}, {"name": "table", "attributes": [], "relations": []}]}'),
    ('the left cup', '{"head": {"name": "cup", "attributes": [], "relations": []}, "others": []}'),
)
DECOMPOSITION_TEMPLATE = SYSTEM_PROMPT + "\n\n" + "\n\n".join(f"Expression: {e}\nJSON: {j}" for e, j in DEMOS) + "\n\nExpression: <expression>\nJSON:"
ATTRIBUTE_TEMPLATE = "Is there a <head> that is <attribute> in this image? Look at the image and answer with a single word: YES or NO. Answer:"


def decomposition_prompt(expression):
    return DECOMPOSITION_TEMPLATE.replace("<expression>", expression)


def attribute_prompt(head, attribute):
    return ATTRIBUTE_TEMPLATE.replace("<head>", head).replace("<attribute>", attribute)


def prompt_hashes():
    return {"decomposition_sha256": hashlib.sha256(DECOMPOSITION_TEMPLATE.encode()).hexdigest(),
            "attribute_sha256": hashlib.sha256(ATTRIBUTE_TEMPLATE.encode()).hexdigest()}


def _normalize_graph(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get("head"), dict):
        raise ValueError("missing head")
    head = graph["head"]
    name = head.get("name")
    attrs = head.get("attributes", [])
    rels = head.get("relations", [])
    others = graph.get("others", [])
    if not isinstance(name, str) or not name.strip() or not isinstance(attrs, list) or not isinstance(rels, list) or not isinstance(others, list):
        raise ValueError("invalid graph schema")
    if not all(isinstance(x, str) for x in attrs):
        raise ValueError("invalid attributes")
    clean_rels, unsupported = [], []
    for rel in rels:
        if not isinstance(rel, dict) or not isinstance(rel.get("type"), str):
            raise ValueError("invalid relation")
        if rel["type"] in RELATIONS and isinstance(rel.get("target"), str):
            clean_rels.append({"type": rel["type"], "target": rel["target"].strip()})
        else:
            unsupported.append(rel)
    return {"head_name": name.strip(), "head_attributes": [x.strip() for x in attrs if x.strip()],
            "head_relations": clean_rels, "others": others, "unsupported_relations": unsupported}


_PATTERN_TYPES = {
    "on the left of": "left_of", "to the left of": "left_of", "left of": "left_of",
    "on the right of": "right_of", "to the right of": "right_of", "right of": "right_of",
    "on top of": "above", "above": "above", "below": "below", "under": "below",
    "beneath": "below", "between": "between", "inside": "inside", "next to": "next_to",
    "beside": "next_to", "near": "near", "close to": "near",
}
assert set(_PATTERN_TYPES) == set(RELATION_PATTERNS)


def rule_fallback(expression):
    head, reason = extract_head(expression)
    matches = []
    for phrase in RELATION_PATTERNS:
        for m in re.finditer(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", expression, re.I):
            matches.append((m.start(), m.end(), phrase))
    # Prefer the longest pattern at a shared start position.
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected = []
    for m in matches:
        if not selected or m[0] >= selected[-1][1]:
            selected.append(m)
    relations = []
    for i, (_, end, phrase) in enumerate(selected):
        stop = selected[i + 1][0] if i + 1 < len(selected) else len(expression)
        target = expression[end:stop].strip(" .,:;!?\t\n")
        if target:
            relations.append({"type": _PATTERN_TYPES[phrase], "target": target})
    return {"head_name": head, "head_attributes": [], "head_relations": relations,
            "others": [], "unsupported_relations": [], "fallback_reason": reason}


def parse_graph(raw, expression):
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1))
    if "{" in text:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
            candidates.append(json.dumps(obj))
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            graph = _normalize_graph(json.loads(candidate))
            return {**graph, "json_parse_success": True, "parse_source": "qwen_json"}
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return {**rule_fallback(expression), "json_parse_success": False, "parse_source": "rule_fallback"}


def parse_yes_no(raw):
    value = raw.strip().upper()
    if value.startswith("YES"):
        return 1.0
    if value.startswith("NO"):
        return 0.0
    return 0.5


def box_key(image_id, box, head, attribute):
    # 1/100-pixel quantization is stable for saved FP32 coordinates.
    rounded = [round(float(x) * 100) for x in box]
    data = json.dumps([int(image_id), rounded, head.casefold(), attribute.casefold()], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def satisfaction(kind, head, target, second=None):
    if kind not in RELATIONS:
        raise ValueError(kind)
    hx, hy = _center(head)
    tx, ty = _center(target)
    tw, th = target[2] - target[0], target[3] - target[1]
    if tw <= 0 or th <= 0 or _area(head) <= 0:
        return 0.0
    if kind in {"left_of", "right_of", "above", "below"}:
        delta = {"left_of": tx-hx, "right_of": hx-tx, "above": ty-hy, "below": hy-ty}[kind]
        margin = 0.15 * (tw if kind in {"left_of", "right_of"} else th)
        return 1.0 if delta > margin else 0.0 if delta < -margin else 0.5
    if kind == "inside":
        ratio = _intersection(head, target) / _area(head)
        return 1.0 if ratio > 0.6 else 0.5 if math.isclose(ratio, 0.6, abs_tol=1e-9) else 0.0
    if kind in {"near", "next_to"}:
        dist = math.hypot(hx-tx, hy-ty)
        limit = 2.5 * (tw + th) / 2
        return 1.0 if dist < limit else 0.5 if math.isclose(dist, limit, abs_tol=1e-9) else 0.0
    if kind == "on":
        overlap_x = max(0, min(head[2], target[2]) - max(head[0], target[0]))
        if hy >= ty or overlap_x <= 0:
            return 0.0
        gap = abs(head[3] - target[1])
        limit = 0.15 * th
        return 1.0 if gap < limit else 0.5 if math.isclose(gap, limit, abs_tol=1e-9) else 0.0
    if second is None:
        return 0.0
    sx, sy = _center(second)
    in_x = min(tx, sx) < hx < max(tx, sx)
    in_y = min(ty, sy) < hy < max(ty, sy)
    if in_x or in_y:
        return 1.0
    boundary = (tx != sx and hx in (tx, sx)) or (ty != sy and hy in (ty, sy))
    return 0.5 if boundary else 0.0


def resolve_target(target, detections):
    key = target.casefold().strip()
    exact = [k for k in detections if k.casefold().strip() == key]
    if exact:
        return exact, "phrase_match"
    partial = [k for k in detections if key in k.casefold() or k.casefold() in key]
    return partial, "substring_fallback" if partial else "missing"


def distinct_target_detections(head_box, detections):
    """A relation's auxiliary object cannot be the exact candidate being scored."""
    return [item for item in detections if item["box_xyxy"] != head_box]


def rank_candidates(candidates):
    return max(candidates, key=lambda c: (c["structured_score"], c["detector_score"])) if candidates else None
