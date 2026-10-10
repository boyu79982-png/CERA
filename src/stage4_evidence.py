"""Pure symmetric constraint evidence from frozen Stage 2 traces."""

import hashlib
import json
import math
import statistics

from src.ccda import box_iou
from src.stage2_protocol import distinct_target_detections, resolve_target, satisfaction


def anchor_attribute_key(image_id, box, head, attribute):
    """Exact frozen-box task identity; head is required because it enters the prompt."""
    payload = json.dumps([int(image_id), [float(x) for x in box], head.casefold(),
                          attribute.casefold()], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def padded_crop_box(width, height, box):
    """Exact Stage 2 crop padding, rounding and clipping."""
    pad_x = (box[2] - box[0]) * .08
    pad_y = (box[3] - box[1]) * .08
    crop = [max(0, math.floor(box[0]-pad_x)), max(0, math.floor(box[1]-pad_y)),
            min(width, math.ceil(box[2]+pad_x)), min(height, math.ceil(box[3]+pad_y))]
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError(f"Invalid candidate crop: {crop}")
    return crop


def same_box(a, b):
    return a is not None and b is not None and len(a) == len(b) == 4 and \
        all(math.isclose(x, y, rel_tol=0, abs_tol=1e-8) for x, y in zip(a, b))


def selected_candidate(row):
    """Locate the frozen structured winner; never re-rank candidates."""
    box = row["structured_box"]
    if box is None:
        if row["structured_total_score"] is not None:
            raise ValueError("Null structured box has a semantic score")
        return None
    matches = [candidate for candidate in row["candidates"] if same_box(candidate["box"], box)]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved structured candidate, got {len(matches)}")
    candidate = matches[0]
    for source, selected in (("attribute_score", "structured_attribute_score"),
                             ("relation_score", "structured_relation_score"),
                             ("structured_score", "structured_total_score")):
        if not math.isclose(candidate[source], row[selected], rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Frozen selected {source} differs from Stage 2 candidate")
    return candidate


def matching_stage2_candidate(anchor_box, row):
    """Reuse attribute evidence only for a head candidate with IoU >= 0.999."""
    if anchor_box is None:
        return None
    eligible = [(box_iou(anchor_box, candidate["box"]), candidate["detector_score"], candidate)
                for candidate in row["candidates"]]
    eligible = [item for item in eligible if item[0] is not None and item[0] >= .999]
    return max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None


def anchor_relation_evidence(anchor_box, row):
    """Evaluate the frozen parsed relations on anchor using saved target detections."""
    detected = {row["head_name"]: row["head_detection"], **row["relation_targets"]}
    answer = []
    for relation in row["relations"]:
        kind, target = relation["type"], relation["target"]
        if anchor_box is None:
            answer.append({"type": kind, "target": target, "target_boxes": [],
                           "satisfactions": [], "score": None, "resolution": "anchor_missing"})
            continue
        if kind == "between":
            names = [item.get("name") for item in row["others"]
                     if isinstance(item, dict) and isinstance(item.get("name"), str)]
            if len(names) < 2:
                names = [part.strip() for part in target.split(" and ") if part.strip()]
            groups = []
            for name in names[:2]:
                keys, _ = resolve_target(name, detected)
                groups.append(distinct_target_detections(anchor_box,
                              [candidate for key in keys for candidate in detected[key]["top6"]]))
            pairs = [(left, right) for left in groups[0] for right in groups[1]
                     if left["box_xyxy"] != right["box_xyxy"]] if len(groups) == 2 else []
            boxes = [[left["box_xyxy"], right["box_xyxy"]] for left, right in pairs]
            values = [satisfaction(kind, anchor_box, left["box_xyxy"], right["box_xyxy"])
                      for left, right in pairs]
            resolution = "two_targets"
        else:
            keys, resolution = resolve_target(target, detected)
            candidates = [candidate for key in keys for candidate in detected[key]["top6"]]
            candidates = distinct_target_detections(anchor_box, candidates)
            boxes = [candidate["box_xyxy"] for candidate in candidates]
            values = [satisfaction(kind, anchor_box, box) for box in boxes]
        answer.append({"type": kind, "target": target, "target_boxes": boxes,
                       "satisfactions": values, "score": max(values, default=0.0),
                       "resolution": resolution})
    return answer


def _component(values, empty_default, pass_threshold):
    if not values:
        return {"mean": empty_default, "min": None, "max": None, "std": None,
                "pass_ratio": None, "count": 0, "missing_count": 0}
    present = [value for value in values if value is not None]
    if len(present) != len(values):
        # Partial evidence cannot be represented as a complete symmetric aggregate.
        return {"mean": None, "min": None, "max": None, "std": None,
                "pass_ratio": None, "count": len(values), "missing_count": len(values)-len(present)}
    return {"mean": statistics.mean(present), "min": min(present), "max": max(present),
            "std": statistics.pstdev(present),
            "pass_ratio": sum(value >= pass_threshold for value in present)/len(present),
            "count": len(values), "missing_count": 0}


def evidence_statistics(attr_scores, relation_scores):
    """Pass: attribute >=0.5, relation ==1; violation: definite score <0.5."""
    attr = _component(attr_scores, .6, .5)
    rel = _component(relation_scores, .6, 1.0)
    present = attr_scores + relation_scores
    if not present:
        constraint_mean = constraint_min = .6
        violations = 0
    elif any(value is None for value in present):
        constraint_mean = constraint_min = violations = None
    else:
        constraint_mean = statistics.mean(present)
        constraint_min = min(present)
        violations = sum(value < .5 for value in present)
    sa, sr = attr["mean"], rel["mean"]
    return {"Sa": sa, "Sr": sr, "S_sem": .5*sa+.5*sr if sa is not None and sr is not None else None,
            "attr_min": attr["min"], "attr_max": attr["max"], "attr_std": attr["std"],
            "attr_pass_ratio": attr["pass_ratio"], "attr_count": attr["count"],
            "attr_missing_count": attr["missing_count"],
            "relation_min": rel["min"], "relation_max": rel["max"],
            "relation_std": rel["std"], "relation_pass_ratio": rel["pass_ratio"],
            "relation_count": rel["count"], "relation_missing_count": rel["missing_count"],
            "constraint_mean": constraint_mean, "constraint_min": constraint_min,
            "violation_count": violations}
