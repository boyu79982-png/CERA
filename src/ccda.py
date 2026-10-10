"""Pure, GT-free DeRecG Equation (5) CCDA-only decision."""

import math

TAU_S = 0.70
TAU_M = 0.10
TAU_IOU = 0.70
BOUNDARY_EPS = 1e-12  # Floating representation, not a tuned gate width.


def box_iou(a, b):
    if a is None or b is None:
        return None
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
                   max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - intersection
    return intersection / union if union > 0 else 0.0


def ccda_decision(anchor_box, anchor_confidence, structured_box, structured_score,
                  head_valid, tau_s=TAU_S, tau_m=TAU_M, tau_iou=TAU_IOU):
    overlap = box_iou(anchor_box, structured_box)
    margin = (structured_score - anchor_confidence if structured_score is not None
              and anchor_confidence is not None else None)
    valid_gate = bool(head_valid)
    score_gate = structured_box is not None and structured_score is not None and \
        structured_score >= tau_s - BOUNDARY_EPS
    margin_gate = margin is not None and margin >= tau_m - BOUNDARY_EPS
    iou_gate = overlap is not None and overlap <= tau_iou + BOUNDARY_EPS
    accept = valid_gate and score_gate and margin_gate and iou_gate
    return {"final_box": structured_box if accept else anchor_box,
            "source": "ccda" if accept else "grounding_dino",
            "gate_valid_head": valid_gate, "gate_score_floor": score_gate,
            "gate_margin": margin_gate, "gate_disagreement": iou_gate,
            "margin_value": margin, "anchor_structured_iou": overlap,
            "all_pass": accept}
