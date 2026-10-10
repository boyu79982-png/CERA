"""Deterministic DeRecG Appendix A.1 parser fallback used for Stage 1B only."""

import re


# The paper specifies a non-stopword content token, but no exact stopword list.
# This fixed, small function-word list is a reproduction assumption, not a noun tagger.
STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "of", "on", "in", "at",
    "to", "from", "with", "by", "and", "or", "is", "are", "was", "were", "be",
    "being", "been", "it", "its", "there", "here",
})

# The English relation patterns enumerated in DeRecG Appendix A.1.
RELATION_PATTERNS = (
    "on the left of", "to the left of", "left of",
    "on the right of", "to the right of", "right of",
    "on top of", "above", "below", "under", "beneath", "between", "inside",
    "next to", "beside", "near", "close to",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
RELATION_RES = tuple(re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.I)
                     for phrase in RELATION_PATTERNS)


def extract_head(expression: str) -> tuple[str, str]:
    """Select one leftmost content token before a relation; never inspect image or GT."""
    matches = [match for pattern in RELATION_RES if (match := pattern.search(expression))]
    span = expression[:min(match.start() for match in matches)] if matches else expression
    tokens = [match.group().casefold() for match in TOKEN_RE.finditer(span)]
    for token in tokens:
        if token not in STOPWORDS:
            return token, "first_non_stopword_before_relation"
    full_tokens = [match.group().casefold() for match in TOKEN_RE.finditer(expression)]
    for token in full_tokens:
        if token not in STOPWORDS:
            return token, "first_token_when_pre_relation_span_empty"
    if full_tokens:
        return full_tokens[0], "all_tokens_are_stopwords_use_first_token"
    raise ValueError("Cannot extract a token from an empty expression")


def patch_row(primary: dict, fallback: dict, head: str, reason: str, model_input: str) -> dict:
    """Keep full-expression candidates and store head-query candidates separately."""
    assert primary["no_anchor"] and primary["anchor"] is None
    assert fallback["no_anchor"] == (fallback["anchor"] is None)
    success = fallback["anchor"] is not None
    result = dict(primary)
    result.update({
        "primary_query_success": False,
        "fallback_used": True,
        "fallback_head": head,
        "fallback_extraction_reason": reason,
        "fallback_query_model_input": model_input,
        "fallback_success": success,
        "anchor_source": "head_fallback" if success else "none",
        "fallback_raw_predictions": fallback["raw_predictions"],
        "fallback_thresholded_predictions": fallback["thresholded_predictions"],
        "fallback_nms_predictions": fallback["nms_predictions"],
        "fallback_latency_ms": fallback["latency_ms"],
        "anchor": fallback["anchor"],
        "anchor_score": fallback["anchor"]["score"] if success else None,
        "anchor_iou": fallback["anchor_iou"],
        "anchor_correct_at_05": fallback["anchor_correct_at_05"],
        "no_anchor": fallback["no_anchor"],
    })
    return result


def preserve_primary(primary: dict) -> dict:
    """Add v2 provenance without changing a successful full-expression prediction."""
    assert not primary["no_anchor"] and primary["anchor"] is not None
    result = dict(primary)
    result.update({
        "primary_query_success": True,
        "fallback_used": False,
        "fallback_head": None,
        "fallback_extraction_reason": None,
        "fallback_query_model_input": None,
        "fallback_success": False,
        "anchor_source": "full_expression",
        "anchor_score": primary["anchor"]["score"],
    })
    return result
