# Stage 3 CCDA head-validity specification

Source: [DeRecG §3.3, §3.6, Equation (5)](https://www.mdpi.com/2078-2489/17/9/832).

**Status: REIMPLEMENTATION_ASSUMPTION.** The paper requires a visual noun appearing in the referring expression, and §3.3 describes lexicon-based LVCE repair. It does not publish the lexicon, morphological rules, or exact substring/token matching implementation. Stage 2/2A did not use a visual-noun lexicon; this stage therefore freezes a new, explicit lexicon before reading full CCDA outcomes. This is not claimed to be the paper's hidden lexicon.

The CCDA-only head is exactly `head_name` saved by the Stage 2 parser. No repaired head or new query is permitted. `src/visual_nouns.txt` contains a fixed physical-object vocabulary, including standard COCO object names and common RefCOCO-style person/clothing/part synonyms. It is selected from the paper's examples and general visual-object vocabulary, not from sample labels, outcomes, or Stage 3 gates. A head is visually valid only if its entire normalized phrase is in this lexicon, allowing a small fixed plural-to-singular normalization. Attribute-prefix phrases such as `black shoe` remain invalid even if the final token is a visual noun.

For appearance in the expression, both strings are case-folded, punctuation replaced with spaces, and whitespace collapsed. A multiword head must match a contiguous full token phrase; `car` does not match `cart`. No embeddings, Qwen, image evidence, GT boxes, or outcome-derived synonyms are used. Exact expression morphology is retained, so `woman` does not match `women` unless that exact wording appears; this conservative choice is another implementation assumption.

`visual_noun_valid`, `appears_in_expression`, and their conjunction `valid` are saved for every sample. A null structured candidate cannot override. If the frozen anchor box or confidence is missing, Equation (5)'s margin/IoU are undefined and the anchor (possibly null) is retained; this is a conservative implementation assumption because the paper does not specify the edge case.
