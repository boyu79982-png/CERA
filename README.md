# CERA frozen method and comparison materials

This repository contains a limited public reference package for CERA (Constraint-Evidence Risk-Aware Arbitration) and the post-hoc ExistingScore-LR (9D) comparison on RefCOCO under a fixed candidate pathway.

## What is included

- Byte-preserved frozen project sources for the shared evidence construction, ordered selector, preprocessing and arbitration primitives, including their local module dependencies.
- The fixed visual-noun lexicon and documented head-validity implementation assumption.
- Historical CPU comparison execution scripts, preserved without editing their recorded scientific choices.
- The 53-column feature order/schema, ordered final 26-column selector, image-level FIT/DEV/INTERNAL_HOLDOUT split, final training recipe and frozen risk policy.
- A clearly labeled public view of the frozen 9D configuration and the recorded CPU dependency versions.
- SHA256 identities for each published file in `public_materials_manifest.json`.

The 9D input order is `head_valid`, `S_st`, `anchor_detector_conf`, `old_margin`, `box_pair_iou`, `structured_detector_conf`, `expression_length`, `num_attributes`, `num_relations` (indices `[3,9,6,10,11,7,0,1,2]` in the frozen 53-column schema). CERA retains the ordered 26-column selector. Benefit=0, Neutral=1, Harm=2; replacement uses `U=P_B-1.5P_H` and strictly `U>0.20`.

## Scope and practical limitations

This is a source-and-protocol reference package, not a self-contained dataset or a claimed one-command reproduction. The historical scripts retain their original absolute paths and artifact guards. They also expect excluded frozen feature/target caches, reports and model archives. Do not execute them blindly: a separate authorized reproduction environment and appropriately obtained inputs are required. Some original project primitives import PyTorch; the six-package requirements file describes the recorded standalone 9D CPU environment, not the complete upstream GPU environment.

COCO/RefCOCO images, dataset annotations, model weights, cached predictions, pickle objects, manuscripts, author contact information, credentials, server configurations and historical review reports are not distributed here. The image split contains benchmark image IDs rather than images. Source manifests are limited to file identities, not a release of those excluded objects. Original inputs must be obtained through their respective sources and permissions; this package does not grant rights to third-party datasets or models.

The comparison uses matching candidates, supervision, LR family and risk workpoint. It changes the evidence group and parameter count (30 versus 81), so it does not isolate a single-feature causal effect or hold model capacity constant. RefCOCO val is development evidence. The additional formal comparison was proposed after the original test results were known; positive testB point estimates have cluster confidence intervals crossing zero. No new training, inference, statistical resampling or tests were performed to prepare this release.

This repository is an author-requested public materials upload; it is not a claim of complete coauthor submission authorization or journal acceptance. No new software license is assigned here. Upstream methods and data retain their own applicable rights.
