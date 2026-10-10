"""CPU-only Stage 5 feature/target schema provenance and parquet audit."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage4_build_features import FEATURES
from scripts.stage4_diagnostics import MODEL_C

OUT = ROOT / "outputs/stage5a10_fast"
DOCS = ROOT / "docs"
IDENTIFIERS = ("sample_id", "image_id", "ref_id", "sentence_id")
TARGETS = ("sample_id", "class_label", "IoU_a", "IoU_s", "delta_iou",
           "neutral_subtype", "anchor_correct", "structured_correct")
FORBIDDEN_TOKENS = {"gt", "label", "correct", "ioua", "ious", "annotation", "category",
                    "neutral", "target", "delta_iou"}


def manifest_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_schema(columns, manifest, stage4_audit):
    names = [row["feature_name"] for row in manifest]
    if names != [*FEATURES, "has_attribute", "has_relation"]:
        raise ValueError("Stage 5 feature list is not Stage 4 plus two declared flags")
    if list(columns) != [*IDENTIFIERS, *names]:
        raise ValueError("Feature parquet has unexpected/missing/reordered columns")
    for row in manifest:
        name = row["feature_name"]
        tokens = set(name.casefold().split("_"))
        if tokens & FORBIDDEN_TOKENS or name in TARGETS:
            raise ValueError(f"Forbidden predictor name: {name}")
        if row["uses_gt"] != "NO" or row["available_at_inference"] != "YES":
            raise ValueError(f"GT/unavailable feature: {name}")
        if not row["source"] or not row["description"]:
            raise ValueError(f"Missing feature provenance: {name}")
        if name in FEATURES:
            old = stage4_audit.get(name)
            if old is None or old["uses_gt"] != "NO" or old["source"] != row["source"]:
                raise ValueError(f"Stage 4 provenance differs: {name}")
        elif row["stage4_equivalent_name"] not in ("num_attributes", "num_relations"):
            raise ValueError(f"Undeclared new feature: {name}")
    if not set(MODEL_C).issubset(names):
        raise ValueError("Frozen Stage 4 Model C columns absent")
    return names


def audit_dataframe(frame, names, manifest):
    import numpy as np
    if list(frame.columns) != [*IDENTIFIERS, *names]:
        raise ValueError("Unexpected feature dataframe columns")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Missing or duplicate sample IDs")
    for key in IDENTIFIERS[1:]:
        if frame[key].isna().any():
            raise ValueError(f"Missing identifier {key}")
    rules = {row["feature_name"]: row for row in manifest}
    for name in names:
        values = frame[name]
        if values.isna().any() and rules[name]["missing_policy"] == "never missing":
            raise ValueError(f"Unexpected NaN/null in {name}")
        numeric = values.dropna().to_numpy(dtype=float)
        if numeric.size and not np.isfinite(numeric).all():
            raise ValueError(f"Infinite/nonfinite feature {name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    manifest = manifest_rows(DOCS / "05a10_feature_manifest.csv")
    old = {row["feature_name"]:row for row in manifest_rows(ROOT / "outputs/stage4/feature_leakage_audit.csv")}
    expected = [*IDENTIFIERS, *[row["feature_name"] for row in manifest]]
    names = validate_schema(expected, manifest, old)
    import pandas as pd
    import pyarrow as pa
    mock = {name: pd.Series([], dtype="object" if name == "sample_id" else "float64")
            for name in expected}
    audit_dataframe(pd.DataFrame(mock), names, manifest)
    pa.Table.from_pandas(pd.DataFrame(mock), preserve_index=False)
    if args.features_parquet:
        audit_dataframe(pd.read_parquet(args.features_parquet), names, manifest)
    report = f"""# Stage 5A10 feature leakage audit

**STATIC_SCHEMA_PASS.** {len(names)} predictor columns were checked against the actual frozen Stage 4 `FEATURES` and provenance audit. The two extra flags are deterministic `num_attributes>0` and `num_relations>0`. An empty pandas/Arrow table passed the same schema audit. Identifiers are not predictors; target columns are physically separate.

Stage 4 names remain unchanged: `detector_conf_delta` is the planned detector margin, and `box_pair_iou` is the planned anchor/structured IoU. The added flags do not change Stage 4 Model C's frozen input list.

Forbidden GT-derived predictor fields, unexpected columns, missing provenance, and unexpected nonfinite data fail the audit. Nullable Stage 4 evidence columns may contain null when a candidate is absent; this is a declared missing value, not a GT signal.

**RUNTIME_DATA_PASS = {'PASS' if args.features_parquet else 'PENDING'}**. A runtime PASS applies only to the explicitly audited feature parquet path; the static result alone is not a claim about later runtime data.
"""
    if args.features_parquet:
        (OUT / "runtime_feature_leakage_audit.json").write_text(json.dumps({
            "status":"PASS", "features_parquet":str(args.features_parquet),
            "predictor_columns":len(names)}, indent=2)+"\n", encoding="utf-8")
    else:
        (DOCS / "05a10_feature_leakage_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"static_schema":"PASS", "predictor_columns":len(names),
                      "runtime_data":"PASS" if args.features_parquet else "PENDING"}))


if __name__ == "__main__":
    main()
