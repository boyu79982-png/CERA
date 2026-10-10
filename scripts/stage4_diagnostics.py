"""Fixed, image-grouped Stage 4 feasibility diagnostics; not a final model."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage4_build_features import FEATURES  # noqa: E402

OUT = ROOT / "outputs/stage4"
MODEL_A = ("head_valid", "S_st", "anchor_detector_conf", "old_margin",
           "box_pair_iou", "structured_detector_conf")
MODEL_B = (*MODEL_A, "Sa_structured", "Sr_structured", "expression_length",
           "num_attributes", "num_relations")
MODEL_C = (*MODEL_B, "Sa_anchor", "Sr_anchor", "delta_Sa", "delta_Sr", "delta_sem",
           "anchor_constraint_min", "structured_constraint_min", "delta_min_constraint",
           "anchor_attr_pass_ratio", "structured_attr_pass_ratio",
           "anchor_relation_pass_ratio", "structured_relation_pass_ratio",
           "anchor_violation_count", "structured_violation_count", "delta_violation")
MODEL_COLUMNS = {"CCDA_FEATURES_ONLY": MODEL_A,
                 "CCDA_PLUS_AGGREGATE": MODEL_B,
                 "SYMMETRIC_CONSTRAINT_EVIDENCE": MODEL_C,
                 "SYMMETRIC_HGB_DIAGNOSTIC": MODEL_C}
CONTINUOUS = tuple(name for name in FEATURES if name not in
                   ("head_valid", "anchor_available", "structured_available",
                    "parse_source_rule_fallback"))


def read_features():
    with (OUT / "pairwise_conflict_features.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1854 or sum(int(row["label"]) for row in rows) != 897:
        raise ValueError("Unexpected conflict feature dataset")
    if set(rows[0]) != {"sample_id", "image_id", "label", *FEATURES}:
        raise ValueError("Unapproved feature column or missing required column")
    return rows


def values(rows, columns):
    return np.array([[float(row[name]) if row[name] not in ("", None) else np.nan
                      for name in columns] for row in rows], dtype=float)


def write_csv(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(y, probability, threshold, anchor_correct):
    from sklearn.metrics import average_precision_score, roc_auc_score
    decisions = probability >= threshold
    benefit = y == 1
    harm = y == 0
    tp = int(np.sum(decisions & benefit))
    fp = int(np.sum(decisions & harm))
    fn = int(np.sum(~decisions & benefit))
    tn = int(np.sum(~decisions & harm))
    return {"threshold": threshold, "tp_benefit_accepted": tp,
            "fp_harm_accepted": fp, "fn_benefit_rejected": fn,
            "tn_harm_rejected": tn, "balanced_accuracy": .5*(tp/(tp+fn)+tn/(tn+fp)),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "benefit_recall": tp/(tp+fn), "harm_rejection_rate": tn/(tn+fp),
            "benefit_precision": tp/(tp+fp) if tp+fp else None,
            "harm_false_accept_rate": fp/(fp+tn),
            "net_gain": tp-fp,
            "simulated_full_acc50": 100*(anchor_correct+tp-fp)/10834,
            "confusion_matrix_TN_FP_FN_TP": [tn,fp,fn,tp]}


def describe(x):
    if len(x) == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q1": None, "q3": None}
    return {"n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "median": float(np.median(x)), "q1": float(np.quantile(x,.25)),
            "q3": float(np.quantile(x,.75))}


def single_feature(rows, y):
    from sklearn.metrics import roc_auc_score
    output = []
    for name in CONTINUOUS:
        x = values(rows, (name,))[:, 0]
        mask = ~np.isnan(x)
        good = x[mask & (y == 1)]
        harm = x[mask & (y == 0)]
        if len(good) == 0 or len(harm) == 0 or np.unique(x[mask]).size < 2:
            raw_auc = oriented = None
            direction = "UNAVAILABLE"
        else:
            raw_auc = float(roc_auc_score(y[mask], x[mask]))
            oriented = max(raw_auc, 1-raw_auc)
            direction = "HIGHER_BENEFIT" if raw_auc >= .5 else "LOWER_BENEFIT"
        a, b = describe(good), describe(harm)
        output.append({"feature": name, "direction": direction,
                       "raw_auroc": raw_auc, "separability_auroc": oriented,
                       "missing": int(np.sum(~mask)),
                       **{f"benefit_{key}": value for key,value in a.items()},
                       **{f"harm_{key}": value for key,value in b.items()}})
    fields = list(output[0])
    write_csv("single_feature_separability.csv", output, fields)
    eligible = [row for row in output if row["separability_auroc"] is not None]
    best = max(eligible, key=lambda row:(row["separability_auroc"],row["feature"]))
    return output, best


def feature_source(name):
    if name in ("expression_length", "num_attributes", "num_relations", "parse_source_rule_fallback", "head_candidate_count"):
        return "Frozen expression / Stage 2 parser and head-candidate trace"
    if name == "head_valid":
        return "Stage 3 lexical head check from parser output and expression"
    if name in ("anchor_available", "anchor_detector_conf"):
        return "Frozen Stage 1 anchor prediction"
    if name in ("structured_available", "structured_detector_conf", "S_st"):
        return "Frozen Stage 2 selected structured candidate"
    if name in ("old_margin", "detector_conf_delta", "box_pair_iou"):
        return "Arithmetic/box geometry from frozen Stage 1 and Stage 2 predictions"
    if name.endswith("_anchor") or name.startswith("anchor_") or name in ("Sa_anchor", "Sr_anchor", "S_sem_anchor"):
        return "Anchor crop Qwen evidence or cached Stage 2 target-box geometry"
    if name.endswith("_structured") or name.startswith("structured_") or name in ("Sa_structured", "Sr_structured", "S_sem_structured"):
        return "Frozen Stage 2 selected candidate constraint evidence"
    if name.startswith("delta_"):
        return "Difference between anchor and structured inference-time evidence"
    raise ValueError(f"Unclassified feature {name}")


def leakage_audit():
    if any(name not in FEATURES for columns in MODEL_COLUMNS.values() for name in columns):
        raise ValueError("A model uses a non-whitelisted column")
    forbidden = {"gt", "label", "correct", "ann", "category", "sample", "image"}
    if any(forbidden.intersection(name.casefold().split("_"))
           for columns in MODEL_COLUMNS.values() for name in columns):
        raise ValueError("Model feature name contains a forbidden GT/metadata term")
    rows = [{"feature_name": name, "source": feature_source(name),
             "available_at_inference": "YES", "uses_gt": "NO",
             "model_a": name in MODEL_A, "model_b": name in MODEL_B, "model_c": name in MODEL_C}
            for name in FEATURES]
    rows += [{"feature_name": "sample_id", "source": "Record key", "available_at_inference": "YES",
              "uses_gt": "NO", "model_a": False, "model_b": False, "model_c": False},
             {"feature_name": "image_id", "source": "Grouping metadata only", "available_at_inference": "YES",
              "uses_gt": "NO", "model_a": False, "model_b": False, "model_c": False},
             {"feature_name": "label", "source": "GT-derived raw conflict outcome; supervision/evaluation only",
              "available_at_inference": "NO", "uses_gt": "YES",
              "model_a": False, "model_b": False, "model_c": False}]
    write_csv("feature_leakage_audit.csv", rows, list(rows[0]))
    return rows


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = read_features()
    audit = leakage_audit()
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    groups = np.array([int(row["image_id"]) for row in rows], dtype=int)
    oracle = json.loads((OUT / "pairwise_oracle.json").read_text(encoding="utf-8"))
    _, best_single = single_feature(rows, y)
    fold_plan = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    folds = np.full(len(rows), -1, dtype=int)
    fold_report = []
    for fold, (train, test) in enumerate(fold_plan):
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise ValueError(f"Image leakage in fold {fold}: {len(overlap)} groups")
        folds[test] = fold
        fold_report.append({"fold": fold, "train_rows": len(train), "test_rows": len(test),
                            "train_images": len(set(groups[train])), "test_images": len(set(groups[test])),
                            "image_overlap": 0, "train_benefit": int(np.sum(y[train])),
                            "test_benefit": int(np.sum(y[test]))})
    if np.any(folds < 0):
        raise ValueError("Incomplete 5-fold OOF assignment")
    probabilities = {}
    diagnostic_rows = []
    for model_name, columns in MODEL_COLUMNS.items():
        X = values(rows, columns)
        oof = np.full(len(rows), np.nan)
        for fold, (train, test) in enumerate(fold_plan):
            if model_name == "SYMMETRIC_HGB_DIAGNOSTIC":
                model = HistGradientBoostingClassifier(random_state=3407)
            else:
                model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0,
                                                    add_indicator=True),
                                      StandardScaler(),
                                      LogisticRegression(class_weight="balanced", max_iter=2000,
                                                         random_state=3407))
            model.fit(X[train], y[train])
            oof[test] = model.predict_proba(X[test])[:, 1]
        if np.any(np.isnan(oof)):
            raise ValueError(f"Incomplete OOF probabilities for {model_name}")
        probabilities[model_name] = oof
        diagnostic_rows.append({"model": model_name,
                                **metrics_for(y, oof, .5, oracle["anchor_correct"])})
    projection = {row["sample_id"]: row for row in
                  (json.loads(line) for line in (OUT / "conflict_projection.jsonl").open(encoding="utf-8"))}
    ccda = np.array([int(projection[row["sample_id"]]["ccda_source"] == "ccda") for row in rows], dtype=float)
    ccda_metrics = {"model": "FROZEN_CCDA", **metrics_for(y, ccda, .5, oracle["anchor_correct"])}
    diagnostic_rows.insert(0, ccda_metrics)
    metric_fields = list(diagnostic_rows[0])
    write_csv("diagnostic_metrics.csv", diagnostic_rows, metric_fields)
    oof_rows = [{"sample_id": row["sample_id"], "image_id": row["image_id"],
                 "label": row["label"], "fold": int(folds[i]),
                 "ccda_replace": int(ccda[i]),
                 **{f"p_benefit_{name}": float(score[i]) for name,score in probabilities.items()}}
                for i,row in enumerate(rows)]
    write_csv("diagnostic_oof_predictions.csv", oof_rows, list(oof_rows[0]))
    curve = []
    model_c = probabilities["SYMMETRIC_CONSTRAINT_EVIDENCE"]
    for step in range(1,20):
        threshold = step*.05
        point = metrics_for(y, model_c, threshold, oracle["anchor_correct"])
        curve.append({"threshold": threshold, "benefit_recall": point["benefit_recall"],
                      "harm_rejection_rate": point["harm_rejection_rate"],
                      "net_gain": point["net_gain"],
                      "simulated_full_acc50": point["simulated_full_acc50"]})
    write_csv("threshold_curve.csv", curve, list(curve[0]))
    best_curve = max(curve, key=lambda point:(point["net_gain"], -abs(point["threshold"]-.5)))
    summary = {"n": len(rows), "benefit": int(sum(y)), "harm": int(len(y)-sum(y)),
               "group_folds": fold_report, "model_columns": {key:list(value) for key,value in MODEL_COLUMNS.items()},
               "leakage_audit": "PASS" if all(row["uses_gt"] == "NO" for row in audit if row["model_c"]) else "FAIL",
               "best_single_feature": best_single,
               "metrics": {row["model"]:row for row in diagnostic_rows},
               "best_diagnostic_curve_point": best_curve,
               "curve_is_internal_oof_threshold_diagnostic": True}
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"leakage": summary["leakage_audit"], "best_single": best_single["feature"],
                      "model_a_auroc": summary["metrics"]["CCDA_FEATURES_ONLY"]["auroc"],
                      "model_c_auroc": summary["metrics"]["SYMMETRIC_CONSTRAINT_EVIDENCE"]["auroc"]}))


if __name__ == "__main__":
    main()
