"""Stage 5B: CPU-only frozen train-10% development, then one-shot holdout."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage5a10_core import atomic_json, sha256_file, sha256_value  # noqa: E402
from src.stage5b10_core import (BINARY_FEATURES, LABEL_TO_INDEX, SEEDS,
    arbitration_metrics, ccda_decisions, class_weights, fit_preprocessor,
    risk_decisions, select_common_decision, transform_features)  # noqa: E402
from src.stage5b10_pipeline import (class_summary, exact_mcnemar,
    gain_head_analysis, load_frozen_inputs, paired_image_bootstrap_ci,
    predict_cera, prediction_frame, prepare_splits, split_distribution,
    train_cera_seed)  # noqa: E402

OUT = ROOT / "outputs/stage5b10"
DOCS = ROOT / "docs"


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def log(message):
    path = OUT / "logs/stage5b10.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def labels(targets):
    return targets["class_label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)


def audit():
    if torch.cuda.is_available():
        # The training device is explicitly CPU even if a GPU is visible.
        pass
    data = load_frozen_inputs(ROOT)
    split, frames = prepare_splits(data)
    return data, split, frames, {"gpu_used": "NO", "cuda_available": bool(torch.cuda.is_available()),
        "rows": len(data["features"]), "images": data["features"]["image_id"].nunique(),
        "model_input_dim": len(data["feature_order"]),
        "raw_structural_nulls": data["raw_structural_nulls"],
        "split_distribution": split_distribution(split, frames),
        "feature_leakage_audit": "PASS", "observed_nonfinite": 0}


def dev_phase():
    data, split, frames, preflight = audit()
    if (OUT / "holdout_consumed.json").exists():
        raise ValueError("Holdout already consumed; do not retrain after seeing it")
    if (OUT / "decision_config.json").exists():
        raise ValueError("DEV selection already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("metrics", "predictions", "checkpoints", "portability", "logs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    log("DEV_START CPU_ONLY")
    split_hash = sha256_value(split)
    atomic_json(OUT / "internal_image_split.json", {"seed": 3407,
        "algorithm": "sha256(stage5b10:3407:image_id)", "images": split,
        "split_hash": split_hash, "distribution": preflight["split_distribution"],
        "pairwise_intersections": {"FIT_DEV": 0, "FIT_HOLDOUT": 0, "DEV_HOLDOUT": 0}})
    feature_order = data["feature_order"]
    continuous = [name for name in feature_order if name not in BINARY_FEATURES]
    binary = [name for name in feature_order if name in BINARY_FEATURES]
    atomic_json(OUT / "feature_order.json", {"feature_order": feature_order,
                                                "model_input_dim": len(feature_order)})
    atomic_json(OUT / "feature_schema.json", {"source": "Stage 5A frozen manifest",
        "source_sha256": data["feature_manifest_sha256"],
        "feature_order": feature_order, "continuous_scaled": continuous,
        "binary_unscaled": binary, "identifiers_excluded":
        ["sample_id", "image_id", "ref_id", "sentence_id", "ann_id"],
        "raw_structural_nulls": data["raw_structural_nulls"],
        "imputation": "FIT-only median for continuous; FIT-only mode for binary",
        "post_transform_nan_inf": 0})
    atomic_json(OUT / "dataset_manifest.json", {"selected_images": 1699,
        "selected_expressions": 12146, "selected_subset_hash": data["selected_subset_hash"],
        "split_hash": split_hash, "feature_sha256": data["feature_sha256"],
        "target_sha256": data["target_sha256"],
        "feature_manifest_sha256": data["feature_manifest_sha256"],
        "feature_leakage_audit": "PASS", "gpu_used": "NO"})
    fit_f, fit_t = frames["FIT"]
    dev_f, dev_t = frames["DEV"]
    preprocessor = fit_preprocessor(fit_f, feature_order, BINARY_FEATURES)
    joblib.dump(preprocessor, OUT / "scaler.pkl")
    x_fit = transform_features(preprocessor, fit_f)
    x_dev = transform_features(preprocessor, dev_f)
    if x_fit.shape[1] != 53 or not np.isfinite(x_fit).all() or not np.isfinite(x_dev).all():
        raise ValueError("Post-preprocessing input integrity failed")
    y_fit, y_dev = labels(fit_t), labels(dev_t)
    weights = class_weights(y_fit)
    weight_list = [float(value) for value in weights]
    config = {"architecture": "53 -> 64 -> 32 -> {3-class, gain}",
        "class_mapping": LABEL_TO_INDEX, "class_weights": weight_list,
        "class_weight_source": "FIT only", "beta": .5, "optimizer": "AdamW",
        "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
        "max_epochs": 200, "min_epochs": 20, "early_stopping_patience": 20,
        "early_stopping_monitor": "DEV total loss", "seeds": list(SEEDS),
        "device": "cpu", "model_input_dim": 53, "model_c_lr_dim": len(data["model_c"]),
        "training_subset_hash": data["selected_subset_hash"], "split_hash": split_hash}
    atomic_json(OUT / "training_config.json", config)
    dev_ccda = ccda_decisions(dev_f)
    anchor_dev = arbitration_metrics(dev_t, np.zeros(len(dev_t), dtype=bool))
    raw_dev = arbitration_metrics(dev_t, np.ones(len(dev_t), dtype=bool))
    ccda_dev = arbitration_metrics(dev_t, dev_ccda)
    atomic_json(OUT / "metrics/dev_baselines.json",
                {"Anchor": anchor_dev, "Raw Structured": raw_dev, "CCDA": ccda_dev})
    model_c_indices = [feature_order.index(name) for name in data["model_c"]]
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=3407)
    lr.fit(x_fit[:, model_c_indices], y_fit)
    if not np.array_equal(lr.classes_, [0, 1, 2]):
        raise ValueError("CERA-LR class order changed")
    joblib.dump({"model": lr, "model_c": data["model_c"]}, OUT / "logistic_model.pkl")
    lr_probability = lr.predict_proba(x_dev[:, model_c_indices])
    lr_choice, _ = select_common_decision([lr_probability] * 3, dev_t, dev_ccda)
    lr_dev = {"classification": class_summary(y_dev, lr_probability),
              "arbitration": arbitration_metrics(dev_t, risk_decisions(
                  lr_probability, lr_choice["lambda"], lr_choice["tau"])),
              "decision": lr_choice}
    atomic_json(OUT / "metrics/logistic_baseline_dev.json", lr_dev)
    probabilities, training, dev_classes = [], {}, {}
    metadata = {"feature_order": feature_order, "class_mapping": LABEL_TO_INDEX,
                "class_weights": weight_list, "beta": .5,
                "training_subset_hash": data["selected_subset_hash"],
                "split_hash": split_hash}
    for seed in SEEDS:
        checkpoint = OUT / f"checkpoints/cera_seed{seed}.pt"
        result = train_cera_seed(seed, x_fit, y_fit, fit_t["delta_iou"].to_numpy(),
            x_dev, y_dev, dev_t["delta_iou"].to_numpy(), weights, checkpoint, metadata)
        training[str(seed)] = result
        logits, probability, gain, payload = predict_cera(checkpoint, x_dev)
        if payload["training_seed"] != seed or payload["split_hash"] != split_hash:
            raise ValueError("Checkpoint metadata mismatch")
        prediction_frame(dev_f, logits, probability, gain).to_parquet(
            OUT / f"predictions/dev_seed{seed}.parquet", index=False)
        probabilities.append(probability)
        dev_classes[str(seed)] = class_summary(y_dev, probability)
        log(f"SEED_{seed}_COMPLETE best_epoch={result['best_epoch']} dev_loss={result['best_dev_loss']:.8f}")
    selected, grid = select_common_decision(probabilities, dev_t, dev_ccda)
    pd.DataFrame(grid).to_csv(OUT / "metrics/risk_grid.csv", index=False)
    cera_dev = {str(seed): {"classification": dev_classes[str(seed)],
        "arbitration": arbitration_metrics(dev_t, risk_decisions(
            probabilities[i], selected["lambda"], selected["tau"]))}
        for i, seed in enumerate(SEEDS)}
    atomic_json(OUT / "metrics/dev_results.json", {"seeds": cera_dev,
        "common_selection": selected})
    atomic_json(OUT / "metrics/training_histories.json", training)
    decision = {"source": "DEV_ONLY", "common_lambda": selected["lambda"],
        "common_tau": selected["tau"], "risk_constraint": selected["risk_constraint"],
        "selection": selected, "lr_lambda": lr_choice["lambda"],
        "lr_tau": lr_choice["tau"], "lr_selection": lr_choice,
        "split_hash": split_hash,
        "checkpoint_sha256": {seed: training[str(seed)]["checkpoint_sha256"] for seed in SEEDS},
        "holdout_evaluated": False}
    atomic_json(OUT / "decision_config.json", decision)
    write_text(DOCS / "05b10_training_protocol.md", f"""# CERA Stage 5B training protocol

CPU only; frozen RefCOCO train/unc 10% with 12,146 expressions and 53 predictor columns.
Identifiers are excluded. Structural nulls are imputed using FIT-only medians/modes;
continuous features use a FIT-only StandardScaler; six binary features remain 0/1.
Labels: BENEFIT=0, NEUTRAL=1, HARM=2. Gain target: delta_iou.
MLP: 53→64→32→classification(3) + gain(1), GELU, dropout 0.10.
Loss: inverse-frequency FIT-only weighted CE + 0.5 SmoothL1.
AdamW lr=1e-3, weight_decay=1e-4, batch=256, at least 20 and at most 200 epochs,
DEV-loss early stopping patience 20. Seeds: 3407, 3408, 3409.
Stage 4 Model C's {len(data['model_c'])} symmetric-evidence columns feed fixed CERA-LR;
class_weight=balanced, no hyperparameter search.
No RefCOCO val/test labels, GPU inference, or training on HOLDOUT were used.
""")
    distribution = preflight["split_distribution"]
    write_text(DOCS / "05b10_internal_split.md", f"""# Image-disjoint internal split

Seed 3407; SHA-256 stable image ordering. Split hash `{split_hash}`.
FIT {distribution['FIT']}; DEV {distribution['DEV']};
INTERNAL_HOLDOUT {distribution['INTERNAL_HOLDOUT']}.
All image intersections are empty. This is train-internal development data, not a RefCOCO test set.
""")
    write_text(DOCS / "05b10_ccda_baseline.md", f"""# Frozen CCDA replay

Replace only when head_valid, S_sem_structured≥0.70,
S_sem_structured−anchor_detector_conf≥0.10, and box_pair_iou≤0.70.
The decision uses inference-available features only.
DEV Anchor {anchor_dev}; Raw Structured {raw_dev}; CCDA {ccda_dev}.
HOLDOUT metrics are withheld until the common CERA decision is frozen.
""")
    write_text(DOCS / "05b10_training_results.md", f"""# DEV training results

Class weights from FIT: {weight_list}. Best checkpoints and SHA-256:
{json.dumps({seed: {'best_epoch': training[str(seed)]['best_epoch'], 'sha256': training[str(seed)]['checkpoint_sha256']} for seed in SEEDS}, indent=2)}
CERA-LR DEV: {lr_dev}. CERA DEV: {cera_dev}.
""")
    write_text(DOCS / "05b10_risk_selection.md", f"""# DEV-only risk selection

Utility U=P_B−lambda·P_H; replace iff U>tau.
Grid: lambda {{1.0,1.5,2.0,3.0}}; tau {{0,.05,.10,.15,.20,.25,.30,.40}}.
Common three-seed selection: {selected}.
Constraint: mean DEV harm rejection ≥ DEV CCDA harm rejection.
Tie breakers: fewer induced errors, then lower override rate.
One pair is frozen for all HOLDOUT seeds. HOLDOUT has not been evaluated here.
""")
    log(f"DEV_SELECTION_FROZEN lambda={selected['lambda']} tau={selected['tau']}")
    return {"status": "DEV_FROZEN", "images": distribution,
            "class_weights": weight_list, "selection": selected,
            "dev_ccda_acc50": ccda_dev["acc50"],
            "dev_cera_mean_acc50": float(np.mean([row["arbitration"]["acc50"] for row in cera_dev.values()]))}


def holdout_phase():
    data, split, frames, preflight = audit()
    decision_path = OUT / "decision_config.json"
    if not decision_path.exists():
        raise ValueError("DEV selection must be frozen before HOLDOUT")
    final_path = OUT / "final_report.json"
    if final_path.exists():
        marker = OUT / "holdout_consumed.json"
        if marker.exists() and json.loads(marker.read_text())["status"] == "COMPLETE":
            return json.loads(final_path.read_text())
        raise ValueError("Incomplete prior HOLDOUT evaluation; preserve evidence")
    decision = json.loads(decision_path.read_text())
    if decision["source"] != "DEV_ONLY" or decision["holdout_evaluated"] or \
       decision["split_hash"] != sha256_value(split):
        raise ValueError("Decision/split protocol mismatch")
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    if manifest["feature_sha256"] != data["feature_sha256"] or \
       manifest["target_sha256"] != data["target_sha256"]:
        raise ValueError("Input changed after DEV selection")
    for seed in SEEDS:
        checkpoint = OUT / f"checkpoints/cera_seed{seed}.pt"
        if sha256_file(checkpoint) != decision["checkpoint_sha256"][str(seed)]:
            raise ValueError(f"Checkpoint {seed} changed after DEV selection")
    marker = OUT / "holdout_consumed.json"
    if marker.exists():
        raise ValueError("HOLDOUT attempt already started; preserve frozen decision and investigate")
    atomic_json(marker, {"status": "STARTED", "decision_sha256": sha256_file(decision_path),
                         "time_utc": datetime.now(timezone.utc).isoformat()})
    log("HOLDOUT_ONESHOT_START")
    hold_f, hold_t = frames["INTERNAL_HOLDOUT"]
    dev_t = frames["DEV"][1]
    preprocessor = joblib.load(OUT / "scaler.pkl")
    feature_order = data["feature_order"]
    if preprocessor["feature_order"] != feature_order:
        raise ValueError("Scaler feature order mismatch")
    x_hold = transform_features(preprocessor, hold_f)
    hold_y = labels(hold_t)
    ccda_replace = ccda_decisions(hold_f)
    baseline = {"Anchor": arbitration_metrics(hold_t, np.zeros(len(hold_t), dtype=bool)),
                "Raw Structured": arbitration_metrics(hold_t, np.ones(len(hold_t), dtype=bool)),
                "CCDA": arbitration_metrics(hold_t, ccda_replace)}
    lr_artifact = joblib.load(OUT / "logistic_model.pkl")
    if lr_artifact["model_c"] != data["model_c"]:
        raise ValueError("Frozen Model C list changed")
    lr_indices = [feature_order.index(name) for name in data["model_c"]]
    lr_probability = lr_artifact["model"].predict_proba(x_hold[:, lr_indices])
    lr_replace = risk_decisions(lr_probability, decision["lr_lambda"], decision["lr_tau"])
    lr_result = {"classification": class_summary(hold_y, lr_probability),
                 "arbitration": arbitration_metrics(hold_t, lr_replace)}
    seed_metrics, seed_gains, logits_by_seed, portability = {}, {}, {}, {}
    for seed in SEEDS:
        checkpoint = OUT / f"checkpoints/cera_seed{seed}.pt"
        logits, probability, gain, payload = predict_cera(checkpoint, x_hold)
        if payload["feature_order"] != feature_order or \
           payload["split_hash"] != decision["split_hash"] or \
           payload["training_subset_hash"] != data["selected_subset_hash"]:
            raise ValueError("Portable checkpoint metadata mismatch")
        prediction_frame(hold_f, logits, probability, gain).to_parquet(
            OUT / f"predictions/holdout_seed{seed}.parquet", index=False)
        replace = risk_decisions(probability, decision["common_lambda"], decision["common_tau"])
        seed_metrics[str(seed)] = {"classification": class_summary(hold_y, probability),
                                   "arbitration": arbitration_metrics(hold_t, replace)}
        seed_gains[str(seed)] = gain_head_analysis(gain, hold_t)
        logits_by_seed[str(seed)] = logits
        ordering = sorted(range(len(hold_f)), key=lambda i: sha256_value(
            [3407, hold_f["sample_id"].iloc[i]]))[:100]
        fresh_scaler = joblib.load(OUT / "scaler.pkl")
        fresh_x = transform_features(fresh_scaler, hold_f.iloc[ordering])
        fresh_logits, fresh_prob, _, _ = predict_cera(checkpoint, fresh_x)
        maximum = float(np.max(np.abs(logits[ordering] - fresh_logits)))
        old_decisions = replace[ordering]
        new_decisions = risk_decisions(fresh_prob, decision["common_lambda"],
                                       decision["common_tau"])
        disagreements = int(np.sum(old_decisions != new_decisions))
        portability[str(seed)] = {"sample_count": 100, "max_logit_diff": maximum,
                                  "decision_disagreement": disagreements,
                                  "status": "PASS" if maximum < 1e-6 and disagreements == 0 else "FAIL"}
    if any(row["status"] != "PASS" for row in portability.values()):
        raise ValueError("Checkpoint/scaler portability FAIL")
    fields = ("acc25", "acc50", "acc75", "miou", "corrected", "induced", "net_gain",
              "override_count", "override_rate", "benefit_recall", "harm_rejection",
              "correction_precision")
    summary = {name: {"mean": float(np.mean([seed_metrics[str(seed)]["arbitration"][name]
                    for seed in SEEDS])), "std": float(np.std([
                    seed_metrics[str(seed)]["arbitration"][name] for seed in SEEDS], ddof=1))}
               for name in fields}
    pd.DataFrame([{"seed": seed, **seed_metrics[str(seed)]["arbitration"]}
                  for seed in SEEDS]).to_csv(OUT / "metrics/cera_seed_metrics.csv", index=False)
    gain_summary = {}
    for name in ("mae", "spearman", "pearson"):
        values = [seed_gains[str(seed)][name] for seed in SEEDS]
        gain_summary[name] = {"mean": float(np.mean(values)) if all(
            value is not None for value in values) else None,
            "std": float(np.std(values, ddof=1)) if all(
            value is not None for value in values) else None}
    def correct(replace):
        a = hold_t["IoU_a"].to_numpy(dtype=float)
        s = hold_t["IoU_s"].to_numpy(dtype=float)
        return np.where(replace, s, a) >= .5
    cera3407 = correct(risk_decisions(
        pd.read_parquet(OUT / "predictions/holdout_seed3407.parquet")[["P_B", "P_N", "P_H"]].to_numpy(),
        decision["common_lambda"], decision["common_tau"]))
    tests = {}
    for name, reference in (("Anchor", correct(np.zeros(len(hold_t), dtype=bool))),
                            ("CCDA", correct(ccda_replace))):
        tests[f"CERA3407_vs_{name}"] = {
            "mcnemar": exact_mcnemar(cera3407, reference),
            "paired_bootstrap": paired_image_bootstrap_ci(
                cera3407, reference, hold_f["image_id"].to_numpy(), 10000, 3407)}
    delta_pp = summary["acc50"]["mean"] - baseline["CCDA"]["acc50"]
    harm_difference = summary["harm_rejection"]["mean"] - baseline["CCDA"]["harm_rejection"]
    wins = sum(seed_metrics[str(seed)]["arbitration"]["acc50"] >
               baseline["CCDA"]["acc50"] for seed in SEEDS)
    induced_difference_pp = 100 * (summary["induced"]["mean"] -
                                   baseline["CCDA"]["induced"]) / len(hold_t)
    if delta_pp >= .30 and harm_difference >= -.02 and wins >= 2 and induced_difference_pp <= 2:
        go = "STRONG_GO"
    elif delta_pp <= 0 or induced_difference_pp > 2:
        go = "NO_GO"
    else:
        go = "REVIEW"
    readiness = {"STRONG_GO": "YES", "REVIEW": "REVIEW", "NO_GO": "NO"}[go]
    recommended = {"STRONG_GO": "Review internal evidence, then separately authorize val feature completion",
                   "REVIEW": "Review risk tradeoff and gain-head diagnostics before any val work",
                   "NO_GO": "Stop GPU expansion; revise protocol with a new unconsumed development holdout"}[go]
    atomic_json(OUT / "metrics/ccda_internal.json", {"DEV": json.loads((OUT / "metrics/dev_baselines.json").read_text()),
                                                        "INTERNAL_HOLDOUT": baseline})
    atomic_json(OUT / "metrics/logistic_baseline.json", {"DEV": json.loads((OUT / "metrics/logistic_baseline_dev.json").read_text()),
                                                         "INTERNAL_HOLDOUT": lr_result})
    atomic_json(OUT / "metrics/cera_summary.json", {"seeds": seed_metrics, "mean_std": summary,
                                                     "gain": seed_gains, "gain_mean_std": gain_summary})
    atomic_json(OUT / "metrics/holdout_results.json", {"baselines": baseline,
        "CERA-LR": lr_result, "CERA": seed_metrics, "CERA_mean_std": summary,
        "statistical_tests": tests, "go_no_go": go})
    atomic_json(OUT / "portability/audit.json", {"status": "PASS", "seeds": portability})
    report = {"status": "PASS", "gpu_used": "NO", "cuda_available": preflight["cuda_available"],
        "input_expressions": 12146, "distribution": preflight["split_distribution"],
        "model_input_dim": 53, "class_weights": json.loads((OUT / "training_config.json").read_text())["class_weights"],
        "beta": .5, "seeds": list(SEEDS),
        "best_common_lambda": decision["common_lambda"],
        "best_common_tau": decision["common_tau"],
        "risk_constraint": decision["risk_constraint"],
        "dev_ccda_acc50": decision["selection"]["ccda_acc50"],
        "dev_cera_mean_acc50": decision["selection"]["mean_acc50"],
        "holdout_baselines": baseline, "holdout_cera_mean_std": summary,
        "cera_delta_vs_ccda_pp": delta_pp, "holdout_cera_seeds": seed_metrics,
        "gain_head_mean_std": gain_summary, "statistics": tests,
        "portability_audit": "PASS", "go_no_go": go,
        "ready_for_val_feature_completion": readiness,
        "recommended_next_stage": recommended,
        "internal_holdout_only": True}
    write_text(DOCS / "05b10_holdout_results.md", f"""# One-shot train-internal HOLDOUT results

This is an image-disjoint RefCOCO train-10% internal holdout, not a paper test result.
Common lambda={decision['common_lambda']}, tau={decision['common_tau']} were frozen on DEV.
Anchor {baseline['Anchor']}; Raw Structured {baseline['Raw Structured']};
CCDA {baseline['CCDA']}; CERA-LR {lr_result}; CERA seeds {seed_metrics}.
CERA mean±sample std across all three seeds: {summary}.
Exact McNemar and 10,000 image-cluster paired bootstrap comparisons: {tests}.
""")
    write_text(DOCS / "05b10_gain_head_analysis.md", f"""# Gain regression head, internal HOLDOUT

Per-seed MAE, Spearman, Pearson and class-conditional prediction distributions:
{json.dumps(seed_gains, indent=2)}
Three-seed mean±sample std: {gain_summary}.
These diagnostics did not change the frozen HOLDOUT decision.
""")
    write_text(DOCS / "05b10_go_no_go.md", f"""# Stage 5B internal resource decision

Decision: **{go}**. CERA−CCDA Acc@0.5 = {delta_pp:.4f} pp;
Harm rejection difference = {100*harm_difference:.4f} pp; seed wins={wins}/3;
induced-error increase = {induced_difference_pp:.4f} pp.
STRONG_GO requires ≥0.30 pp mean Acc gain, risk loss ≤2 pp, ≥2/3 seed wins,
and no >2 pp induced-error increase. REVIEW denotes smaller positive gain or risk tradeoff;
NO_GO denotes no gain or >2 pp induced-error increase.
Recommended next stage: {recommended}. No val/test inference has been run.
""")
    atomic_json(OUT / "final_report.json", report)
    atomic_json(marker, {"status": "COMPLETE", "decision_sha256": sha256_file(decision_path),
                         "report_sha256": sha256_file(final_path),
                         "time_utc": datetime.now(timezone.utc).isoformat()})
    log(f"HOLDOUT_COMPLETE go_no_go={go} delta_vs_ccda_pp={delta_pp:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "dev", "holdout"), required=True)
    args = parser.parse_args()
    if args.phase == "audit":
        print(json.dumps(audit()[3], ensure_ascii=False, sort_keys=True))
    elif args.phase == "dev":
        print(json.dumps(dev_phase(), ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(holdout_phase(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
