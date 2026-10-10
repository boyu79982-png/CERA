"""CPU-only Stage 5B loading, training and one-shot evaluation helpers."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from scripts.audit_feature_leakage import (IDENTIFIERS, TARGETS, audit_dataframe,
                                           manifest_rows, validate_schema)
from scripts.stage4_diagnostics import MODEL_C
from src.stage5a10_core import sha256_file
from src.stage5b10_core import (BINARY_FEATURES, CeraNet, LABEL_TO_INDEX,
                               arbitration_metrics, split_images)


def load_frozen_inputs(root):
    root = Path(root)
    stage5a = root / "outputs/stage5a10_fast"
    merged = stage5a / "merged"
    manifest = manifest_rows(root / "docs/05a10_feature_manifest.csv")
    previous = {row["feature_name"]: row for row in manifest_rows(
        root / "outputs/stage4/feature_leakage_audit.csv")}
    feature_order = validate_schema(
        [*IDENTIFIERS, *[row["feature_name"] for row in manifest]], manifest, previous)
    if len(feature_order) != 53 or not set(MODEL_C).issubset(feature_order):
        raise ValueError("Frozen Stage 5A/Stage 4 feature dimensions changed")
    manifest_on_disk = json.loads((merged / "merge_manifest.json").read_text())
    report = json.loads((stage5a / "final_report.json").read_text())
    prep = json.loads((stage5a / "prep_manifest.json").read_text())
    if report["status"] != "PASS" or report["processed_expressions"] != 12146 or \
       manifest_on_disk["status"] != "COMPLETE" or manifest_on_disk["rows"] != 12146:
        raise ValueError("Stage 5A frozen completion gate absent")
    paths = {"features": merged / "training_features_10pct.parquet",
             "targets": merged / "training_targets_10pct.parquet"}
    for key, path in paths.items():
        if sha256_file(path) != manifest_on_disk[key]["sha256"]:
            raise ValueError(f"Frozen {key} parquet hash mismatch")
    features = pd.read_parquet(paths["features"])
    targets = pd.read_parquet(paths["targets"])
    audit_dataframe(features, feature_order, manifest)
    if list(targets.columns) != list(TARGETS) or len(features) != 12146 or len(targets) != 12146:
        raise ValueError("Stage 5B input shape/schema mismatch")
    if not features["sample_id"].equals(targets["sample_id"]) or \
       features["sample_id"].duplicated().any() or targets["sample_id"].duplicated().any():
        raise ValueError("Feature/target IDs mismatch or duplicate")
    if features["image_id"].nunique() != 1699:
        raise ValueError("Selected image count changed")
    if targets["class_label"].isna().any() or \
       set(targets["class_label"]) != set(LABEL_TO_INDEX):
        raise ValueError("Class mapping invalid")
    numeric = targets[["IoU_a", "IoU_s", "delta_iou"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or \
       not np.allclose(numeric[:, 2], numeric[:, 1] - numeric[:, 0], atol=1e-10):
        raise ValueError("Nonfinite or inconsistent targets")
    anchor, structured = numeric[:, 0] >= .5, numeric[:, 1] >= .5
    derived = np.where(~anchor & structured, "BENEFIT",
                       np.where(anchor & ~structured, "HARM", "NEUTRAL"))
    if not np.array_equal(derived, targets["class_label"].to_numpy()) or \
       not np.array_equal(anchor, targets["anchor_correct"].to_numpy()) or \
       not np.array_equal(structured, targets["structured_correct"].to_numpy()):
        raise ValueError("Target correctness/class inconsistency")
    null_counts = features[feature_order].isna().sum()
    return {"features": features, "targets": targets, "feature_order": feature_order,
            "manifest": manifest, "model_c": list(MODEL_C),
            "raw_structural_nulls": {key: int(value) for key, value in
                                     null_counts[null_counts > 0].items()},
            "selected_subset_hash": prep["selected_image_hash"],
            "feature_sha256": manifest_on_disk["features"]["sha256"],
            "target_sha256": manifest_on_disk["targets"]["sha256"],
            "feature_manifest_sha256": sha256_file(root / "docs/05a10_feature_manifest.csv")}


def prepare_splits(data):
    features, targets = data["features"], data["targets"]
    split = split_images(features["image_id"])
    sets = [set(values) for values in split.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)) or \
       set.union(*sets) != set(features["image_id"]):
        raise ValueError("Image overlap/missing image in internal split")
    frames = {}
    for key, images in split.items():
        mask = features["image_id"].isin(images).to_numpy()
        f, t = features.loc[mask].reset_index(drop=True), targets.loc[mask].reset_index(drop=True)
        if len(f) == 0 or not f["sample_id"].equals(t["sample_id"]):
            raise ValueError(f"Empty or misjoined {key} split")
        counts = t["class_label"].value_counts()
        if any(counts.get(label, 0) == 0 for label in LABEL_TO_INDEX):
            raise ValueError(f"Class missing from {key}; split must not be retuned")
        frames[key] = (f, t)
    return split, frames


def split_distribution(split, frames):
    result = {}
    for key, images in split.items():
        targets = frames[key][1]
        result[key] = {"images": len(images), "expressions": len(targets),
                       "labels": {label: int((targets["class_label"] == label).sum())
                                  for label in LABEL_TO_INDEX},
                       "both_correct": int((targets["neutral_subtype"] == "BOTH_CORRECT").sum()),
                       "both_wrong": int((targets["neutral_subtype"] == "BOTH_WRONG").sum())}
    return result


def class_summary(true_indices, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    return {"accuracy": float(accuracy_score(true_indices, predicted)),
            "macro_f1": float(f1_score(true_indices, predicted, labels=[0, 1, 2],
                                       average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(true_indices, predicted,
                                                  labels=[0, 1, 2]).tolist()}


def train_cera_seed(seed, x_fit, y_fit, gain_fit, x_dev, y_dev, gain_dev,
                    weights, checkpoint_path, metadata):
    """Best DEV-loss checkpoint; HOLDOUT is absent from this interface."""
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = CeraNet(x_fit.shape[1]).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32))
    gain_loss = torch.nn.SmoothL1Loss()
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_fit), torch.as_tensor(y_fit, dtype=torch.long),
        torch.as_tensor(gain_fit, dtype=torch.float32).reshape(-1, 1))
    batches = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True,
                                          generator=torch.Generator().manual_seed(seed), num_workers=0)
    dev_x = torch.as_tensor(x_dev)
    dev_y = torch.as_tensor(y_dev, dtype=torch.long)
    dev_gain = torch.as_tensor(gain_dev, dtype=torch.float32).reshape(-1, 1)
    history, best_loss, best_epoch, stale = [], math.inf, 0, 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 201):
        model.train()
        total = 0.0
        for inputs, labels, gains in batches:
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_gain = model(inputs)
            loss = ce(logits, labels) + .5 * gain_loss(predicted_gain, gains)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(inputs)
        model.eval()
        with torch.inference_mode():
            dev_logits, dev_predicted_gain = model(dev_x)
            dev_loss = float((ce(dev_logits, dev_y) +
                              .5 * gain_loss(dev_predicted_gain, dev_gain)).item())
        history.append({"epoch": epoch, "fit_loss": total / len(dataset),
                        "dev_loss": dev_loss})
        if dev_loss < best_loss - 1e-10:
            best_loss, best_epoch, stale = dev_loss, epoch, 0
            payload = {"state_dict": model.state_dict(), "architecture": "53-64-32-3+1",
                       "input_dim": x_fit.shape[1], "training_seed": seed,
                       "best_dev_epoch": epoch, "best_dev_loss": dev_loss, **metadata}
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
        else:
            stale += 1
        if epoch >= 20 and stale >= 20:
            break
    return {"seed": seed, "best_epoch": best_epoch, "best_dev_loss": best_loss,
            "trained_epochs": len(history), "history": history,
            "checkpoint_sha256": sha256_file(checkpoint_path)}


def predict_cera(checkpoint_path, matrix):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CeraNet(payload["input_dim"]).to("cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        logits, gain = model(torch.as_tensor(matrix, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1)
    return (logits.numpy(), probabilities.numpy(), gain.numpy().reshape(-1), payload)


def prediction_frame(features, logits, probabilities, gain):
    return pd.DataFrame({"sample_id": features["sample_id"],
                         "image_id": features["image_id"],
                         "logit_B": logits[:, 0], "logit_N": logits[:, 1],
                         "logit_H": logits[:, 2],
                         "P_B": probabilities[:, 0], "P_N": probabilities[:, 1],
                         "P_H": probabilities[:, 2], "delta_iou_hat": gain})


def exact_mcnemar(first_correct, second_correct):
    first, second = np.asarray(first_correct, dtype=bool), np.asarray(second_correct, dtype=bool)
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    return {"first_only": first_only, "second_only": second_only,
            "discordant": discordant,
            "p_exact_two_sided": float(binomtest(first_only, discordant, .5).pvalue)
            if discordant else 1.0}


def paired_image_bootstrap_ci(first_correct, second_correct, image_ids,
                              resamples=10000, seed=3407):
    first, second = np.asarray(first_correct, dtype=np.float64), np.asarray(second_correct, dtype=np.float64)
    images = np.asarray(image_ids)
    unique, inverse = np.unique(images, return_inverse=True)
    delta_by_image = np.bincount(inverse, weights=first-second)
    count_by_image = np.bincount(inverse)
    samples = np.random.default_rng(seed).integers(0, len(unique),
                                                   size=(resamples, len(unique)))
    differences = 100 * delta_by_image[samples].sum(axis=1) / count_by_image[samples].sum(axis=1)
    return {"unit": "image", "resamples": resamples, "seed": seed,
            "difference_pp": float(100 * np.mean(first-second)),
            "ci95_pp": [float(np.quantile(differences, .025)),
                         float(np.quantile(differences, .975))]}


def gain_head_analysis(predicted_gain, targets):
    truth = targets["delta_iou"].to_numpy(dtype=np.float64)
    predicted = np.asarray(predicted_gain, dtype=np.float64)
    if len(truth) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Invalid gain prediction")
    def correlation(fn):
        if np.std(predicted) == 0 or np.std(truth) == 0:
            return None
        value = float(fn(predicted, truth).statistic)
        return value if np.isfinite(value) else None
    distribution = {}
    for label in LABEL_TO_INDEX:
        values = predicted[(targets["class_label"] == label).to_numpy()]
        distribution[label] = {"n": int(len(values)), "mean": float(values.mean()),
                               "std": float(values.std()),
                               "p05": float(np.quantile(values, .05)),
                               "p50": float(np.quantile(values, .50)),
                               "p95": float(np.quantile(values, .95))}
    return {"mae": float(np.mean(np.abs(predicted-truth))),
            "spearman": correlation(spearmanr), "pearson": correlation(pearsonr),
            "predicted_distribution_by_class": distribution}
