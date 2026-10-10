"""Frozen CPU-only data, decision and model primitives for CERA Stage 5B."""

import hashlib

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


LABEL_TO_INDEX = {"BENEFIT": 0, "NEUTRAL": 1, "HARM": 2}
SEEDS = (3407, 3408, 3409)
LAMBDA_GRID = (1.0, 1.5, 2.0, 3.0)
TAU_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
BINARY_FEATURES = ("head_valid", "anchor_available", "structured_available",
                   "parse_source_rule_fallback", "has_attribute", "has_relation")


def split_images(image_ids, seed=3407):
    """Domain-separated SHA-256 ordering, independent of input/hash ordering."""
    images = sorted({int(value) for value in image_ids})
    order = sorted(images, key=lambda image: (
        hashlib.sha256(f"stage5b10:{seed}:{image}".encode("ascii")).digest(), image))
    n_fit = round(len(order) * .8)
    n_dev = round(len(order) * .1)
    return {"FIT": order[:n_fit], "DEV": order[n_fit:n_fit+n_dev],
            "INTERNAL_HOLDOUT": order[n_fit+n_dev:]}


def class_weights(y_fit):
    counts = np.bincount(np.asarray(y_fit, dtype=np.int64), minlength=3)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError("All three FIT classes are required")
    raw = 1.0 / counts.astype(np.float64)
    return raw / raw.mean()


def _numeric(frame, names):
    return frame[list(names)].to_numpy(dtype=np.float64, na_value=np.nan)


def fit_preprocessor(fit_frame, feature_order, binary_features=BINARY_FEATURES):
    names = list(feature_order)
    binary = [name for name in names if name in binary_features]
    continuous = [name for name in names if name not in binary]
    if len(names) != len(set(names)) or not continuous:
        raise ValueError("Invalid predictor order")
    if not np.isfinite(_numeric(fit_frame, names)[~np.isnan(_numeric(fit_frame, names))]).all():
        raise ValueError("Nonfinite FIT feature")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    filled = imputer.fit_transform(_numeric(fit_frame, continuous))
    scaler = StandardScaler().fit(filled)
    binary_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if binary:
        binary_imputer.fit(_numeric(fit_frame, binary))
    return {"feature_order": names, "continuous": continuous, "binary": binary,
            "imputer": imputer, "scaler": scaler, "binary_imputer": binary_imputer}


def transform_features(preprocessor, frame):
    names = preprocessor["feature_order"]
    if not set(names).issubset(frame.columns):
        raise ValueError("Missing predictor column")
    values = _numeric(frame, names)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("Nonfinite predictor")
    continuous = preprocessor["continuous"]
    scaled = preprocessor["scaler"].transform(
        preprocessor["imputer"].transform(_numeric(frame, continuous)))
    by_name = {name: scaled[:, i] for i, name in enumerate(continuous)}
    binary = preprocessor["binary"]
    if binary:
        transformed = preprocessor["binary_imputer"].transform(_numeric(frame, binary))
        for i, name in enumerate(binary):
            if not np.isin(transformed[:, i], [0.0, 1.0]).all():
                raise ValueError(f"Binary feature outside 0/1: {name}")
            by_name[name] = transformed[:, i]
    matrix = np.column_stack([by_name[name] for name in names]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed NaN/Inf")
    return matrix


def ccda_decisions(features):
    """Frozen four-gate CCDA replay; no target column is consulted."""
    head = _numeric(features, ["head_valid"])[:, 0]
    score = _numeric(features, ["S_sem_structured"])[:, 0]
    confidence = _numeric(features, ["anchor_detector_conf"])[:, 0]
    overlap = _numeric(features, ["box_pair_iou"])[:, 0]
    eps = 1e-12
    return ((head == 1) & (score >= .70 - eps) &
            (score - confidence >= .10 - eps) & (overlap <= .70 + eps))


def arbitration_metrics(targets, replace):
    decision = np.asarray(replace, dtype=bool)
    if len(decision) != len(targets):
        raise ValueError("Decision/target length mismatch")
    anchor = targets["IoU_a"].to_numpy(dtype=np.float64)
    structured = targets["IoU_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(anchor).all() or not np.isfinite(structured).all():
        raise ValueError("Nonfinite target IoU")
    final = np.where(decision, structured, anchor)
    labels = targets["class_label"].to_numpy()
    benefit, harm = labels == "BENEFIT", labels == "HARM"
    corrected = int(np.sum(decision & benefit))
    induced = int(np.sum(decision & harm))
    overrides = int(decision.sum())
    return {
        "acc25": float(100 * np.mean(final >= .25)),
        "acc50": float(100 * np.mean(final >= .50)),
        "acc75": float(100 * np.mean(final >= .75)),
        "miou": float(100 * np.mean(final)),
        "corrected": corrected, "induced": induced,
        "net_gain": corrected - induced,
        "override_count": overrides,
        "override_rate": float(overrides / len(decision)),
        "benefit_recall": float(corrected / benefit.sum()) if benefit.any() else None,
        "harm_rejection": float(1 - induced / harm.sum()) if harm.any() else None,
        "correction_precision": float(corrected / overrides) if overrides else None,
    }


def risk_decisions(probabilities, risk_lambda, tau):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or \
       not np.isfinite(probabilities).all():
        raise ValueError("Expected finite BENEFIT/NEUTRAL/HARM probabilities")
    return probabilities[:, 0] - risk_lambda * probabilities[:, 2] > tau


def select_common_decision(probabilities_by_seed, dev_targets, ccda_replace):
    """Select one pair on DEV only, with frozen CCDA harm rejection as constraint."""
    if len(probabilities_by_seed) != 3:
        raise ValueError("Three seeds required for common risk selection")
    ccda = arbitration_metrics(dev_targets, ccda_replace)
    rows = []
    for risk_lambda in LAMBDA_GRID:
        for tau in TAU_GRID:
            metrics = [arbitration_metrics(dev_targets,
                       risk_decisions(probability, risk_lambda, tau))
                       for probability in probabilities_by_seed]
            rows.append({"lambda": risk_lambda, "tau": tau,
                         "mean_acc50": float(np.mean([item["acc50"] for item in metrics])),
                         "mean_corrected": float(np.mean([item["corrected"] for item in metrics])),
                         "mean_induced": float(np.mean([item["induced"] for item in metrics])),
                         "mean_override_rate": float(np.mean([item["override_rate"] for item in metrics])),
                         "mean_benefit_recall": float(np.mean([item["benefit_recall"] for item in metrics])),
                         "mean_harm_rejection": float(np.mean([item["harm_rejection"] for item in metrics]))})
    eligible = [row for row in rows if row["mean_harm_rejection"] >=
                ccda["harm_rejection"] - 1e-12]
    pool = eligible if eligible else rows
    if eligible:
        rank = lambda row: (row["mean_acc50"], -row["mean_induced"],
                            -row["mean_override_rate"])
    else:
        rank = lambda row: (row["mean_harm_rejection"], row["mean_acc50"],
                            -row["mean_induced"], -row["mean_override_rate"])
    chosen = max(pool, key=rank)
    return {**chosen, "risk_constraint": "MET" if eligible else "RISK_CONSTRAINT_NOT_MET",
            "ccda_harm_rejection": ccda["harm_rejection"],
            "ccda_acc50": ccda["acc50"]}, rows


class CeraNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.GELU(), torch.nn.Dropout(.10),
            torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Dropout(.10))
        self.risk = torch.nn.Linear(32, 3)
        self.gain = torch.nn.Linear(32, 1)

    def forward(self, values):
        embedding = self.shared(values)
        return self.risk(embedding), self.gain(embedding)
