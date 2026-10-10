"""Stage 7 CPU-only train/val ablations; frozen test metrics are read only for tables."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.base import clone  # noqa: E402

from src.stage5a10_core import sha256_file  # noqa: E402
from src.stage5b10_core import (BINARY_FEATURES, CeraNet, LABEL_TO_INDEX,
    arbitration_metrics, ccda_decisions, fit_preprocessor, risk_decisions,
    transform_features)  # noqa: E402
from src.stage5b10_pipeline import paired_image_bootstrap_ci  # noqa: E402

OUT = ROOT / "outputs/stage7"
S5A = ROOT / "outputs/stage5a10_fast"
S5B = ROOT / "outputs/stage5b10"
S5C = ROOT / "outputs/stage5c"
S5D = ROOT / "outputs/stage5d_final"
S6 = ROOT / "outputs/stage6_test"
FROZEN_TEST = S6 / "evaluation/final_test_results.json"
MASTER = ROOT / "outputs/final_archive/paper_main_results.csv"
GROUPS = {
    "CONSTRAINT_MASK": {"head_valid"},
    "STRUCTURED_SIDE_EVIDENCE": {"S_st", "Sa_structured", "Sr_structured",
        "structured_constraint_min", "structured_attr_pass_ratio",
        "structured_relation_pass_ratio", "structured_violation_count"},
    "ANCHOR_SIDE_EVIDENCE": {"Sa_anchor", "Sr_anchor", "anchor_constraint_min",
        "anchor_attr_pass_ratio", "anchor_relation_pass_ratio", "anchor_violation_count"},
    "PAIRWISE_DELTA": {"old_margin", "delta_Sa", "delta_Sr", "delta_sem",
        "delta_min_constraint", "delta_violation"},
    "DETECTOR_CONFIDENCE": {"anchor_detector_conf", "structured_detector_conf"},
    "PAIR_GEOMETRY": {"box_pair_iou"},
    "EXPRESSION_METADATA": {"expression_length", "num_attributes", "num_relations"},
}
ANCHOR_SEMANTIC_DELTAS = {"delta_Sa", "delta_Sr", "delta_sem",
                          "delta_min_constraint", "delta_violation"}
NO_CONSTRAINT_GROUPS = {"DETECTOR_CONFIDENCE", "PAIR_GEOMETRY",
                        "EXPRESSION_METADATA"}
TAU_B_GRID = (.30, .35, .40, .45, .50, .55, .60, .65, .70)


def jread(path):
    return json.loads(Path(path).read_text())


def jwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def input_audit():
    required = [S5A / "merged/training_features_10pct.parquet",
        S5A / "merged/training_targets_10pct.parquet",
        S5A / "shards/shard_00_000_005/training_features.parquet",
        S5A / "shards/shard_00_000_005/training_targets.parquet",
        S5A / "shards/shard_01_005_010/manifest.full.json",
        S5B / "internal_image_split.json", S5B / "feature_order.json",
        S5B / "logistic_model.pkl", S5B / "scaler.pkl",
        S5B / "metrics/cera_summary.json", S5C / "val_features_complete.parquet",
        S5C / "frozen_val_evaluation_trace.parquet", S5C / "metrics/cera_lr.json",
        S5D / "cera_lr_final.pkl", S5D / "final_scaler.pkl",
        S6 / "final_lr_26_feature_columns.json", MASTER, FROZEN_TEST]
    if any(not p.is_file() for p in required):
        raise ValueError("Stage7 required input missing")
    merge = jread(S5A / "merged/merge_manifest.json")
    if sha256_file(required[0]) != merge["features"]["sha256"] or \
       sha256_file(required[1]) != merge["targets"]["sha256"]:
        raise ValueError("Train10 merged parquet hash mismatch")
    shard = jread(S5A / "shards/shard_00_000_005/manifest.full.json")
    if sha256_file(required[2]) != shard["features_sha256"] or \
       sha256_file(required[3]) != shard["targets_sha256"] or \
       (shard["image_count"], shard["expression_count"]) != (849, 6049):
        raise ValueError("Frozen train5 shard mismatch")
    if sha256_file(S5D / "cera_lr_final.pkl") != \
       "75b9cc451404de044a41a954e9e4a5de0a867341227f3462a0db2aadba3881fc":
        raise ValueError("Final model hash changed")
    freeze = jread(S6 / "final_lr_26_feature_columns.json")
    if freeze["selector_hash"] != \
       "510a10f30efde12d0e371ed74edffb0e10caf03fb8f918c02b681f22b3086f1d":
        raise ValueError("LR selector changed")
    return {"status": "PASS", "input_hashes": {str(p.relative_to(ROOT)): sha256_file(p)
        for p in required if p.suffix in (".parquet", ".pkl", ".json", ".csv")},
        "test_result_hash": sha256_file(FROZEN_TEST),
        "test_prediction_hashes": {s: sha256_file(S6 / f"predictions/{s}_predictions_frozen.parquet")
                                   for s in ("testA", "testB")}}


def freeze_protocol():
    if OUT.exists():
        raise ValueError("Stage7 output already exists; preserve frozen protocol")
    audit = input_audit()
    order = jread(S5B / "feature_order.json")["feature_order"]
    selector = jread(S6 / "final_lr_26_feature_columns.json")["ordered_columns"]
    if len(order) != 53 or len(selector) != 26 or len(set(selector)) != 26 or \
       not set(selector).issubset(order):
        raise ValueError("Frozen feature order mismatch")
    manifest = {row["feature_name"]: row for row in csv.DictReader(
        (ROOT / "docs/05a10_feature_manifest.csv").open(encoding="utf-8"))}
    assigned = {name: group for group, names in GROUPS.items() for name in names}
    if len(assigned) != sum(map(len, GROUPS.values())) or set(selector) != set(assigned):
        raise ValueError("Unproven feature grouping; stop instead of guessing")
    rows = []
    for i, name in enumerate(selector):
        source = manifest[name]["source"]
        group = assigned[name]
        if manifest[name]["uses_gt"] != "NO" or manifest[name]["model_c_input"] != "YES":
            raise ValueError(f"Selector provenance mismatch: {name}")
        anchor_semantic = group == "ANCHOR_SIDE_EVIDENCE" or name in ANCHOR_SEMANTIC_DELTAS
        constraint = group in ("ANCHOR_SIDE_EVIDENCE", "STRUCTURED_SIDE_EVIDENCE",
                               "CONSTRAINT_MASK") or group == "PAIRWISE_DELTA"
        rows.append({"feature_name": name, "feature_index": i, "feature_group": group,
                     "source": source, "depends_anchor_semantic": int(anchor_semantic),
                     "depends_constraint_evidence": int(constraint),
                     "stage4_code": "scripts/stage4_build_features.py:104-138"})
    no_sym = [r["feature_name"] for r in rows if not r["depends_anchor_semantic"]]
    no_constraint = [r["feature_name"] for r in rows if
                     r["feature_group"] in NO_CONSTRAINT_GROUPS]
    if len(no_sym) != 15 or len(no_constraint) != 6:
        raise ValueError("Expected preregistered NoSym=15, NoConstraint=6 features")
    OUT.mkdir(parents=True)
    for name in ("models", "metrics", "predictions_val", "tables", "figdata", "manifests"):
        (OUT / name).mkdir()
    write_csv(OUT / "lr_feature_group_manifest.csv", rows)
    jwrite(OUT / "manifests/input_audit.json", audit)
    protocol = {"status": "FROZEN_BEFORE_VAL_ABLATION", "test_phase": "CLOSED",
        "gpu_used": "NO", "train_policy": "FIT_IMAGES_ONLY; no refit after DEV threshold selection",
        "full_control": "Stage5B frozen FIT-only LR/scaler (same training protocol)",
        "no_sym_features": no_sym, "no_constraint_features": no_constraint,
        "no_risk_features": selector, "full_features": selector,
        "no_sym_removed": [x for x in selector if x not in no_sym],
        "no_constraint_removed": [x for x in selector if x not in no_constraint],
        "lambda": 1.5, "tau": .20,
        "no_risk_tau_grid": list(TAU_B_GRID),
        "no_risk_dev_rank": ["Acc50 descending", "Induced ascending", "Override ascending", "tau ascending"],
        "val_policy": "one predeclared evaluation after FIT training and DEV threshold selection",
        "train5_policy": "full frozen shard00 6049 expressions; same 26 columns/recipe, lambda/tau fixed",
        "bootstrap": {"unit": "image_id", "resamples": 10000, "seed": 3407},
        "feature_group_manifest_sha256": sha256_file(OUT / "lr_feature_group_manifest.csv"),
        "frozen_inputs": audit}
    jwrite(OUT / "manifests/ablation_protocol.json", protocol)
    (ROOT / "docs/07_ablation_protocol.md").write_text(
        "# Stage 7 frozen ablation protocol\n\nFull, NoSym and NoConstraint use the Stage5B "
        "image-disjoint FIT split, FIT-fitted imputer/scaler and identical LR estimator recipe. "
        "The Full control is the persisted Stage5B FIT-only LR. NoSym removes anchor-side "
        "semantic evidence and its five derived deltas; NoConstraint keeps only two detector "
        "confidence, one pair geometry and three expression metadata columns. All three use "
        "λ=1.5, τ=0.20. NoRisk reuses Full probabilities; its P(BENEFIT) threshold is "
        "selected once on DEV from the frozen grid by Acc50, induced, override, then tau. "
        "No INTERNAL_HOLDOUT or val is used to choose features or policy. Train5 uses all "
        "6049 expressions in shard00 with the frozen full LR recipe and decision rule. "
        "The test phase is closed. See outputs/stage7/manifests/ablation_protocol.json and "
        "lr_feature_group_manifest.csv.\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "no_sym_dim": len(no_sym),
                      "no_constraint_dim": len(no_constraint),
                      "protocol_sha256": sha256_file(OUT / "manifests/ablation_protocol.json")}))


def select_no_risk_tau(dev_prob, dev_targets):
    rows = []
    for tau in TAU_B_GRID:
        m = arbitration_metrics(dev_targets, dev_prob[:, 0] > tau)
        rows.append({"tau": tau, "acc50": m["acc50"], "induced": m["induced"],
                     "override_count": m["override_count"]})
    selected = max(rows, key=lambda r: (r["acc50"], -r["induced"],
                                         -r["override_count"], -r["tau"]))
    return selected, rows


def val_metrics(targets, decision):
    m = arbitration_metrics(targets, decision)
    labels = targets["class_label"].to_numpy()
    benefit, harm = labels == "BENEFIT", labels == "HARM"
    conflict = benefit | harm
    m["conflict_decision_accuracy"] = float((np.sum(benefit & decision) +
        np.sum(harm & ~decision)) / np.sum(conflict))
    m["balanced_conflict_accuracy"] = (m["benefit_recall"] + m["harm_rejection"]) / 2
    return m


def load_train_val():
    f = pd.read_parquet(S5A / "merged/training_features_10pct.parquet")
    t = pd.read_parquet(S5A / "merged/training_targets_10pct.parquet")
    v = pd.read_parquet(S5C / "val_features_complete.parquet")
    vt = pd.read_parquet(S5C / "frozen_val_evaluation_trace.parquet")
    if len(f) != 12146 or len(v) != 10834 or len(vt) != 10834 or \
       not f["sample_id"].equals(t["sample_id"]) or \
       not v["sample_id"].equals(vt["sample_id"]):
        raise ValueError("Frozen train/val row alignment mismatch")
    split = jread(S5B / "internal_image_split.json")["images"]
    fit = f["image_id"].isin(split["FIT"]).to_numpy()
    dev = f["image_id"].isin(split["DEV"]).to_numpy()
    hold = f["image_id"].isin(split["INTERNAL_HOLDOUT"]).to_numpy()
    if fit.sum() != 9694 or dev.sum() != 1219 or hold.sum() != 1233 or \
       np.any(fit & dev) or np.any(fit & hold) or np.any(dev & hold):
        raise ValueError("Frozen image split changed")
    return f, t, v, vt, fit, dev


def save_prediction(name, ids, probability, decision):
    pd.DataFrame({"sample_id": ids, "P_B": probability[:, 0],
        "P_N": probability[:, 1], "P_H": probability[:, 2],
        "replace_decision": decision}).to_parquet(
        OUT / f"predictions_val/{name}.parquet", index=False)


def bench(fn, n):
    for _ in range(100): fn()
    times = []
    for _ in range(100):
        begin = time.perf_counter_ns(); fn(); times.append((time.perf_counter_ns()-begin)/1e6)
    ms = np.asarray(times)
    return {"repeats": 100, "warmups": 100, "post_feature_only": True,
        "p50_ms_full_val": float(np.median(ms)), "p95_ms_full_val": float(np.quantile(ms, .95)),
        "p50_us_per_sample": float(1000*np.median(ms)/n),
        "p95_us_per_sample": float(1000*np.quantile(ms,.95)/n),
        "p50_ms_per_10k": float(np.median(ms)*10000/n),
        "p95_ms_per_10k": float(np.quantile(ms,.95)*10000/n)}


def run():
    protocol_path = OUT / "manifests/ablation_protocol.json"
    if not protocol_path.exists() or (OUT / "manifests/final_audit.json").exists():
        raise ValueError("Frozen Stage7 protocol missing or run already complete")
    protocol = jread(protocol_path)
    if protocol["status"] != "FROZEN_BEFORE_VAL_ABLATION" or \
       sha256_file(OUT / "lr_feature_group_manifest.csv") != protocol["feature_group_manifest_sha256"]:
        raise ValueError("Ablation protocol/feature grouping changed")
    audit = input_audit()
    if audit != protocol["frozen_inputs"]:
        raise ValueError("Frozen input hashes changed after protocol freeze")
    if torch.cuda.is_available():
        raise ValueError("Stage7 must be no-GPU")
    torch.set_num_threads(1)
    f,t,v,vt,fit,dev = load_train_val()
    order = jread(S5B / "feature_order.json")["feature_order"]
    selector = protocol["full_features"]
    prep = joblib.load(S5B / "scaler.pkl")
    full_artifact = joblib.load(S5B / "logistic_model.pkl")
    if prep["feature_order"] != order or full_artifact["model_c"] != selector:
        raise ValueError("Stage5B persisted FIT control recipe mismatch")
    x_fit = transform_features(prep, f.loc[fit])
    x_dev = transform_features(prep, f.loc[dev])
    x_val = transform_features(prep, v)
    idx = lambda names: [order.index(name) for name in names]
    y_fit = t.loc[fit,"class_label"].map(LABEL_TO_INDEX).to_numpy()
    results, decisions = {}, {}
    val_targets = vt[["sample_id","IoU_a","IoU_s","class_label"]].copy()
    full_model = full_artifact["model"]
    full_prob = full_model.predict_proba(x_val[:,idx(selector)])
    full_decision = risk_decisions(full_prob,1.5,.20)
    saved = pd.read_parquet(S5C / "predictions/cera_lr_val.parquet")
    if saved["sample_id"].tolist() != v["sample_id"].tolist() or \
       not np.allclose(full_prob, saved[["P_B","P_N","P_H"]].to_numpy(), rtol=0,atol=1e-12) or \
       not np.array_equal(full_decision,saved["replace_decision"].to_numpy(dtype=bool)):
        raise ValueError("FIT Full control does not replay frozen Stage5C LR")
    results["CERA-Full"] = val_metrics(val_targets,full_decision)
    frozen_lr = jread(S5C / "metrics/cera_lr.json")
    if any(abs(results["CERA-Full"][k]-frozen_lr[k])>1e-9 for k in
           ("acc50","miou","override_rate","harm_rejection")):
        raise ValueError("Full ablation control differs from frozen val metrics")
    decisions["CERA-Full"] = full_decision
    save_prediction("cera_full_control",v["sample_id"],full_prob,full_decision)
    trained = {}
    for name,key in (("CERA-NoSym","no_sym_features"),
                     ("CERA-NoConstraint","no_constraint_features")):
        names = protocol[key]
        model = clone(full_model)
        model.fit(x_fit[:,idx(names)],y_fit)
        prob = model.predict_proba(x_val[:,idx(names)])
        choice = risk_decisions(prob,1.5,.20)
        results[name] = val_metrics(val_targets,choice)
        decisions[name] = choice
        save_prediction(name.lower().replace("-","_"),v["sample_id"],prob,choice)
        path = OUT / f"models/{name.lower().replace('-','_')}.pkl"
        joblib.dump({"model":model,"feature_names":names,"training_split":"FIT_ONLY",
                     "preprocessor_path":str(S5B / 'scaler.pkl')},path)
        trained[name] = {"model_path":str(path),"sha256":sha256_file(path),"features":names}
    dev_prob = full_model.predict_proba(x_dev[:,idx(selector)])
    no_risk_selected,grid = select_no_risk_tau(dev_prob,t.loc[dev].reset_index(drop=True))
    write_csv(OUT / "metrics/no_risk_dev_grid.csv",grid)
    jwrite(OUT / "manifests/no_risk_dev_selection.json",no_risk_selected)
    no_risk_choice = full_prob[:,0] > no_risk_selected["tau"]
    results["CERA-NoRisk"] = val_metrics(val_targets,no_risk_choice)
    decisions["CERA-NoRisk"] = no_risk_choice
    save_prediction("cera_no_risk",v["sample_id"],full_prob,no_risk_choice)
    ccda = jread(S5C / "metrics/baseline_ccda.json")
    results["CCDA"] = {**ccda,"conflict_decision_accuracy":val_metrics(
        val_targets,vt["ccda_replace"].to_numpy(dtype=bool))["conflict_decision_accuracy"],
        "balanced_conflict_accuracy":(ccda["benefit_recall"]+ccda["harm_rejection"])/2}
    # Frozen train5 shard: fit its own scaler and LR, with no DEV/val policy selection.
    f5 = pd.read_parquet(S5A / "shards/shard_00_000_005/training_features.parquet")
    t5 = pd.read_parquet(S5A / "shards/shard_00_000_005/training_targets.parquet")
    if len(f5)!=6049 or f5["image_id"].nunique()!=849 or not f5["sample_id"].equals(t5["sample_id"]):
        raise ValueError("Train5 shard identity mismatch")
    subset = f[f["sample_id"].isin(f5["sample_id"])]["sample_id"]
    if set(subset)!=set(f5["sample_id"]) or len(subset)!=6049:
        raise ValueError("Train5 not exact subset of train10")
    prep5 = fit_preprocessor(f5,order,BINARY_FEATURES)
    x5 = transform_features(prep5,f5)
    model5 = clone(full_model)
    model5.fit(x5[:,idx(selector)],t5["class_label"].map(LABEL_TO_INDEX).to_numpy())
    x5_val = transform_features(prep5,v)
    p5 = model5.predict_proba(x5_val[:,idx(selector)])
    d5 = risk_decisions(p5,1.5,.20)
    m5 = val_metrics(val_targets,d5)
    joblib.dump({"model":model5,"feature_names":selector},OUT / "models/cera_lr_train5.pkl")
    joblib.dump(prep5,OUT / "models/train5_scaler.pkl")
    save_prediction("cera_lr_train5",v["sample_id"],p5,d5)
    ten = jread(S5D / "final_refit_val_sanity.json")["all_metrics"]
    # The 10% value comes from the previously frozen Stage5D full10 val sanity.
    data_scale = {"5%":{"train_images":849,"train_expressions":6049,"metrics":m5,
                        "source":"Stage7 full shard00 train, frozen policy"},
                  "10%":{"train_images":1699,"train_expressions":12146,"metrics":ten,
                         "source":"Stage5D frozen full10 refit val sanity"},
                  "ten_minus_five_acc50_pp":ten["acc50"]-m5["acc50"],
                  "ten_minus_five_miou_pp":ten["miou"]-m5["miou"],
                  "ten_minus_five_harm_rejection":ten["harm_rejection"]-m5["harm_rejection"]}
    jwrite(OUT / "metrics/data_scale.json",data_scale)
    jwrite(OUT / "metrics/ablation_val.json",results)
    jwrite(OUT / "manifests/trained_models.json",trained)
    # Development val inference, clustered by image: supportive, not independent test inference.
    stats = {}
    image_ids=vt["image_id"].to_numpy()
    a=val_targets["IoU_a"].to_numpy(); s=val_targets["IoU_s"].to_numpy()
    full_iou=np.where(full_decision,s,a)
    for name in ("CERA-NoSym","CERA-NoRisk","CERA-NoConstraint"):
        other=np.where(decisions[name],s,a)
        stats[name]={"acc50":paired_image_bootstrap_ci(full_iou>=.5,other>=.5,image_ids,
            resamples=10000,seed=3407),"miou":paired_image_bootstrap_ci(full_iou,other,image_ids,
            resamples=10000,seed=3407)}
    jwrite(OUT / "metrics/ablation_statistics.json",stats)
    # Capacity and post-feature CPU overhead: frozen FIT scaler/model, MLP seed 3407.
    checkpoint=S5B / "checkpoints/cera_seed3407.pt"
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    mlp=CeraNet(payload["input_dim"]).to("cpu")
    mlp.load_state_dict(payload["state_dict"]);mlp.eval()
    mlp_params=sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    lr_params=int(full_model.coef_.size+full_model.intercept_.size)
    with torch.inference_mode():
        def mlp_run():
            x=transform_features(prep,v)
            logits,_=mlp(torch.as_tensor(x,dtype=torch.float32))
            p=torch.softmax(logits,dim=1).numpy()
            return risk_decisions(p,1.0,.15)
        latency={"CCDA":bench(lambda:ccda_decisions(v),len(v)),
                 "CERA-LR":bench(lambda:risk_decisions(
                     full_model.predict_proba(transform_features(prep,v)[:,idx(selector)]),1.5,.20),len(v)),
                 "CERA-MLP":bench(mlp_run,len(v))}
    complexity={"CCDA":{"input_dim":4,"trainable_params":0,
                 "artifact_size_bytes":(ROOT / 'src/ccda.py').stat().st_size,
                 "artifact_kind":"rule source code", "latency":latency["CCDA"]},
                "CERA-LR":{"input_dim":26,"trainable_params":lr_params,
                 "artifact_size_bytes":(S5B / 'logistic_model.pkl').stat().st_size,
                 "artifact_kind":"Stage5B FIT LR pickle", "latency":latency["CERA-LR"]},
                "CERA-MLP":{"input_dim":53,"trainable_params":mlp_params,
                 "artifact_size_bytes":checkpoint.stat().st_size,
                 "artifact_kind":"Stage5B seed3407 checkpoint", "latency":latency["CERA-MLP"]}}
    jwrite(OUT / "metrics/complexity.json",complexity)
    # Write paper tables from frozen val/test reports and newly completed val ablations.
    make_tables(results,data_scale,complexity,stats)
    write_docs(results,data_scale,complexity,stats,no_risk_selected)
    after=input_audit()
    if after!=audit or sha256_file(FROZEN_TEST)!=protocol["frozen_inputs"]["test_result_hash"]:
        raise ValueError("Frozen source/test results changed during Stage7")
    expected_test={"testA":"4dec7f4b6c9296c5e65b849b8eb8636d1f39a3fbde0257bca2d738a3b5000093",
                   "testB":"36431066683fbf8443a970a0bdb6f9c63e7995f1db5bdf9604a8b7ed72a3a9af"}
    if after["test_prediction_hashes"]!=expected_test:
        raise ValueError("Frozen test prediction hash changed")
    jwrite(OUT / "manifests/final_audit.json",{"status":"PASS","GPU_USED":"NO",
        "TEST_PHASE":"CLOSED","NEW_TEST_INFERENCE":"NO","TEST_RETUNING":"NO",
        "TEST_RESULT_FILES_MODIFIED":"NO","TEST_RESULT_CONSISTENCY":"PASS",
        "ABLATION_PROTOCOL":"PASS","ABLATION_STATISTICS":"PASS",
        "protocol_sha256":sha256_file(protocol_path),"source_hashes_unchanged":True})
    print(json.dumps({"status":"PASS","metrics":{k:{"acc50":v["acc50"],
        "miou":v["miou"],"harm_rejection":v["harm_rejection"]} for k,v in results.items()},
        "no_risk_tau":no_risk_selected["tau"],"data_scale":data_scale,
        "params":{"LR":lr_params,"MLP":mlp_params},
        "latency_us":{"LR":latency["CERA-LR"]["p50_us_per_sample"],
                      "MLP":latency["CERA-MLP"]["p50_us_per_sample"]}}))


def make_tables(results,scale,complexity,stats):
    with MASTER.open(encoding="utf-8") as stream: master=list(csv.DictReader(stream))
    main=[{k:r[k] for k in ("Split","Method","Acc25","Acc50","Acc75","mIoU")} for r in master]
    write_csv(OUT / "tables/table_main_results.csv",main)
    frozen=jread(FROZEN_TEST)
    for split in ("testA","testB"):
        for method,key in (("CCDA","CCDA"),("CERA-LR","CERA-LR")):
            source=frozen[split]["metrics"][key]
            direct=next(r for r in main if r["Split"]==split and r["Method"]==method)
            if any(float(direct[col])!=source[src] for col,src in
                   (("Acc25","acc25"),("Acc50","acc50"),("Acc75","acc75"),("mIoU","miou"))):
                raise ValueError("Stage6D master/test frozen result discrepancy")
    risk=[]
    for split in ("testA","testB"):
        for method in ("CCDA","CERA-LR"):
            m=frozen[split]["metrics"][method]
            risk.append({"Split":split,"Method":method,"Corrected":m["corrected"],
                "Induced":m["induced"],"NetGain":m["net_gain"],
                "OverrideRate":m["override_rate"],"BenefitRecall":m["benefit_recall"],
                "HarmRejection":m["harm_rejection"],
                "CorrectionPrecision":m["correction_precision"],
                "ConflictDecisionAccuracy":frozen[split]["conflict"][method]["decision_accuracy"]})
    write_csv(OUT / "tables/table_risk_analysis.csv",risk)
    abl=[]
    for name in ("CCDA","CERA-NoConstraint","CERA-NoSym","CERA-NoRisk","CERA-Full"):
        m=results[name]
        abl.append({"Method":name,"Acc50":m["acc50"],"mIoU":m["miou"],
          "Corrected":m["corrected"],"Induced":m["induced"],"NetGain":m["net_gain"],
          "BenefitRecall":m["benefit_recall"],"HarmRejection":m["harm_rejection"],
          "OverrideRate":m["override_rate"],"CorrectionPrecision":m["correction_precision"],
          "ConflictDecisionAccuracy":m["conflict_decision_accuracy"],
          "BalancedConflictAccuracy":m["balanced_conflict_accuracy"],
          "Source":"Stage5C frozen" if name in ("CCDA","CERA-Full") else "Stage7 FIT-only; val once"})
    write_csv(OUT / "tables/table_ablation_val.csv",abl)
    mlp=jread(S5C / "metrics/cera_mlp_summary.json")
    frozen_lr=jread(S5C / "metrics/cera_lr.json")
    capacity=[]
    for name,m in (("LR",frozen_lr),("MLP",{k:v["mean"] for k,v in mlp.items()})):
        c=complexity["CERA-LR" if name=="LR" else "CERA-MLP"]
        capacity.append({"Method":name,"Acc50":m["acc50"],"mIoU":m["miou"],
            "Induced":m["induced"],"HarmRejection":m["harm_rejection"],
            "OverrideRate":m["override_rate"],"Params":c["trainable_params"],
            "CPU_us_per_sample_p50":c["latency"]["p50_us_per_sample"],
            "Source":"Stage5C development val; MLP 3-seed mean"})
    write_csv(OUT / "tables/table_arbitrator_capacity.csv",capacity)
    scale_rows=[]
    for name in ("5%","10%"):
        d=scale[name];m=d["metrics"]
        scale_rows.append({"TrainFraction":name,"TrainImages":d["train_images"],
            "TrainExpressions":d["train_expressions"],"Acc50":m["acc50"],
            "mIoU":m["miou"],"BenefitRecall":m["benefit_recall"],
            "HarmRejection":m["harm_rejection"],"Induced":m["induced"],
            "Source":d["source"]})
    write_csv(OUT / "tables/table_data_scale.csv",scale_rows)
    comp=[]
    for name,c in complexity.items():
        comp.append({"Method":name,"InputDim":c["input_dim"],
            "TrainableParams":c["trainable_params"],
            "ArtifactSizeBytes":c["artifact_size_bytes"],
            "PostFeatureLatency_us_per_sample_p50":c["latency"]["p50_us_per_sample"],
            "PostFeatureLatency_us_per_sample_p95":c["latency"]["p95_us_per_sample"],
            "ArtifactKind":c["artifact_kind"]})
    write_csv(OUT / "tables/table_complexity.csv",comp)
    write_csv(OUT / "figdata/figure_risk_tradeoff_data.csv",[
        {"Split":r["Split"],"Method":r["Method"],"BenefitRecall":r["BenefitRecall"],
         "HarmRejection":r["HarmRejection"],"OverrideRate":r["OverrideRate"],
         "NetGain":r["NetGain"]} for r in risk])
    write_csv(OUT / "figdata/figure_data_scale.csv",scale_rows)


def write_docs(results,scale,complexity,stats,tau):
    docs=ROOT / "docs"
    j= lambda x: json.dumps(x,ensure_ascii=False,indent=2)
    (docs / "07_ablation_results.md").write_text(
        "# Stage7 FIT-only ablation on RefCOCO val\n\nFull control replays the frozen Stage5C "
        "FIT-only LR. NoSym/NoConstraint use the same FIT split, preprocessing and recipe. "
        f"NoRisk tau_B={tau['tau']:.2f} was selected on DEV. Val was evaluated once.\n\n"+
        j(results)+"\n",encoding="utf-8")
    (docs / "07_data_scale.md").write_text(
        "# Train 5% versus 10%\n\nBoth use the frozen 26 columns, LR recipe, λ=1.5 and "
        "τ=.20. The 5% model is fitted on shard00; the 10% figure is the previously frozen "
        "Stage5D full10 val sanity. Only the examined 5–10% range is described.\n\n"+
        j(scale)+"\n",encoding="utf-8")
    (docs / "07_complexity.md").write_text(
        "# Arbitration complexity\n\n100 warmups and 100 CPU repeats over 10,834 val "
        "feature rows. Timings include feature preprocessing for LR/MLP and the arbitration "
        "decision, but exclude GDINO, Qwen, disk I/O and feature extraction. MLP timing uses "
        "the frozen seed3407 checkpoint; val quality uses the frozen 3-seed mean. The CCDA "
        "artifact size denotes rule source code.\n\n"+j(complexity)+"\n",encoding="utf-8")
    (docs / "07_statistical_analysis.md").write_text(
        "# Development-val ablation statistics\n\n10,000 image-clustered bootstrap draws, "
        "seed3407, comparing Full minus each ablation. RefCOCO val was used during method "
        "development; these intervals are supportive and are not independent test inference.\n\n"+
        j(stats)+"\n",encoding="utf-8")
    (docs / "07_claim_boundaries.md").write_text(
        "# Claim boundaries\n\nThe frozen Stage6C results support improvement over frozen CCDA "
        "on RefCOCO testA/testB and a substantial reduction in harmful replacement. "
        "Stage7 val ablations may support a contribution from symmetric evidence and the "
        "risk term only to the extent that the reported deltas and intervals do. LR offers "
        "strong observed risk control with measured post-feature overhead. No SOTA, universal "
        "superiority, all-dataset generalization, or full-data saturation claim is supported. "
        "Val is development data, and testA/testB are closed to new model selection.\n",
        encoding="utf-8")
    (docs / "07_experiment_section_outline.md").write_text(
        "# Experiment section outline\n\n"
        "## 4.1 Experimental Setup\nData splits, frozen 53D/26D schema, CPU protocols.\n\n"
        "## 4.2 Anchor / Raw / CCDA comparison\nOfficial split-specific main table.\n\n"
        "## 4.3 Risk-aware arbitration\nCorrected, induced, overrides, conflict decisions.\n\n"
        "## 4.4 Ablations\nFull FIT control, NoSym, NoRisk, NoConstraint on val.\n\n"
        "## 4.5 Data efficiency\nFrozen shard00 5% versus full10 val sanity.\n\n"
        "## 4.6 Complexity and runtime\nParameter counts, artifact sizes, post-feature latency.\n\n"
        "## 4.7 Statistical evidence\nOfficial test CIs and supportive val ablation CIs, labeled separately.\n\n"
        "## 4.8 Discussion\nRisk/benefit tradeoffs and evidence limits.\n",
        encoding="utf-8")
    (docs / "07_stage_summary.md").write_text(
        "# Stage7 summary\n\nCPU-only FIT/DEV/val ablations are in outputs/stage7. "
        "Frozen test results were read only to populate paper tables. The final primary method "
        "remains CERA-LR, test phase remains closed, and no test inference or retuning occurred. "
        "Stage8 paper writing is a proposed next stage and was not started.\n",
        encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-protocol",action="store_true")
    modes.add_argument("--run",action="store_true")
    args=parser.parse_args()
    freeze_protocol() if args.freeze_protocol else run()
