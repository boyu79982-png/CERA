"""One authorized local CPU run. Refuses to overwrite or rerun existing artifacts."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import ast, csv, hashlib, json, pathlib, platform, subprocess, sys, time, warnings, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
sys.path.insert(0, r"D:\codex_temp\cera_cpu_runtime")
sys.path.insert(0, r"D:\codex_temp\cera_parquet_reader")
import joblib, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, scipy, sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._logistic import _check_multi_class
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = pathlib.Path(r"D:\CERA_local_writing\experiments\score_only_9d_cpu")
INPUT = ROOT / "inputs"
OUT = ROOT / "run_01"
SOURCE = pathlib.Path(r"C:\Users\57017\Documents\ChatGPT\论文2\paper2_derecg")
R2ROOT = pathlib.Path(r"D:\CERA_local_writing\drafts\cera_academic_revision_20261010")
EXPECTED = json.loads(r'''[{"relative_path":"outputs/stage5a10_fast/merged/training_features_10pct.parquet","size":796892,"sha256":"9862eab40d1fcb3e8c04dc0210e1236169d42e3a20a0fd0d906e16264676feaa"},{"relative_path":"outputs/stage5a10_fast/merged/training_targets_10pct.parquet","size":359838,"sha256":"9226cab2ea8a1c9b9913963f3b5338ac981b09547663dc7023d28785b9f43ec7"},{"relative_path":"outputs/stage5c/val_features_complete.parquet","size":718098,"sha256":"54a39e6c0ebd0a968287b1d934b39cc3d96a1640d4a2a5512d28e799e60060ac"},{"relative_path":"outputs/stage5c/frozen_val_evaluation_trace.parquet","size":736195,"sha256":"55ab5735628d494fb8b6ba1613605bf10244136bdafd993a092196298193ba4c"},{"relative_path":"outputs/stage5b10/internal_image_split.json","size":12227,"sha256":"5b6f12182c4059a39e4c68825581c5235194b6a073d066ee930f4b18d5a9df1c"},{"relative_path":"outputs/stage5b10/feature_order.json","size":1136,"sha256":"7058c251655c7292133e35580eeb27c13be46691ed6a999889dd3a15128cf578"},{"relative_path":"outputs/stage5b10/feature_schema.json","size":3666,"sha256":"fe746c83b7be39541c8b10f75f2201e52f899dc1fa0673f5cd34abfb47427efe"},{"relative_path":"outputs/stage5b10/decision_config.json","size":1141,"sha256":"ba3984e2b24b70c9446de886ea914dc0bb3d2afe737a901a031d50e3596a563e"},{"relative_path":"outputs/stage5b10/metrics/logistic_baseline_dev.json","size":796,"sha256":"a3a0fe98a6b846a71ed6f197cc55142dc78d19087ec33001e7ef8f11c5cbe283"},{"relative_path":"outputs/stage7/predictions_val/cera_full_control.parquet","size":383195,"sha256":"7e0b9c3e9ee6f49004cce2466fa7a642db2bc3b9bfe7b2a17cf9a710ee42d161"},{"relative_path":"outputs/stage5a10_fast/merged/merge_manifest.json","size":676,"sha256":"6b5b2587911a997bca12a9813426c55b950852d9f17370df5296f25862809026"},{"relative_path":"outputs/stage5d_final/training_recipe.json","size":725,"sha256":"70df1256c749720689af36d26dad9420bc6db1e35916231634a0062784127010"},{"relative_path":"outputs/stage7/manifests/ablation_protocol.json","size":6258,"sha256":"fba3ddbc7090be6095406ef9e5b5e0b0b3c7096698c879f08e241505dd4cee0b"},{"relative_path":"outputs/stage7/manifests/output_manifest.json","size":6039,"sha256":"c104d5f686481c7313828068031b5eff1ffcf20007c23b2e0f59d63799eb29db"},{"relative_path":"docs/05a10_feature_manifest.csv","size":14479,"sha256":"7282998d33413f80804114cf8692c8033ae14912a933d2d9ecff0ba310e12d29"},{"relative_path":"outputs/stage5b10/scaler.pkl","size":3825,"sha256":"840705b9b939e45623410d4336b67bd7734096c9eb3a6aa1be8bc0b3584566bc"},{"relative_path":"outputs/stage5b10/logistic_model.pkl","size":2070,"sha256":"7983cad3ce59446f68594d7cb9b89bf3f32f1572535d0431c41731f4818b0391"}]''')
SOURCE_HASHES = json.loads(r'''{"src/stage5b10_core.py":"121da427702ec3d1722bcae7a3b6e634c9f70409915ed5f1ab68b0b683dbab5b","src/stage5b10_pipeline.py":"de1beb1805b7c8a481b070f65e02954ca6ce4b595076518069ca66a804b81f0c","scripts/run_stage5b10_nogpu.py":"37c94c7364b8f94de7594a530335d45d56e733877ab6bfce9de08977c1c8181b","scripts/audit_feature_leakage.py":"42e400465bb7e07292784df0b6b7f38d3347c739b5d00009e06fda30c9630c64","scripts/stage4_build_features.py":"47bf7edd142deee5b0e529d4475bc59207a445d29096424a6db9637e1ad670a5","scripts/evaluate_stage5c_val_frozen.py":"412570f6e3badf96a9ef9a7d9ec29eef9874167695394b95f42ea57fb27551ff","scripts/run_stage7_nogpu_ablations.py":"dac0ceda53d2297d923fb66bed438d07faf14278515f7561dc6a19b292be2de3"}''')
NINE = ["head_valid", "S_st", "anchor_detector_conf", "old_margin", "box_pair_iou",
        "structured_detector_conf", "expression_length", "num_attributes", "num_relations"]
NINE_IDX = [3, 9, 6, 10, 11, 7, 0, 1, 2]
R2_HASHES = {
    "CERA_\u4e2d\u6587\u8bba\u6587_\u5b66\u672f\u4fee\u8ba2_R2.docx": "83b530670c158268285cb0cc437501b3cf3b6b3330cf7e72c90a19f6a7a1954d",
    "CERA_\u4e2d\u6587\u8bba\u6587_\u5b66\u672f\u4fee\u8ba2_R2_QC.pdf": "367d7bf9330621d1498e2daa69fd6f11fe7fa5dd567271b02f43bcb9b3447b52"}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def stamp(): return datetime.now(timezone.utc).isoformat()
def jread(p): return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
def jwrite(name, data):
    p = OUT / name
    assert not p.exists(), "Refuse to overwrite " + str(p)
    p.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def log(event, **data): print(json.dumps({"event":event, **data}, ensure_ascii=True), flush=True)
def read_parquet(rel): return pq.ParquetFile(INPUT / rel).read(use_threads=False).to_pandas()
def literal(rel, name):
    for n in ast.parse((SOURCE / rel).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise ValueError(name)
def source_functions(rel, names, namespace):
    tree = ast.parse((SOURCE / rel).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / rel), "exec"), namespace)

def main():
    assert sklearn.__version__ == "1.7.2", "Do not silently change sklearn objective"
    assert not any(p.name != pathlib.Path(__file__).name for p in OUT.iterdir()), "Run already started; do not retrain"
    identities = {}
    for item in EXPECTED:
        p = INPUT / item["relative_path"]
        assert p.stat().st_size == item["size"] and sha(p) == item["sha256"], str(p)
        identities[item["relative_path"]] = {"sha256":sha(p), "size_bytes":p.stat().st_size}
    for rel, h in SOURCE_HASHES.items(): assert sha(SOURCE / rel) == h, rel
    for name,h in R2_HASHES.items(): assert sha(R2ROOT / name) == h, name
    assert sha(ROOT / "cache_identity_preflight_report.md") == "bb85793d7855fe416714619cf964d4713ef15e6956a45c0d820b1482a9d90f9a"
    order = jread(INPUT / "outputs/stage5b10/feature_order.json")["feature_order"]
    schema = jread(INPUT / "outputs/stage5b10/feature_schema.json")
    split = jread(INPUT / "outputs/stage5b10/internal_image_split.json")
    recipe = jread(INPUT / "outputs/stage5d_final/training_recipe.json")
    protocol = jread(INPUT / "outputs/stage7/manifests/ablation_protocol.json")
    assert len(order)==53 and order==schema["feature_order"]
    assert order==list(literal("scripts/stage4_build_features.py","FEATURES"))+["has_attribute","has_relation"]
    assert [order[i] for i in NINE_IDX] == NINE
    assert protocol["train_policy"] == "FIT_IMAGES_ONLY; no refit after DEV threshold selection"
    assert recipe["source_code_sha256"] == sha(SOURCE / "scripts/run_stage5b10_nogpu.py")
    namespace = {"np":np, "SimpleImputer":SimpleImputer, "StandardScaler":StandardScaler}
    for name in ("BINARY_FEATURES","LAMBDA_GRID","TAU_GRID"):
        namespace[name] = literal("src/stage5b10_core.py",name)
    source_functions("src/stage5b10_core.py",
                     ["_numeric","fit_preprocessor","transform_features","arbitration_metrics",
                      "risk_decisions","ccda_decisions","select_common_decision"], namespace)
    source_functions("scripts/run_stage7_nogpu_ablations.py",["val_metrics"],namespace)
    source_functions("src/stage5b10_pipeline.py",["paired_image_bootstrap_ci"],namespace)
    fit_preprocessor, transform, metrics, risk, ccda, select, boot = (
        namespace[k] for k in ("fit_preprocessor","transform_features","val_metrics","risk_decisions",
                               "ccda_decisions","select_common_decision","paired_image_bootstrap_ci"))
    # Scientific sanity checks before any real-data fit or prediction.
    fixture=pd.DataFrame({"IoU_a":[0,.8,.8,0],"IoU_s":[.8,0,.9,0],
                          "class_label":["BENEFIT","HARM","NEUTRAL","NEUTRAL"]})
    m=metrics(fixture,np.array([True,True,False,False]))
    assert (m["corrected"],m["induced"],m["net_gain"],m["acc50"],m["miou"],
            m["harm_rejection"],m["conflict_decision_accuracy"])==(1,1,0,50.,40.,0.,.5)
    assert not risk(np.array([[.2,.8,0.]]),1.5,.2)[0], "Strict greater than, never >="
    z=boot(np.ones(4),np.zeros(4),[1,1,2,2],resamples=10000,seed=3407)
    assert z["ci95_pp"]==[100.,100.] and z["difference_pp"]==100.
    f=read_parquet("outputs/stage5a10_fast/merged/training_features_10pct.parquet")
    t=read_parquet("outputs/stage5a10_fast/merged/training_targets_10pct.parquet")
    v=read_parquet("outputs/stage5c/val_features_complete.parquet")
    vt=read_parquet("outputs/stage5c/frozen_val_evaluation_trace.parquet")
    full=read_parquet("outputs/stage7/predictions_val/cera_full_control.parquet")
    assert len(f)==len(t)==12146 and len(v)==len(vt)==len(full)==10834
    assert f.columns.tolist()==v.columns.tolist()==["sample_id","image_id","ref_id","sentence_id"]+order
    assert f.sample_id.tolist()==t.sample_id.tolist()
    assert v.sample_id.tolist()==vt.sample_id.tolist()==full.sample_id.tolist()
    assert v.image_id.tolist()==vt.image_id.tolist()
    for frame in (f,t,v,vt,full): assert not frame.sample_id.isna().any() and not frame.sample_id.duplicated().any()
    assert not set(f.image_id) & set(v.image_id)
    sets={k:set(ids) for k,ids in split["images"].items()}
    assert not sets["FIT"]&sets["DEV"] and not sets["FIT"]&sets["INTERNAL_HOLDOUT"] and not sets["DEV"]&sets["INTERNAL_HOLDOUT"]
    assert set.union(*sets.values())==set(f.image_id)
    fit_mask=f.image_id.isin(sets["FIT"]).to_numpy()
    dev_mask=f.image_id.isin(sets["DEV"]).to_numpy()
    hold_mask=f.image_id.isin(sets["INTERNAL_HOLDOUT"]).to_numpy()
    assert (int(fit_mask.sum()),int(dev_mask.sum()),int(hold_mask.sum()))==(9694,1219,1233)
    for frame in (t,vt):
        a,s=frame.IoU_a.to_numpy(float),frame.IoU_s.to_numpy(float)
        assert np.isfinite(a).all() and np.isfinite(s).all()
        assert np.all((a>=0)&(a<=1)&(s>=0)&(s<=1))
        labels=np.where((a<.5)&(s>=.5),"BENEFIT",np.where((a>=.5)&(s<.5),"HARM","NEUTRAL"))
        assert np.array_equal(labels,frame.class_label.to_numpy())
    with (INPUT/"docs/05a10_feature_manifest.csv").open(encoding="utf-8",newline="") as stream:
        rules=list(csv.DictReader(stream))
    assert [r["feature_name"] for r in rules]==order
    assert all(r["uses_gt"]=="NO" and r["available_at_inference"]=="YES" for r in rules)
    assert not set(order)&{"GT","IoU_a","IoU_s","class_label","anchor_correct","structured_correct","delta_iou"}
    full_m=metrics(vt,full.replace_decision.to_numpy(bool))
    expected_full={"acc50":53.2952,"miou":53.1858,"harm_rejection":92.7900,"induced":69}
    assert round(full_m["acc50"],4)==expected_full["acc50"]
    assert round(full_m["miou"],4)==expected_full["miou"]
    assert round(100*full_m["harm_rejection"],4)==expected_full["harm_rejection"]
    assert full_m["induced"]==69
    assert np.array_equal(full.replace_decision.to_numpy(bool),vt.lr_replace.to_numpy(bool))
    log("PREFLIGHT_AND_FULL_REAGGREGATION_PASS",full=full_m)
    cpu=json.loads(subprocess.check_output(["powershell","-NoProfile","-Command",
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],text=True))
    env={"python":platform.python_version(),"platform":platform.platform(),"cpu":cpu,
         "numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,
         "pandas":pd.__version__,"pyarrow":pa.__version__,"joblib":joblib.__version__,
         "cpu_threads":1,"cuda_visible_devices":"","torch_imported":False}
    with threadpool_limits(limits=1):
        env["native_threadpools"]=threadpool_info()
        begin=time.perf_counter()
        prep=fit_preprocessor(f.loc[fit_mask],order)
        prep_seconds=time.perf_counter()-begin
        # The historical object is trusted only after all source and artifact hash gates above.
        old_prep=joblib.load(INPUT/"outputs/stage5b10/scaler.pkl")
        assert prep["feature_order"]==old_prep["feature_order"]==order
        assert prep["continuous"]==old_prep["continuous"]==schema["continuous_scaled"]
        assert prep["binary"]==old_prep["binary"]==schema["binary_unscaled"]
        comparisons={}
        for key in ("imputer","scaler","binary_imputer"):
            attrs=("statistics_",) if key!="scaler" else ("mean_","scale_","var_","n_samples_seen_")
            comparisons[key]={a:bool(np.array_equal(getattr(prep[key],a),getattr(old_prep[key],a))) for a in attrs}
        x_fit=transform(prep,f.loc[fit_mask]); x_dev=transform(prep,f.loc[dev_mask])
        comparisons["FIT_output_bitwise_equal"]=bool(np.array_equal(x_fit,transform(old_prep,f.loc[fit_mask])))
        comparisons["DEV_output_bitwise_equal"]=bool(np.array_equal(x_dev,transform(old_prep,f.loc[dev_mask])))
        assert all(all(vv.values()) if isinstance(vv,dict) else vv for vv in comparisons.values()), comparisons
        assert x_fit.shape==(9694,53) and x_dev.shape==(1219,53)
        mapping=recipe["class_mapping"]
        assert mapping=={"BENEFIT":0,"NEUTRAL":1,"HARM":2}
        y=t.loc[fit_mask,"class_label"].map(mapping).to_numpy(np.int64)
        assert set(y)=={0,1,2}
        params=recipe["lr_estimator_params"]
        model=LogisticRegression(**params)
        assert model.get_params()==params
        assert _check_multi_class("auto",params["solver"],3)=="multinomial"
        begin=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x_fit[:,NINE_IDX],y)  # The only LR fit in this run.
        fit_seconds=time.perf_counter()-begin
        warnings_record=[{"category":w.category.__name__,"message":str(w.message)} for w in caught]
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught), "LR did not converge; no retuning permitted"
        assert model.classes_.tolist()==[0,1,2]
        assert model.coef_.shape==(3,9) and model.intercept_.shape==(3,) and model.n_features_in_==9
        p_dev=model.predict_proba(x_dev[:,NINE_IDX])
        assert np.isfinite(p_dev).all() and np.allclose(p_dev.sum(axis=1),1,rtol=0,atol=1e-12)
        dev_t=t.loc[dev_mask].reset_index(drop=True)
        ccda_dev=ccda(f.loc[dev_mask])
        ccda_dev_metrics=metrics(dev_t,ccda_dev)
        frozen_decision=jread(INPUT/"outputs/stage5b10/decision_config.json")
        assert abs(ccda_dev_metrics["harm_rejection"]-frozen_decision["lr_selection"]["ccda_harm_rejection"])<1e-12
        assert abs(ccda_dev_metrics["acc50"]-frozen_decision["lr_selection"]["ccda_acc50"])<1e-12
        chosen,source_grid=select([p_dev,p_dev,p_dev],dev_t,ccda_dev)
        grid=[]
        for i,row in enumerate(source_grid):
            m=metrics(dev_t,risk(p_dev,row["lambda"],row["tau"]))
            grid.append({"grid_index":i,"lambda":row["lambda"],"tau":row["tau"],**m,
                         "risk_feasible":m["harm_rejection"]>=ccda_dev_metrics["harm_rejection"]-1e-12})
        assert len(grid)==32 and len({(r["lambda"],r["tau"]) for r in grid})==32
        eligible=[r for r in grid if r["risk_feasible"]]
        for r in grid: r["selected"]=False
        if eligible:
            best=max(eligible,key=lambda r:(r["acc50"],-r["induced"],-r["override_rate"]))
            assert (best["lambda"],best["tau"])==(chosen["lambda"],chosen["tau"])
            best["selected"]=True
        with (OUT/"dev_risk_grid.csv").open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)
        if not eligible:
            jwrite("model_manifest.json",{"status":"NO_FEASIBLE_POLICY","dev_ccda":ccda_dev_metrics,"env":env})
            raise RuntimeError("NO_FEASIBLE_POLICY; val comparison is forbidden")
        assert chosen["risk_constraint"]=="MET"
        frozen_at=stamp()
        artifact=OUT/"score_only_9d_model.pkl"
        joblib.dump({"model":model,"preprocessor":prep,"feature_names":NINE,"indices_in_53":NINE_IDX,
                     "class_mapping":mapping,"fitted_image_ids":sorted(sets["FIT"]),
                     "policy":{"lambda":best["lambda"],"tau":best["tau"],"operator":">"}},artifact)
        manifest={"name":"CERA-ExistingScore-LR (9D)","status":"FROZEN_BEFORE_NEW_9D_VAL_EVALUATION",
                  "frozen_at_utc":frozen_at,"env":env,"input_identities":identities,"source_hashes":SOURCE_HASHES,
                  "execution_script_sha256":sha(__file__),"training_fit_images":1359,"training_fit_expressions":9694,
                  "dev_images":170,"dev_expressions":1219,"holdout_used":False,"fit_count":1,
                  "fit_class_counts":{label:int((y==index).sum()) for label,index in mapping.items()},
                  "class_mapping":mapping,"classes_":model.classes_.tolist(),"coefficient_shape":list(model.coef_.shape),
                  "intercept_shape":list(model.intercept_.shape),"input_dim":9,"parameter_count":30,
                  "estimator_params":model.get_params(),"objective":"multinomial softmax logistic regression, balanced class weighting",
                  "feature_names":NINE,"indices_in_53":NINE_IDX,"preprocessing_order":order,
                  "preprocessor_fit_scope":"FIT ONLY, full 53 columns before fixed 9-column selection",
                  "preprocessor_fit_seconds":prep_seconds,"lr_fit_seconds":fit_seconds,
                  "n_iter_":model.n_iter_.tolist(),"converged":True,"warnings":warnings_record,
                  "preprocessor_comparison_to_original_FIT":comparisons,
                  "original_fit_scaler_sha256":identities["outputs/stage5b10/scaler.pkl"]["sha256"],
                  "model_sha256":sha(artifact),"dev_grid_sha256":sha(OUT/"dev_risk_grid.csv"),
                  "dev_policy":{"lambda":best["lambda"],"tau":best["tau"],"operator":">","risk_constraint":"MET",
                      "dev_selected_metrics":{k:v for k,v in best.items() if k not in ("selected","grid_index","risk_feasible")},
                      "ccda_reference":ccda_dev_metrics,"eligible_policy_count":len(eligible),
                      "rank":"Acc50 descending; Induced ascending; Override Rate ascending; stable lambda-then-tau grid order"},
                  "auxiliary_policy":{"lambda":1.5,"tau":.20,"status":"PREDECLARED_DIAGNOSTIC_NOT_SELECTION_CANDIDATE"},
                  "new_test_data_read":False,"new_val_tuning":False,
                  "study_boundary":"Post hoc supplemental developmental control; historical test results were already known",
                  "r2_before_hashes":R2_HASHES}
        jwrite("model_manifest.json",manifest)
        manifest_hash=sha(OUT/"model_manifest.json")
        log("MODEL_AND_DEV_POLICY_FROZEN",lambda_selected=best["lambda"],tau_selected=best["tau"],
            iterations=model.n_iter_.tolist(),fit_seconds=fit_seconds,dev=best,
            preprocessing=comparisons,manifest_sha256=manifest_hash)
        # First and only call to predict_proba on the new model's val input.
        val_start=stamp()
        x_val=transform(prep,v)
        assert np.array_equal(x_val,transform(old_prep,v)), "Val preprocessing must equal historical FIT-only transform"
        p_val=model.predict_proba(x_val[:,NINE_IDX])
        assert p_val.shape==(10834,3) and np.isfinite(p_val).all()
        assert np.allclose(p_val.sum(axis=1),1,rtol=0,atol=1e-12)
        primary=risk(p_val,best["lambda"],best["tau"])
        aux=risk(p_val,1.5,.20)
        full_decision=full.replace_decision.to_numpy(bool)
        anchor,structured=vt.IoU_a.to_numpy(float),vt.IoU_s.to_numpy(float)
        full_iou=np.where(full_decision,structured,anchor)
        primary_iou=np.where(primary,structured,anchor)
        aux_iou=np.where(aux,structured,anchor)
        result_metrics={"9D_DEV_SELECTED":metrics(vt,primary),"9D_FIXED_FULL_WORKPOINT":metrics(vt,aux),
                        "FROZEN_26D_FULL":full_m,"FROZEN_CCDA":metrics(vt,vt.ccda_replace.to_numpy(bool))}
        paired={
            "acc50":boot(full_iou>=.5,primary_iou>=.5,v.image_id.to_numpy(),resamples=10000,seed=3407),
            "miou":boot(full_iou,primary_iou,v.image_id.to_numpy(),resamples=10000,seed=3407)}
        diffs={}
        for name in ("9D_DEV_SELECTED","9D_FIXED_FULL_WORKPOINT"):
            m=result_metrics[name]
            diffs[name]={"Full_minus_9D_acc50_pp":full_m["acc50"]-m["acc50"],
                         "Full_minus_9D_miou_pp":full_m["miou"]-m["miou"],
                         "Full_minus_9D_induced_count":full_m["induced"]-m["induced"],
                         "Full_minus_9D_harm_rejection_pp":100*(full_m["harm_rejection"]-m["harm_rejection"])}
        output=pd.DataFrame({"sample_id":v.sample_id,"image_id":v.image_id,"ref_id":v.ref_id,
                             "sentence_id":v.sentence_id,"class_label":vt.class_label,
                             "IoU_a":anchor,"IoU_s":structured,"P_B":p_val[:,0],"P_N":p_val[:,1],"P_H":p_val[:,2],
                             "primary_utility":p_val[:,0]-best["lambda"]*p_val[:,2],
                             "primary_replace":primary,"primary_output_iou":primary_iou,"primary_correct":primary_iou>=.5,
                             "aux_utility":p_val[:,0]-1.5*p_val[:,2],"aux_replace":aux,
                             "aux_output_iou":aux_iou,"aux_correct":aux_iou>=.5,
                             "full_replace":full_decision,"full_output_iou":full_iou,"full_correct":full_iou>=.5,
                             "ccda_replace":vt.ccda_replace})
        assert not (OUT/"val_predictions.parquet").exists()
        pq.write_table(pa.Table.from_pandas(output,preserve_index=False),OUT/"val_predictions.parquet")
        # Only read manuscript Table 3 for the old, immutable developmental ablation summaries.
        ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(R2ROOT/"CERA_\u4e2d\u6587\u8bba\u6587_\u5b66\u672f\u4fee\u8ba2_R2.docx") as archive:
            xml=ET.fromstring(archive.read("word/document.xml"))
        old_table=xml.findall(".//w:tbl",ns)[2]
        old_rows=[["".join(x.text or "" for x in c.findall(".//w:t",ns)) for c in row.findall("w:tc",ns)] for row in old_table.findall("w:tr",ns)]
        assert old_rows==[["\u53d8\u4f53","Acc@0.5","mIoU","Harm Rejection","Induced"],
                         ["CCDA","51.9291","51.9847","75.1306","238"],
                         ["CERA-NoConstraint","52.6121","52.5435","83.0721","162"],
                         ["CERA-NoSym","52.7044","52.6210","88.8192","107"],
                         ["CERA-NoRisk","53.3783","53.2674","91.4316","82"],
                         ["CERA-Full","53.2952","53.1858","92.7900","69"]]
        stats={"status":"RESULTS_FROZEN","model_manifest_sha256":manifest_hash,
               "val_prediction_sha256":sha(OUT/"val_predictions.parquet"),"val_evaluation_start_utc":val_start,
               "val_evaluation_end_utc":stamp(),"val_prediction_calls":1,"val_sample_count":10834,
               "image_cluster_count":int(v.image_id.nunique()),"val_development_only":True,
               "metrics":result_metrics,"differences":diffs,"paired_bootstrap":paired,
               "bootstrap_method":"Percentile 95% CI; 10000 image_id clusters draws with replacement; seed 3407; preserve all within-image expressions; expression-weighted ratio in each draw",
               "bootstrap_direction":"Full minus 9D DEV-selected","bootstrap_interval_units":"percentage points",
               "bootstrap_multiplicity":"Two marginal descriptive developmental intervals; no multiplicity correction or independent-test significance claim",
               "old_frozen_ablation_table":{"source":"R2 DOCX Table 3 (read only)","rows":old_rows},
               "old_ablation_not_retrained":True,"holdout_not_evaluated":True,
               "no_gt_input":True,"no_test_read":True,"no_gpu":True,"no_upstream_rerun":True,
               "post_hoc_developmental_baseline":True}
        assert sha(OUT/"model_manifest.json")==manifest_hash
        for item in EXPECTED: assert sha(INPUT/item["relative_path"])==item["sha256"]
        for name,h in R2_HASHES.items(): assert sha(R2ROOT/name)==h
        jwrite("paired_statistics.json",stats)
        log("VAL_AND_PAIRED_BOOTSTRAP_FROZEN",metrics=result_metrics,differences=diffs,paired=paired)
if __name__=="__main__":
    try: main()
    except Exception as error:
        log("BLOCKED",error=type(error).__name__+": "+str(error))
        raise

