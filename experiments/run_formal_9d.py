"""Frozen-protocol, single local CPU train10 refit and once-per-split prediction.

Execution plan: verify identities; sanity-check frozen pure helpers; fit 53D
preprocessing and one 9D LR; save/hash model and pretest manifest; evaluate
testA and testB once each; compute paired statistics; freeze results; stop.
An interrupted run is never automatically retrained or re-predicted. Existing
events/results require human review. --self-check performs synthetic checks only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import ast, csv, hashlib, json, pathlib, platform, re, sys, time, warnings
from datetime import datetime, timezone
sys.dont_write_bytecode = True
sys.path.insert(0, r'D:\codex_temp\cera_cpu_runtime')
sys.path.insert(0, r'D:\codex_temp\cera_parquet_reader')
import joblib, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, scipy, sklearn
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = pathlib.Path(r'D:\CERA_local_writing\experiments\score_only_9d_cpu')
FORMAL = ROOT / 'formal_test_02'
OUT = pathlib.Path(__file__).resolve().parent
TRAIN = ROOT / 'inputs'
CACHE = FORMAL / 'inputs'
SOURCE = pathlib.Path(r'C:\Users\57017\Documents\ChatGPT\论文2\paper2_derecg')
R2 = pathlib.Path(r'D:\CERA_local_writing\drafts\cera_academic_revision_20261010')
PROTECTED_REPORTS = {
    'run_01/experiment_and_review_report.md': 'b5b1d8e934220632c745e187791f0be0fef8760d6d051e879db9b415fe29b062',
    'run_01/artifact_manifest.json': 'a92887c5492d2eb7fa5a56065a86b01b4d76188a3559319d7ad3646f5d4086c7',
    'formal_test_02/experiment_and_review_report.md': 'bff49f55b7e6f4e9a93269b3860c83cfdbe569994c95bae81d823c27d6b9663c',
    'formal_test_02/formal_protocol.json': 'fa7b4291421f2c3b619c8e848fe87401d3460403a225112b64ddd31fc1c57b10',
    'formal_test_02/artifact_manifest.json': '90b364f6c865f69ba703d3fb83b8036a1fd0f866a2c885f785a85d5f11ee1467',
    'formal_test_02/test_cache_recovery_preflight.md': '749ca681081fed37b6faee8768188168ff723fd903baac4ad25c318036f14f78',
}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(name, data):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def event(name, **data):
    record = {'event': name, 'at_utc': stamp(), **data}
    with (OUT / 'execution_events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=True), flush=True)


def parquet(path, columns=None):
    return pq.ParquetFile(path).read(columns=columns, use_threads=False).to_pandas()


def pure_helpers(source_hashes):
    """Extract exact frozen functions without importing torch/project entrypoints."""
    ns = {'np': np, 'SimpleImputer': SimpleImputer, 'StandardScaler': StandardScaler,
          'binomtest': binomtest}
    for rel, names in [
        ('src/stage5b10_core.py', ['_numeric', 'fit_preprocessor', 'transform_features',
                                 'arbitration_metrics', 'risk_decisions']),
        ('scripts/run_stage7_nogpu_ablations.py', ['val_metrics']),
        ('src/stage5b10_pipeline.py', ['exact_mcnemar', 'paired_image_bootstrap_ci']),
    ]:
        assert sha(SOURCE / rel) == source_hashes[rel], rel
        tree = ast.parse((SOURCE / rel).read_text(encoding='utf-8'))
        if rel == 'src/stage5b10_core.py':
            ns['BINARY_FEATURES'] = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == 'BINARY_FEATURES' for t in n.targets))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in funcs} == set(names)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SOURCE / rel), 'exec'), ns)
    return ns


def self_check(ns):
    """Synthetic known-answer checks; no real-data fitting or prediction."""
    truth = pd.DataFrame({'IoU_a': [0., .8, .8, 0.], 'IoU_s': [.8, 0., .9, 0.],
                          'class_label': ['BENEFIT', 'HARM', 'NEUTRAL', 'NEUTRAL']})
    m = ns['val_metrics'](truth, np.array([True, True, False, False]))
    assert (m['corrected'], m['induced'], m['net_gain'], m['acc50'], m['miou'],
            m['harm_rejection'], m['conflict_decision_accuracy']) == (1, 1, 0, 50., 40., 0., .5)
    assert not ns['risk_decisions'](np.array([[.2, .8, 0.]]), 1.5, .2)[0]
    assert ns['risk_decisions'](np.array([[.3, .7, 0.]]), 1.5, .2)[0]
    mc = ns['exact_mcnemar']([True, True, False], [False, False, True])
    assert (mc['first_only'], mc['second_only'], mc['discordant'], mc['p_exact_two_sided']) == (2, 1, 3, 1.)
    z = ns['paired_image_bootstrap_ci'](np.ones(4), np.zeros(4), [1, 1, 2, 2], resamples=10000, seed=3407)
    assert z['difference_pp'] == 100. and z['ci95_pp'] == [100., 100.]


def identities():
    for rel, digest in PROTECTED_REPORTS.items():
        assert sha(ROOT / rel) == digest, rel
    previous = read_json(FORMAL / 'artifact_manifest.json')
    for rel, v in previous['protected_run01_files'].items():
        assert sha(ROOT / 'run_01' / rel) == v['sha256'], rel
    for rel, v in previous['protected_input_identities'].items():
        assert (TRAIN / rel).stat().st_size == v['size_bytes'] and sha(TRAIN / rel) == v['sha256'], rel
    for rel, digest in previous['protected_source_hashes'].items():
        assert sha(SOURCE / rel) == digest, rel
    for rel, v in previous['source_path_evidence'].items():
        assert sha(SOURCE / rel) == v['sha256'], rel
    for name, digest in previous['protected_r2_sha256'].items():
        assert sha(R2 / name) == digest, name
    recovered = {}
    report = (FORMAL / 'test_cache_recovery_preflight.md').read_text(encoding='utf-8')
    for line in report.splitlines():
        cols = [c.strip() for c in line.strip('|').split('|')]
        if len(cols) == 4 and cols[0].startswith('outputs/') and cols[1].isdigit() and re.fullmatch('[0-9a-f]{64}', cols[2]):
            recovered[cols[0]] = {'size_bytes': int(cols[1]), 'sha256': cols[2]}
    assert len(recovered) == 18
    for rel, v in recovered.items():
        assert (CACHE / rel).stat().st_size == v['size_bytes'] and sha(CACHE / rel) == v['sha256'], rel
    return previous, recovered


def main():
    assert OUT == FORMAL / 'run_formal_01'
    protocol = read_json(FORMAL / 'formal_protocol.json')
    prev, cache_ids = identities()
    ns = pure_helpers(prev['protected_source_hashes'])
    self_check(ns)
    assert sklearn.__version__ == protocol['estimator']['sklearn_version'] == '1.7.2'
    if sys.argv[1:] == ['--self-check']:
        print(json.dumps({'self_check': 'PASS', 'identity_check': 'PASS',
                          'fit_calls': 0, 'test_prediction_calls': 0}), flush=True)
        return
    assert not sys.argv[1:]
    assert sorted(p.name for p in OUT.iterdir()) == [pathlib.Path(__file__).name], 'Existing run; no automatic retraining or repeat evaluation'
    event('RUN_STARTED', protocol_sha256=sha(FORMAL / 'formal_protocol.json'), script_sha256=sha(__file__))
    order, nine, indexes = (protocol['preprocessing']['full_feature_order'],
                           protocol['feature_names'], protocol['indices_in_53'])
    assert indexes == [3, 9, 6, 10, 11, 7, 0, 1, 2] and [order[i] for i in indexes] == nine
    assert protocol['training']['expressions'] == 12146 and protocol['training']['images'] == 1699
    assert protocol['training']['expected_fit_calls'] == 1 and protocol['policy']['operator'] == '>'
    assert (protocol['policy']['lambda'], protocol['policy']['tau']) == (1.5, .2)
    assert protocol['statistics']['difference_direction'] == 'Full(26D) - ExistingScore-LR(9D)'
    schema = read_json(TRAIN / 'outputs/stage5b10/feature_schema.json')
    assert read_json(TRAIN / 'outputs/stage5b10/feature_order.json')['feature_order'] == order == schema['feature_order']
    recipe = read_json(TRAIN / 'outputs/stage5d_final/training_recipe.json')
    dev_manifest = read_json(ROOT / 'run_01/model_manifest.json')
    params = protocol['estimator']['params']
    assert params == recipe['lr_estimator_params'] == dev_manifest['estimator_params']
    mapping = protocol['class_mapping']
    assert mapping == recipe['class_mapping'] == {'BENEFIT': 0, 'NEUTRAL': 1, 'HARM': 2}
    f = parquet(TRAIN / 'outputs/stage5a10_fast/merged/training_features_10pct.parquet')
    t = parquet(TRAIN / 'outputs/stage5a10_fast/merged/training_targets_10pct.parquet')
    assert len(f) == len(t) == 12146 and f.image_id.nunique() == 1699
    assert f.sample_id.is_unique and t.sample_id.is_unique and f.sample_id.tolist() == t.sample_id.tolist()
    assert f.columns.tolist() == ['sample_id', 'image_id', 'ref_id', 'sentence_id'] + order
    for col in ('image_id', 'ref_id', 'sentence_id'):
        if col in t:
            assert f[col].tolist() == t[col].tolist()
    images = set(map(int, f.image_id))
    split = read_json(TRAIN / 'outputs/stage5b10/internal_image_split.json')
    subsets = {k: set(v) for k, v in split['images'].items()}
    assert set.union(*subsets.values()) == images
    assert not subsets['FIT'] & subsets['DEV'] and not subsets['FIT'] & subsets['INTERNAL_HOLDOUT'] and not subsets['DEV'] & subsets['INTERNAL_HOLDOUT']
    assert [int(f.image_id.isin(subsets[k]).sum()) for k in ['FIT', 'DEV', 'INTERNAL_HOLDOUT']] == [9694, 1219, 1233]
    train_anchor, train_structured = t.IoU_a.to_numpy(float), t.IoU_s.to_numpy(float)
    assert np.isfinite(train_anchor).all() and np.isfinite(train_structured).all()
    assert ((train_anchor >= 0) & (train_anchor <= 1) & (train_structured >= 0) & (train_structured <= 1)).all()
    labels = np.where((train_anchor < .5) & (train_structured >= .5), 'BENEFIT',
                      np.where((train_anchor >= .5) & (train_structured < .5), 'HARM', 'NEUTRAL'))
    assert np.array_equal(labels, t.class_label.to_numpy())
    y = t.class_label.map(mapping).to_numpy(np.int64)
    class_counts = {name: int((y == index).sum()) for name, index in mapping.items()}
    assert class_counts == {'BENEFIT': 937, 'NEUTRAL': 10082, 'HARM': 1127}
    with (TRAIN / 'docs/05a10_feature_manifest.csv').open(encoding='utf-8', newline='') as stream:
        feature_rules = list(csv.DictReader(stream))
    assert [r['feature_name'] for r in feature_rules] == order
    assert all(r['uses_gt'] == 'NO' and r['available_at_inference'] == 'YES' for r in feature_rules)
    assert not set(order) & {'GT', 'gt_box', 'gt_box_xyxy', 'IoU_a', 'IoU_s', 'delta_iou', 'class_label', 'anchor_correct', 'structured_correct'}
    # Metadata and identity only before final freeze. No test target/prediction values are read here.
    test_identity = {}
    test_images = {}
    for name, count in protocol['evaluation']['split_rows'].items():
        paths = [CACHE / f'outputs/stage6_test/{name}/features_53d.parquet',
                 CACHE / f'outputs/stage6_test/predictions/{name}_predictions_frozen.parquet',
                 CACHE / f'outputs/stage6_test/evaluation/{name}_per_sample_eval.parquet']
        index = [json.loads(line) for line in (CACHE / f'outputs/stage6_test/{name}_sample_index.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(index) == count
        ids = [r['sample_id'] for r in index]
        image_ids = [r['image_id'] for r in index]
        assert len(ids) == len(set(ids)) and len(set(image_ids)) == 750
        for path in paths:
            metadata = pq.ParquetFile(path)
            assert metadata.metadata.num_rows == count
            identity = parquet(path, ['sample_id', 'image_id'])
            assert identity.sample_id.tolist() == ids and identity.image_id.tolist() == image_ids
        assert pq.ParquetFile(paths[0]).schema_arrow.names == ['sample_id', 'image_id', 'ref_id', 'sentence_id'] + order
        test_images[name] = set(image_ids)
        assert not images & test_images[name]
        test_identity[name] = {'expressions': count, 'images': 750, 'sample_id_unique': True,
                              'sample_id_semantic_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                              'train_image_overlap': 0, 'aligned_to_original_index': True}
    assert not test_images['testA'] & test_images['testB']
    assert 'torch' not in sys.modules
    event('PREFLIGHT_PASS', training_rows=12146, training_images=1699, classes=class_counts, test_identity=test_identity)
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        prep = ns['fit_preprocessor'](f, order)
        prep_seconds = time.perf_counter() - start
        assert prep['continuous'] == schema['continuous_scaled'] and prep['binary'] == schema['binary_unscaled']
        assert prep['scaler'].n_samples_seen_ == 12146
        assert prep['imputer'].keep_empty_features and prep['binary_imputer'].keep_empty_features
        x = ns['transform_features'](prep, f)
        assert x.shape == (12146, 53) and x.dtype == np.float32 and np.isfinite(x).all()
        model = LogisticRegression(**params)
        assert model.get_params() == params
        event('LR_FIT_BEGIN', fit_call_number=1, images=1699, expressions=12146)
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x[:, indexes], y)  # The unique LR fit authorized for this run.
        fit_seconds = time.perf_counter() - begin
        warning_record = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'Convergence failure; stop without retuning'
        assert np.array_equal(model.classes_, [0, 1, 2]) and model.coef_.shape == (3, 9) and model.intercept_.shape == (3,)
        assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
        assert model.n_features_in_ == 9 and (model.n_iter_ < params['max_iter']).all()
        event('LR_FIT_COMPLETE', n_iter=model.n_iter_.tolist(), warnings=warning_record)
        model_path = OUT / 'final_9d_model.pkl'
        assert not model_path.exists()
        artifact = {'model': model, 'preprocessor': prep, 'feature_names': nine,
                    'indices_in_53': indexes, 'class_mapping': mapping,
                    'fitted_image_ids': sorted(images), 'training_sample_count': 12146,
                    'training_sample_ids_sha256': hashlib.sha256(json.dumps(f.sample_id.tolist(), separators=(',', ':')).encode()).hexdigest(),
                    'policy': protocol['policy'], 'protocol_sha256': sha(FORMAL / 'formal_protocol.json')}
        joblib.dump(artifact, model_path)
        frozen_at = stamp()
        manifest = {
            'task': 'CERA-FORMAL-TEST-03', 'name': protocol['model_name'],
            'status': 'FROZEN_BEFORE_TEST_RESULTS_OR_NEW_PREDICTIONS', 'frozen_at_utc': frozen_at,
            'protocol_path': str(FORMAL / 'formal_protocol.json'), 'protocol_sha256': sha(FORMAL / 'formal_protocol.json'),
            'execution_script_sha256': sha(__file__), 'model_sha256': sha(model_path),
            'training_images': 1699, 'training_expressions': 12146, 'fit_calls': 1,
            'class_counts': class_counts, 'class_mapping': mapping, 'classes_': model.classes_.tolist(),
            'coef_shape': list(model.coef_.shape), 'intercept_shape': list(model.intercept_.shape),
            'input_dim': 9, 'parameter_count': int(model.coef_.size + model.intercept_.size),
            'feature_names': nine, 'indices_in_53': indexes, 'preprocessing': protocol['preprocessing'],
            'estimator_params': model.get_params(), 'training_recipe': recipe,
            'training_input_identities': protocol['training']['training_input_identities'],
            'training_sample_ids_sha256': artifact['training_sample_ids_sha256'],
            'test_cache_identities': cache_ids, 'test_identity_checks': test_identity,
            'source_sha256': prev['protected_source_hashes'], 'protected_reports': PROTECTED_REPORTS,
            'protected_r2_sha256': prev['protected_r2_sha256'],
            'policy': protocol['policy'], 'statistics_protocol': protocol['statistics'],
            'comparison': protocol['comparison'], 'study_boundary': protocol['study_boundary'],
            'evidence_limitations': ['testA/testB per_sample_eval.parquet have matching remote/local hashes and checked schema/IDs but no independent historical byte hashes',
                                   'Post-hoc supplementary comparison; historical test results already known',
                                   'Marginal unadjusted tests/intervals across endpoints and splits; expression-level McNemar does not account for within-image correlation',
                                   'Evidence-group and parameter-count differences do not identify a single symmetric feature causally'],
            'n_iter_': model.n_iter_.tolist(), 'converged': True, 'warnings': warning_record,
            'preprocessor_fit_seconds': prep_seconds, 'LR_fit_seconds': fit_seconds,
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                            'pandas': pd.__version__, 'pyarrow': pa.__version__, 'joblib': joblib.__version__,
                            'cpu_threads': 1, 'threadpools': threadpool_info(), 'CUDA_VISIBLE_DEVICES': '',
                            'torch_imported': False},
            'new_test_predictions_before_freeze': 0, 'test_performance_computed_before_freeze': False,
            'remote_server_used': False, 'no_original_CERA_modified': True,
        }
        write_json('formal_pretest_freeze_manifest.json', manifest)
        frozen_manifest_hash = sha(OUT / 'formal_pretest_freeze_manifest.json')
        event('MODEL_AND_PROTOCOL_FROZEN', model_sha256=sha(model_path), manifest_sha256=frozen_manifest_hash)
        evaluation_start = stamp()
        event('EVALUATION_PHASE_STARTED', evaluation_start_utc=evaluation_start)
        summary, paired, evaluation_records, prediction_hashes = {}, {}, {}, {}
        for name, count in protocol['evaluation']['split_rows'].items():
            assert sha(model_path) == manifest['model_sha256'] and sha(OUT / 'formal_pretest_freeze_manifest.json') == frozen_manifest_hash
            features = parquet(CACHE / f'outputs/stage6_test/{name}/features_53d.parquet')
            predictions = parquet(CACHE / f'outputs/stage6_test/predictions/{name}_predictions_frozen.parquet')
            trace = parquet(CACHE / f'outputs/stage6_test/evaluation/{name}_per_sample_eval.parquet')
            assert len(features) == len(predictions) == len(trace) == count
            assert features.sample_id.tolist() == predictions.sample_id.tolist() == trace.sample_id.tolist()
            assert features.image_id.tolist() == predictions.image_id.tolist() == trace.image_id.tolist()
            anchor, structured = trace.anchor_iou.to_numpy(float), trace.structured_iou.to_numpy(float)
            truth = pd.DataFrame({'IoU_a': anchor, 'IoU_s': structured, 'class_label': trace.class_label})
            label_check = np.where((anchor < .5) & (structured >= .5), 'BENEFIT',
                                  np.where((anchor >= .5) & (structured < .5), 'HARM', 'NEUTRAL'))
            assert np.array_equal(label_check, trace.class_label.to_numpy())
            full_decision = predictions.cera_replace.to_numpy(bool)
            ccda_decision = predictions.ccda_replace.to_numpy(bool)
            assert np.array_equal(full_decision, trace.cera_replace.to_numpy(bool))
            assert np.array_equal(ccda_decision, trace.ccda_replace.to_numpy(bool))
            full_iou = np.where(full_decision, structured, anchor)
            assert np.array_equal(full_iou, trace.cera_final_iou.to_numpy(float))
            full_metrics = ns['val_metrics'](truth, full_decision)
            reference = protocol['comparison']['reference_metrics_percent_4dp'][name]
            assert all(round(full_metrics[k], 4) == v for k, v in reference.items()), 'Frozen Full metrics not reproduced; stop'
            event('FULL_REFERENCE_REPRODUCED', split=name, metrics=full_metrics)
            test_x = ns['transform_features'](prep, features)[:, indexes]
            assert test_x.shape == (count, 9) and np.isfinite(test_x).all()
            start_prediction = stamp()
            event('TEST_PREDICT_BEGIN', split=name, call_number=1, rows=count)
            probabilities = model.predict_proba(test_x)  # Exactly one new-model prediction per split.
            assert probabilities.shape == (count, 3) and np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)
            utility = probabilities[:, 0] - protocol['policy']['lambda'] * probabilities[:, 2]
            decision = ns['risk_decisions'](probabilities, protocol['policy']['lambda'], protocol['policy']['tau'])
            assert np.array_equal(decision, utility > .20)
            # Missing candidates retain the original zero-IoU/None semantics. No added gate.
            assert all(anchor[i] == 0. for i, box in enumerate(predictions.anchor_box) if box is None)
            assert all(structured[i] == 0. for i, box in enumerate(predictions.structured_box) if box is None)
            final_iou = np.where(decision, structured, anchor)
            selected_boxes = [s if replace else a for replace, a, s in
                              zip(decision, predictions.anchor_box, predictions.structured_box)]
            result = pd.DataFrame({
                'sample_id': features.sample_id, 'split': name, 'image_id': features.image_id,
                'ref_id': features.ref_id, 'sentence_id': features.sentence_id,
                'anchor_box': predictions.anchor_box, 'structured_box': predictions.structured_box,
                'class_label': trace.class_label, 'anchor_iou': anchor, 'structured_iou': structured,
                'P_B': probabilities[:, 0], 'P_N': probabilities[:, 1], 'P_H': probabilities[:, 2],
                'utility': utility, 'replace_decision': decision, 'output_box': selected_boxes,
                'output_iou': final_iou, 'correct50': final_iou >= .5,
                'full_replace': full_decision, 'full_output_iou': full_iou, 'full_correct50': full_iou >= .5,
                'ccda_replace': ccda_decision, 'ccda_output_iou': np.where(ccda_decision, structured, anchor),
            })
            dest = OUT / f'{name}_9d_predictions.parquet'
            assert not dest.exists()
            pq.write_table(pa.Table.from_pandas(result, preserve_index=False), dest)
            prediction_hashes[name] = sha(dest)
            evaluation_records[name] = {'prediction_calls': 1, 'prediction_start_utc': start_prediction,
                                        'prediction_saved_utc': stamp(), 'rows': count, 'images': 750,
                                        'sha256': prediction_hashes[name]}
            method_metrics = {
                'ExistingScore-LR (9D)': ns['val_metrics'](truth, decision),
                'CERA-Full (26D)': full_metrics,
                'Anchor': ns['val_metrics'](truth, np.zeros(count, dtype=bool)),
                'Raw Structured': ns['val_metrics'](truth, np.ones(count, dtype=bool)),
                'CCDA': ns['val_metrics'](truth, ccda_decision),
            }
            baseline = method_metrics['ExistingScore-LR (9D)']
            differences = {k + ('_pp' if k in ['acc50', 'miou'] else '_count' if k in
                ['corrected', 'induced', 'net_gain', 'override_count'] else '_pp'):
                full_metrics[k] - baseline[k] if k in ['acc50', 'miou', 'corrected', 'induced', 'net_gain', 'override_count']
                else 100 * (full_metrics[k] - baseline[k]) for k in
                ['acc50', 'miou', 'corrected', 'induced', 'net_gain', 'harm_rejection',
                 'conflict_decision_accuracy', 'benefit_recall', 'override_count', 'override_rate']}
            summary[name] = {'rows': count, 'images': 750, 'metrics': method_metrics,
                             'Full_minus_9D': differences,
                             'consequence_counts': {label: int((trace.class_label == label).sum()) for label in mapping}}
            mc = ns['exact_mcnemar'](full_iou >= .5, final_iou >= .5)
            mc.update({'paired_expressions': count, 'net_correct_difference': mc['first_only'] - mc['second_only'],
                       'direction': 'Full correct / 9D wrong minus Full wrong / 9D correct',
                       'unit': 'expression', 'within_image_dependence_not_adjusted': True})
            paired[name] = {
                'mcnemar': mc,
                'acc50_cluster_bootstrap': ns['paired_image_bootstrap_ci'](full_iou >= .5, final_iou >= .5,
                    features.image_id.to_numpy(), resamples=protocol['statistics']['bootstrap']['resamples'],
                    seed=protocol['statistics']['bootstrap']['seed']),
                'miou_cluster_bootstrap': ns['paired_image_bootstrap_ci'](full_iou, final_iou,
                    features.image_id.to_numpy(), resamples=protocol['statistics']['bootstrap']['resamples'],
                    seed=protocol['statistics']['bootstrap']['seed']),
            }
            event('TEST_PREDICT_COMPLETE', split=name, prediction_sha256=sha(dest), metrics=method_metrics,
                  differences=differences, paired_statistics=paired[name])
        assert sha(model_path) == manifest['model_sha256'] and sha(OUT / 'formal_pretest_freeze_manifest.json') == frozen_manifest_hash
        identities()  # All original data/protocol/manuscript hashes must still match.
        val = read_json(ROOT / 'run_01/paired_statistics.json')
        result_identity = {'task': 'CERA-FORMAL-TEST-03', 'status': 'RESULTS_FROZEN',
                           'model_sha256': sha(model_path), 'pretest_manifest_sha256': frozen_manifest_hash,
                           'protocol_sha256': sha(FORMAL / 'formal_protocol.json'), 'script_sha256': sha(__file__),
                           'evaluation_phase_start_utc': evaluation_start,
                           'test_prediction_records': evaluation_records, 'completed_at_utc': stamp(),
                           'post_hoc': True, 'GPU_USED': 'NO', 'REMOTE_SERVER_USED': 'NO',
                           'UPSTREAM_INFERENCE_RERUN': 'NO', 'PARAMETER_RESEARCH_REOPENED': 'NO'}
        write_json('test_metrics.json', {**result_identity, 'splits': summary,
            'units': {'acc50': 'percent', 'miou': 'percent', 'rates': 'fraction', 'differences': 'percentage points or expression counts'},
            'development_comparison': {'source_sha256': sha(ROOT / 'run_01/paired_statistics.json'),
                'training_scope': 'FIT-only 9694 expressions, unlike final train10-refit test models',
                'metrics': val['metrics'], 'differences': val['differences'],
                'absolute_val_test_scores_not_same_model': True}})
        write_json('paired_statistics.json', {**result_identity, 'statistics_protocol': protocol['statistics'],
            'splits': paired, 'multiplicity': 'All marginal p-values/CIs unadjusted, exploratory post-hoc evidence; both splits and metrics retained',
            'mcnemar_limitation': 'Expression-level exact binomial test does not adjust dependence among expressions on the same image; image-cluster CIs are also reported'})
        event('RESULTS_FROZEN', test_metrics_sha256=sha(OUT / 'test_metrics.json'),
              statistics_sha256=sha(OUT / 'paired_statistics.json'), prediction_sha256=prediction_hashes,
              original_protected_hashes_unchanged=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--self-check' not in sys.argv:
            event('BLOCKED', error=type(error).__name__ + ': ' + str(error))
        raise
