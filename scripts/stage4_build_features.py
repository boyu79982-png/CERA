"""Build inference-available pairwise features from frozen and verified evidence."""

import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ccda import box_iou  # noqa: E402
from src.stage2_protocol import parse_yes_no  # noqa: E402
from src.stage4_evidence import anchor_relation_evidence, evidence_statistics  # noqa: E402

OUT = ROOT / "outputs/stage4"
FEATURES = (
    "expression_length", "num_attributes", "num_relations", "head_valid",
    "anchor_available", "structured_available", "anchor_detector_conf",
    "structured_detector_conf", "detector_conf_delta", "S_st", "old_margin",
    "box_pair_iou", "parse_source_rule_fallback", "head_candidate_count",
    "Sa_anchor", "Sa_structured", "delta_Sa", "Sr_anchor", "Sr_structured", "delta_Sr",
    "S_sem_anchor", "S_sem_structured", "delta_sem",
    "anchor_attr_mean", "anchor_attr_min", "anchor_attr_max", "anchor_attr_std",
    "anchor_attr_pass_ratio", "structured_attr_mean", "structured_attr_min",
    "structured_attr_max", "structured_attr_std", "structured_attr_pass_ratio",
    "anchor_relation_mean", "anchor_relation_min", "anchor_relation_max",
    "anchor_relation_std", "anchor_relation_pass_ratio", "structured_relation_mean",
    "structured_relation_min", "structured_relation_max", "structured_relation_std",
    "structured_relation_pass_ratio", "anchor_constraint_mean", "structured_constraint_mean",
    "anchor_constraint_min", "structured_constraint_min", "delta_min_constraint",
    "anchor_violation_count", "structured_violation_count", "delta_violation",
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def difference(left, right):
    return left-right if left is not None and right is not None else None


def unavailable_stats(num_attrs, num_rels):
    result = {name: None for name in evidence_statistics([], [])}
    result["attr_count"] = num_attrs
    result["relation_count"] = num_rels
    result["attr_missing_count"] = num_attrs
    result["relation_missing_count"] = num_rels
    return result


def build_one(record, cache):
    anchor, structured = record["anchor"], record["structured"]
    attr_details = []
    for item in anchor["attribute_evidence"]:
        copy = dict(item)
        if item["source"] == "stage4_cache_or_new_qwen":
            cached = cache.get(item["key"])
            if cached is None:
                raise ValueError(f"Missing new anchor Qwen evidence {item['key']}")
            if cached["attribute"] != item["attribute"] or cached["head"] != record["head_name"]:
                raise ValueError(f"Anchor Qwen cache prompt mismatch {item['key']}")
            if not math.isclose(parse_yes_no(cached["raw_response"]), cached["score"], abs_tol=1e-9):
                raise ValueError(f"Anchor Qwen response parse mismatch {item['key']}")
            copy["score"] = cached["score"]
            copy["raw_response"] = cached["raw_response"]
        attr_details.append(copy)
    if len(attr_details) != len(record["attributes"]):
        raise ValueError(f"Anchor attribute count mismatch {record['sample_id']}")
    relation_details = anchor_relation_evidence(anchor["box"], record)
    if len(relation_details) != len(record["relations"]):
        raise ValueError(f"Anchor relation count mismatch {record['sample_id']}")
    a_scores = [item.get("score") for item in attr_details]
    ar_scores = [item["score"] for item in relation_details]
    if anchor["box"] is None:
        astats = unavailable_stats(len(a_scores), len(ar_scores))
    else:
        astats = evidence_statistics(a_scores, ar_scores)
    if structured["box"] is None:
        s_scores = [None]*len(record["attributes"])
        sr_scores = [None]*len(record["relations"])
        sstats = unavailable_stats(len(s_scores), len(sr_scores))
    else:
        sattrs = structured["attribute_evidence"]
        srels = structured["relation_evidence"]
        if len(sattrs) != len(record["attributes"]) or len(srels) != len(record["relations"]):
            raise ValueError(f"Structured evidence count mismatch {record['sample_id']}")
        s_scores = [item["score"] for item in sattrs]
        sr_scores = [item["best_score"] for item in srels]
        sstats = evidence_statistics(s_scores, sr_scores)
        for key, saved in (("Sa", structured["Sa"]), ("Sr", structured["Sr"]),
                           ("S_sem", structured["S"])):
            if not math.isclose(sstats[key], saved, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Structured {key} differs from frozen score at {record['sample_id']}")
    conf_a = anchor["confidence"]
    conf_s = structured["detector_confidence"]
    features = {
        "expression_length": record["expression_word_count"],
        "num_attributes": len(record["attributes"]), "num_relations": len(record["relations"]),
        "head_valid": int(record["head_valid"]),
        "anchor_available": int(anchor["box"] is not None),
        "structured_available": int(structured["box"] is not None),
        "anchor_detector_conf": conf_a, "structured_detector_conf": conf_s,
        "detector_conf_delta": difference(conf_s, conf_a),
        "S_st": structured["S"], "old_margin": difference(structured["S"], conf_a),
        "box_pair_iou": box_iou(anchor["box"], structured["box"]),
        "parse_source_rule_fallback": int(record["parse_source"] == "rule_fallback"),
        "head_candidate_count": record["head_candidate_count"],
        "Sa_anchor": astats["Sa"], "Sa_structured": sstats["Sa"],
        "delta_Sa": difference(sstats["Sa"], astats["Sa"]),
        "Sr_anchor": astats["Sr"], "Sr_structured": sstats["Sr"],
        "delta_Sr": difference(sstats["Sr"], astats["Sr"]),
        "S_sem_anchor": astats["S_sem"], "S_sem_structured": sstats["S_sem"],
        "delta_sem": difference(sstats["S_sem"], astats["S_sem"]),
        "anchor_attr_mean": astats["Sa"], "anchor_attr_min": astats["attr_min"],
        "anchor_attr_max": astats["attr_max"], "anchor_attr_std": astats["attr_std"],
        "anchor_attr_pass_ratio": astats["attr_pass_ratio"],
        "structured_attr_mean": sstats["Sa"], "structured_attr_min": sstats["attr_min"],
        "structured_attr_max": sstats["attr_max"], "structured_attr_std": sstats["attr_std"],
        "structured_attr_pass_ratio": sstats["attr_pass_ratio"],
        "anchor_relation_mean": astats["Sr"], "anchor_relation_min": astats["relation_min"],
        "anchor_relation_max": astats["relation_max"], "anchor_relation_std": astats["relation_std"],
        "anchor_relation_pass_ratio": astats["relation_pass_ratio"],
        "structured_relation_mean": sstats["Sr"], "structured_relation_min": sstats["relation_min"],
        "structured_relation_max": sstats["relation_max"], "structured_relation_std": sstats["relation_std"],
        "structured_relation_pass_ratio": sstats["relation_pass_ratio"],
        "anchor_constraint_mean": astats["constraint_mean"],
        "structured_constraint_mean": sstats["constraint_mean"],
        "anchor_constraint_min": astats["constraint_min"],
        "structured_constraint_min": sstats["constraint_min"],
        "delta_min_constraint": difference(sstats["constraint_min"], astats["constraint_min"]),
        "anchor_violation_count": astats["violation_count"],
        "structured_violation_count": sstats["violation_count"],
        "delta_violation": difference(astats["violation_count"], sstats["violation_count"]),
    }
    if set(features) != set(FEATURES):
        raise ValueError("Feature schema mismatch")
    dataset_row = {"sample_id": record["sample_id"], "image_id": record["image_id"],
                   "label": record["label"], **features}
    detail = {"sample_id": record["sample_id"], "image_path": record["image_path"],
              "expression": record["expression"], "GT": record["GT"],
              "anchor_box": anchor["box"], "structured_box": structured["box"],
              "label": record["label"], "attributes": record["attributes"],
              "relations": record["relations"], "anchor_attr": attr_details,
              "structured_attr": structured["attribute_evidence"],
              "anchor_relation": relation_details,
              "structured_relation": structured["relation_evidence"]}
    return dataset_row, detail


def visualize(details):
    from PIL import Image, ImageDraw
    folder = OUT / "constraint_pair_visualizations"
    folder.mkdir(exist_ok=True)
    rng = random.Random(3407)
    by_label = {label: [item for item in details if item["label"] == label] for label in (1, 0)}
    selected = {"benefit": rng.sample(by_label[1], min(15, len(by_label[1]))),
                "harm": rng.sample(by_label[0], min(15, len(by_label[0])))}
    manifest = {"seed": 3407, "groups": {name: [x["sample_id"] for x in rows] for name,rows in selected.items()}}
    for group, items in selected.items():
        for ordinal, row in enumerate(items, 1):
            with Image.open(row["image_path"]) as image_file:
                image = image_file.convert("RGB")
            lines = [row["sample_id"], row["expression"][:140],
                     "GT=green  Anchor=red  Structured=orange"]
            for index, attr in enumerate(row["attributes"]):
                av = row["anchor_attr"][index].get("score")
                sv = row["structured_attr"][index]["score"] if row["structured_box"] is not None else None
                lines.append(f"attr {attr}: A={av} S={sv} delta={difference(sv,av)}")
            for index, relation in enumerate(row["relations"]):
                av = row["anchor_relation"][index]["score"]
                sv = row["structured_relation"][index]["best_score"] if row["structured_box"] is not None else None
                lines.append(f"rel {relation['type']} {relation['target']}: A={av} S={sv} delta={difference(sv,av)}")
            header = max(120, len(lines)*26+12)
            canvas = Image.new("RGB", (max(1200, image.width), header+image.height), "white")
            canvas.paste(image, (0, header))
            draw = ImageDraw.Draw(canvas)
            for k, line in enumerate(lines):
                draw.text((8, 8+k*26), line, fill="black")
            for name, box, color in (("GT", row["GT"], "green"),
                                     ("A", row["anchor_box"], "red"),
                                     ("S", row["structured_box"], "orange")):
                if box is not None:
                    draw.rectangle([box[0], box[1]+header, box[2], box[3]+header], outline=color, width=3)
                    draw.text((box[0]+3, box[1]+header+3), name, fill=color)
            canvas.save(folder / f"{group}_{ordinal:02d}_{row['sample_id']}.png")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    cache = {}
    cache_path = OUT / "anchor_attribute_cache.jsonl"
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            if row["key"] in cache:
                raise ValueError(f"Duplicate Stage 4 Qwen cache key {row['key']}")
            cache[row["key"]] = row
    rows, details = [], []
    for record in read_jsonl(OUT / "conflict_projection.jsonl"):
        feature, detail = build_one(record, cache)
        rows.append(feature)
        details.append(detail)
    if len(rows) != 1854 or sum(row["label"] for row in rows) != 897 or len({row["sample_id"] for row in rows}) != 1854:
        raise ValueError("Feature dataset conflict counts fail")
    with (OUT / "pairwise_conflict_features.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_id", "label", *FEATURES])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "constraint_pair_evidence.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = visualize(details)
    print(json.dumps({"rows": len(rows), "benefit": sum(row["label"] for row in rows),
                      "harm": len(rows)-sum(row["label"] for row in rows),
                      "visualizations": sum(len(x) for x in manifest["groups"].values())}))


if __name__ == "__main__":
    main()
