"""CPU-only deterministic Stage 5A index, cache and durability primitives."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

SEED = 3407
PROTOCOL_VERSION = "cera-stage5a10-fast-v1"
SHARD_NAMES = ("shard_00_000_005", "shard_01_005_010")


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Durable same-directory replace; old file survives failures before replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def stable_image_order(image_ids, seed=SEED):
    """SHA-256 seeded permutation; independent of Python random/hash version."""
    ids = sorted({int(value) for value in image_ids})
    return sorted(ids, key=lambda value: (hashlib.sha256(
        f"{seed}:{value}".encode("ascii")).digest(), value))


def select_tenth(image_ids, seed=SEED):
    order = stable_image_order(image_ids, seed)
    return order[:math.floor(len(order) * .10)]


def shard_images(selected):
    boundary = len(selected) // 2
    return selected[:boundary], selected[boundary:]


def sample_id(split, sentence_id):
    if split not in ("train", "val", "testA", "testB"):
        raise ValueError(split)
    return f"refcoco_unc_{split}_sent_{int(sentence_id)}"


def stable_processing_order(rows, selected_images):
    by_image = {}
    for row in rows:
        by_image.setdefault(int(row["image_id"]), []).append(row)
    if set(by_image) != set(selected_images):
        raise ValueError("Selected image set differs from expression index")
    order = []
    for image in selected_images:
        order.extend(sorted(by_image[image], key=lambda row:
                            (int(row["ref_id"]), int(row["sentence_id"]), row["sample_id"])))
    if len({row["sample_id"] for row in order}) != len(order):
        raise ValueError("Duplicate sample ID")
    return order


def preflight_ids(rows, count=100, seed=SEED):
    if len(rows) < count:
        raise ValueError("Insufficient expressions")
    return [row["sample_id"] for row in sorted(rows, key=lambda row:
            (hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).digest(), row["sample_id"]))[:count]]


def normalize_query(query):
    return re.sub(r"\s+", " ", query.strip()).casefold()


def gdino_query_key(image_id, query, protocol_hash):
    return sha256_value([int(image_id), normalize_query(query), protocol_hash])


def qwen_parse_key(expression, prompt_hash, revision):
    return sha256_value([expression, prompt_hash, revision])


def quantized_box(box):
    """Stage 2's 1/100-pixel banker-rounding cache convention; lookup only."""
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("Invalid cache box")
    return [round(float(value) * 100) for value in box]


def attribute_key(image_id, box, attribute, head, prompt_hash, revision):
    return sha256_value([int(image_id), quantized_box(box), normalize_query(attribute),
                         normalize_query(head), prompt_hash, revision])


def checkpoint_template(protocol_hash, subset_hash, order_hash):
    return {"protocol_hash": protocol_hash, "selected_subset_hash": subset_hash,
            "processing_order_hash": order_hash, "current_shard": None,
            "last_durable_index": 0, "processed_count": 0,
            "cache_offsets": {"gdino": 0, "parse": 0, "attribute": 0},
            "runtime_counters": {}, "status": "PREPARED_NOT_RUN"}


def truncate_jsonl_to_count(path, keep):
    """Discard non-durable suffix after a crash; reject a missing durable prefix."""
    path = Path(path)
    if not path.exists():
        if keep:
            raise ValueError(f"Missing durable file: {path}")
        return
    with path.open("rb") as stream:
        lines = stream.readlines()
    if len(lines) < keep or any(not line.endswith(b"\n") for line in lines[:keep]):
        raise ValueError(f"Incomplete durable prefix: {path}")
    if len(lines) == keep:
        return
    temporary = path.with_name(path.name + ".recover.tmp")
    with temporary.open("wb") as stream:
        stream.writelines(lines[:keep])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def recover_pair(features_path, targets_path, durable_index):
    truncate_jsonl_to_count(features_path, durable_index)
    truncate_jsonl_to_count(targets_path, durable_index)
    features = list(read_jsonl(features_path))
    targets = list(read_jsonl(targets_path))
    if len(features) != durable_index or len(targets) != durable_index:
        raise ValueError("Durable pair count mismatch")
    left = [row["sample_id"] for row in features]
    right = [row["sample_id"] for row in targets]
    if left != right or len(set(left)) != len(left):
        raise ValueError("Durable feature/target sample IDs mismatch")
    return left


def enforce_hard_cap(order, allowed_ids):
    allowed = set(allowed_ids)
    if len(order) != len(allowed) or {row["sample_id"] for row in order} != allowed:
        raise ValueError("Processing order exceeds frozen 10% subset")
    return order
