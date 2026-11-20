"""Strict, read-only loader for the canonical full-leaf dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DATASET_SCHEMA = "riceblast.full_leaf_dataset.v1"
RELATION_SCHEMA = "riceblast.full_leaf_relation.v1"
CLASS_COUNT_COLUMNS = (
    "background_pixels",
    "healthy_leaf_pixels",
    "susceptible_lesion_pixels",
    "resistant_lesion_pixels",
)
RELATION_COUNT_KEYS = (
    "background",
    "healthy_leaf",
    "susceptible_lesion",
    "resistant_lesion",
)
REQUIRED_MANIFEST_COLUMNS = {
    "sample_id", "split", "source_image_id", "leaf_id", "image_path",
    "mask_path", "relation_path", "width", "height", *CLASS_COUNT_COLUMNS,
}


@dataclass(frozen=True)
class FullLeafRecord:
    sample_id: str
    split: str
    source_image_id: str
    leaf_id: str
    image_path: str
    mask_path: str
    relation_path: str
    width: int
    height: int
    class_counts: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["class_counts"] = list(self.class_counts)
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_dataset_path(root: Path, relative: str, field: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes dataset_root: {relative}") from exc
    return candidate


def compute_dataset_fingerprint(root: str | Path, rows: Iterable[dict] | None = None) -> dict:
    root = Path(root).resolve()
    manifest_path = root / "manifest.csv"
    summary_path = root / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if rows is None:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = list(rows)
    content = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        for field in ("image_path", "mask_path", "relation_path"):
            path = _safe_dataset_path(root, row[field], field)
            content.update(row["sample_id"].encode("utf-8"))
            content.update(field.encode("ascii"))
            content.update(_sha256(path).encode("ascii"))
    by_split = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row.get("split") == split]
        by_split[split] = {
            "sample_count": len(selected),
            "source_count": len({row.get("source_image_id") for row in selected}),
        }
    return {
        "manifest_sha256": _sha256(manifest_path),
        "split_summary_sha256": _sha256(summary_path),
        "content_sha256": content.hexdigest(),
        "schema_version": summary.get("schema_version"),
        "splits": by_split,
    }


def fingerprints_match(expected: dict, actual: dict) -> bool:
    return expected == actual


def assert_resume_fingerprint(expected: dict, actual: dict, allow_mismatch: bool = False) -> None:
    if fingerprints_match(expected, actual):
        return
    if allow_mismatch:
        return
    raise ValueError(
        "Dataset fingerprint differs from the resumed run. Pass the explicit "
        "fingerprint-mismatch override only after auditing the dataset change."
    )


def load_full_leaf_manifest(
    dataset_root: str | Path,
    split: str,
    *,
    validate_pixels: bool = True,
) -> tuple[list[FullLeafRecord], dict]:
    """Validate the whole manifest and return records for one immutable split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    root = Path(dataset_root).resolve()
    manifest_path = root / "manifest.csv"
    summary_path = root / "split_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("dataset_root must contain manifest.csv and split_summary.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != DATASET_SCHEMA:
        raise ValueError(f"Unsupported dataset schema: {summary.get('schema_version')}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = list(reader)

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest.csv contains duplicate sample_id values")
    invalid_splits = sorted({row["split"] for row in rows} - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Invalid manifest split values: {invalid_splits}")

    assignments = summary.get("group_assignments", {})
    sources: dict[str, str] = {}
    records: list[FullLeafRecord] = []
    for row in rows:
        sample_id = row["sample_id"]
        row_split = row["split"]
        source_id = row["source_image_id"]
        previous = sources.setdefault(source_id, row_split)
        if previous != row_split:
            raise ValueError(f"Source split leakage for {source_id}: {previous}, {row_split}")
        if assignments.get(source_id) != row_split:
            raise ValueError(f"split_summary assignment mismatch for source {source_id}")

        image_path = _safe_dataset_path(root, row["image_path"], "image_path")
        mask_path = _safe_dataset_path(root, row["mask_path"], "mask_path")
        relation_path = _safe_dataset_path(root, row["relation_path"], "relation_path")
        for path in (image_path, mask_path, relation_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing full-leaf triplet file: {path}")

        width, height = int(row["width"]), int(row["height"])
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask = np.asarray(mask_image, dtype=np.int64) if validate_pixels else None
        if image_size != mask_size or image_size != (width, height):
            raise ValueError(f"Dimension mismatch for {sample_id}: manifest={(width, height)}, image={image_size}, mask={mask_size}")

        counts = tuple(int(row[key]) for key in CLASS_COUNT_COLUMNS)
        relation = json.loads(relation_path.read_text(encoding="utf-8"))
        if relation.get("schema_version") != RELATION_SCHEMA:
            raise ValueError(f"Unsupported relation schema for {sample_id}")
        expected_relation = {
            "sample_id": sample_id, "split": row_split,
            "source_image_id": source_id, "leaf_id": row["leaf_id"],
        }
        for key, value in expected_relation.items():
            if str(relation.get(key)) != str(value):
                raise ValueError(f"Relation {key} mismatch for {sample_id}")
        if relation.get("size") != {"width": width, "height": height}:
            raise ValueError(f"Relation size mismatch for {sample_id}")
        files = relation.get("files", {})
        for key, row_key in (("image", "image_path"), ("mask", "mask_path"), ("relation", "relation_path")):
            if files.get(key) != row[row_key]:
                raise ValueError(f"Relation {key} path mismatch for {sample_id}")
        relation_counts = relation.get("class_pixel_counts", {})
        if tuple(int(relation_counts.get(key, -1)) for key in RELATION_COUNT_KEYS) != counts:
            raise ValueError(f"Relation class counts mismatch for {sample_id}")
        geometry = relation.get("geometry", {})
        if "source_to_export_3x3" not in geometry or "export_to_source_3x3" not in geometry:
            raise ValueError(f"Relation geometry is incomplete for {sample_id}")
        if validate_pixels:
            values, actual_counts = np.unique(mask, return_counts=True)
            if not set(values.tolist()).issubset({0, 1, 2, 3}):
                raise ValueError(f"Invalid mask values for {sample_id}: {values.tolist()}")
            recount = tuple(int(actual_counts[values.tolist().index(i)]) if i in values else 0 for i in range(4))
            if recount != counts:
                raise ValueError(f"Manifest class counts mismatch for {sample_id}")

        if row_split == split:
            records.append(FullLeafRecord(
                sample_id=sample_id, split=row_split, source_image_id=source_id,
                leaf_id=row["leaf_id"], image_path=str(image_path), mask_path=str(mask_path),
                relation_path=str(relation_path), width=width, height=height, class_counts=counts,
            ))

    if not records:
        raise ValueError(f"Requested split is empty: {split}")
    expected_split = summary.get("splits", {}).get(split, {})
    if expected_split.get("leaf_count") != len(records):
        raise ValueError(f"split_summary leaf count mismatch for {split}")
    for split_name in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split_name]
        expected = summary.get("splits", {}).get(split_name, {})
        if expected.get("leaf_count") != len(selected) or expected.get("source_count") != len({row["source_image_id"] for row in selected}):
            raise ValueError(f"split_summary counts mismatch for {split_name}")
    fingerprint = compute_dataset_fingerprint(root, rows)
    return records, fingerprint
