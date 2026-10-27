"""Paper analyses on saved full-leaf prediction masks.

Computes, for one or more sets of class-label prediction PNGs (as written by
test.py or full_leaf_inference.py), the analyses reported in the paper in
addition to the metrics in test.py's metrics.json:

* one-to-one component recall, precision and F1 at IoU >= 0.10 and >= 0.50,
  using maximum-cardinality bipartite matching between same-class components,
  by reference-area stratum (<=32, 33-128, 129-512, >512 pixels; Supplementary S4.1);
* leaf-level lesion-area agreement: MAE, RMSE and bias in percentage points for
  susceptible-type (S), resistant-type (R) and total lesion area fractions
  (Section 2.5, "Leaf-level area agreement"), plus the sensitivity analysis that
  restricts predictions to the reference leaf support;
* the same component metrics after removing predicted components smaller than
  24 pixels (filtering sensitivity, Supplementary S4.4);
* optional source-photograph bootstrap intervals (Supplementary S4.2), including
  paired differences against a reference prediction set.

Example (HiBlastNet versus the refinement control, seed 100):

    python scripts/analyze_predictions.py --dataset-root data/blastleaf --split test \\
        --predictions hiblastnet=outputs/hiblastnet/seed100 gcnet_bc=outputs/gcnet_bc/seed100 \\
        --reference gcnet_bc --bootstrap 10000 --output outputs/analysis_seed100.json

Each --predictions entry is NAME=PATH, where PATH is a test.py output folder
(the newest run inside it is used) or a folder of <sample_id>.png masks.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blastseg.data import load_full_leaf_manifest  # noqa: E402

LESION_CLASSES = (2, 3)
AREA_BINS = (32, 128, 512)
STRATA = ("le32", "33_128", "129_512", "gt512")
IOU_THRESHOLDS = (0.10, 0.50)
FILTER_MIN_AREA = 24
STRUCTURE = ndimage.generate_binary_structure(2, 2)  # eight-connectivity


def label_components(binary):
    labels, count = ndimage.label(binary, structure=STRUCTURE)
    return labels, count


def stratum_index(areas):
    return np.searchsorted(AREA_BINS, areas, side="left")


def remove_small_components(prediction, min_area):
    """Reassign predicted lesion components smaller than min_area pixels to healthy tissue."""
    filtered = prediction.copy()
    for class_id in LESION_CLASSES:
        labels, count = label_components(prediction == class_id)
        if not count:
            continue
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        small = np.flatnonzero(areas < min_area)
        small = small[small > 0]
        if len(small):
            filtered[np.isin(labels, small)] = 1
    return filtered


def component_statistics(prediction, target):
    """Per-leaf component counts by class, stratum and IoU threshold."""
    stats = {
        "ref": np.zeros((2, 4), np.int64),           # reference components per class/stratum
        "pred": np.zeros(2, np.int64),               # predicted components per class
        "survived": np.zeros((2, 4), np.int64),      # any same-class overlap
        "matched_any": np.zeros((2, 4), np.int64),   # IoU >= 0.10, many-to-one allowed
        "zero_overlap_fp": np.zeros(2, np.int64),    # predicted components with no reference overlap
        "one_to_one": np.zeros((2, 2, 4), np.int64),  # threshold x class x stratum
    }
    for c, class_id in enumerate(LESION_CLASSES):
        gt_labels, gt_count = label_components(target == class_id)
        pred_labels, pred_count = label_components(prediction == class_id)
        gt_areas = np.bincount(gt_labels.ravel(), minlength=gt_count + 1)
        pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
        gt_strata = stratum_index(gt_areas[1:])
        stats["ref"][c] = np.bincount(gt_strata, minlength=4)
        stats["pred"][c] = pred_count
        overlap = (gt_labels > 0) & (pred_labels > 0)
        if not overlap.any():
            stats["zero_overlap_fp"][c] = pred_count
            continue
        base = pred_count + 1
        codes, inter = np.unique(
            gt_labels[overlap].astype(np.int64) * base + pred_labels[overlap], return_counts=True
        )
        gt_ids, pred_ids = codes // base, codes % base
        iou = inter / (gt_areas[gt_ids] + pred_areas[pred_ids] - inter)
        survived = np.unique(gt_ids)
        stats["survived"][c] = np.bincount(gt_strata[survived - 1], minlength=4)
        matched = np.unique(gt_ids[iou >= 0.10])
        stats["matched_any"][c] = np.bincount(gt_strata[matched - 1], minlength=4)
        stats["zero_overlap_fp"][c] = pred_count - len(np.unique(pred_ids))
        for t, threshold in enumerate(IOU_THRESHOLDS):
            keep = iou >= threshold
            if not keep.any():
                continue
            graph = csr_matrix(
                (np.ones(keep.sum(), np.int8), (gt_ids[keep] - 1, pred_ids[keep] - 1)),
                shape=(gt_count, pred_count),
            )
            graph.sort_indices()
            match = maximum_bipartite_matching(graph, perm_type="column")
            matched_gt = np.flatnonzero(match >= 0)
            stats["one_to_one"][t, c] = np.bincount(gt_strata[matched_gt], minlength=4)
    return stats


def pixel_statistics(prediction, target):
    valid = target >= 0
    p, g = prediction[valid], target[valid]
    tp = np.array([np.count_nonzero((p == k) & (g == k)) for k in LESION_CLASSES], np.int64)
    pc = np.array([np.count_nonzero(p == k) for k in LESION_CLASSES], np.int64)
    gc = np.array([np.count_nonzero(g == k) for k in LESION_CLASSES], np.int64)
    return tp, pc, gc


def area_fractions(prediction, target):
    """Lesion-area fractions (%) for S, R and total; own supports and reference-support variant."""
    pred_support = np.count_nonzero(prediction > 0)
    ref_support = np.count_nonzero(target > 0)
    ref_mask = target > 0

    def fractions(mask, support, restrict=None):
        values = []
        for classes in ((2,), (3,), (2, 3)):
            lesion = np.isin(mask, classes)
            if restrict is not None:
                lesion &= restrict
            values.append(100.0 * np.count_nonzero(lesion) / support if support else 0.0)
        return values

    return (
        np.array(fractions(prediction, pred_support)),
        np.array(fractions(target, ref_support)),
        np.array(fractions(prediction, ref_support, restrict=ref_mask)),
    )


def leaf_statistics(prediction, target):
    tp, pc, gc = pixel_statistics(prediction, target)
    pred_frac, ref_frac, restricted_frac = area_fractions(prediction, target)
    return {
        "tp": tp, "pc": pc, "gc": gc,
        "components": component_statistics(prediction, target),
        "area_error": pred_frac - ref_frac,
        "area_error_ref_support": restricted_frac - ref_frac,
    }


def stack(leaves):
    """Stack per-leaf statistics into arrays with a leading leaf axis."""
    out = {key: np.stack([leaf[key] for leaf in leaves]) for key in ("tp", "pc", "gc", "area_error", "area_error_ref_support")}
    out["components"] = {
        key: np.stack([leaf["components"][key] for leaf in leaves]) for key in leaves[0]["components"]
    }
    return out


def _ratio(a, b):
    return float(a / b) if b else None


def summarize(stats, weights):
    """Pooled metrics for leaves weighted by bootstrap multiplicity (1 for the original set)."""
    w = np.asarray(weights, dtype=np.float64)
    n_leaves = w.sum()
    tp, pc, gc = (np.tensordot(w, stats[k], axes=1) for k in ("tp", "pc", "gc"))
    dice = [2 * tp[i] / (pc[i] + gc[i]) if pc[i] + gc[i] else 0.0 for i in range(2)]
    comp = {k: np.tensordot(w, v, axes=1) for k, v in stats["components"].items()}
    ref = comp["ref"]                    # class x stratum
    ref_all = ref.sum()
    result = {
        "lesion_dice": 100 * float(np.mean(dice)),
        "small_survival": 100 * _ratio(comp["survived"][:, 0].sum(), ref[:, 0].sum()) if ref[:, 0].sum() else None,
        "matched_recall_iou010": 100 * _ratio(comp["matched_any"].sum(), ref_all) if ref_all else None,
        "fp_per_leaf": float(comp["zero_overlap_fp"].sum() / n_leaves),
        "one_to_one": {},
    }
    n_pred = comp["pred"].sum()
    for t, threshold in enumerate(IOU_THRESHOLDS):
        matched = comp["one_to_one"][t]  # class x stratum
        entry = {
            "recall_all": 100 * _ratio(matched.sum(), ref_all) if ref_all else None,
            "precision": 100 * _ratio(matched.sum(), n_pred) if n_pred else None,
            "f1": 100 * _ratio(2 * matched.sum(), n_pred + ref_all) if n_pred + ref_all else None,
            "recall_by_stratum": {
                name: (100 * _ratio(matched[:, s].sum(), ref[:, s].sum()) if ref[:, s].sum() else None)
                for s, name in enumerate(STRATA)
            },
            "recall_by_class_stratum": {
                str(class_id): {
                    name: (100 * _ratio(matched[c, s], ref[c, s]) if ref[c, s] else None)
                    for s, name in enumerate(STRATA)
                }
                for c, class_id in enumerate(LESION_CLASSES)
            },
        }
        result["one_to_one"][f"iou{threshold:.2f}"] = entry
    for key, name in (("area_error", "area"), ("area_error_ref_support", "area_reference_support")):
        err = stats[key]  # leaves x (S, R, total), percentage points
        mean = np.tensordot(w, err, axes=1) / n_leaves
        mae = np.tensordot(w, np.abs(err), axes=1) / n_leaves
        rmse = np.sqrt(np.tensordot(w, err ** 2, axes=1) / n_leaves)
        result[name] = {
            label: {"mae_pp": float(mae[i]), "rmse_pp": float(rmse[i]), "bias_pp": float(mean[i])}
            for i, label in enumerate(("S", "R", "total"))
        }
    return result


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def resolve_prediction_dir(path, split):
    path = Path(path)
    if any(path.glob("*.png")):
        return path
    runs = sorted(p for p in path.glob(f"*/full_leaf_predictions/{split}") if p.is_dir())
    if not runs:
        runs = sorted(p for p in path.glob(f"full_leaf_predictions/{split}") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No prediction PNGs found under {path}")
    return runs[-1]


def load_mask(path, shape):
    with Image.open(path) as handle:
        mask = np.asarray(handle, dtype=np.int16).copy()
    if mask.shape != shape or not np.isin(mask, (0, 1, 2, 3)).all():
        raise ValueError(f"Invalid prediction mask {path}: expected {shape} with labels 0-3")
    return mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--predictions", nargs="+", required=True, help="NAME=PATH entries")
    parser.add_argument("--reference", default=None, help="NAME of the set that paired differences are taken against")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of source-photograph bootstrap draws (paper: 10000)")
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--min-area", type=int, default=FILTER_MIN_AREA, help="Filtering threshold in pixels (paper: 24)")
    parser.add_argument("--output", required=True, help="JSON file to write")
    args = parser.parse_args(argv)

    sets = {}
    for item in args.predictions:
        if "=" not in item:
            parser.error(f"--predictions entries must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        sets[name] = resolve_prediction_dir(path, args.split)
    if args.reference is not None and args.reference not in sets:
        parser.error("--reference must name one of the --predictions entries")

    records, _ = load_full_leaf_manifest(args.dataset_root, args.split, validate_pixels=False)
    raw = {name: [] for name in sets}
    filtered = {name: [] for name in sets}
    for record in records:
        with Image.open(record.mask_path) as handle:
            target = np.asarray(handle, dtype=np.int16).copy()
        for name, folder in sets.items():
            prediction = load_mask(folder / f"{record.sample_id}.png", target.shape)
            raw[name].append(leaf_statistics(prediction, target))
            filtered[name].append(leaf_statistics(remove_small_components(prediction, args.min_area), target))
    stats = {name: {"raw": stack(raw[name]), "filtered": stack(filtered[name])} for name in sets}
    ones = np.ones(len(records))
    report = {
        "split": args.split, "leaves": len(records),
        "filter_min_area_px": args.min_area,
        "prediction_dirs": {name: str(folder) for name, folder in sets.items()},
        "results": {
            name: {variant: summarize(stats[name][variant], ones) for variant in ("raw", "filtered")}
            for name in sets
        },
    }

    if args.bootstrap > 0:
        sources = sorted({record.source_image_id for record in records})
        leaf_source = np.array([sources.index(record.source_image_id) for record in records])
        rng = np.random.default_rng(args.bootstrap_seed)
        draws = {name: {"raw": [], "filtered": []} for name in sets}
        for _ in range(args.bootstrap):
            counts = np.bincount(rng.integers(0, len(sources), len(sources)), minlength=len(sources))
            weights = counts[leaf_source]
            for name in sets:
                for variant in ("raw", "filtered"):
                    draws[name][variant].append(flatten(summarize(stats[name][variant], weights)))

        def interval(values):
            values = np.array([v for v in values if v is not None], dtype=np.float64)
            if not len(values):
                return None
            return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

        bootstrap = {"draws": args.bootstrap, "seed": args.bootstrap_seed, "unit": "source photograph", "intervals": {}}
        for name in sets:
            for variant in ("raw", "filtered"):
                keys = draws[name][variant][0].keys()
                bootstrap["intervals"].setdefault(name, {})[variant] = {
                    key: interval([d[key] for d in draws[name][variant]]) for key in keys
                }
        if args.reference is not None:
            bootstrap["paired_differences_vs"] = args.reference
            bootstrap["differences"] = {}
            for name in sets:
                if name == args.reference:
                    continue
                for variant in ("raw", "filtered"):
                    base = flatten(report["results"][args.reference][variant])
                    point = flatten(report["results"][name][variant])
                    entry = {}
                    for key in point:
                        if not isinstance(point[key], (int, float)) or base.get(key) is None or point[key] is None:
                            continue
                        diffs = [
                            a[key] - b[key]
                            for a, b in zip(draws[name][variant], draws[args.reference][variant])
                            if a.get(key) is not None and b.get(key) is not None
                        ]
                        entry[key] = {"difference": point[key] - base[key], "interval": interval(diffs)}
                    bootstrap["differences"].setdefault(name, {})[variant] = entry
        report["bootstrap"] = bootstrap

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in sets:
        result = report["results"][name]["raw"]
        o2o = result["one_to_one"]
        print(
            f"{name}: Dice {result['lesion_dice']:.2f}  Small survival {result['small_survival']:.2f}  "
            f"FP/leaf {result['fp_per_leaf']:.2f}  one-to-one recall@0.50 {o2o['iou0.50']['recall_all']:.2f}  "
            f"area MAE total {result['area']['total']['mae_pp']:.3f} pp"
        )
    print(f"Written: {output}")


if __name__ == "__main__":
    main()
