"""Spatial metrics, compact notebook reports and final deliverables."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import iter_analysis_images
from src.evaluation import accuracy_from_predictions, save_predictions
from .methods import STAGES, crop_intersection, map_status, sample_grid

GROUPS = ("all", "both_correct", "resnet_only", "vit_only", "both_wrong")


def entropy(cam):
    if map_status(cam) not in ("ok", "near_constant") or np.size(cam) <= 1:
        return float("nan")
    p = np.asarray(cam, dtype=float).ravel() / np.max(cam)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(np.size(cam)))


def agreement(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Maps must share the same grid")
    if map_status(a) != "ok" or map_status(b) != "ok":
        return {"spearman": float("nan"), "top20_iou": float("nan")}
    ra, rb = (pd.Series(v.ravel()).rank(method="average") for v in (a, b))
    count = math.ceil(.2 * a.size)
    ia, ib = (set(np.argsort(-v.ravel(), kind="stable")[:count]) for v in (a, b))
    return {"spearman": float(np.corrcoef(ra, rb)[0, 1]), "top20_iou": len(ia & ib) / len(ia | ib)}


def _sample(row, maps, name, grid, region=None):
    if name not in maps or row.get("map_status", {}).get(name) != "ok":
        return None
    value = sample_grid(maps[name], row["geometry"], grid, region)
    return value if map_status(value) == "ok" else None


def analyze(first, second=None):
    """E1: ResNet/ViT pair; layerwise_gradcam: one store.

    Return per-image metrics, summary and coverage tables.
    Invalid pairs remain rows with NaN metrics, never artificial zero scores.
    """
    experiment = first.config["experiment"]
    if experiment not in ("E1", "layerwise_gradcam"):
        raise ValueError("Wrong experiment family")
    if first.done != set(first.config["image_ids"]):
        raise ValueError("Cannot summarize an incomplete run")
    if experiment == "E1" and second is None:
        raise ValueError("A paired run is required")
    if second is not None:
        if second.done != set(second.config["image_ids"]):
            raise ValueError("Paired run is incomplete")
        if (second.config["experiment"] != "E1" or second.config["model"] != "vit_b_16"
                or (experiment == "E1" and first.config["model"] != "resnet50")):
            raise ValueError("Expected ResNet E1/ViT E1")
        for key in ("code", "shared_code", "manifest", "baseline", "dataset_config", "grids", "preprocessing", "dtype", "torch", "torchvision"):
            if first.config[key] != second.config[key]:
                raise ValueError(f"Paired run mismatch: {key}")
        if not set(first.config["image_ids"]).issubset(second.done):
            raise ValueError("Paired run does not contain all required IDs")
    partner = {row["image_id"]: (row, maps) for row, maps in second.items()} if second else {}
    rows, coverage = [], []
    stores = [first, second] if experiment == "E1" else [first]
    for store in stores:
        names = list(STAGES) if store.config["experiment"] == "layerwise_gradcam" else ["near_head"]
        for row, maps in store.items():
            for name in names:
                status = row.get("map_status", {}).get(name, row["status"])
                coverage.append({"image_id": row["image_id"], "group": row["group"],
                                 "model": store.config["model"], "stage": name, "status": status,
                                 "baseline_match": row.get("baseline_match", np.nan),
                                 "degenerate": status not in ("ok", "decode_error", "extraction_error"),
                                 "failed": status in ("decode_error", "extraction_error")})
    for row, maps in first.items():
        other, other_maps = partner.get(row["image_id"], ({}, {}))
        if other and (row["label"] != other["label"] or row["group"] != other["group"]):
            raise ValueError("Paired labels or baseline correctness groups differ")
        for grid in (7, 14):
            base = {"image_id": row["image_id"], "group": row["group"], "grid": grid,
                    "model": first.config["model"]}
            if experiment == "E1":
                base["model"] = "resnet50_vs_vit_b_16"
                a = b = None
                if "geometry" in row and "geometry" in other:
                    region = crop_intersection(row["geometry"], other["geometry"])
                    a = _sample(row, maps, "near_head", grid, region)
                    b = _sample(other, other_maps, "near_head", grid, region)
                entry = {**base, "comparison": "resnet_vs_vit", "entropy_resnet": np.nan,
                         "entropy_vit": np.nan, "entropy_delta_vit_minus_resnet": np.nan,
                         "spearman": np.nan, "top20_iou": np.nan}
                if a is not None and b is not None:
                    entry.update(agreement(a, b), entropy_resnet=entropy(a), entropy_vit=entropy(b),
                                 entropy_delta_vit_minus_resnet=entropy(b) - entropy(a))
                rows.append(entry)
            else:
                names = list(STAGES)
                previous = None
                for name in names:
                    cam = _sample(row, maps, name, grid)
                    rows.append({**base, "stage": name, "entropy": entropy(cam) if cam is not None else np.nan,
                                 "native_shape": str(row.get("native_shapes", {}).get(name)),
                                 **(agreement(previous, cam) if experiment == "layerwise_gradcam" and previous is not None
                                    and cam is not None else {"spearman": np.nan, "top20_iou": np.nan})})
                    previous = cam
    table, coverage = pd.DataFrame(rows), pd.DataFrame(coverage)
    summaries, rates = [], []
    keys = [key for key in ("grid", "model", "stage", "comparison") if key in table]
    metrics = [key for key in table if key.startswith("entropy") or key in ("spearman", "top20_iou")]
    for group in ["all", "both_correct", "resnet_only", "vit_only", "both_wrong"]:
        frame = table if group == "all" else table[table.group.eq(group)]
        for identity, part in frame.groupby(keys, dropna=False):
            identity = identity if isinstance(identity, tuple) else (identity,)
            for metric in metrics:
                valid = part[metric].dropna()
                summaries.append({**dict(zip(keys, identity)), "group": group, "metric": metric,
                                  "n_total": len(part), "n_valid": len(valid),
                                  "mean": valid.mean() if len(valid) else np.nan,
                                  "median": valid.median() if len(valid) else np.nan})
        frame = coverage if group == "all" else coverage[coverage.group.eq(group)]
        for (model, stage), part in frame.groupby(["model", "stage"]):
            rates.append({"group": group, "model": model, "stage": stage, "n_images": len(part),
                          "degenerate_rate": part.degenerate.mean(), "failure_rate": part.failed.mean(),
                          "n_predictions": int(part.baseline_match.notna().sum()),
                          "baseline_mismatch_rate": 1 - part.baseline_match.dropna().mean()
                          if part.baseline_match.notna().any() else np.nan})
    return {"per_image": table, "summary": pd.DataFrame(summaries), "coverage": coverage,
            "coverage_summary": pd.DataFrame(rates)}


def show_results(results):
    """Display E1 metrics and coverage without dumping per-image rows."""
    from IPython.display import display
    summary = results["E1"]["summary"]
    print("E1: mean spatial metrics on paired valid maps; grids 7 (primary), 14 (sensitivity)")
    display(summary[summary.group.eq("all")].pivot(index="metric", columns="grid", values="mean").round(4))
    print("E1: metrics by baseline correctness group, primary grid")
    display(summary[summary.grid.eq(7)][["group", "metric", "n_total", "n_valid", "mean", "median"]].round(4))
    print("Coverage: failure/degenerate/mismatch rates are fractions (0..1); counts include failed attempts")
    display(pd.concat([r["coverage_summary"].assign(condition=c)
                       for c, r in results.items()], ignore_index=True).round(4))


def plot_summary(result, condition):
    """Quantitative figure for exactly one experiment, with explicit model labels."""
    import matplotlib.pyplot as plt
    if condition != "E1":
        raise ValueError("Choose E1")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = {"spearman": "CAM rank agreement (Spearman)",
              "top20_iou": "Top-20% CAM overlap (IoU)",
              "entropy_delta_vit_minus_resnet": "CAM entropy: ViT minus ResNet"}
    for ax, (metric, label) in zip(axes, labels.items()):
        frame = result["per_image"]
        for grid in (7, 14):
            values = frame.loc[frame.grid.eq(grid), metric].dropna()
            ax.hist(values, bins=30, alpha=.5, label=f"{grid}x{grid}, n={len(values)}")
        ax.set(title=label, xlabel="Metric value", ylabel="Number of images")
        ax.legend(fontsize=8)
    fig.suptitle("E1 | Grad-CAM comparison: ResNet50 vs ViT-B/16 | Ground-truth target, crop intersection")
    fig.tight_layout()
    return fig


def plot_examples(root, stores, selected, limit=4, *, data_dir=None):
    """One paired panel: first ID per available baseline group, independent of maps."""
    import matplotlib.pyplot as plt
    from PIL import Image
    selected = selected.sort_values("image_id").groupby("group", sort=True).head(1).head(limit)
    ids = set(selected.image_id)
    captured = [{row["image_id"]: (row, maps) for row, maps in store.items()
                 if row["image_id"] in ids} for store in stores]
    originals = {image_id: image for image, _, image_id in iter_analysis_images(
        selected.to_dict("records"), data_dir=Path(data_dir) if data_dir is not None else Path(root) / "data/imagenet/data")}
    condition = stores[0].config["experiment"]
    if condition != "E1":
        raise ValueError("Choose E1 stores")
    panels = [(0, None), (0, "near_head"), (1, None), (1, "near_head")]
    if any(store.config["experiment"] != condition for store in stores):
        raise ValueError("Do not mix experiments in one example figure")
    fig, axes = plt.subplots(len(selected), len(panels), squeeze=False,
                             figsize=(3.5 * len(panels), 3.4 * len(selected)))
    for axes_row, record in zip(axes, selected.to_dict("records")):
        for ax, (index, stage) in zip(axes_row, panels):
            row, maps = captured[index][record["image_id"]]
            config = stores[index].config
            ax.set_xticks([])
            ax.set_yticks([])
            status = row.get("map_status", {}).get(stage, row["status"])
            view = "Input crop" if stage is None else ("Grad-CAM" if condition == "E1" else stage)
            ax.set_title(f"{config['model']} | {view}\npred={row.get('prediction', '?')} | {status}", fontsize=8)
            if "geometry" not in row:
                ax.text(.1, .5, row["status"])
                continue
            geometry = row["geometry"]
            image = originals[record["image_id"]].resize(tuple(geometry["resized_size"]), Image.Resampling.BILINEAR)
            x, y = geometry["crop_offset"]
            w, h = geometry["crop_size"]
            ax.imshow(image.crop((x, y, x + w, y + h)))
            if stage in maps:
                heatmap = sample_grid(maps[stage], geometry, 224)
                if np.isfinite(heatmap).all():
                    ax.imshow(heatmap, cmap="magma", vmin=0, vmax=1, alpha=.45)
        axes_row[0].set_ylabel(f"{record['image_id']}\nGT={record['label']}\n{record['group']}", fontsize=8)
    title = "E1 | ResNet50 vs ViT-B/16 Grad-CAM | Ground-truth target"
    fig.suptitle(title + "\nEach input/overlay pair uses the same official crop; IDs selected by baseline group")
    fig.tight_layout()
    return fig


def export_final(stores, results, figures, output, figure_output):
    """Save metric/provenance files and experiment-specific figures; never overwrite finals.

    metrics.csv uses long rows (metric/value); per_image.csv combines spatial
    metrics and coverage rows, distinguished by record_type and condition.
    """
    output, figure_output = Path(output), Path(figure_output)
    by_model, identities, exclusions, accuracy = {}, [], [], []
    for store in stores:
        if store.done != set(store.config["image_ids"]):
            raise ValueError("Cannot export an incomplete run")
        config, rows = store.config, []
        identities.append(config)
        for row, _ in store.items():
            if row["status"] != "ok":
                exclusions.append(dict(image_id=row["image_id"], model=config["model"],
                                       condition=config["experiment"], reason=row.get("error", row["status"])))
                continue
            predicted, label = row["top5"], int(row["label"])
            rows.append(dict(image_id=row["image_id"], model_name=config["model"], label=label,
                             **{f"pred_top{i}": p for i, p in enumerate(predicted, 1)},
                             top1_correct=int(predicted[0] == label), top5_correct=int(label in predicted),
                             experiment="gradcam", condition=config["experiment"], severity=""))
        scores = accuracy_from_predictions(rows) if rows else {"num_images": 0}
        accuracy.append(dict(model=config["model"], condition=config["experiment"],
                             n_requested=len(store.done), n_failed=len(store.done) - len(rows), **scores))
        by_model.setdefault(config["model"], []).extend(rows)
    accuracy = pd.DataFrame(accuracy)
    tables, details = [], []
    for condition, result in results.items():
        tables.append(result["summary"].rename(columns={"mean": "value"}).assign(kind="spatial", condition=condition))
        coverage = result["coverage_summary"]
        keys = ["group", "model", "stage"]
        tables.append(coverage.melt(id_vars=keys, var_name="metric", value_name="value")
                      .assign(kind="coverage", condition=condition))
        for kind in ("per_image", "coverage"):
            details.append(result[kind].assign(record_type=kind, condition=condition))
    tables.append(accuracy.melt(id_vars=["model", "condition"], var_name="metric", value_name="value")
                  .assign(kind="accuracy", group="all"))
    metrics, per_image = pd.concat(tables, ignore_index=True), pd.concat(details, ignore_index=True)
    predictions = {name: rows for name, rows in by_model.items() if rows}
    paths = ([output / f"{name}_predictions.csv" for name in predictions]
             + [output / name for name in ("metrics.csv", "per_image.csv", "metadata.json")]
             + [figure_output / f"{name}.png" for name in figures])
    if any(path.exists() for path in paths):
        raise FileExistsError("Final files already exist. Set SAVE_FINAL=False to inspect, or choose a new run ID.")
    output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    for name, rows in predictions.items():
        save_predictions(rows, output / f"{name}_predictions.csv")
    metrics.to_csv(output / "metrics.csv", index=False)
    per_image.to_csv(output / "per_image.csv", index=False)
    metadata = dict(experiment="gradcam", runs=identities, exclusions=exclusions,
                    groups="saved baseline predictions", severity="not applicable (clean inputs)",
                    schemas={"metrics.csv": "kind=spatial/coverage/accuracy; metric/value; median/n_valid for spatial rows",
                             "per_image.csv": "record_type=per_image (spatial metrics) or coverage (status/rates); condition=E1/E2"},
                    units="accuracy: percent; coverage rates, IoU and normalized entropy: 0..1; Spearman: -1..1",
                    figures={name: ("E1: ResNet50 vs ViT-B/16 ground-truth Grad-CAM" if name.startswith("E1")
                                   else "E2: ViT-B/16 class-agnostic CLS attention") for name in figures})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for name, figure in figures.items():
        figure.savefig(figure_output / f"{name}.png", dpi=160, bbox_inches="tight")
    print(f"Saved {len(paths)} final files:\n  Metrics: {output}\n  Figures: {figure_output}")
    return accuracy
