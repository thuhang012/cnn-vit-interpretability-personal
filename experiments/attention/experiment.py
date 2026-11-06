"""Attention cohorts and resumable experiment execution."""

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
import pandas as pd

from src.data import load_analysis_manifest, iter_analysis_images
from src.models import load_model
from .methods import attention, map_status, preprocess

MODELS = ("resnet50", "vit_b_16")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_selection(root):
    """Reuse the shared manifest; enrich IDs with baseline correctness groups."""
    root = Path(root)
    path = root / "data/manifests/analysis_10000_seed42.csv"
    config = json.loads((root / "configs/analysis_set.json").read_text(encoding="utf-8"))
    if checksum(path) != config["manifest_sha256"]:
        raise ValueError("Analysis manifest checksum differs")
    selected = pd.DataFrame(load_analysis_manifest(path))
    for name in MODELS:
        predictions = pd.read_csv(root / f"results/metrics/baseline_notebook/full/{name}_predictions.csv")
        joined = selected.merge(predictions, on="image_id", validate="one_to_one", suffixes=("", "_pred"))
        if len(joined) != len(selected) or not joined.label.eq(joined.label_pred).all():
            raise ValueError(f"Baseline IDs/labels differ: {name}")
        selected[name + "_prediction"] = joined.pred_top1.to_numpy()
    a = selected.resnet50_prediction.eq(selected.label)
    b = selected.vit_b_16_prediction.eq(selected.label)
    selected["group"] = np.select([a & b, a, b], ["both_correct", "resnet_only", "vit_only"], default="both_wrong")
    return selected.sort_values("image_id").reset_index(drop=True)


def subsets(selected, seed=42):
    """Select one image per class with the original seeded selection rule."""
    if selected.label.nunique() != 1000 or selected.image_id.duplicated().any():
        raise ValueError("Subset selection requires all 1,000 classes and unique IDs")
    rng = random.Random(seed)
    indices = [rng.choice(g.sort_values("image_id").index.tolist())
               for _, g in selected.groupby("label", sort=True)]
    return selected.loc[indices].sort_values("image_id").reset_index(drop=True)


def images(root, selected, *, data_dir=None):
    """Join experiment annotations by ID; shared decoding errors stop the run."""
    records = selected.to_dict("records")
    lookup = {row["image_id"]: row for row in records}
    for image, label, image_id in iter_analysis_images(records, data_dir=Path(data_dir) if data_dir is not None else Path(root) / "data/imagenet/data"):
        yield lookup[image_id], image


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    """Small chunk checkpoints; JSON is the commit marker for its map archive.

    Single writer per run directory. Incomplete archives from interruption are
    ignored. IDs in committed chunks are never rerun or appended twice.
    """
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        path = self.directory / "config.json"
        if config is None:
            self.config = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
                raise ValueError(f"Checkpoint configuration mismatch: {directory}")
            if not path.exists():
                _json(path, config)
            self.config = config
        self.done = set()
        for row, _ in self.items():
            if row["image_id"] in self.done:
                raise ValueError("Duplicate committed image ID")
            self.done.add(row["image_id"])
        if not self.done.issubset(set(self.config["image_ids"])):
            raise ValueError("Checkpoint contains unexpected IDs")

    def append(self, records):
        ids = [row["image_id"] for row, _ in records]
        if len(set(ids)) != len(ids) or set(ids) & self.done:
            raise ValueError("Refusing duplicate checkpoint IDs")
        if not set(ids).issubset(self.config["image_ids"]):
            raise ValueError("Unexpected checkpoint IDs")
        if not records:
            return
        stem = self.directory / ("chunk-" + uuid.uuid4().hex)
        arrays, rows = {}, []
        for index, (record, maps) in enumerate(records):
            row = {**record, "map_keys": {}}
            for name, cam in maps.items():
                key = f"r{index}_{name}"
                arrays[key] = cam
                row["map_keys"][name] = key
            rows.append(row)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        _json(stem.with_suffix(".json"), rows)
        self.done.update(ids)

    def items(self):
        for path in sorted(self.directory.glob("chunk-*.json")):
            rows = json.loads(path.read_text(encoding="utf-8"))
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                for row in rows:
                    yield row, {key: archive[value].copy() for key, value in row["map_keys"].items()}


def _identity(root, selected, name, experiment, batch_size, device, *, seed=42, run_id="run", model_metadata=None, data_dir=None):
    import torch
    import torchvision
    import pyarrow
    from PIL import __version__ as pillow_version
    root = Path(root)
    code = {name: checksum(Path(__file__).with_name(name))
            for name in ("methods.py", "experiment.py", "analysis.py")}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL))
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = None, None  # A Drive copy may not include Git history.
    baseline = root / "results/metrics/baseline_notebook/full"
    return {"protocol": 1, "code": code,
            "shared_code": {p: checksum(root / "src" / p) for p in ("models.py", "data.py", "evaluation.py", "evaluate_baseline.py")},
            "experiment": experiment, "model": name,
            "run_id": run_id, "seed": seed, "git_revision": revision, "git_dirty": dirty,
            "model_metadata": model_metadata,
            "target": "class_agnostic",
            "preprocessing": "official",
            "manifest_path": "data/manifests/analysis_10000_seed42.csv",
            "selection_rule": "committed 10k cohort; optional class-balanced subset",
            "parameters": {"attention": "CLS-to-patch, mean heads; rollout adds identity and row-normalizes",
                           "alignment": "original-coordinate crop intersection", "tie_break": "row-major"},
            "image_ids": sorted(selected.image_id.tolist()),
            "records": hashlib.sha256(selected.sort_values("image_id").to_csv(index=False).encode()).hexdigest(),
            "manifest": checksum(root / "data/manifests/analysis_10000_seed42.csv"),
            "dataset_config": checksum(root / "configs/analysis_set.json"),
            "data_dir": str((Path(data_dir) if data_dir is not None else root / "data/imagenet/data").resolve()),
            "baseline": {n: checksum(baseline / f"{n}_predictions.csv") for n in MODELS},
            "batch_size": batch_size, "dtype": "float32", "device": str(device),
            "threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "pillow": pillow_version, "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "grids": [7, 14], "top_fraction": .2}


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract(model, x, labels, name, experiment):
    return attention(model, x, rollout=experiment == "attention_rollout")


def _records(batch, result, name, elapsed=0., read_seconds=0.):
    records = []
    for i, (record, _, geometry) in enumerate(batch):
        maps = {key: value[i] for key, value in result["maps"].items()}
        diagnostics = {key: {k: v if k.endswith("shape") else v[i] for k, v in value.items()}
                       for key, value in result.get("diagnostics", {}).items()}
        for diagnostic in diagnostics.values():
            if not np.isfinite(diagnostic["gradient_abs_mean"]):
                diagnostic["gradient_abs_mean"] = None
        statuses = {key: map_status(cam) for key, cam in maps.items()}
        for key, diagnostic in diagnostics.items():
            if not diagnostic["finite"]:
                statuses[key] = "nonfinite_features"
            elif diagnostic["gradient_abs_mean"] == 0:
                statuses[key] = "zero_gradient"
        prediction = int(result["logits"][i].argmax())
        top5 = result["logits"][i].topk(5).indices.tolist()
        records.append(({**record, "geometry": geometry, "status": "ok", "prediction": prediction,
                         "baseline_match": prediction == int(record[name + "_prediction"]), "top5": top5,
                         "target_logit": float(result["logits"][i, int(record["label"])]),
                         "map_status": statuses, "diagnostics": diagnostics,
                         "native_shapes": {key: list(cam.shape) for key, cam in maps.items()},
                         "compute_seconds": elapsed / len(batch), "read_seconds": read_seconds / len(batch)}, maps))
    return records


def run(root, selected, name, experiment, output, batch_size=1, device="cpu", chunk_size=256, *, seed=42, allow_download=False, data_dir=None):
    """Run/resume one model. OOM and mapping errors stop, never shrink the dataset."""
    import torch
    if (experiment not in ("E2", "attention_rollout") or batch_size < 1 or selected.empty
            or selected.image_id.duplicated().any() or chunk_size < 1):
        raise ValueError("Invalid experiment, selection or batch size")
    if name != "vit_b_16":
        raise ValueError("Attention experiments require ViT")
    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, transform, metadata = load_model(name, device, allow_download=allow_download)
    store = RunStore(Path(output) / experiment / name,
                     _identity(root, selected, name, experiment, batch_size, device,
                               seed=seed, run_id=Path(output).parent.name, model_metadata=metadata, data_dir=data_dir))
    remaining = selected[~selected.image_id.isin(store.done)]
    if remaining.empty:
        del model
        return store
    buffer, batch = [], []

    def flush(force=False):
        if len(buffer) >= chunk_size or (force and buffer):
            store.append(buffer)
            buffer.clear()

    def process(batch, read_seconds):
        x = torch.stack([item[1] for item in batch]).to(device)
        labels = torch.tensor([item[0]["label"] for item in batch], device=device)
        _sync(device)
        start = time.perf_counter()
        try:
            result = _extract(model, x, labels, name, experiment)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as error:
            if len(batch) > 1:
                for item in batch:
                    process([item], read_seconds / len(batch))
            else:
                buffer.append(({**batch[0][0], "status": "extraction_error", "error": str(error)}, {}))
            flush()
            return
        _sync(device)
        buffer.extend(_records(batch, result, name, time.perf_counter() - start, read_seconds))
        flush()

    try:
        start = time.perf_counter()
        for record, image in images(root, remaining, data_dir=data_dir):
            try:
                tensor, geometry = preprocess(image, transform)
            except (OSError, ValueError) as error:
                buffer.append(({**record, "status": "decode_error", "error": "preprocess: " + str(error)}, {}))
                flush()
                continue
            batch.append((record, tensor, geometry))
            if len(batch) == batch_size:
                process(batch, time.perf_counter() - start)
                batch = []
                start = time.perf_counter()
        if batch:
            process(batch, time.perf_counter() - start)
        flush(True)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if store.done != set(selected.image_id):
        raise ValueError("Incomplete experiment")
    return store


def prepare_cohorts(root, analysis_size=10000, seed=42):
    """Return the experiment cohort without writing extra CSV files."""
    if analysis_size not in (5000, 10000):
        raise ValueError("analysis_size must be 5000 or 10000")
    selected = load_selection(root)
    if analysis_size != 10000:
        selected = (selected.groupby("label", group_keys=False)
                    .sample(n=analysis_size // 1000, random_state=seed)
                    .sort_values("image_id").reset_index(drop=True))
    return subsets(selected, seed=seed)


def experiment_stores(root, selected, condition, output, *, batch_sizes=None, device="cpu",
                      seed=42, allow_download=False, data_dir=None):
    """Run/resume or inspect one experiment, independently of the other."""
    if condition != "E2":
        raise ValueError("Choose E2 (attention)")
    stores = []
    names = ("vit_b_16",)
    for name in names:
        path = Path(output) / condition / name
        if batch_sizes is not None:
            store = run(root, selected, name, condition, output, batch_sizes[name], device,
                        seed=seed, allow_download=allow_download, data_dir=data_dir)
        else:
            store = RunStore(path) if (path / "config.json").is_file() else None
        expected = set(selected.image_id)
        complete = store is not None and store.done == expected and set(store.config["image_ids"]) == expected
        count = len(store.done & expected) if store else 0
        print(f"{condition}/{name}: {count}/{len(expected)} processed; complete={complete}")
        stores.append(store if complete else None)
    return stores if all(store is not None for store in stores) else None
