"""Prepare Colab storage and dependencies; code synchronization is in the notebook."""

from pathlib import Path
import csv
import errno
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile


SHARDS = {f"validation-{i:05d}-of-00014.parquet" for i in range(14)}


def _drive_io(operation, drive_root, mount_drive):
    """Retry disconnected Drive operations at most twice; propagate other errors."""
    for attempt in range(3):
        try:
            return operation()
        except OSError as error:
            if error.errno not in (errno.EIO, errno.ENOTCONN, 107):
                raise
            if not mount_drive or attempt == 2:
                raise RuntimeError(
                    "Google Drive is disconnected. Reconnect Drive and rerun setup; "
                    "existing checkpoints and completed local data are unchanged."
                ) from error
            from google.colab import drive
            print(f"Drive disconnected; reconnecting ({attempt + 1}/2)...", flush=True)
            drive.mount(str(drive_root), force_remount=True)


def _cache_archive(archive, destination, stat, drive_root, mount_drive):
    """Copy Drive ZIP sequentially, publishing only a complete local copy."""
    if destination.is_file() and destination.stat().st_size == stat.st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")

    def copy():
        print("Copying dataset ZIP from Drive to local Colab storage...", flush=True)
        with archive.open("rb") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        current = archive.stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Dataset ZIP changed during transfer. Rerun setup.")
        if partial.stat().st_size != stat.st_size:
            raise ValueError("Dataset ZIP transfer is incomplete. Rerun setup.")
        partial.replace(destination)

    _drive_io(copy, drive_root, mount_drive)
    return destination


def validate_data(root, data):
    """Check packaged manifest and selected Parquet IDs/labels without decoding images."""
    import pyarrow.parquet as pq

    def records(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return sorted((int(row["dataset_index"]), row["image_id"], int(row["label"]))
                          for row in csv.DictReader(stream))

    expected_rows = records(root / "data/manifests/analysis_10000_seed42.csv")
    packaged = data / "analysis_10000_seed42.csv"
    if packaged.is_file() and records(packaged) != expected_rows:
        raise ValueError("ZIP manifest does not match the repository analysis manifest.")
    expected = {image_id: label for _, image_id, label in expected_rows}
    seen = set()
    for name in sorted(SHARDS):
        path = data / name
        image_type = pq.ParquetFile(path).schema_arrow.field("image").type
        if not {"bytes", "path"}.issubset(field.name for field in image_type):
            raise ValueError(f"Unsupported image schema in {name}.")
        ids, labels = pq.read_table(path, columns=["image.path", "label"]).columns
        for image_id, label in zip(ids.to_pylist(), labels.to_pylist()):
            if image_id not in expected:
                continue
            if image_id in seen or label != expected[image_id]:
                raise ValueError(f"Duplicate ID or label mismatch: {image_id}")
            seen.add(image_id)
    if seen != expected.keys():
        raise ValueError(f"Dataset is missing {len(expected.keys() - seen)} analysis IDs.")
    print(f"Dataset verified: {len(seen):,} analysis IDs and labels match the repository manifest.")


def setup(root, *, data_dir=None, results_dir=None,
          project_zip="cv_2026/data_10k.zip", mount_drive=True,
          install_dependencies=True):
    root = Path(root).resolve()
    drive_root = Path("/content/drive")
    mydrive = drive_root / "MyDrive"

    def connect():
        if not mydrive.is_dir():
            if not mount_drive:
                raise RuntimeError("Mount Google Drive before setup.")
            from google.colab import drive
            drive.mount(str(drive_root))
        # A stale mount can still appear to be a directory; perform a real read.
        next(mydrive.iterdir(), None)

    _drive_io(connect, drive_root, mount_drive)

    def resolve(value, base):
        value = Path(value).expanduser()
        return _drive_io(lambda: (value if value.is_absolute() else base / value).resolve(),
                         drive_root, mount_drive)

    results = resolve(results_dir or "cv_2026", mydrive)
    if not any(results.is_relative_to(path.resolve()) for path in
               (mydrive, drive_root / "Shareddrives")):
        raise ValueError("RESULTS_DIR must be on mounted Google Drive.")
    _drive_io(lambda: results.mkdir(parents=True, exist_ok=True), drive_root, mount_drive)
    os.environ["TORCH_HOME"] = str(mydrive / "cv_2026/torch-cache")

    if data_dir is not None:
        data = resolve(data_dir, root)
    else:
        archive = resolve(project_zip, mydrive)
        try:
            stat = _drive_io(archive.stat, drive_root, mount_drive)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset ZIP is missing: {archive}. Set PROJECT_ZIP or DATA_DIR.") from None
        else:
            signature = hashlib.sha256(
                f"{archive}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()[:16]
            data = Path("/content/imagenet-cache") / signature / "data"
            ready = data.parent / ".ready"
            if not ready.is_file() or not all((data / name).is_file() for name in SHARDS):
                local_archive = _cache_archive(archive, data.parent / "dataset.zip", stat,
                                               drive_root, mount_drive)
                ready.unlink(missing_ok=True)
                with zipfile.ZipFile(local_archive) as bundle:
                    members = {}
                    for member in bundle.infolist():
                        parts = Path(member.filename).parts
                        if (len(parts) == 2 and parts[0] == "data_10k") or (
                            len(parts) >= 4 and tuple(parts[-4:-1]) == ("data", "imagenet", "data")
                        ):
                            if parts[-1] in SHARDS:
                                if parts[-1] in members:
                                    raise ValueError(f"Duplicate dataset shard: {parts[-1]}")
                                members[parts[-1]] = member
                    if set(members) != SHARDS:
                        raise ValueError("ZIP must contain all 14 validation shards under data_10k/.")
                    print("Extracting ImageNet from local ZIP (once per runtime)...", flush=True)
                    data.mkdir(parents=True, exist_ok=True)
                    for name, member in sorted(members.items()):
                        # Write only known filenames; never extract arbitrary ZIP paths.
                        with bundle.open(member) as source, (data / name).open("wb") as target:
                            shutil.copyfileobj(source, target)
                    for name in ("analysis_10000_seed42.csv", "subset_info.json", "README.md"):
                        member_name = f"data_10k/{name}"
                        if member_name in bundle.namelist():
                            with bundle.open(member_name) as source, (data / name).open("wb") as target:
                                shutil.copyfileobj(source, target)
                    ready.write_text("complete", encoding="utf-8")
    missing = sorted(name for name in SHARDS if not (data / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ImageNet shards in {data}. Put the existing "
            "cv_2026/data_10k.zip in MyDrive, or set DATA_DIR to the folder "
            "containing all 14 validation Parquet files."
        )

    if install_dependencies:
        digest = hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()
        marker = Path("/content") / f".cnn-vit-dependencies-{digest}"
        if not marker.is_file():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                            str(root / "requirements.txt")], check=True)
            marker.touch()
    validate_data(root, data)
    return root, data.resolve(), results
