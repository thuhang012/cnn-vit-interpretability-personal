"""Prepare Colab storage and dependencies; code synchronization is in the notebook."""

from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile


SHARDS = {f"validation-{i:05d}-of-00014.parquet" for i in range(14)}


def setup(root, *, data_dir=None, results_dir=None,
          project_zip="cnn-vit-interpretability.zip", mount_drive=True,
          install_dependencies=True):
    root = Path(root).resolve()
    drive_root = Path("/content/drive")
    if mount_drive and not (drive_root / "MyDrive").is_dir():
        from google.colab import drive
        drive.mount(str(drive_root))
    mydrive = drive_root / "MyDrive"
    if not mydrive.is_dir():
        raise RuntimeError("Mount Google Drive before setup.")

    def resolve(value, base):
        value = Path(value).expanduser()
        return (value if value.is_absolute() else base / value).resolve()

    results = resolve(results_dir or "cnn-vit-interpretability-results", mydrive)
    if not any(results.is_relative_to(path.resolve()) for path in
               (mydrive, drive_root / "Shareddrives")):
        raise ValueError("RESULTS_DIR must be on mounted Google Drive.")
    results.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(mydrive / "torch-cache")

    if data_dir is not None:
        data = resolve(data_dir, root)
    else:
        archive = resolve(project_zip, mydrive)
        if archive.is_file():
            stat = archive.stat()
            signature = hashlib.sha256(
                f"{archive}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()[:16]
            data = Path("/content/imagenet-cache") / signature / "data"
            ready = data.parent / ".ready"
            if not ready.is_file() or not all((data / name).is_file() for name in SHARDS):
                with zipfile.ZipFile(archive) as bundle:
                    members = {}
                    for member in bundle.infolist():
                        parts = Path(member.filename).parts
                        if len(parts) >= 4 and tuple(parts[-4:-1]) == ("data", "imagenet", "data"):
                            if parts[-1] in SHARDS:
                                if parts[-1] in members:
                                    raise ValueError(f"Duplicate dataset shard: {parts[-1]}")
                                members[parts[-1]] = member
                    if set(members) != SHARDS:
                        raise ValueError("ZIP must contain all 14 shards under data/imagenet/data/.")
                    print("Extracting ImageNet from Drive ZIP to local Colab storage (once per runtime)...", flush=True)
                    data.mkdir(parents=True, exist_ok=True)
                    for name, member in sorted(members.items()):
                        # Write only known filenames; never extract arbitrary ZIP paths.
                        with bundle.open(member) as source, (data / name).open("wb") as target:
                            shutil.copyfileobj(source, target)
                    ready.write_text("complete", encoding="utf-8")
        else:
            data = mydrive / "cnn-vit-data/imagenet/data"
    missing = sorted(name for name in SHARDS if not (data / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ImageNet shards in {data}. Put the existing "
            "cnn-vit-interpretability.zip in MyDrive, or set DATA_DIR to the folder "
            "containing all 14 validation Parquet files."
        )

    if install_dependencies:
        digest = hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()
        marker = Path("/content") / f".cnn-vit-dependencies-{digest}"
        if not marker.is_file():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                            str(root / "requirements.txt")], check=True)
            marker.touch()
    return root, data.resolve(), results
