"""Preprocess every archived HDF5 shot into one ``.pt`` file for ``scrape_data.py``.

Mirrors ``IpDataset``: signals are normalized by the dataset mean/std (mean of
per-shot means, as in ``get_means``) and zero-padded/truncated to the model's input
length. Saves ``{"shot_no", "data"}`` to ``HISTORICAL_DATA_PATH``.
"""

import os

# HDF5 file locking is slow and flaky on Lustre (Eagle); reads are read-only anyway.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from util.historical import read_h5_shot

_REPO = Path(__file__).resolve().parents[1]
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")

hdf5_dir = Path(os.environ["HDF5_DIR"])
out_path = Path(os.environ["HISTORICAL_DATA_PATH"])
# The CNN's input length is fixed by the training dataset (mmap reads only the header).
max_length = int(
    torch.load(os.environ["DATA_PATH"], mmap=True, weights_only=True).shape[1]
)

logger.remove()
logger.add(sys.stderr, level="INFO")


def _load(path: Path):
    """Return ``(shot_no, padded raw current, mean, mean of squares)`` or ``None``."""
    try:
        _, ip = read_h5_shot(path)
    except Exception as e:
        logger.warning(f"Skipping {path}: {e}")
        return None
    # One NaN would poison the dataset mean/std and therefore every shot.
    if not np.isfinite(ip).all():
        logger.warning(f"Skipping {path}: non-finite current")
        return None
    padded = np.zeros(max_length, dtype=np.float32)
    n = min(len(ip), max_length)
    padded[:n] = ip[:n]
    return int(path.stem), n, padded, float(np.mean(ip)), float(np.mean(ip**2))


if __name__ == "__main__":
    files = sorted(p for p in hdf5_dir.rglob("*.h5") if p.stem.isdigit())
    workers = len(os.sched_getaffinity(0))
    logger.info(f"Preprocessing {len(files)} shots with {workers} workers")

    # One pass over the files: hold raw padded signals, normalize in place after.
    data = torch.zeros(len(files), max_length)
    shot_no = torch.zeros(len(files), dtype=torch.int64)
    length = torch.zeros(len(files), dtype=torch.int64)
    means, sq_means = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(
            executor.map(_load, files, chunksize=16),
            total=len(files),
            desc="Preprocessing",
            unit="shot",
        ):
            if result is None:
                continue
            k = len(means)
            shot_no[k], length[k], data[k] = result[0], result[1], torch.from_numpy(result[2])
            means.append(result[3])
            sq_means.append(result[4])

    n_ok = len(means)
    # torch.save writes a view's whole storage, so trim skipped rows with a copy.
    if n_ok < len(files):
        shot_no, length, data = (x[:n_ok].clone() for x in (shot_no, length, data))

    mean = float(np.mean(means))
    std = float(max(0.0, float(np.mean(sq_means)) - mean**2) ** 0.5)
    logger.info(f"Dataset statistics: mean={mean:.6f}, std={std:.6f}")
    data.sub_(mean).div_(std)
    # Padding stays 0 after normalization, as in load_and_pad_norm.
    for k, n in enumerate(length.tolist()):
        data[k, n:] = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"shot_no": shot_no, "data": data}, out_path)
    logger.info(f"Saved {n_ok}/{len(files)} shots to {out_path}")
