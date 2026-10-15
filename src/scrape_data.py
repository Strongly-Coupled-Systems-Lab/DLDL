"""Apply the best CNN to every preprocessed historical shot, then time the disruptive ones.

Run ``preprocess_historical.py`` first. Shots already in ``all_shots.csv`` are
skipped, and results are appended per batch so a walltime kill loses little.
"""

from __future__ import annotations

import os

# HDF5 file locking is slow and flaky on Lustre (Eagle); reads are read-only anyway.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from util.best_model import load_best_model_cnn, load_best_model_env
from util.disruption_predict import predict_disruption_time
from util.historical import h5_path, read_h5_shot

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "all_shots.csv"
COLUMNS = ["shot_no", "P_disrupt", "t_0", "t_root", "t_f"]
DISRUPT_PROB = 0.5
BATCH_SIZE = 256
NO_TIME = (-1.0, -1.0, -1.0)


class _ShapeOnlyDataset:
    """Stands in for ``IpDataset`` when building ``IpCNN``, which only needs the
    signal length. Avoids loading the full training dataset into memory."""

    def __init__(self, max_length: int) -> None:
        self.data = torch.empty(0, max_length)

    def __len__(self) -> int:
        return 0


def _disruption_time(shot_no: int) -> tuple[float, float, float]:
    """Predict disruption times from the raw signal, as in ``predict.py``."""
    try:
        t, ip = read_h5_shot(h5_path(os.environ["HDF5_DIR"], shot_no))
        return tuple(float(v) for v in predict_disruption_time(ip, t))
    except Exception as e:
        logger.warning(f"Disruption time failed for shot {shot_no}: {e}")
        return NO_TIME


def main() -> None:
    os.chdir(REPO)
    load_best_model_env()
    shots = torch.load(
        os.environ["HISTORICAL_DATA_PATH"], mmap=True, weights_only=True
    )
    data, shot_nos = shots["data"], shots["shot_no"]

    done = (
        set(pd.read_csv(OUT_PATH)["shot_no"]) if OUT_PATH.exists() else set()
    )
    todo = [i for i, s in enumerate(shot_nos.tolist()) if s not in done]
    logger.info(f"{len(todo)} new shots ({len(shot_nos) - len(todo)} already in CSV)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_best_model_cnn(_ShapeOnlyDataset(data.shape[1]))
    if model is None:
        raise FileNotFoundError("No *_best_params.pt checkpoint in best_model/")
    model.to(device).eval()

    # Spawned (not forked) workers: the parent has initialized CUDA. Each re-imports
    # torch, so cap the count; the per-shot timing work is small.
    with ProcessPoolExecutor(
        max_workers=min(16, len(os.sched_getaffinity(0))),
        mp_context=mp.get_context("spawn"),
    ) as executor, torch.inference_mode():
        for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Predicting", unit=" batch"):
            idx = todo[start : start + BATCH_SIZE]
            x = data[idx]
            probs = torch.sigmoid(model(x.to(device))[:, 0]).tolist()
            hits = [k for k, p in enumerate(probs) if p > DISRUPT_PROB]
            times = dict(
                zip(
                    hits,
                    executor.map(_disruption_time, [int(shot_nos[idx[k]]) for k in hits]),
                )
            )
            rows = [
                (int(shot_nos[i]), p, *times.get(k, NO_TIME))
                for k, (i, p) in enumerate(zip(idx, probs))
            ]
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False
            )

    df = pd.read_csv(OUT_PATH).drop_duplicates("shot_no", keep="last")
    df.sort_values("shot_no").to_csv(OUT_PATH, index=False)
    logger.info(
        f"{len(df)} shots ({(df['P_disrupt'] > DISRUPT_PROB).sum()} disruptive) in {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
