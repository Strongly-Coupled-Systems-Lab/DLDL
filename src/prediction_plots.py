"""Evaluate the best model on the dev holdout set."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import matplotlib.axes

# Prevent macOS OpenMP thread deadlock on CPU inference
if not torch.cuda.is_available():
    torch.set_num_threads(1)

from loguru import logger
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import pandas as pd

from model.dataset import IpDataset
from util.data_loading import _read_signal_file
from util.disruption_predict import predict_disruption_time, PredictionType
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env

_REPO = Path(__file__).resolve().parents[1]
# Env paths are relative to the repo root; run from there so they resolve directly.
os.chdir(_REPO)
load_best_model_env()
data_path = Path(os.environ["DATA_PATH"])
labels_path = Path(os.environ["TRAIN_LABELS_PATH"])
model_dir = best_model_dir()
predictions_csv = model_dir / "predictions.csv"


def gaussian(x: np.ndarray, mu: float, sigma: float):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_gaussian(
    ax: matplotlib.axes.Axes,
    arr: np.ndarray,
    label="",
    color=None,
):
    sigma = arr.std()
    mu = arr.mean()

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 500)
    ax.plot(
        x,
        gaussian(x, mu, sigma),
        linewidth=2,
        color=color,
        label=label,
    )


def get_pred_type_label(prediction_type: "t_root" | "t_0" | "t_f"):
    if prediction_type == "t_0":
        return "t_0"
    elif prediction_type == "t_root":
        return "t_\\mathrm{root}"
    else:
        return "t_f"


def generate_err_histogram(df: pd.DataFrame, prediction_type: str) -> None:
    # Errors in seconds; keep those within +/-10 ms, then convert to milliseconds.
    diff = df["diff"][(df["diff"] < 10e-3) & (df["diff"] > -10e-3)] * 1e3
    sigma = diff.std()
    mu = np.abs(diff.mean())
    logger.success(
        f"Disruption time error (milliseconds, n={len(diff)}): "
        f"mean={mu:.3f}, median={diff.median():.3f}, variance={diff.var():.3f}, stddev={sigma:.3f}"
    )
    first_quartile = diff[np.abs(diff) < sigma]
    second_quartile = diff[np.abs(diff) < 2 * sigma]
    third_quartile = diff[np.abs(diff) < 3 * sigma]
    logger.success(
        f"{100*len(first_quartile) / len(diff):2f}% shots within 1 stddev, {100*len(second_quartile) / len(diff):2f}% shots within 2 stddev, {(100*len(third_quartile) / len(diff)):2f}% shots within 3 stddev"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    start = mu - 5 * sigma
    end = mu + 5 * sigma
    ax.hist(
        diff,
        bins=60,
        density=True,
        color="#B8C4D0",
        edgecolor="#5A6B7B",
        linewidth=0.5,
        alpha=0.9,
        range=(start, end),
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(f"${get_pred_type_label(prediction_type)}$ (ms)")
    ax.set_ylabel("Fraction of shots")

    # Overlay the best-fit Gaussian. Explicit high-contrast accents against the
    # neutral gray bars: full-data fit in blue, the 3-sigma-trimmed fit in orange.
    plot_gaussian(
        ax,
        diff,
        color="#0072B2",
        label=f"Full test set",
    )

    three_sigma = diff[np.abs(diff) < (3 * sigma)]
    logger.info(f"mu={(three_sigma.mean()):.1f};sigma={three_sigma.std():.1f}")
    plot_gaussian(
        ax,
        three_sigma,
        label=f"Excluding outliers",
        color="#D55E00",
    )
    ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / f"disruption_time_diff_{prediction_type}.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def generate_range_histogram(df: pd.DataFrame) -> None:
    # Errors in seconds; keep those within +/-10 ms, then convert to milliseconds.
    df["range"] = df["t_f"] - df["t_0"]
    disrupt_range = 1e3 * df["range"][(df["range"] > 0) & (df["range"] < 0.02)]
    sigma = disrupt_range.std()
    mu = np.abs(disrupt_range.mean())
    logger.success(
        f"Disruption time range (milliseconds, n={len(disrupt_range)}): "
        f"mean={mu:.3f}, median={disrupt_range.median():.3f}, variance={disrupt_range.var():.3f}, stddev={sigma:.3f}"
    )
    first_quartile = disrupt_range[np.abs(disrupt_range) < sigma]
    second_quartile = disrupt_range[np.abs(disrupt_range) < 2 * sigma]
    third_quartile = disrupt_range[np.abs(disrupt_range) < 3 * sigma]
    logger.success(
        f"{100*len(first_quartile) / len(disrupt_range):2f}% shots within 1 stddev, {100*len(second_quartile) / len(disrupt_range):2f}% shots within 2 stddev, {(100*len(third_quartile) / len(disrupt_range)):2f}% shots within 3 stddev"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        disrupt_range,
        bins=50,
        density=True,
        color="#B8C4D0",
        edgecolor="#5A6B7B",
        linewidth=0.5,
        alpha=0.9,
        range=(0, 20),
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(f"$t_f-t_0$ (ms)")
    ax.set_ylabel("Fraction of shots")

    # Overlay the best-fit Gaussian. Explicit high-contrast accents against the
    # neutral gray bars: full-data fit in blue, the 3-sigma-trimmed fit in orange.
    plot_gaussian(
        ax,
        disrupt_range,
        color="#0072B2",
        label=f"Full test set",
    )

    three_sigma = disrupt_range[np.abs(disrupt_range) < (3 * sigma)]
    logger.info(f"mu={(three_sigma.mean()):.1f};sigma={three_sigma.std():.1f}")
    plot_gaussian(
        ax,
        three_sigma,
        label=f"Excluding outliers",
        color="#D55E00",
    )
    ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / f"disruption_time_range.png"
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def generate_scatter_plot(
    df: pd.DataFrame, prediction_type: "t_root" | "t_0" | "t_f"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[prediction_type], df["t_D"], alpha=0.85)

    _, hi = ax.get_xlim()
    ax.plot(
        [0, hi],
        [0, hi],
        "r--",
        linewidth=1,
        label=f"$t_D={get_pred_type_label(prediction_type)}$",
    )

    ax.set_xlabel(f"${get_pred_type_label(prediction_type)}$ (s)")
    ax.set_ylabel("$t_D$ (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = model_dir / f"predictions_scatter_{prediction_type}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction plots")
    load_best_model_env()

    df = pd.read_csv(predictions_csv)

    shots_in_range = df[(df["t_D"] < df["t_f"]) & (df["t_D"] > df["t_0"])]
    logger.info(
        f"{len(shots_in_range)} / {len(df["t_D"])} shots in range ({len(shots_in_range) / len(df["t_D"])})."
    )
    generate_range_histogram(df)

    for prediction_type in ["t_root", "t_0", "t_f"]:
        df["diff"] = df[prediction_type] - df["t_D"]
        generate_scatter_plot(df, prediction_type)
        generate_err_histogram(df, prediction_type)


if __name__ == "__main__":
    main()
