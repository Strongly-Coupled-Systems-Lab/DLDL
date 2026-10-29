"""Disruption time prediction from smoothed current residuals."""

from __future__ import annotations
from enum import IntEnum
import numpy as np

DEFAULT_SMOOTHING = 300


class PredictionType(IntEnum):
    START = 0
    ROOT = 1
    END = 2


def get_window_size(current: np.ndarray):
    return max(1, len(current) // DEFAULT_SMOOTHING)


def get_t_equal_zero_idx(time: np.ndarray):
    return np.argmin(np.abs(time))


def clean_zeros(current: np.ndarray, time: np.ndarray):
    """
    D3D data has a jump to 0 at the end when the shot terminates.
    We want to remove that jump and shift up by the last value to flatten the curve
    """
    # remove trailing zeroes
    processed_current = np.trim_zeros(current).copy()
    processed_time = time[: len(processed_current)].copy()
    return (
        processed_current[get_t_equal_zero_idx(processed_time) :],
        processed_time[get_t_equal_zero_idx(processed_time) :],
    )


# 2026-08-11 18:52:40,315 INFO __main__: shot 176705 (index 35710): P_disrupt=0.926, disruptive=True, t_D=2.065 s, t_root=6.484 s, window=[6.484, 6.498] s, t_D-t_root=-4.419 s,
# 2026-08-11 18:53:55,073 INFO __main__: shot 176705 (index 35710): P_disrupt=0.926, disruptive=True, t_D=2.065 s, t_root=2.081 s, window=[2.068, 2.247] s, t_D-t_root=-0.016 s,


def apply_smoothing(current: np.ndarray):
    window_size = get_window_size(current)
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(current, weights, mode="same")
    return smoothed


def get_oriented_current(current: np.ndarray):
    return -current if np.isclose(np.max(current), 0) else current


def apply_filter(current: np.ndarray):
    oriented = get_oriented_current(current)
    smoothed = apply_smoothing(oriented)

    return oriented - smoothed, smoothed


def predict_disruption_time(current: np.ndarray, time: np.ndarray) -> float:
    diff, _ = apply_filter(current)

    # mask out zeroes
    idx_peak = np.where(time < 0, -np.inf, diff).argmax()
    idx_trough = np.argmin(diff[idx_peak:]) + idx_peak

    # first root after peak
    arr = diff[idx_peak:idx_trough]
    idx_root = next(iter(np.flatnonzero(np.diff(np.signbit(arr)))), 0)

    return (
        time[idx_peak],
        time[idx_peak + idx_root],
        time[idx_trough],
    )
