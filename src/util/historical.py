"""Read archived D3D HDF5 shots."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def h5_path(hdf5_dir: str | Path, shot_no: int) -> Path:
    return Path(hdf5_dir) / f"{shot_no // 1000}xxx" / f"{shot_no}.h5"


SAMPLE_RATE = 4000  # [Hz], matches the training .txt signals


def read_h5_shot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time [s], current [MA])`` for one shot, linearly interpolated to 4 kHz."""
    with h5py.File(path, "r") as f:
        t = f["ip"]["times"][:] / 1e3  # [s]
        ip = f["ip"]["data"][:] / 1e6  # [MA]
    t_uniform = t[0] + np.arange(int((t[-1] - t[0]) * SAMPLE_RATE) + 1) / SAMPLE_RATE
    return t_uniform, np.interp(t_uniform, t, ip)
