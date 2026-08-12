"""Browse preprocessed shots with a slider and index text box."""

import argparse
import logging
import math
import os
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider, TextBox

from model.dataset import IpDataset, ShotView
from util.data_loading import _read_signal_file
from util.disruption_predict import (
    predict_disruption_time,
    apply_filter,
    get_oriented_current,
    clean_zeros,
)
from util.best_model import load_best_model_cnn, load_best_model_env

logger = logging.getLogger(__name__)

repo = Path(__file__).resolve().parents[1]
load_best_model_env()


def abs_path(p: str) -> str:
    return p if os.path.isabs(p) else str(repo / p)


dataset = IpDataset(
    data_file=abs_path(os.environ["DATA_PATH"]),
    labels_file=abs_path(os.environ["TRAIN_LABELS_PATH"]),
    labels_path=abs_path(os.environ["LABELS_PATH"]),
    data_dir=abs_path(os.environ["DATA_DIR"]),
    labels_type="scaled",
    cpu_use=float(os.environ["CPU_USE"]),
    preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
)
model = load_best_model_cnn(dataset)
data_dir = dataset.data_dir


def simple_draw(ax: plt.Axes, shot: ShotView):
    """
    Renders one shot into the current-signal
    axis ``ax1`` and the heuristic axis ``ax2``. It returns the shot view so
    callers can title panels as they see fit.
    """
    signal = torch.tensor(shot.current).float().reshape(1, -1)
    cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()

    # Plot the raw shot file directly: column 0 is time (s), column 1 is
    # current (SI), so the axes carry physical units without de-normalizing.
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    current, time = clean_zeros(raw_current, raw_time)

    # t_disrupt is stored normalized (disruption_index / max_length); map it
    # back onto the SI time axis via the raw time samples.
    max_length = dataset.data.shape[1]
    t_disrupt_si = (
        float(raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)])
        if shot.disruptive
        else None
    )

    t_disrupt_str = f"{t_disrupt_si:.3f}" if t_disrupt_si is not None else "n/a"
    logger.info(
        f"shot {shot.shot_no} (index {shot.index}): P_disrupt={cnn_prob:.3f}, "
        f"disruptive={shot.disruptive}, t_D={t_disrupt_str} s"
    )

    ax.clear()
    ax.set_title(
        shot.title + ": $P_\\mathrm{disrupt} = " + f"{100*cnn_prob:.0f}\\%$",
    )
    ax.set_ylim(min(-2.5, 2.5 * np.min(current)), max(2, 1.5 * np.max(current)))
    ax.plot(time, current, label="$I_\\mathrm{raw}$")
    flipped_current = get_oriented_current(current)
    if not np.array_equal(current, flipped_current):
        ax.plot(
            time,
            flipped_current,
            color="C0",
            label="$I_\\mathrm{raw}$",
            linestyle=" = ",
        )

    if shot.disruptive:
        ax.axvline(
            t_disrupt_si,
            color="black",
            ls="--",
            label=f"$t_D$",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (MA)")
    ax.legend(loc="lower left")
    ax.grid(True)

    return shot


def draw(ax1, ax2, shot: ShotView, zoom=False):
    signal = torch.tensor(shot.current).float().reshape(1, -1)
    cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()

    # Plot the raw shot file directly: column 0 is time (s), column 1 is
    # current (SI), so the axes carry physical units without de-normalizing.
    raw_path = os.path.join(dataset.data_dir, f"{shot.shot_no}.txt")
    raw_current = _read_signal_file(raw_path, col=1)
    raw_time = _read_signal_file(raw_path, col=0)
    current, time = clean_zeros(raw_current, raw_time)
    predicted_time_start, pred_time, predicted_time_end = predict_disruption_time(
        raw_current, raw_time
    )

    # t_disrupt is stored normalized (disruption_index / max_length); map it
    # back onto the SI time axis via the raw time samples.
    max_length = dataset.data.shape[1]
    t_disrupt_si = (
        float(raw_time[min(round(shot.t_disrupt * max_length), len(raw_time) - 1)])
        if shot.disruptive
        else None
    )

    diff = (t_disrupt_si - pred_time) if t_disrupt_si is not None else None
    t_disrupt_str = f"{t_disrupt_si:.3f}" if t_disrupt_si is not None else "n/a"
    diff_str = f"{diff:.3f}" if diff is not None else "n/a"
    logger.info(
        f"shot {shot.shot_no} (index {shot.index}): P_disrupt={cnn_prob:.3f}, "
        f"disruptive={shot.disruptive}, t_D={t_disrupt_str} s, "
        f"t_root={pred_time:.3f} s, "
        f"window=[{predicted_time_start:.3f}, {predicted_time_end:.3f}] s, "
        f"t_D-t_root={diff_str} s, "
    )

    ax1.clear()
    ax1.set_title(shot.title + ": $P_\\mathrm{disrupt} = " + f"{100*cnn_prob:.0f}\\%$")
    ax1.plot(time, current, label="$I_\\mathrm{raw}$", color="C0")
    flipped_current = get_oriented_current(current)
    if not np.array_equal(current, flipped_current):
        ax1.plot(
            time,
            flipped_current,
            color="C0",
            label="Flipped $I_\\mathrm{raw}$",
            linestyle=" = ",
        )

    filtered, smoothed = apply_filter(current)
    ax1.plot(
        time,
        smoothed,
        color="C1",
        label="$I_\\mathrm{smooth}$",
        linestyle="--",
    )
    if shot.disruptive:
        ax1.axvline(
            t_disrupt_si,
            color="black",
            ls="--",
            label=f"$t_D$",
        )
        ax2.axvline(
            t_disrupt_si,
            color="black",
            ls="--",
            label=f"$t_D$",
        )

    ax2.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (MA)")
    ax1.legend(loc="lower left")
    ax1.grid(True)

    ax2.clear()
    ax2.set_ylabel("Heuristic")
    ax2.plot(time, filtered, label="$f$")
    heuristic_label = "$t_\\mathrm{root}$"
    ax2.axvline(pred_time, color="C3", ls="--", label=heuristic_label)
    ax1.axvline(pred_time, color="C3", ls="--", label=heuristic_label)
    ax2.axvspan(
        xmin=predicted_time_start,
        xmax=predicted_time_end,
        alpha=0.2,
        label="$t_0 \\leq t \\leq t_f$",
        color="C3",
    )
    ax1.axvspan(
        xmin=predicted_time_start,
        xmax=predicted_time_end,
        alpha=0.2,
        label="$t_0 \\leq t \\leq t_f$",
        color="C3",
    )
    ax2.legend(loc="lower left")
    ax2.grid()

    # Zoom to a window centered on the disruption (true time if known,
    # else the heuristic prediction), clamped to the available time range.
    if zoom:
        window = 0.05
        center = t_disrupt_si if t_disrupt_si is not None else pred_time
        lo = max(time[0], center - (2 * window / 3))
        hi = min(time[-1], center + (1 * window / 3))
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(-0.1, 1.2 * current[time == predicted_time_start][0])
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(-2, 1)

    return shot


def run_interactive(zoom=False, simple=False) -> None:
    """Slider/text-box browser over all shots (one shot at a time).

    ``simple`` picks the single-panel :func:`simple_draw`; otherwise the shot is
    rendered as a stacked (current, heuristic) pair via :func:`draw`.
    """
    matplotlib.use("QtAgg")
    # Interactive figures are small (10x7 @ 100 dpi), so the default matplotlib
    # font sizes render oversized; shrink them here. Exports use large figures
    # at 300 dpi and are unaffected (separate CLI invocation).
    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "lines.linewidth": 1,
            "lines.markersize": 2,
        }
    )
    num_rows = len(dataset)
    if simple:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=150)
    fig.subplots_adjust(bottom=0.2)

    with torch.no_grad():

        def redraw(i: int) -> None:
            shot = dataset.load_shot_view(max(0, min(int(i), num_rows - 1)))
            if simple:
                simple_draw(ax, shot)
            else:
                draw(ax1, ax2, shot, zoom=zoom)
            fig.canvas.draw_idle()

        redraw(0)

        index_slider = Slider(
            fig.add_axes([0.12, 0.05, 0.55, 0.03]),
            "index",
            0,
            num_rows - 1,
            valinit=0,
            valstep=1,
        )
        index_slider.valtext.set_visible(False)
        box = TextBox(fig.add_axes([0.72, 0.05, 0.12, 0.03]), "", initial="0")

        index_slider.on_changed(lambda v: (redraw(v), box.set_val(str(int(v)))))
        box.on_submit(
            lambda t: index_slider.set_val(int(t)) if t.strip().isdigit() else None
        )

        plt.show()


def save_grid(shots: list[ShotView], out_path: Path, zoom=False, simple=False) -> None:
    """Render the given shots as a grid of shot panels.

    In ``simple`` mode each shot occupies a single current-signal panel via
    ``simple_draw``. Otherwise each shot occupies one grid column with two
    stacked rows (current signal on top, heuristic below) via ``draw``.
    """
    matplotlib.use("Agg")
    n = len(shots)
    ncols = min(n, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    if simple:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )
        flat = axes.flatten()

        with torch.no_grad():
            for cell, shot in enumerate(shots):
                simple_draw(flat[cell], shot)

        # Blank any unused panels.
        for cell in range(n, nrows * ncols):
            flat[cell].axis("off")
    else:
        # Two physical rows (current + heuristic) per shot row.
        fig, axes = plt.subplots(
            nrows * 2,
            ncols,
            figsize=(6 * ncols, 8 * nrows),
            squeeze=False,
        )

        with torch.no_grad():
            for cell, shot in enumerate(shots):
                row, col = divmod(cell, ncols)
                ax1 = axes[row * 2][col]
                ax2 = axes[row * 2 + 1][col]
                draw(ax1, ax2, shot, zoom=zoom)

        # Blank any unused panels in the final grid row.
        for cell in range(n, nrows * ncols):
            row, col = divmod(cell, ncols)
            axes[row * 2][col].axis("off")
            axes[row * 2 + 1][col].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse raw shot currents, or save a grid of specific shots."
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        metavar="SHOT",
        help="Shot numbers to render as a grid of subplots instead of browsing.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help="Path of file to save to. Must be .png or .jpeg",
    )
    parser.add_argument("--zoom", default=False, action="store_true")
    parser.add_argument("--simple", default=False, action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if model is None:
        return

    if args.shots:
        if not args.out_path or not Path(args.out_path).parent.exists():
            raise ValueError("Must provide a valid out_path")
        missing = [s for s in args.shots if not dataset.has_shot(s)]
        if missing:
            raise KeyError(f"Shots not found in dataset: {missing}")

        shots = [dataset.shot_view(s) for s in args.shots]
        save_grid(shots, Path(args.out_path), zoom=args.zoom, simple=args.simple)
    else:
        run_interactive(zoom=args.zoom, simple=args.simple)


if __name__ == "__main__":
    main()
