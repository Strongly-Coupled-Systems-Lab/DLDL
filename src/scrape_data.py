import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from util.disruption_predict import predict_disruption_time, PredictionType
from util.best_model import load_best_model_cnn, load_best_model_env
from model.dataset import IpDataset
import torch
import torch.nn.functional as F

# Set the directory path ('.' means current directory)
repo = Path(__file__).resolve().parents[1]
target_dir = Path("/eagle/fusiondl_aesp/d3d_raw_data_hdf5")
out_path = Path("./all_shots.csv")


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


disruption_times: list[tuple[int, float, float]] = []
model = load_best_model_cnn(dataset)

for file_path in tqdm(
    target_dir.rglob("*.h5"), desc="Create .txt shot files", unit=" shots"
):
    with h5py.File(file_path, "r") as f:
        # downsample
        t = f["ip"]["times"][:] / 1e3  # [s]
        ip = f["ip"]["data"][:] / 1e6  # [MA]

        # 1. Convert raw array directly into a 3D tensor format for interpolation: (Batch, Channels, Length)
        raw_signal_tensor = torch.tensor(ip).float().reshape(1, 1, -1)
        raw_time_tensor = torch.tensor(t).float().reshape(1, 1, -1)

        # 2. Resample the length dimension smoothly from 68608 down to 60201
        target_length = dataset.data.shape[1]  # 60201
        downsampled_signal = F.interpolate(
            raw_signal_tensor, size=target_length, mode="linear", align_corners=False
        )
        downsampled_time = F.interpolate(
            raw_time_tensor, size=target_length, mode="linear", align_corners=False
        )

        # 3. Strip back the extra channel dimension so it fits your model's forward path
        signal = downsampled_signal.squeeze(1)  # Outputs shape: torch.Size([1, 60201])

        cnn_prob = torch.sigmoid(model.forward(signal)[0, 0]).item()

        t_0, pred_time, t_f = (
            predict_disruption_time(signal, downsampled_time)
            if cnn_prob > 0.5
            else (-1, -1, -1)
        )
        disruption_times.append((file_path.stem, cnn_prob, t_0, pred_time, t_f))
        break


df = pd.DataFrame(
    disruption_times,
    columns=[
        "shot_no",
        "P_disrupt",
        "t_0",
        "t_root",
        "t_f",
    ],
)
df.to_csv(out_path, sep=",", header=True, index=False)
