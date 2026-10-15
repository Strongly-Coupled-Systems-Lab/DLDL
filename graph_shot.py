import h5py
import numpy as np
import matplotlib.pyplot as plt
from util.best_model import best_model_dir, load_best_model_cnn, load_best_model_env

shot = 189993
path = f"/eagle/fusiondl_aesp/d3d_raw_data_hdf5/{shot//1000}xxx/{shot}.h5"


with h5py.File(path, "r") as f:
    t = f["ip"]["times"][:]  # ms
    ip = f["ip"]["data"][:]  # A


plt.plot(t / 1000, ip / 1e6)
plt.xlabel("t [s]")
plt.ylabel("Ip [MA]")
disrupt = 0.05111655635833845
plt.axvline(x=disrupt, color="red", linestyle="--")
plt.savefig(f"./{shot}.png")
