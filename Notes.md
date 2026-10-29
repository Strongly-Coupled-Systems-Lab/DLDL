Available keys:

['betap', 'bt', 'density', 'e567dn', 'e567up', 'e89dn', 'e89up', 'ece1', 'ece10', 'ece11', 'ece12', 'ece13', 'ece14', 'ece15', 'ece16', 'ece17', 'ece18', 'ece19', 'ece2', 'ece20', 'ece21', 'ece22', 'ece23', 'ece24', 'ece25', 'ece26', 'ece27', 'ece28', 'ece29', 'ece3', 'ece30', 'ece31', 'ece32', 'ece33', 'ece34', 'ece35', 'ece36', 'ece37', 'ece38', 'ece39', 'ece4', 'ece40', 'ece5', 'ece6', 'ece7', 'ece8', 'ece9', 'ech_total', 'ecoila', 'ecoilb', 'edensfit', 'etempfit', 'f1a', 'f1b', 'f2a', 'f2b', 'f3a', 'f3b', 'f4a', 'f4b', 'f5a', 'f5b', 'f6a', 'f6b', 'f7a', 'f7b', 'f8a', 'f8b', 'f9a', 'f9b', 'gasa_cal', 'gasc_cal', 'gasd_cal', 'ip', 'ipspr15v', 'itempfit', 'kappa', 'li', 'n1rms', 'n2rms', 'pinj', 'pinj_15l', 'pinj_15r', 'pinj_21l', 'pinj_21r', 'pinj_30l', 'pinj_30r', 'pinj_33l', 'pinj_33r', 'pohm', 'prad_divl', 'prad_divu', 'prmtan_neped', 'rmaxis', 'rvsin', 'rvsout', 'seconds_since_bzn', 'seconds_since_disruption', 'tinj', 'tinj_15l', 'tinj_15r', 'tinj_21l', 'tinj_21r', 'tinj_30l', 'tinj_30r', 'tinj_33l', 'tinj_33r', 'tribot', 'tritop', 'trotfit', 'zmaxis', 'zvsin', 'zvsout']

Found with

```
module use /soft/modulefiles
module use conda
conda activate base
```

```
import h5py, numpy as np
shot = 170000
path = f"/eagle/fusiondl_aesp/d3d_raw_data_hdf5/{shot//1000}xxx/{shot}.h5"
with h5py.File(path, "r") as f:
    print(list(f.keys()))          # all signals in this shot
```
