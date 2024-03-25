import numpy as np
import time
import os
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
try:
    import torch
except:
    pass

################################################################################
## Utility Functions and Globals
################################################################################
def check_file(file_path, verbose = False):
    if os.path.exists(file_path):
        if verbose:
            file_size = os.path.getsize(file_path)
            print(f"File {file_path} exists. Size: {file_size} bytes.")
        return True
    else:
        if verbose:
            print(f"File {file_path} does not exist.")
        return False


def get_length(filename, data_dir):
    """
    Get length of time series for a single file.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
    """
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path)

    return data.shape[0]


def get_means(filename, data_dir):
    """
    Get mean and mean of squares of time series for a single file.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
    """
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path)

    mean = np.mean(data)
    mean2 = np.mean(data**2)

    return [mean, mean2]


def load_and_pad(filename, data_dir, max_length):
    """
    Loads a single current series and pads it with zeros up to the max length,
    then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, use_cols=1, dtype=np.float32)
    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded data)


def load_and_pad_norm(filename, data_dir, max_length, mean = None, std = None):
    """
    Loads a single current series and pads it with zeros up to the max length,
    normalizes the signal values then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
        mean: float, supply if you want to use dataset-wide statistics
        std: float, " "
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, use_cols=1, dtype=np.float32)
    
    if mean == None:
        mean = np.mean(data)
        std = np.std(data)

    data = (data - mean)/std

    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded data)


def load_and_pad_scale(filename, data_dir, max_length):
    """
    Loads a single current series and pads it with zeros up to the max length,
    scales data values to [0,1], then returns it.

    Args:
        filename: str, file name
        data_dir: str, path to file directory
        max_length: int, maximum length of the time series across all shots
    """
    shot_no = int(filename[:-4])
    file_path = os.path.join(data_dir, filename)
    data = np.loadtxt(file_path, use_cols=1, dtype=np.float32)
    data = data - np.min(data)
    data = data/np.max(data)
    N = min(len(data), max_length)
    padded_data = np.zeros(max_length)
    padded_data[:N] = data

    return (shot_no, padded data)


################################################################################
## Preprocessor Class
################################################################################
class Preprocessor:
    def __init__(self, dataset_dir, data_dir, labels_path):
        self.data_dir = data_dir
        self.dataset_path = os.path.join(dataset_dir,'processed_dataset.pt')
        self.labels_pt_path = os.path.join(dataset_dir,'processed_labels.pt')
        self.max_length_file = os.path.join(self.dataset_dir,'max_length.txt')
        self.mean_std_file = os.path.join(self.dataset_dir,'mean_std.txt')
        self.labels_path = labels_path


    def Get_Max_Length(self, save = True, cpu_use = 0.8):
        """
        Acquires the maximum length of the current time series across the
        entire dataset.

        Args:
            save: bool, True to save result in data_dir
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        file_list = [f for f in os.listdir(self.data_dir) if\
                (f.endswith('.txt') and not f.startswith('t_end'))]
        num_shots = len(file_list)
        print("Finding N_max for the {} shots in ".format(int(num_shots))\
                +self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(list(executor.map(get_length, file_list,\
                           [self.data_dir]*num_shots)))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        maximum = np.max(results)

        print("Finished getting end timesteps in {} seconds.".format(T))

        if save:
            np.savetxt(self.max_length_file, np.array([maximum]))

        return maximum

        
    def Get_Mean_Std(self, save = True, cpu_use = 0.8):
        """
        Acquires the mean and std. dev. of the entire dataset.

        Args:
            save: bool, True to save result in data_dir
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        file_list = [f for f in os.listdir(self.data_dir) if\
                (f.endswith('.txt') and not f.startswith('t_end'))]
        num_shots = len(file_list)
        print("Finding the mean and std. dev. for the {} shots in ".format(\
                int(num_shots))+self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                results = np.asarray(list(executor.map(get_means, file_list,\
                           [self.data_dir]*num_shots)))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        mean = np.mean(results[:,0])
        std = (np.mean(results[:,1])-mean**2)**0.5

        print("Finished getting stats in {} seconds.".format(T))

        if save:
            np.savetxt(self.mean_std_file, np.array([mean, std]))

        return np.array([mean, std])

        
    def Make_Dataset(self, normalization = None, mean = None, std = None,\
                     max_length = None, cpu_use = 0.8):
        """
        Acquires the maximum length of the current time series across the
        entire dataset.

        Args:
            dataset_dir: str, name of a directory that will house the complete
                         dataset
            normalization: str, specifies normalization type, options are
                           'scale' 'meanvar-whole' and 'meanvar-single', leave 
                           blank for no normalization
            mean, std: float, dataset-wide statistics if desired
            cpu_use: float in (0,1], fraction of cpu cores to use
        """
        if max_length == None:
            if check_file(self.max_length_file):
                max_length = np.loadtxt(self.max_length_file).astype(int)[0]
            else:
                max_length = self.Get_Max_Length(cpu_use = cpu_use)

        if normalization == "meanvar-whole":
            if check_file(self.mean_std_file):
                stats = np.loadtxt(self.max_length_file)
                mean = stats[0]
                std = stats[1]
            else:
                stats = self.Get_Mean_Std(cpu_use = cpu_use)
                mean = stats[0]
                std = stats[1]

        file_list = [f for f in os.listdir(self.data_dir) if\
                (f.endswith('.txt') and not f.startswith('t_end'))]
        num_shots = len(file_list)
        print("Building dataset for the {} shots in ".format(int(num_shots))\
                +self.data_dir)
        t_b = time.time()

        assert cpu_use <= 1
        use_cores = max(1, int((cpu_use)*mp.cpu_count()))
        print(f"Running on {use_cores} processes.")
        with ProcessPoolExecutor(max_workers = use_cores) as executor:
            # Process all files in parallel and collect results
            try:
                if normalization == None:
                    results = list(executor.map(load_and_pad,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots))
                elif normalization == "scale":
                    results = list(executor.map(load_and_pad_scale,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots))
                elif normalization.startswith("meanvar"):
                    results = list(executor.map(load_and_pad_norm,\
                            file_list, [self.data_dir]*num_shots,\
                            [max_length]*num_shots, [mean]*num_shots,\
                            [std]*num_shots))
            except Exception as e:
                print(f"An error occurred: {e}")

        t_e = time.time()
        T = t_e-t_b

        labels = torch.tensor(np.loadtxt(self.labels_path).reshape((num_shots,1)))
        sorted_data = sorted(results, key=lambda x: x[0])
        dataset = torch.zeros((num_shots, max_length))
        for i in range(num_shots):
            dataset[i,:] = sorted_data[i]

        print("Finished loading and preparing data in {} seconds.".format(T))

        torch.save(self.dataset_path, dataset)
        torch.save(self.labels_pt_path, labels)