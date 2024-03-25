from Preprocessor import Preprocessor

data_dir = '/eagle/fusiondl_aesp/signal_data/d3d/ipspr15V/'
dataset_dir = '/eagle/fusiondl_aesp/jrodriguez/processed_data/'
labels_path = '/eagle/fusiondl_aesp/jrodriguez/shot_lists/ips_labels.txt'

PP = Preprocessor(dataset_dir, data_sir, labels_path)
stats = PP.Get_Mean_Std(cpu_use = 1)
PP.Make_Dataset(cpu_use = 1)
