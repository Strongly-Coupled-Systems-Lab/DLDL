from Preprocessor import Preprocessor

data_dir = '/eagle/fusiondl_aesp/signal_data/d3d/ipspr15V/'
dataset_dir = '/eagle/fusiondl_aesp/jrodriguez/processed_data/'
labels_path = '/eagle/fusiondl_aesp/jrodriguez/shot_lists/ips_labels.txt'

#PP = Preprocessor(dataset_dir, data_dir, labels_path)
PP_s = Preprocessor(dataset_dir, data_dir, labels_path, ID = '_scaled_labels')
#labels = PP.Make_Labels_Naive(save = True)
#labels = PP_s.Make_Labels_Scaled(save = True)
#PP.Check_Dataset(scale_labels = False, verbose = True)
PP_s.Check_Dataset(dset_path = dataset_dir+'processed_dataset.pt', scale_labels = True, verbose = True)
#stats = PP.Get_Mean_Std(cpu_use = 1)
#labels = PP.Make_Labels_Scaled(save = True)
#PP.Make_Dataset(cpu_use = 1)
