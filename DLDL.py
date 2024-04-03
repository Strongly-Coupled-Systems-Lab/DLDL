import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, Subset
    import torch.nn as nn
    import torch.nn.functional as F
except:
    print("WARNING: pytorch not installed!")
    pass

classification_loss = nn.BCEWithLogitsLoss()
time_prediction_loss = nn.MSELoss()

def loss(outputs, labels):
    # outputs = [classification_logits, time_predictions]
    # labels = [binary_class_labels, normalized_time_steps]
    class_loss = classification_loss(outputs[:, 0], labels[:, 0])
    
    # Apply time prediction loss only to disruptive shots
    disruptive_mask = labels[:, 0] == 1
    if disruptive_mask.any():
        time_loss = time_prediction_loss(outputs[disruptive_mask, 1],\
                labels[disruptive_mask, 1])
    else:
        time_loss = 0

    return class_loss + time_loss


def split(dataset, train_size = 0.8):
    dev_size = (1 - train_size)/2

    total_size = len(dataset)
    train_end = int(train_size * total_size)
    dev_end = int((train_size+dev_size) * total_size)
    train_indices = range(0, train_end)
    dev_indices = range(train_end, dev_end)
    test_indices = range(dev_end, total_size)

    train = Subset(dataset, train_indices)
    dev = Subset(dataset, dev_indices)
    test = Subset(dataset, test_indices)

    return train, dev, test


class ipDataset(Dataset):
    def __init__(self, data_file, labels_file):
        self.data = torch.load(data_file)
        self.labels = torch.load(labels_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class ipCNN(nn.Module):
    def __init__(self, conv1 = (16, 9, 4), conv2 = (32, 5, 2),\
                 conv3 = (64, 3, 1), pool_size = 4):
        super(ipCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, conv1[0], kernel_size=conv1[1], stride=1,\
                               padding=conv1[2])
        self.conv2 = nn.Conv1d(conv1[0], conv2[0], kernel_size=conv2[1],\
                               stride=1, padding=conv2[2])
        self.conv3 = nn.Conv1d(conv2[0], conv3[0], kernel_size=conv3[1],\
                               stride=1, padding=conv3[2])
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        self.fc1 = nn.Linear(conv3[0] * max_length // pool_size**3, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)  # Two outputs: classification and time of disruption

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
