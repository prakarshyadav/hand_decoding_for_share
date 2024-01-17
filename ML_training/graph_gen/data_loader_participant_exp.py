import numpy as np
import torch
from torch.utils import data
import glob, os
import pickle
from torch_geometric.loader import DataLoader

def pkl_load(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

class MyDataset(data.Dataset):
    def __init__(self, path, ext, exclusion_list_participant, exclusion_list_class):
        file_path = glob.glob(path + r'\**\*'+ext, recursive=True)
        self.data_path = []
        for i, fname in enumerate(file_path):
            _, participant, date_name, class_name, samp_name = fname.split('\\')
            if participant not in exclusion_list_participant:
                if class_name not in exclusion_list_class:
                    self.data_path.append((fname,int(class_name)))
        
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        X_path,y = self.data_path[index]
        X = pkl_load(X_path)

        num_nodes = X.num_nodes
        order = np.arange(num_nodes)
        np.random.shuffle(order)
        X.x=X.x[order,:]
        X.edge_index = X.edge_index.T
        # For testing set, return only x
        return (X,num_nodes), y
    
if __name__ == "__main__":
    exclusion_list_participant = []
    exclusion_list_class = []
    dataset = MyDataset("./data",".pkl",exclusion_list_participant,exclusion_list_class)



    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    from torch.utils.data.sampler import SubsetRandomSampler

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(25, 15))
    axes = axes.flatten()
    for i, inp_train in enumerate(train_loader):
        # print(inp_train)
        
        inputs_list, labels_train = inp_train[0], inp_train[1]
        batch_items = labels_train.size()[0]
        inputs = inputs_list[0]
        num_nodes = inputs_list[1]
        axes[int(labels_train[0])].plot(inputs.x[2,:],alpha = 0.5, linewidth = 0.2)
    plt.show()
