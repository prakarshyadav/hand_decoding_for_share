import numpy as np
import torch
from torch.utils import data
import glob, os
import pickle
from torch_geometric.loader import DataLoader
from gifs.gesture_master_list import GESTURE_KEYS, GESTURE_LIST_MASTER
import posixpath

def pkl_load(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


GESTURE_SIM_MAT_FUNCTIONAL = np.zeros((len(GESTURE_LIST_MASTER),len(GESTURE_LIST_MASTER)))
for i in range(GESTURE_SIM_MAT_FUNCTIONAL.shape[0]):
    for j in range(GESTURE_SIM_MAT_FUNCTIONAL.shape[1]):
        if i!=j:
            if GESTURE_LIST_MASTER[i][1] == GESTURE_LIST_MASTER[j][1]:
                GESTURE_SIM_MAT_FUNCTIONAL[i,j] = 1


class MyDataset_loss_triplet(data.Dataset):
    def __init__(self, path, ext, exclusion_list_participant, exclusion_list_class):
        file_path = glob.glob(path + r'/**/*'+ext, recursive=True)
        self.data_path = []
        self.data_dict = {}
        self.pos_pairs = {}
        self.neg_pairs = {}

        self.participants = []
        self.other_participants =  {}
        for k in list(GESTURE_KEYS.values()):
            if str(k) not in exclusion_list_class:
                self.data_dict[k] = {}
                for i,fname in enumerate(os.listdir(path)):
                    fname = fname.replace(os.sep, posixpath.sep)
                    participant = fname.split('/')[-1]
                    if participant not in exclusion_list_participant:
                        self.participants.append(participant)
                        self.other_participants[participant] = []
                        self.data_dict[k][participant] = []

        for part in self.other_participants.keys():
            for next_participant in list(self.other_participants.keys()):
                if next_participant != part:
                    self.other_participants[part].append(next_participant)
        self.num_class = len(GESTURE_KEYS.keys())
        self.classes_id = np.arange(self.num_class)
        for i, fname in enumerate(file_path):
            fname = fname.replace(os.sep, posixpath.sep)
            file_list = fname.split('/')
            participant, class_name, samp_name = file_list[-3],file_list[-2],file_list[-1]
            if participant not in exclusion_list_participant:
                if str(GESTURE_KEYS[class_name]) not in exclusion_list_class:
                    self.data_path.append((fname,int(GESTURE_KEYS[class_name]),participant))
                    self.data_dict[int(GESTURE_KEYS[class_name])][participant].append(fname)

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        anchor_path,y, positive_id = self.data_path[index]

        anchor, num_nodes = pkl_load(anchor_path)
        anchor.edge_index = anchor.edge_index.T

        pos_path = np.random.choice(self.data_dict[y][np.random.choice(self.other_participants[positive_id],1)[0]],1)[0]
        positive, num_nodes = pkl_load(pos_path)
        positive.edge_index = positive.edge_index.T

        neg_wt = np.ones(self.num_class)/(self.num_class-1)
        neg_wt[y] = 0
        neg_label = np.random.choice(self.classes_id,size = 1, p = neg_wt)[0]
        neg_path = np.random.choice(self.data_dict[neg_label][positive_id],1)[0]
        negative, num_nodes = pkl_load(neg_path)
        negative.edge_index = negative.edge_index.T

        # return (anchor,num_nodes), positive, negative, num_nodes
        return anchor, positive, negative, num_nodes, y, int(neg_label)


class MyDataset_loss_paired(data.Dataset):
    def __init__(self, path, ext, exclusion_list_participant, exclusion_list_class):
        file_path = glob.glob(path + r'/**/*'+ext, recursive=True)
        self.data_path = []
        self.data_dict = {}
        self.pos_pairs = {}

        self.participants = []
        self.other_participants =  {}
        for k in list(GESTURE_KEYS.values()):
            if str(k) not in exclusion_list_class:
                self.data_dict[k] = {}
                for i,fname in enumerate(os.listdir(path)):
                    fname = fname.replace(os.sep, posixpath.sep)
                    participant = fname.split('/')[-1]
                    if participant not in exclusion_list_participant:
                        self.participants.append(participant)
                        self.other_participants[participant] = []
                        self.data_dict[k][participant] = []

        for part in self.other_participants.keys():
            for next_participant in list(self.other_participants.keys()):
                if next_participant != part:
                    self.other_participants[part].append(next_participant)
        self.num_class = len(GESTURE_KEYS.keys())
        
        for i, fname in enumerate(file_path):
            fname = fname.replace(os.sep, posixpath.sep)
            file_list = fname.split('/')
            participant, class_name, samp_name = file_list[-3],file_list[-2],file_list[-1]
            if participant not in exclusion_list_participant:
                if str(GESTURE_KEYS[class_name]) not in exclusion_list_class:
                    self.data_path.append((fname,int(GESTURE_KEYS[class_name]),participant))
                    self.data_dict[int(GESTURE_KEYS[class_name])][participant].append(fname)

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        anchor_path,y, positive_id = self.data_path[index]
        anchor, num_nodes = pkl_load(anchor_path)
        anchor.edge_index = anchor.edge_index.T

        pos_path = np.random.choice(self.data_dict[y][np.random.choice(self.other_participants[positive_id],1)[0]],1)[0]
        positive, num_nodes = pkl_load(pos_path)
        positive.edge_index = positive.edge_index.T

        # return (anchor,num_nodes), positive, negative, num_nodes
        return anchor, positive, num_nodes, y
