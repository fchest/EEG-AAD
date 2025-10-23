import os
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as scio

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]


def getData(data_path, label_path, id): 
    onedata = np.load(data_path +  "/data/" + f'S{id}.npy')
    onelabel = np.load(label_path +  "/label/" + f'S{id}.npy')
    onedata = onedata.transpose(0,2,1)
    onelabel = np.squeeze(onelabel)
    return onedata,  onelabel

# ========================= model =====================================
def save_model(args, subject_name, best_acc, val_acc, model, epoch, model_name = None):
    print(f'Validation acc increase ({best_acc:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
    model_save_path = args.model_save_path + subject_name + ".pt"
    makePath(args.model_save_path)
    torch.save(model, model_save_path)
    

def load_model(path, subject_name, model_name = None):
    model_save_path = path + subject_name + ".pt"
    model = torch.load(model_save_path)
    return model 

