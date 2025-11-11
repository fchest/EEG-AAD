from __future__ import division
from __future__ import print_function
import os
import argparse
import csv
import numpy as np
from typing import Tuple, Any, List
from pathlib import Path

import torch
from torch import nn
import pandas as pd
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from model_module import DARNet

# 可选：指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device is", device)

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_model(args) -> nn.Module:
    name = args.model_name.lower()
    if name == "darnet":
        return DARNet(args)
    raise ValueError(f"Unknown model: {name}")

def getData(args, id):
    onedata = np.load(args.data_path +  f'S{id}.npy')
    onedata = onedata.transpose(0,2,1)
    onedata = np.array(onedata)
    return onedata

class CustomDatasets(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        return data


class Trynetwork():
    def __init__(self, model, batch_size):
        self.model = model
        self.best_test = 0
        self.batch_size = batch_size

    def inference(self, test_loader):
        self.model.eval()
        Epochs_pred = []
        with torch.no_grad():
            for batch_data in test_loader:
                input = batch_data.cuda().float()
                pred = self.model(input)         
                pred = torch.argmax(pred, dim=1)
                pred = pred.detach().cpu().numpy()
                Epochs_pred.append(pred)
            return Epochs_pred
            

    def __getModel__(self):
        return self.model

def run_inference(args, testdata):
    # get tast data
    test_loader = DataLoader(
        dataset=CustomDatasets(testdata),
        batch_size= args.batch_size,
        drop_last=False,
    )

    # build the model and load the weights
    model = get_model(args)
    model.to(device)
    model = torch.load(args.model_path)
    model.eval()

    # inference
    BaselineNet = Trynetwork(
        model,
        batch_size = args.batch_size)
    pre_lables = BaselineNet.inference(test_loader)
    pre_lables = np.concatenate(pre_lables, axis=0)
    return pre_lables



def parse_args():
    p = argparse.ArgumentParser(description="EEG Inference script")
    p.add_argument("--model_name", 
                   type=str, 
                   default="DARNet", #Your model_name
                   help="Model name (e.g., DARNet)")
    p.add_argument("--model_path", 
                   type=str, 
                   default="/savemodel/DARNet.pt",  #Your model_path
                   help="Checkpoint path (.pt/.pth)")
    p.add_argument( "--data_path", 
                   type=str, 
                   default="/testdata/", #Your data_path
                   help="EEG source (# eeg-aad-challenge2025-task1-baselines-master/dataset/test_data/)")
    p.add_argument("--out_csv", 
                   type=str, 
                   default="./results_task1/cross_subject", 
                   help="Output CSV path (id,predictions)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    options = {'MM-AAD':[10 ,32, 20, 128]} 
    args.subject_number = options['MM-AAD'][0]
    args.eeg_channel = options['MM-AAD'][1]
    args.trail_number = options['MM-AAD'][2]
    args.fs  = options['MM-AAD'][3]

    args.batch_size = 64
    args.num_workers = 4
    args.num_classes = 2
    args.device = "cuda"

    sub_ids = [31,32,33,34,35,36,37,38,39,40]
    for id in sub_ids:    
        testdata = getData(args, id)
        pre_lables = run_inference(args, testdata)

        # write CSV（2 columns: id, label）
        out_path = f'{args.out_csv}_{id}.csv'
        out_dir  = os.path.dirname(out_path)
        if out_dir:                                
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for i, y in enumerate(pre_lables):
                writer.writerow([i, int(y)])

        print(f"Write predictions to: {out_path}")

