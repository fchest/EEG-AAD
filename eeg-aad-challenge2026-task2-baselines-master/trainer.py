from __future__ import division
from __future__ import print_function
import os
import math

import numpy as np
import pandas as pd

from datetime import datetime
import argparse
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from typing import Optional
import torchinfo
import matplotlib.pyplot as plt

from utils import *
from collections import OrderedDict
from model_module import DARNet
from mne.decoding import CSP

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device is", device)

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

class StepwiseLR_GRL: 
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.001,
                 gamma: Optional[float] = 0.01, decay_rate: Optional[float] = 0.1,max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        if lr <= 1e-8:
            lr = 1e-8
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1

class Trynetwork():
    def __init__(self, model, train_loader, valid_loader, test_loader, batch_size, lr, weight_decay):
        self.model = model
        self.datasets = OrderedDict((("train", train_loader), ("valid", valid_loader), ("test", test_loader)))
        if valid_loader is None:
            self.datasets.pop("valid")
        if test_loader is None:
            self.datasets.pop("test")
        self.best_test = 0
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 35], gamma=0.5)
        self.scheduler_down = StepwiseLR_GRL(self.optimizer, init_lr= args.lr, gamma= 10, decay_rate=args.lr_decayrate,max_iter=args.max_epoch)
        self.criterion = nn.CrossEntropyLoss()
        
        # initialize epoch dataFrame instead of loss and acc for train and test
        self.val_df = pd.DataFrame()  
        self.train_df = pd.DataFrame()  
        self.epoch_df = pd.DataFrame()  
        

    def __getModel__(self):
        return self.model

    def train_step(self):
        self.model.train()
        train_dicts_per_epoch = OrderedDict()
        Batch_size, Cls_loss, Train_acc = [], [], []
        for i_batch, batch_data  in enumerate(self.datasets['train']):
            train_data,  train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data,  train_label = train_data.cuda().float(), train_label.cuda().long()

            source_softmax = self.model(train_data)
            nll_loss = self.criterion(source_softmax, train_label)  

            Batch_size.append(len(train_label))
            _, predicted = torch.max(source_softmax.data, 1)
            batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(
                train_label)
            # Forward pass
            Train_acc.append(batch_acc)
            cls_loss_np = nll_loss.cpu().detach().numpy()
            Cls_loss.append(cls_loss_np)

            # Backward and optimize
            self.optimizer.zero_grad()
            nll_loss.backward()
            self.optimizer.step()

        epoch_acc = sum(Train_acc) / len(Train_acc) * 100
        epoch_loss = sum(Cls_loss) / len(Cls_loss)

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)
        train_dicts_per_epoch = {k: [v] for k, v in train_dicts_per_epoch.items()}
        self.train_df = pd.concat([self.train_df, pd.DataFrame(train_dicts_per_epoch)], ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  
        return epoch_loss , epoch_acc
    
    def test_batch(self, input, label):
        self.model.eval()
        with torch.no_grad():
            val_input = input.cuda().float()
            val_label = label.cuda().long()
            val_fc1 = self.model(val_input)
            loss = self.criterion(val_fc1, val_label)
            _, preds = torch.max(val_fc1.data, 1)  
            preds = preds.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
        return preds, loss

    def evaluate_step(self, Flag_test):
        if Flag_test and "test" in self.datasets:
            setname = 'test'
        else:
            setname = 'valid'
        result_dicts_per_monitor = OrderedDict()  
        with torch.no_grad():
            Batch_size, Epochs_loss, Epochs_acc = [], [], []
            for i_batch, batch_data in enumerate(self.datasets[setname]):
                seq_input,  target = batch_data
                target = target.squeeze(-1)
                pred, loss = self.test_batch(seq_input,  target)  
                Epochs_loss.append(loss)
                Batch_size.append(len(target))
                Epochs_acc.append(np.equal(pred, target.numpy()).sum())  
        epoch_acc = sum(Epochs_acc) / sum(Batch_size) * 100
        epoch_loss = sum(Epochs_loss) / len(Epochs_loss)
        key_loss = setname + '_loss'
        key_acc = setname + '_acc'
        loss = {key_loss: epoch_loss}
        acc = {key_acc: epoch_acc}
        result_dicts_per_monitor.update(loss)
        result_dicts_per_monitor.update(acc)
        result_dicts_per_monitor = {k: [v] for k, v in result_dicts_per_monitor.items()}
        self.val_df = pd.concat([self.val_df, pd.DataFrame(result_dicts_per_monitor)], ignore_index=True)
        self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]  
        return epoch_loss, epoch_acc


    def train(self, args, testsub_id):
        torchinfo.summary(self.model)
        testsub_name = 'S' + str(testsub_id)
        
        best_epoch = 0
        best_acc = 0
        for epoch in range(1, args.max_epoch + 1):
            train_loss,train_acc = self.train_step( )
            val_loss, val_acc = self.evaluate_step(False)
            # self.scheduler_down.step()
            print('TestSub:', testsub_name,
                  'Epoch {:2d} Finsh | Now_lr {:2.4f}/{:2.4f}|Train Loss {:2.4f} | Valid Loss {:2.4f} | Train Acc {:5.4f}| Valid Acc {:5.4f}'.format(epoch,
                                                                                                                                                self.optimizer.param_groups[0]["lr"], args.lr,
                                                                                                                                                train_loss,
                                                                                                                                                val_loss,
                                                                                                                                                train_acc,
                                                                                                                                                val_acc))
           
            if  val_acc > best_acc:
                save_model(args, testsub_name, best_acc, val_acc, self.model, epoch, args.model)
                best_acc = val_acc
                best_epoch = epoch
                # save_model(args, model, name=args.name)
                stale = 0
            else:
                stale += 1
                if stale > args.patience:
                    print(f"Early stopping at epoch {epoch}!")
                    break
    
            self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)
        model = load_model(args.model_save_path, testsub_name, args.model)
        self.model = model
        test_loss, model_test_acc = self.evaluate_step(True)
        # self.save_acc_loss_fig(args, testsub_id)
        print("-" * 50)
        print('Test_Subject :{:s} |Best epoch:{:d} | Test Loss:{:2.4f} | Best Acc {:2.4f} | Savemodel Acc {:2.4f}'.format(testsub_name,
                                                                                                    best_epoch,
                                                                                                    test_loss,
                                                                                                    best_acc,
                                                                                                    model_test_acc))
        print("-" * 50)
        return model_test_acc
    

def cross_session(args, sub_id, train_data,  train_label, val_data,  val_label):
    train_label = train_label.reshape(-1,1)
    val_label = val_label.reshape(-1,1)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]
    val_data, val_label = val_data[indices], val_label[indices]
    
    print(f"train_data_shape{train_data.shape},val_data_shape{val_data.shape}")

    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=False, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label),
                              batch_size=args.batch_size, drop_last=False, pin_memory=True)
    
    #####################################################################################
    #2.define model
    #####################################################################################
    BaselineNet =Trynetwork(
        model = DARNet(args).cuda(),
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader=None,
        batch_size = args.batch_size, 
        lr = args.lr,
        weight_decay = args.weight_decay)
    model_val_acc = BaselineNet.train(args, sub_id)
    return model_val_acc

    
if __name__ == '__main__':
    # Training settings
    args = argparse.ArgumentParser()
    args.seed = 42
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False

    # data
    args.dataset = 'MM-AAD'
    args.start_time = datetime.now().strftime(f"task2_AAD_{args.dataset}_%Y-%m-%d-%H-%M-%S")
    print('start time:',args.start_time)

    options = {'MM-AAD':[30 ,32, 20, 128]} #30 10
    args.subject_number = options[args.dataset][0]
    args.eeg_channel = options[args.dataset][1]
    args.trail_number = options[args.dataset][2]
    args.fs  = options[args.dataset][3]
    args.source_data_path = "/media/pre_aoall" #Your source_data_path
    args.source_label_path = "/media/pre_aoall"#Your source_label_path

    args.target_data_path = "/media/pre_avall"#Your target_data_path
    args.target_label_path = "/media/pre_avall"#Your target_label_path

    args.win_time = 1
    args.win_len = math.ceil(args.fs * args.win_time)
    args.overlap = 0.5
    args.window_lap = args.win_len * (1 - args.overlap)

    # basic info of the model
    args.model = "DARNet"
    args.batch_size = 128
    args.lr = 5e-4
    args.lam = 0.2
    args.lr_decayrate = 0.5
    args.weight_decay = 3e-4
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 10
    args.csp_comp = args.eeg_channel
    
    # save to 
    filename = "./exps/cross-session/%s/" % args.model
    args.model_save_path = f'{filename}baseline_%s/' % args.start_time
    makePath(args.model_save_path)

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
   
    sub_ids =  list(range(1, args.subject_number+1)) 
    sub_ids = [sub_id for sub_id in sub_ids]

    all_acc = []
    for id in sub_ids:  
        source_alldata,  source_alllabel = getData(args.source_data_path, args.source_label_path, id)
        target_alldata,  target_alllabel = getData(args.target_data_path, args.target_label_path, id)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_data = csp.fit_transform(source_alldata, source_alllabel)
        val_data = csp.transform(target_alldata)

        savemodel_acc = cross_session(args, id, train_data,  source_alllabel, val_data,  target_alllabel)
        all_acc.append(savemodel_acc)

    print('=' * 108)
    for i, acc in enumerate(all_acc):
        print("sub{} -> acc: {:.4f}%".format(sub_ids[i], acc))
    print('=' * 108)

    print(f"lr:{args.lr } -> bs:{args.batch_size}")
    print(f"Cross Session -> Result ACC mean: {np.mean(all_acc):.4f}")
    print('=' * 108)
    now1 = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
    print('end time:',now1)


