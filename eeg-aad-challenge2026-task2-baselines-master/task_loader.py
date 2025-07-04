import math

import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from mne.decoding import CSP
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import random

def get_MMAAD_data(name="s1", time_len=1, data_document_path1="/assert",data_document_path2="/assert2"):
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
    
    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()  #eeg.shape 21120x32 
        mat_event_value = mat_event[0]['value']  # 1*20 1=male, 2=female
        mat_eeg = matstruct_contents[0, 0]['eeg']  # 20 trials 21120*32
        for i in range(mat_eeg.shape[1]):  
            mat_eeg_data.append(mat_eeg[0, i])
            mat_event_data.append(mat_event_value[i][0][0])

        return mat_eeg_data, mat_event_data
    
    def sliding_window2(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))
        # train_eeg = []
        test_eeg = []
        # train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i+window_size, :]
                windows.append(window)
                new_label.append(label)
            test_eeg.append(np.array(windows))
            test_label.append(np.array(new_label))
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)
        return test_eeg, test_label
    
    args = DotMap()
    args.name = name
    args.data_document_path1 = data_document_path1
    args.data_document_path2 = data_document_path2
    args.trade_off = 1
    args.subject_number = int(args.name[1:])
    args.ConType = ["No"]      
    args.fs = 128        
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size =64
    args.max_epoch = 200
    args.patience = 15
    args.randomized = False    
    args.image_size = 32
    args.people_number = 50
    args.eeg_channel = 32 
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 20    
    args.cell_number = 21120 
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.csp_comp = 32  
    args.label_col = 0

    args.delta_low = 1
    args.delta_high = 3
    args.theta_low = 4
    args.theta_high = 7
    args.alpha_low = 8
    args.alpha_high = 13
    args.beta_low = 14
    args.beta_high = 30
    args.gamma_low = 31
    args.gamma_high = 50
    args.log_path = "result/1s"
    args.frequency_resolution = args.fs / args.window_length   
    args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
    args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
    args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
    args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
    args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
    args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
    args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
    args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    
    #Loading source domain data
    data1_path = args.data_document_path1 + "/data/" + str(args.name) + ".npy"
    label1_path = args.data_document_path1 + "/label/" + str(args.name) + ".npy" 

    train_eeg = np.load(data1_path)
    train_label = np.load(label1_path)
    data1 = train_eeg.transpose(0, 2, 1) 
    data1 = data1[:, :args.eeg_channel, :]
    label1 = np.array(train_label)
    train_label = np.squeeze(label1) 
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space', norm_trace=True)
    data1 = csp.fit_transform(data1, train_label)
    train_eeg = data1.transpose(0, 2, 1)
    args.n_train = len(train_label)
   
    print(train_eeg.shape, 5)

    train_data = train_eeg.transpose(0, 2, 1)

    indices = np.arange(train_data.shape[0]) 
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]
    
    #Loading target domain data
    data2_path = args.data_document_path2 + "/data/" + str(args.name) + ".npy"
    label2_path = args.data_document_path2 + "/label/" + str(args.name) + ".npy" 

    test_eeg2 =np.load(data2_path)
    test_label2 = np.load(label2_path)

    data2 = test_eeg2.transpose(0, 2, 1)
    data2 = data2[:, :args.eeg_channel, :]
    label2 = np.array(test_label2)
    test_label2 = np.squeeze(label2) 
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space', norm_trace=True)
    data2 = csp.fit_transform(data2, test_label2)
    test_eeg2 = data2.transpose(0, 2, 1)

    args.n_test = len(test_label2)  
   

    print("len of test_label", len(test_label2))
    test_data2 = test_eeg2.transpose(0, 2, 1)   

    indices = np.arange(test_data2.shape[0]) 
    np.random.shuffle(indices)
    test_data2, test_label2 = test_data2[indices], test_label2[indices]
    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=False, pin_memory=True)
    valid_loader = None
    test_loader2 = DataLoader(dataset=CustomDatasets(test_data2, test_label2),
                             batch_size=args.batch_size, drop_last=False, pin_memory=True)
    return train_loader, valid_loader, test_loader2





