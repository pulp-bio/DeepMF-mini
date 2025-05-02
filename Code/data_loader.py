# Copyright (C) 2024 ETH Zurich. All rights reserved.   
# Author: Carlos Santos, ETH Zurich           

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.   
# SPDX-License-Identifier: Apache-2.0


# Imports
import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

#############################################################
# Dataloader and transformations
#############################################################

class DeepMFDataset(Dataset):

    """ Dataset implementation:
        Returns a tuple of (f1, f2, f3) and ecg gt.
        It assumes all data is stored in .mat files that contain the features and ground truth ECG waveforms."""

    def __init__(self, data, feature_names, task, transform_in_ear = None, transform_ecg = None):
        self.data = data # list of .mat file locations
        self.feature_names = feature_names # list of str with features to use
        self.task = task # encode/classify
        self.transform_in_ear = transform_in_ear
        self.transform_ecg = transform_ecg

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        file_name = self.data[index]
        file_data = sio.loadmat(file_name)  # Load .mat file

        # Ear biopotentials
        in_ear = []
        for i in range(len(self.feature_names)):  # Iterate over feature names
            feature_name = self.feature_names[i]
            feature = torch.tensor(np.squeeze(file_data[feature_name]), dtype = torch.float32) # Load feature from file
            if self.transform_in_ear: # Transform feature 
                feature = self.transform_in_ear(feature)
            in_ear.append(feature)  # Append to list
        in_ear = torch.stack(in_ear, dim = 0)
        
        # ECG
        if self.task == 'encode':
            ecg = torch.tensor(np.squeeze(file_data['LeadI_chest']), dtype=torch.float32)
            if self.transform_ecg: # Transform ECG
                ecg = self.transform_ecg(ecg)
            ecg = torch.unsqueeze(ecg, dim=0)
            
        else:
            ecg = torch.tensor(np.squeeze(file_data['LeadI_chest_ones']), dtype=torch.float32)
        
        return in_ear, ecg
    

class tanhNormalize:
    def __init__(self, scale_factor = 0.5):
        self.scale_factor = scale_factor
    
    def __call__(self, signal):
        return torch.tanh(signal * self.scale_factor)
    

class Normalize:
    def __init__(self):
        pass

    def __call__(self, signal):
        mean = torch.mean(signal)
        std = torch.std(signal)
        return (signal - mean) / std
    
class MinMaxNormalize:
    def __init__(self):
        pass

    def __call__(self, signal):
        min_val = torch.min(signal)
        max_val = torch.max(signal)
        return (signal - min_val) / (max_val - min_val)
    
class Invert:
    def __init__(self):
        pass

    def __call__(self, signal):
        return -signal
    

# === Custom Data Reader for Calibration ===
# class DeepMFDataReader(CalibrationDataReader):
#     def __init__(self, dataloader):
#         self.data = []
#         for in_ear, _ in dataloader:
#             self.data.append({"input": in_ear.numpy().astype(np.float32)})  # Adjust "input" based on your model's input name
#         self.iter = iter(self.data)
    
#     def get_next(self):
#         return next(self.iter, None)  # Return data until exhausted
    
#     def rewind(self):
#         self.iter = iter(self.data)  # Reset iterator