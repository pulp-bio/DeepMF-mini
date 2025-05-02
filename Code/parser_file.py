# Copyright (C) 2024 ETH Zurich. All rights reserved.   
# Author: Carlos Santos, ETH Zurich           

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.   
# SPDX-License-Identifier: Apache-2.0


# Imports
import argparse
import os

#############################################################
# Specifices the arguments needed to run the high-level scripts
#############################################################

def parse_train():
    parser = argparse.ArgumentParser(description='DeepMF algorithm for feature extraction and classification of ECG from in ear biopotential signals.')
    
    ############ Mandatory arguments ############
    parser.add_argument('data_loading_dir', type=str, help='Directory where the data is stored.')
    parser.add_argument('all_model_dir', type=str, help='Directory where models are saved.')
    parser.add_argument('device', type=str, help='Device: "cpu"/"cuda:0"/"cuda:1"/"cuda:2"/"cuda:3".')
    parser.add_argument('model', type =str, help='Model that you want to train. Options: "DeepMF"/"DeepMFMini".')

    ############ Optional arguments ############
    
    # Training
    parser.add_argument('--n_epochs_enc', type=int, default = 30, help='Number of epochs for encoder training.')
    parser.add_argument('--n_epochs_clas', type=int, default = 20, help='Number of epochs for classifier training.')
    parser.add_argument('--lr_enc', type=float, default = 0.001, help='Learning rate for encoder training.')
    parser.add_argument('--lr_clas', type=float, default = 0.001, help='Learning rate for classifier training.')
    parser.add_argument('--wd_enc', type=float, default = 1e-4, help='Weight decay for encoder training.')
    parser.add_argument('--wd_clas', type=float, default = 1e-6, help='Weight decay for classifier training.')
    parser.add_argument('--batch_size', type=int, default = 32, help='Batch size.')
    parser.add_argument('--print_every_iters', type=int, default = 600, help='Print every iterations.')
    parser.add_argument('--training_summary', type=str, default = 'training_summary.txt', help='Log file.')
    
    # Inference
    parser.add_argument('--jump', type=float, default = 0.4, help= 'Seconds to jump in sliding window. Options: 0.4/0.8. Default: 0.4.')
    parser.add_argument('--window_size', type=int, default = 2, help='Window size in seconds. Only implemented value: 2. Default: 2.')
    parser.add_argument('--not_trust', type=float, default = 0.4, help='Not trustable inference. Default: 0.4 seconds.')
    parser.add_argument('--fs', type=int, default = 250, help='Sampling frequency. Default: 250Hz.')
    parser.add_argument('--method', type=str, default = 'max', help='Method to use in sliding window. Only implemented option: "max". Default: "max".')
    parser.add_argument('--mask', type=int, default = 70, help='Mask in indexes between peaks. Default: 70')
    parser.add_argument('--tolerance', type=int, default = 3, help='Tolerance in indexes between peaks. Default: 3.')
    parser.add_argument('--av_time', type=int, default = 10, help='Time in seconds for HR/HRV computation. Defaullt: 10.')
    
    return parser.parse_args()

def log_arguments(args):
    log_file_name = os.path.join(args.all_model_dir, args.training_summary)
    log_file = open(log_file_name, 'w')
    for arg, value in vars(args).items():
        log_file.write(f"{arg}: {value}\n")
    log_file.close()