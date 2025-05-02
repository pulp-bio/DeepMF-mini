# Copyright (C) 2024 ETH Zurich. All rights reserved.   
# Author: Carlos Santos, ETH Zurich           

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.   
# SPDX-License-Identifier: Apache-2.0


# Imports
import os
import numpy as np
import scipy.io as sio
import torch

# Custom imports
from train import train_encoder, train_classifier
from utils import read_model_lines, get_LOO_sets
from parser_file import parse_train, log_arguments

#############################################################
# Script to train models with leave-one-out recording
# For each model, for each subject, for each recording, train encoder-decoder and encoder-classifier based on same-subject recordings
#############################################################

def main():

    # Load argparse arguments
    args = parse_train()
    log_arguments(args)

    # Connect to GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(314)
    torch.cuda.manual_seed(314)

    # Read training models from saving directory
    trainings_file = os.path.join(args.all_model_dir, 'Trainings.txt')
    if not os.path.exists(trainings_file):
        raise ValueError("Trainings file not found.")
    
    # Divide lines into training names and features
    trainings, features = read_model_lines(trainings_file)

    # Get training and validation paths
    subject_ids = os.listdir(args.data_loading_dir)
    
    # LOSO CV - n subjects, m recordings
    num_recordings = 4 # TODO: specify for your dataset
    enc_mat = np.zeros((len(subject_ids), num_recordings))
    cla_mat = np.zeros((len(subject_ids), num_recordings))

    for i in range(len(trainings)): # for each model

        training = trainings[i] # "Model 1/Model 2/Model n"
        feature_names = features[i] # "L_inear/R_inear"/...

        # Create model directory - for current model save
        model_dir = os.path.join(args.all_model_dir, training)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir) 

        for j in range(len(subject_ids)): # for each subject

            subject_id = subject_ids[j]
            print(training + ". Subject: " + subject_id)

            # Read recordings for each subject
            subject_data_folder = os.path.join(args.data_loading_dir, subject_id)
            recordings = os.listdir(subject_data_folder)

            for k in range(len(recordings)): # for each recording

                recording_id = recordings[k]
                print("LOO Recording: " + recording_id) # Model ith, subject jth, recording kth out

                # Create recording fold directory - for current model and fold save
                recording_fold = os.path.join(model_dir, subject_id, recording_id)
                if not os.path.exists(recording_fold):
                    os.makedirs(recording_fold)

                log_file_name = os.path.join(recording_fold, 'log_file.txt')
                log_file = open(log_file_name, 'w')

                ###### Training #####
                # Training and validation files -> n recordings vs 1 recording
                train_set, val_set = get_LOO_sets(subject_data_folder, recording_id)

                # Train encoder
                encoder_weights, last_loss_encoder = train_encoder(device, recording_fold, feature_names, train_set, val_set, log_file, args, weight_init = False)
                enc_mat[j, k] = last_loss_encoder

                # Train classifier
                last_loss_classifier = train_classifier(device, recording_fold, feature_names, train_set, val_set, log_file, encoder_weights, args, weight_init = True)
                cla_mat[j, k] = last_loss_classifier

                log_file.close() # Close log file

        # Save encoder and classifier loss matrices
        sio.savemat(os.path.join(model_dir, 'loss_' + training + '.mat'), {'enc_mat': enc_mat, 'cla_mat': cla_mat})

if __name__ == "__main__":
    main()