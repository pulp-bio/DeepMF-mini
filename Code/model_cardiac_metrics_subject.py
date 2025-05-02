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
from tqdm import tqdm
import onnxruntime as ort

# Custom imports
from DeepMF import DeepMFClassifier, DeepMFMiniClassifier
from inference import validate_classification
from utils import read_model_lines, get_LOO_sets, save_ROC_curve, save_cardiac_curve
from parser_file import parse_train

#############################################################
# Script to get model metrics in LOO subject
#############################################################

def main():

    # Load argparse arguments
    args = parse_train()

    # Connect to GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(314)

    # Read training models from saving directory
    trainings_file = os.path.join(args.all_model_dir, 'Trainings.txt')
    if not os.path.exists(trainings_file):
        raise ValueError("Trainings file not found.")
    
    # Divide lines into training names and features
    trainings, features = read_model_lines(trainings_file)

    # Ids and thresholds
    subject_ids = os.listdir(args.data_loading_dir)
    thresholds = np.linspace(0.6, 0.2, 21)

    for i in range(len(trainings)): # for each model

        training = trainings[i] # Model name ("Model 1/Model 2/Model n")
        feature_names = features[i] # Feature names

        # Access model directory
        model_dir = os.path.join(args.all_model_dir, training)
        if not os.path.exists(model_dir):
            raise ValueError("Model directory not found.")
        
        ROC_dict = {} # Save ROC curves for each model
        cardiac_dict = {} # Save cardiac metrics for each model

        for j in range(len(subject_ids)): # for each subject

            subject_id = subject_ids[j]
            print(f'Training: {training} - Patient: {subject_id}')

            # Get sets
            train_set, val_set = get_LOO_sets(args.data_loading_dir, subject_id)

            # Load subject folder
            subject_fold = os.path.join(model_dir, subject_id)
            if not os.path.exists(subject_fold):
                raise ValueError("Subject folder not found.")

            # Instantiate classifier
            # Instantiate and freeze classifier
            if args.model == 'DeepMF':
                classifier = DeepMFClassifier(in_channels = len(feature_names))
            elif args.model == 'DeepMFMini':
                classifier = DeepMFMiniClassifier(in_channels = len(feature_names))
            else:
                raise(ValueError('Model not found'))
                
            path_model = os.path.join(subject_fold, 'EC_checkpoint.pt')
            classifier.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
            classifier.to(device)
            for param in classifier.parameters():
                param.requires_grad = False

            # Load onnx file
            # path_model = os.path.join(subject_fold, 'EC_QUANT.onnx')
            # classifier = ort.InferenceSession(path_model)

            # ROC - cardiac metrics tensor for model and subject - measure evolution per threshold
            ROC_subject = np.zeros((len(thresholds), len(val_set), 2)) # thresholds x n_recordings x [prec, rec]
            cardiac_subject = np.zeros((len(thresholds), len(val_set), 4)) # thresholds x n_recordings x [HR_err, HRV_err, HR_corr_err, HRV_corr_err]

            for t in tqdm(range(len(thresholds))): # for each threshold
                save_path = os.path.join(subject_fold, 'Recordings', 'Thresholds', f'{t}')
                prec, rec, der, HR_err, HRV_err, HR_corr_err, HRV_corr_err = validate_classification(device, classifier, val_set, feature_names, args, save_path, cardiac_metrics = True, threshold = thresholds[t])

                ROC_subject[t, :, 0] = np.array(prec)
                ROC_subject[t, :, 1] = np.array(rec)
                cardiac_subject[t, :, 0] = np.array(HR_err)
                cardiac_subject[t, :, 1] = np.array(HRV_err)
                cardiac_subject[t, :, 2] = np.array(HR_corr_err)
                cardiac_subject[t, :, 3] = np.array(HRV_corr_err)
                print(f'\nThreshold: {thresholds[t]:.4f} - Precision: {np.mean(prec):.4f} - Recall: {np.mean(rec):.4f} - DER: {np.mean(der):.4f} - HR_err: {np.mean(HR_err):.4f} - HRV_err: {np.mean(HRV_err):.4f} - HR_corr_err: {np.mean(HR_corr_err):.4f} - HRV_corr_err: {np.mean(HRV_corr_err):.4f}')

            # Save subject in dict  
            ROC_dict[subject_id] = ROC_subject
            cardiac_dict[subject_id] = cardiac_subject

        # Save ROC_dict and cardiac_dict in model_dir
        sio.savemat(os.path.join(model_dir, 'ROC_dict.mat'), ROC_dict)
        sio.savemat(os.path.join(model_dir, 'cardiac_dict.mat'), cardiac_dict)

        # Save progress curves
        dir = os.path.join(model_dir, "threshold_curves")
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Save ROC and cardiac metrics curves
        for key in ROC_dict.keys(): # for each subject
            dir_subject = os.path.join(dir, key)
            if not os.path.exists(dir_subject):
                os.makedirs(dir_subject)

            ROC_subject = ROC_dict[key]
            ROC_subject = np.mean(ROC_subject, axis = 1) # mean accross recordings (len(thresholds) x 2)
            save_ROC_curve(dir_subject, key, ROC_subject)

            cardiac_subject = cardiac_dict[key]
            cardiac_subject = np.mean(cardiac_subject, axis = 1) # mean accross recordings (len(thresholds) x 4)
            save_cardiac_curve(dir_subject, key, cardiac_subject, thresholds)

if __name__ == "__main__":
    main()