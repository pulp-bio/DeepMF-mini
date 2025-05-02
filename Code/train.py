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
from matplotlib import pyplot as plt
import scipy.io as sio

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import copy

# Import custom modules
from data_loader import DeepMFDataset, tanhNormalize, Normalize, Invert, MinMaxNormalize
from DeepMF import DeepMFEncoderDecoder, DeepMFClassifier, DeepMFMiniEncoderDecoder, DeepMFMiniClassifier
from inference import validate_classification
from utils import save_images, save_training_curves

#############################################################
#  Script that holds the logic for training of the networks
#############################################################

def train_loop(device, save_directory, model, feature_names, train_set, val_set, log_file, criterion, optimizer, scheduler, args, task):

    """ Training loop for DeepMF model """
    """ Inputs:
            device: torch.device
            save_directory: str - directory where to save model weights
            model: nn.Module - PyTorch model
            feature_names: list - features onto which to train the model (must be a valid key contained in the recording .mat files)
            train_set: list of tuples (files: list of .mat files, peaks: np array)
            val_set: list of tuples (files: list of .mat files, peaks: np array)
            criterion: nn.Module
            optimizer: torch.optim
            scheduler: torch.optim.lr_scheduler
            args: argparse.Namespace - arguments as specified in parser_file.py
            task: str - 'encode' to train the encoder-decoder or 'classify' to train the encoder-classifier
        
        Outputs:
            val_loss_per_epoch: float
            encoder_weights: dict"""

    # Saving
    if task == 'encode':
        model_checkpoint = 'ED_checkpoint.pt'
        n_epochs = args.n_epochs_enc
    else:
        model_checkpoint = 'EC_checkpoint.pt'
        n_epochs = args.n_epochs_clas

    # Define list for training and validation files
    train_files = []
    val_files = []

    for i in range(len(train_set)):
        files, peaks = train_set[i]
        train_files.extend(files)

    for i in range(len(val_set)):
        files, peaks = val_set[i]
        val_files.extend(files)

    print(len(train_files))
    print(len(val_files))
    
    # Transformations and Dataloaders
    transform_in_ear = Invert()
    if args.model == 'DeepMFMini':
        transform_ecg = Compose([tanhNormalize(scale_factor = 0.5), MinMaxNormalize()])
    else:
        transform_ecg = None
    
    encoder_decoder_train = DeepMFDataset(train_files, feature_names, task, transform_in_ear = transform_in_ear, transform_ecg = transform_ecg)
    train_loader = DataLoader(encoder_decoder_train, args.batch_size, shuffle=True, num_workers=2)

    encoder_decoder_val = DeepMFDataset(val_files, feature_names, task, transform_in_ear = transform_in_ear, transform_ecg = transform_ecg)
    val_loader = DataLoader(encoder_decoder_val, args.batch_size, shuffle=True, num_workers=2)

    # Training metrics: L2, sensitivity, recall, DER, HR_error, HRV_error, HR_corr_error, HRV_corr_err
    train_MSE_loss, train_prec, train_rec, train_der = [None] * n_epochs, [None] * n_epochs, [None] * n_epochs, [None] * n_epochs
    val_MSE_loss, val_prec, val_rec, val_der = [None] * n_epochs, [None] * n_epochs, [None] * n_epochs, [None] * n_epochs
    
    # Training loop
    for epoch in range(n_epochs):

        print(f'Start of epoch n: {str(epoch)}')

        # Training mode
        model.train() 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # inputs
            in_ear, ecg = data
            in_ear = in_ear.to(device)
            ecg = ecg.to(device)

            # zero gradients + forward + loss + backward + optimize
            optimizer.zero_grad()
            outputs = model(in_ear)
            loss = criterion(outputs, ecg)
            loss.backward()
            optimizer.step()

            # training metrics
            running_loss += loss.item()
            if (i + 1) % args.print_every_iters == 0:
                print(
                    f'[Epoch: {epoch + 1} / {n_epochs},'
                    f' Iter: {i + 1:5d} / {len(train_loader)}]'
                    f' Training loss: {running_loss / (i + 1):.3f}',
                    file = log_file
                )

        mean_train_loss = running_loss / len(train_loader)
        train_MSE_loss[epoch] = mean_train_loss

        # Validation mode
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # inputs
                in_ear, ecg = data
                in_ear = in_ear.to(device)
                ecg = ecg.to(device)
                
                # forward + loss
                outputs = model(in_ear)
                loss = criterion(outputs, ecg)

                # print statistics
                running_loss += loss.item()

        mean_val_loss = running_loss / len(val_loader)
        val_MSE_loss[epoch] = mean_val_loss

        scheduler.step()

        print(
            f'[Epoch: {epoch + 1} / {n_epochs}]'
            f' Training loss: {mean_train_loss:.3f}'
            f' Validation loss: {mean_val_loss:.3f}',
            file = log_file
        )

        # In classification task, record precision, recall and DER metrics 
        if task == 'classify':
            
            path_train = None
            path_val = None

            if epoch == n_epochs - 1: # Save peaks file
                # Record peak metrics for train and validation files
                path_train = os.path.join(save_directory, 'Recordings', 'train')
                path_val = os.path.join(save_directory, 'Recordings', 'val')
                # save_directory + 'Recordings/train/'
                # path_val = save_directory + 'Recordings/val/'
                if not os.path.exists(path_train):
                    os.makedirs(path_train)
                    os.makedirs(path_val)

            # Precision, recall and DER curves
            precision, recall, der = validate_classification(device, model, train_set, feature_names, args, path_train)
            train_prec[epoch], train_rec[epoch], train_der[epoch] = np.mean(precision), np.mean(recall), np.mean(der)
            precision, recall, der = validate_classification(device, model, val_set, feature_names, args, path_val)
            val_prec[epoch], val_rec[epoch], val_der[epoch] = np.mean(precision), np.mean(recall), np.mean(der)

    print('Finished Training', file = log_file)

    if task == 'encode':
        sio.savemat(os.path.join(save_directory, task + '_loss_curve.mat'), {'train_MSE_loss': train_MSE_loss, 'val_MSE_loss': val_MSE_loss})
    else:
        sio.savemat(os.path.join(save_directory, task + '.mat'), {'train_MSE_loss': train_MSE_loss, 'val_MSE_loss': val_MSE_loss, 'train_prec': train_prec, 'train_rec': train_rec, 'train_der': train_der, 'val_prec': val_prec, 'val_rec': val_rec, 'val_der': val_der})
    
    # Save loss and encoder weight
    encoder_weights = copy.deepcopy(model.encoder.state_dict())
    model.write_weights(os.path.join(save_directory, model_checkpoint))

    # Save images better
    image_directory = os.path.join(save_directory, 'images')
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    save_images(device, model, val_loader, image_directory, feature_names, task)
    save_training_curves(image_directory, task, train_MSE_loss, val_MSE_loss, train_prec, val_prec, train_rec, val_rec, train_der, val_der)
    
    return encoder_weights, val_MSE_loss[-1]


def train_encoder(device, model_fold, feature_names, train_set, val_set, log_file, args, weight_init):

    """ Train the encoder-decoder model """
    """ Inputs:
            device: torch.device
            model_fold: str - directory where to save model weights
            feature_names: list of str - features onto which to train the model
            train_set: list of tuples (files: list of .mat files, peaks: np array)
            val_set: list of tuples (files: list of .mat files, peaks: np array)
            log_file: str - .txt where to print training losses
            args: argparse.Namespace - arguments as specified in parser_file.py
            weight_init: bool - weight initialization

        Outputs:
            encoder_weights: dict - weights of the encoder to later initialize the encoder-classifier model
            last_loss: float"""

    # Instantiate model
    num_features = len(feature_names)  # Number of features in the input
    if args.model == 'DeepMF':
        deepMF_encoderdecoder = DeepMFEncoderDecoder(num_features)
    elif args.model == 'DeepMFMini':
        deepMF_encoderdecoder = DeepMFMiniEncoderDecoder(num_features)
    else:
        raise(ValueError('Model not found'))

    # TODO: weight_init logic not implemented
    if weight_init:
        pass

    # Optimizer and loss functions
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(deepMF_encoderdecoder.parameters(), lr = args.lr_enc, weight_decay=args.wd_enc)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    deepMF_encoderdecoder.to(device)

    # Train
    encoder_weights, last_loss = train_loop(device, model_fold, deepMF_encoderdecoder, feature_names, train_set, val_set, log_file, criterion, optimizer, scheduler, args, task = 'encode')

    return encoder_weights, last_loss

def train_classifier(device, model_fold, feature_names, train_set, val_set, log_file, encoder_weights, args, weight_init):
    
    """ Train the encoder-classifier model """
    """ Inputs:
            device: torch.device
            model_fold: str - directory where to save model weights
            feature_names: list of str - features onto which to train the model
            train_set: list of tuples (files: list of .mat files, peaks: np array)
            val_set: list of tuples (files: list of .mat files, peaks: np array)
            log_file: str - .txt where to print training losses
            encoder_weights: dict
            args: argparse.Namespace - arguments as specified in parser_file.py
            weight_init: bool - weight initialization

        Outputs:
            last_loss: float"""

    # instantiate model
    num_features = len(feature_names)  # Number of features in the input

    if args.model == 'DeepMF':
        deepMF_encoderclassifier = DeepMFClassifier(num_features)
    elif args.model == 'DeepMFMini':
        deepMF_encoderclassifier = DeepMFMiniClassifier(num_features)
    else:
        raise(ValueError('Model not found'))

    deepMF_encoderclassifier.to(device)

    if weight_init:
        # Load encoder weights
        deepMF_encoderclassifier.encoder.load_state_dict(encoder_weights)

        # Freeze encoder weights
        layer_names = ['classifier']
        for name, param in deepMF_encoderclassifier.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Get optimizer and loss functions
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(deepMF_encoderclassifier.parameters(), lr=args.lr_clas, weight_decay=args.wd_clas) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    _, last_loss = train_loop(device, model_fold, deepMF_encoderclassifier, feature_names, train_set, val_set, log_file, criterion, optimizer, scheduler, args, task = 'classify')
            
    return last_loss