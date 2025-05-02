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
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import torch

def ignore_prefix(file_list, prefix):
    filtered_files = []
    for file_name in file_list:
        # Extract the base file name without the extension
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        # Check if the base name starts with the specified prefix
        if not base_name.startswith(prefix):
            filtered_files.append(file_name)

    return filtered_files


def get_prefix(file_list, prefix):
    filtered_files = []
    for file_name in file_list:
        # Extract the base file name without the extension
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        # Check if the base name starts with the specified prefix
        if base_name.startswith(prefix):
            filtered_files.append(file_name)

    return filtered_files


def get_recording(file_list, recording):
    filtered_files = []
    for file_name in file_list:
        # Extract the base file name without the extension
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        # Check if the base name starts with the specified prefix
        if recording in base_name:
            filtered_files.append(file_name)
            
    return filtered_files


def read_model_lines(file_path):
    training = []
    features = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(': ')
            training_name = parts[0]
            training_features = parts[1].split(', ')
            training.append(training_name)
            features.append(training_features)
    return training, features


def get_files_from_str(files, string):

    filtered_files = []

    for file in files:
        if string in file:
            filtered_files.append(file)

    return filtered_files


def get_files_ignore_str(files, string):
    
        filtered_files = []
    
        for file in files:
            if string not in file:
                filtered_files.append(file)
    
        return filtered_files


def LOO_folders_str(data_loading_dir, string):
    """
    Returns train and validation folder paths

    Input:
        - data_loading_dir: path to LOO decision
        - string: LOO id

    Output:
        - List of ordered file names
    """

    train_folders = []
    valid_folder = []

    folders = os.listdir(data_loading_dir) # read all folders in the loading directory
    
    for folder in folders:
        if string not in folder:
            train_folders.append(folder)
        else:
            valid_folder.append(folder)
    
    train_folders = append_path(train_folders, data_loading_dir) # Append path to access files
    valid_folder = append_path(valid_folder, data_loading_dir)

    return train_folders, valid_folder[0]


def append_path(file_list, path):

    new_list = []

    for file in file_list:
        new_file = os.path.join(path, file)
        new_list.append(new_file)

    return new_list


def skip_files(file_list, jump):
    """
    Returns ordered file names list from test_directory folder
    skipping elements according to the given `step`.

    Input:
        - file_list: List containing the ordered test files
        - jump: Step value determining how many elements to skip

    Output:
        - List of ordered file names
    """

    jump = int(np.round(jump/0.2))
    
    if jump < 1:
        raise ValueError("Step must be a positive integer greater than 20ms.")
    
    skipped_files = file_list[::jump]

    return skipped_files


def get_LOO_sets(data_loading_dir, id):
    """
    Returns train and validation dictionaries for DataLoader

    Input:
        - data_loading_dir: Directory containing folders (S1, S2, ... / R1, R2, ...)
        - id: LOO id, Sx/Rx

    Output:
        - train_set: Training set list
        - val_set: Validation set list
    """
    train_set = []
    val_set = []

    train_folder_paths, valid_folder_path = LOO_folders_str(data_loading_dir, id) # S1, S2 - S3 / R1, R2 - R3

    for folder_path in train_folder_paths:
        items = sorted(os.listdir(folder_path), key=len) # R2, R3 / recordings
        subfolders_paths = [os.path.join(folder_path, item) for item in items if os.path.isdir(os.path.join(folder_path, item))] # R1, R2, R3 / None

        if subfolders_paths: # Subject LOO
            for subfolder_path in subfolders_paths: # R1, R2, R3...
                # Recording files
                items = sorted(os.listdir(subfolder_path), key=len)
                items.remove('gt.mat') # recordings
                train_files = append_path(items, subfolder_path)
                # GT R-Peaks
                gt_data = sio.loadmat(os.path.join(subfolder_path, 'gt.mat'))
                gt_peaks = gt_data['chest_peaks'] - 1 # Takes care of Matlab 1 indexing
                # Dictionary
                train_set.append((train_files, gt_peaks))
        else: # Record LOO
            items.remove('gt.mat') # recordings
            train_files = append_path(items, folder_path)
            # GT R-Peaks
            gt_data = sio.loadmat(os.path.join(folder_path, 'gt.mat'))  # Load .mat file
            gt_peaks = gt_data['chest_peaks'] - 1 # Takes care of Matlab 1 indexing
            # Dictionary
            train_set.append((train_files, gt_peaks))

    # Validation files
    items = sorted(os.listdir(valid_folder_path), key=len) # R1, R2, R3 / recordings
    subfolders_paths = [os.path.join(valid_folder_path, item) for item in items if os.path.isdir(os.path.join(valid_folder_path, item))] # R2, R3 / None

    if subfolders_paths: # Subject LOO
            for subfolder_path in subfolders_paths: # R1, R2, R3...
                items = sorted(os.listdir(subfolder_path), key=len)
                items.remove('gt.mat')
                val_files = append_path(items, subfolder_path)
                gt_data = sio.loadmat(os.path.join(subfolder_path, 'gt.mat')) # Load .mat file
                gt_peaks = gt_data['chest_peaks'] - 1 # Takes care of Matlab 1 indexing
                # Dictionary
                val_set.append((val_files, gt_peaks))
                
    else: # Record LOO
        items.remove('gt.mat')
        val_files = append_path(items, valid_folder_path)
        gt_data = sio.loadmat(os.path.join(valid_folder_path, 'gt.mat')) # Load .mat file
        gt_peaks = gt_data['chest_peaks'] - 1 # Takes care of Matlab 1 indexing
        # Dictionary
        val_set.append((val_files, gt_peaks))

    return train_set, val_set


def save_images(device, model, val_loader, image_directory, feature_names, task):
    """ Save training curves and inference results as images """
    """ Inputs:
            device: torch.device
            model: nn.Module
            val_loader: DataLoader
            image_directory: str
            training_loss_per_epoch: list
            val_loss_per_epoch: list
            feature_names: list
            task: str

        Outputs:
            None"""

    # Inference examples
    # Select a batch of data
    dataiter = iter(val_loader)
    in_ear, ecg = next(dataiter)  # Using validation set for examples

    # Forward pass through the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        reconstructed = model(in_ear.to(device)).cpu()
        in_ear = in_ear.cpu() # Ensure input is on CPU for plotting
        ecg = ecg.cpu()  # Ensure ground truth ECG is also on CPU for plotting

    # Set the number of samples you want to display
    num_samples_to_display = 6

    # Plotting
    fig, axes = plt.subplots(nrows=num_samples_to_display, figsize=(10, 2 * num_samples_to_display))
    for i in range(num_samples_to_display):
        # Plot both Ground Truth ECG and Reconstructed Output on the same plot
        ax = axes[i]
        ax.plot(ecg[i].squeeze().numpy(), label='Ground Truth ECG', color='blue')
        if task == 'encode':
            for j in range(len(feature_names)):
                ax.plot(in_ear[:, j][i].numpy()*((np.std(ecg[i].squeeze().numpy()))/(np.std(in_ear[:, j][i].numpy()))), label=feature_names[j])
        ax.plot(reconstructed[i].squeeze().numpy(), label='Reconstructed Output', color='red')
        ax.set_title('Comparison for Sample {}'.format(i+1))
        ax.legend()

    plt.subplots_adjust(top=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Encoder-' + str(task) + ' inference results', size=16)
    plt.savefig(os.path.join(image_directory, str(task) + '_Inference results.png'))
    plt.close()


def save_training_curves(image_directory, task, training_loss_per_epoch, val_loss_per_epoch, train_prec, val_prec, train_rec, val_rec, train_der, val_der):
    """ Save training curves as images
        Inputs:
            image_directory:
            task: 
            training_loss_per_epoch: torch.device
            val_loss_per_epoch: nn.Module
            

        Outputs:
            None"""
    
    # Training curves
    save_curve(image_directory, task + '_MSE', training_loss_per_epoch, val_loss_per_epoch)
    if task == 'classify':
        save_curve(image_directory, 'Precision', train_prec, val_prec)
        save_curve(image_directory, 'Recall', train_rec, val_rec)
        save_curve(image_directory, 'DER', train_der, val_der)


def save_curve(save_directory, figure_title, train_list, val_list):
    """Save training curve
        Inputs:
            save_directory:
            figure_title:
            train_list:
            val_list:

        Outputs:
            None
    """
    # Convert to np arrays
    train_array = np.array(train_list)
    val_array = np.array(val_list)

    plt.figure()
    plt.plot(train_array)
    plt.plot(val_array)
    plt.title(figure_title)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_directory, figure_title + '.png'))
    plt.close()


def save_ROC_curve(save_directory, patient_id, metrics, threshold_flag = False):
    """Save ROC curve with exactly 7 ticks and prevent clipping
        Inputs:
            save_directory: Path to save the ROC curve
            patient_id: Identifier for the patient
            metrics: Array with precision and recall values

        Outputs:
            None
    """
    precision = metrics[:, 0]
    recall = metrics[:, 1]

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision)

    if threshold_flag:
        thresholds = np.linspace(0.9, 0.1, 21) # ROC curves
        threshold = best_threshold(precision, recall, thresholds)
        plt.scatter(recall[threshold], precision[threshold], color='red', 
                    s=100, edgecolor='black', zorder=5)
        
        print(f'Best threshold: {thresholds[threshold]} - Precision: {precision[thresholds == threshold]} - Recall: {recall[thresholds == threshold]}')
    
    # plt.scatter(recall[thresholds == threshold_2], precision[thresholds == threshold_2], color='green', 
    #            s=100, edgecolor='black', zorder=5)
    
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.xticks(fontsize=8)  # Smaller tick labels
    plt.yticks(fontsize=8)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))  # Ensure 7 ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))  # Ensure 7 ticks on y-axis

    
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', pad=10)
    plt.savefig(os.path.join(save_directory, 'ROC_curve.pdf'), dpi=600, bbox_inches='tight')
    plt.close()

    if threshold_flag:
        return threshold


def best_threshold(precisions, recalls, thresholds):
    """Find the threshold that maximizes the F1 score."""
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_index = np.argmax(f1_scores)
    return best_index


def best_threshold_distance(precisions, recalls, thresholds):
    """Find the threshold that minimizes the Euclidean distance from (1, 1)."""
    distances = np.sqrt((1 - precisions)**2 + (1 - recalls)**2)
    best_index = np.argmin(distances)
    return thresholds[best_index]


def save_cardiac_curve(save_directory, patient_id, metrics, thresholds):
    
    """Save HR err vs HRV err curve
        Inputs:
            save_directory: Path to save the ROC curve
            patient_id: Identifier for the patient
            metrics: Array with HR err, HRV err, HR corr err, HRV corr err values

        Outputs:
            None
    """

    HR_err = metrics[:, 0]
    HRV_err = metrics[:, 1]
    HR_corr_err = metrics[:, 2]
    HRV_corr_err = metrics[:, 3]


    # HR error
    plt.figure(figsize=(5.5, 5))
    plt.plot(thresholds, HR_err, label='Prediction')
    plt.plot(thresholds, HR_corr_err, label='Corrected Prediction')
    
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('HR Error (bpm)', fontsize=10)
    plt.legend()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))

    plt.tight_layout()
    plt.tick_params(axis='both', which='major', pad=10)
    plt.savefig(os.path.join(save_directory, 'HR_error.pdf'), dpi=600, bbox_inches='tight')
    plt.close()

    # HRV error
    plt.figure(figsize=(5.5, 5))
    plt.plot(thresholds, HRV_err, label='Prediction')
    plt.plot(thresholds, HRV_corr_err, label='Corrected Prediction')
    
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('HRV Error (ms)', fontsize=10)
    plt.legend()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))

    plt.tight_layout()
    plt.tick_params(axis='both', which='major', pad=10)
    plt.savefig(os.path.join(save_directory, 'HRV_error.pdf'), dpi=600, bbox_inches='tight')
    plt.close()


def print_statistics(save_path, threshold, precision, recall, der, gt_peaks = None, prediction_peaks = None, tp = None, fp = None, fn = None):
    
    # Print to log file
    rec_log_file_name = os.path.join(save_path, 'peak_metrics_file.txt') # Record log file
    rec_log_file = open(rec_log_file_name, 'w')

    # Print identifier
    print(f'Threshold: {threshold}', file = rec_log_file)
    if isinstance(precision, float): # single value from 1 recording
        print(f'Tot peaks: {len(gt_peaks)}', file = rec_log_file)
        print(f'Tot pred: {len(prediction_peaks)}', file = rec_log_file)
        print(f'TP: {tp}', file = rec_log_file)
        print(f'FP: {fp}', file = rec_log_file)
        print(f'FN: {fn}', file = rec_log_file)
        print(f'Precision: {precision:.4f}', file = rec_log_file)
        print(f'Recall: {recall:.4f}', file = rec_log_file)
        print(f'DER: {der:.4f}', file = rec_log_file)
    else: # multiple values from multiple recordings
        print(f'Precision: {np.mean(precision):.4f} +/- {np.std(precision):.4f}', file = rec_log_file)
        print(f'Recall: {np.mean(recall):.4f} +/- {np.std(precision):.4f}', file = rec_log_file)
        print(f'DER: {np.mean(der):.4f} +/- {np.std(der):.4f}', file = rec_log_file)

    rec_log_file.close()


def print_cardiac_metrics(save_path, HR_err, HRV_err, HR_corr_err, HRV_corr_err, HR_err_std = None, HRV_err_std = None, HR_corr_err_std = None, HRV_corr_err_std = None):

    # Print to log file
    rec_log_file_name = os.path.join(save_path, 'peak_metrics_file.txt') # Record log file
    rec_log_file = open(rec_log_file_name, 'a')

    if isinstance(HR_err, float): # single value from 1 recording
        print(f'HR error: {HR_err:.4f} + {HR_err_std:.4f}', file = rec_log_file)
        print(f'HRV error: {HRV_err:.4f} + {HRV_err_std:.4f}', file = rec_log_file)
        print(f'HR correction error: {HR_corr_err:.4f} + {HR_corr_err_std:.4f}', file = rec_log_file)
        print(f'HRV correction error: {HRV_corr_err:.4f} + {HRV_corr_err_std:.4f}', file = rec_log_file)

    else: # multiple values from multiple recordings
        print(f'HR error: {np.mean(HR_err):.4f} + {np.std(HR_err):.4f}', file = rec_log_file)
        print(f'HRV error: {np.mean(HRV_err):.4f} + {np.std(HRV_err):.4f}', file = rec_log_file)
        print(f'HR correction error: {np.mean(HR_corr_err):.4f} + {np.std(HR_corr_err):.4f}', file = rec_log_file)
        print(f'HRV correction error: {np.mean(HRV_corr_err):.4f} + {np.std(HRV_corr_err):.4f}', file = rec_log_file)

    rec_log_file.close()
