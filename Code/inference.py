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
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
import onnxruntime as ort

# Import custom modules
from utils import print_statistics, print_cardiac_metrics
from data_loader import DeepMFDataset, Invert
from utils import skip_files

#############################################################
# Functions that contain the logic for rolling output inference and metric computation
#############################################################

def validate_classification(device, model, file_set, feature_names, args, save_path, cardiac_metrics = False, threshold = 0.5):
    
    """ 
    Validate rolling performance on set of recordings. 
    Output performance metrics (precision, recall, DER, HR error, HRV error, HR correction error, HRV correction error).
    
    Inputs:
            device: torch.device 
            model: nn.Module - PyTorch model
            file_set: list of tuples (files: list of .mat files, peaks: np array)
            feature_names: list of str - features onto which to train the model (must be a valid key contained in the recording .mat files)
            args - argparse.Namespace - arguments as specified in parser_file.py
            save_path: str - directory to save inference results

        Outputs:
            prec: float - precision
            rec: float - recall
            der: float - DER
            HR_err: float - HR error
            HRV_err: float - HRV error
            HR_corr_err: float - HR correction error
            HRV_corr_err: float - HRV correction error"""

    save_rec = None
    n_files = len(file_set) # 1 (LOO_rec) / 4 (LOO_patient)

    transform_in_ear = Invert()
    
    prec, rec, der = [None] * n_files, [None] * n_files, [None] * n_files
    if cardiac_metrics:
        HR_err, HRV_err, HR_corr_err, HRV_corr_err = [None] * n_files, [None] * n_files, [None] * n_files, [None] * n_files

    # Loop over all recordings in the set
    for i in range(n_files):
        file_list, gt_peaks = file_set[i]
        files = skip_files(file_list, args.jump) # Jump files for inference
        dataset = DeepMFDataset(files, feature_names, task = 'classify', transform_in_ear = transform_in_ear, transform_ecg = None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        if save_path:
            if n_files > 1:
                save_rec = os.path.join(save_path, f'{i+1}')
                # save_rec = save_path + f'R_{i+1}/'
            else:
                save_rec = save_path

            if not os.path.exists(save_rec):
                os.makedirs(save_rec)

        if cardiac_metrics:
            prec[i], rec[i], der[i], HR_err[i], HRV_err[i], HR_corr_err[i], HRV_corr_err[i] = inference(device, model, loader, gt_peaks, args, save_rec, cardiac_metrics, threshold)
        else:
            prec[i], rec[i], der[i] = inference(device, model, loader, gt_peaks, args, save_rec, cardiac_metrics, threshold)

    
    if cardiac_metrics:
        return prec, rec, der, HR_err, HRV_err, HR_corr_err, HRV_corr_err
    else:
        return prec, rec, der
    

def inference(device, model, loader, gt_peaks, args, save_path, cardiac_metrics = False, threshold = 0.5, quantize = False):
    
    """ 
    Perform rolling window inference on the loader. The loader corresponds to the already skipped files of a recording. 
    Output predicted and gt peaks list and rolling output array.
    
    Inputs:
            device: torch.device
            model: nn.Module - PyTorch model
            loader: nn.DataLoader - iterates over ordered sequence of .mat file representing one recording
            gt_peaks: np.array - GT array containing the peak locations, saved during preprocessing
            args: argparse.Namespace - arguments as specified in parser_file.py
            save_path: str - path to save inference results
            cardiac_metrics: bool - flag to compute HR and HRV metrics
            threshold: float - threshold for R-Peak detection
            quantize: bool

        Outputs:
            precision: float - precision 
            recall: float - recall
            der: float - DER
            HR_mean: float - HR error
            HRV_median: float - HRV error
            HR_corr_mean: float - HR correction error
            HRV_corr_median: float - HRV correction error""" 

    # Arguments
    fs = args.fs
    jump = int(args.jump*args.fs)
    window_size = int(args.window_size*args.fs)
    not_trust = int(args.not_trust*args.fs)
    method = args.method
    mask = args.mask
    averaging_time = int(args.av_time*args.fs)
    tolerance = args.tolerance
    
    if cardiac_metrics:
        # Time, HR and HRV arrays
        time = np.array([])

        HRV_preds = np.array([])
        HRV_corrs = np.array([])
        HRV_trues = np.array([])

        HR_preds = np.array([])
        HR_corrs = np.array([])
        HR_trues = np.array([])

    start_index = 0
    end_index = window_size - not_trust - jump # 500 - 100 - 100 = 300 samples
    last_height = 0
    dist = 0

    rolling_output = np.zeros(end_index)
    prediction_peaks = np.array([])

    # Validation mode
    if isinstance(model, ort.InferenceSession): # ONNX model
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
    # elif isinstance(model, nntool.graph.nngraph.NNGraph): # G graph (GAP9)
        # pass
    else: # Pytorch model
        model.eval()

    ######################## Inference Start ########################
    with torch.no_grad():

        for i, data in enumerate(loader, 0):
            # inputs
            in_ear, ecg = data
            
            # Model inference
            if isinstance(model, ort.InferenceSession): # ONNX model
                in_ear = in_ear.numpy().astype(np.float32)
                result = model.run([output_name], {input_name: in_ear})
                output = result[0].squeeze()

            else:
                in_ear = in_ear.to(device)
                output = model(in_ear).squeeze().detach().cpu().numpy()
            
            output = output[:-not_trust] # Discard not trusted samples
            
            if method == 'max':
                rolling_output[start_index:end_index] = np.maximum(rolling_output[start_index:end_index], output[:-jump]) # Refinement
                rolling_output = np.concatenate((rolling_output, output[-jump:])) # Extend

                peak_window = rolling_output[start_index:start_index + jump] # Window jump
                window_peaks, updated_prediction_peaks, dist, last_height = get_peaks(peak_window, prediction_peaks, threshold, dist, last_height, mask) # New peaks

            # TODO: implement your own R-Peak detection scheme

            else:
                raise ValueError("Method not recognized.")
            
            window_peaks += start_index
            prediction_peaks = np.append(updated_prediction_peaks, window_peaks, axis = 0)
            
            # HR and HRV computation
            if cardiac_metrics and start_index >= averaging_time:
                
                # Peaks in last averaging time
                last_gt_peaks = gt_peaks[np.where((gt_peaks >= start_index - averaging_time) & (gt_peaks < start_index))]
                last_pred_peaks = prediction_peaks[np.where((prediction_peaks >= start_index - averaging_time) & (prediction_peaks < start_index))]

                # HRV
                HRV_pred, HRV_corr, HRV_gt, n_missed, n_excess = get_HRV(last_pred_peaks, last_gt_peaks, fs) # RMSSD
                # HR
                HR_pred, HR_corr, HR_gt = get_HR(last_pred_peaks, last_gt_peaks, n_missed, n_excess, fs, args.av_time) #n_peaks/10s
                
                # Time stamp
                time = np.append(time, start_index/fs)
                HRV_preds = np.append(HRV_preds, HRV_pred)
                HRV_corrs = np.append(HRV_corrs, HRV_corr)
                HRV_trues = np.append(HRV_trues, HRV_gt)
                HR_preds = np.append(HR_preds, HR_pred)
                HR_corrs = np.append(HR_corrs, HR_corr)
                HR_trues = np.append(HR_trues, HR_gt)

            start_index += jump
            end_index += jump
        
        ######################## Inference End ########################

    # GT
    gt_peaks = gt_peaks[gt_peaks <= start_index + jump] # Ignore last window

    # Predicted peak statistics
    tp, fp, fn, precision, recall, der = get_statistics(prediction_peaks, gt_peaks, tolerance)

    if cardiac_metrics:
        # Compute errors
        HR_err_mean, HR_err_std, HRV_err_mean, HRV_err_std, HR_corr_err_mean, HR_corr_err_std, HRV_corr_err_mean, HRV_corr_err_std = get_errors(HR_preds, HR_corrs, HR_trues, HRV_preds, HRV_corrs, HRV_trues)
    
    # Save results
    if save_path:
        
        # Save rolling output and peaks
        sio.savemat(os.path.join(save_path, 'rolling_output.mat'), {'rolling_output': rolling_output, 'prediction_peaks': prediction_peaks, 'gt_peaks': gt_peaks})
        # sio.savemat(save_path + 'rolling_output.mat', {'rolling_output': rolling_output, 'prediction_peaks': prediction_peaks, 'gt_peaks': gt_peaks})
        print_statistics(save_path, threshold, precision, recall, der, gt_peaks, prediction_peaks, tp, fp, fn)

        if cardiac_metrics:

            # Save HR and HRV metrics
            sio.savemat(os.path.join(save_path, 'HR_metrics.mat'), {'time': time, 'HR_trues': HR_trues, 'HR_preds': HR_preds, 'HR_corrs': HR_corrs, 'HR_err_mean': HR_err_mean, 'HR_err_std': HR_err_std, 'HR_corr_err_mean': HR_corr_err_mean, 'HR_corr_err_std': HR_corr_err_std, 'HRV_trues': HRV_trues, 'HRV_preds': HRV_preds, 'HRV_corrs': HRV_corrs, 'HRV_err_mean': HRV_err_mean, 'HRV_err_std': HRV_err_std, 'HRV_corr_err_mean': HRV_corr_err_mean, 'HRV_corr_err_std': HRV_corr_err_std})
            # sio.savemat(save_path + 'HR_metrics.mat', {'time': time, 'HR_trues': HR_trues, 'HR_preds': HR_preds, 'HR_corrs': HR_corrs, 'HR_err_mean': HR_err_mean, 'HR_err_std': HR_err_std, 'HR_corr_err_mean': HR_corr_err_mean, 'HR_corr_err_std': HR_corr_err_std, 'HRV_trues': HRV_trues, 'HRV_preds': HRV_preds, 'HRV_corrs': HRV_corrs, 'HRV_err_mean': HRV_err_mean, 'HRV_err_std': HRV_err_std, 'HRV_corr_err_mean': HRV_corr_err_mean, 'HRV_corr_err_std': HRV_corr_err_std})
            print_cardiac_metrics(save_path, HR_err_mean, HRV_err_mean, HR_corr_err_mean, HRV_corr_err_mean, HR_err_std, HRV_err_std, HR_corr_err_std, HRV_corr_err_std)
            save_HR_HRV_images(save_path, time, HR_trues, HR_preds, HR_corrs, HRV_trues, HRV_preds, HRV_corrs)

    if cardiac_metrics:
        return precision, recall, der, HR_err_mean, HRV_err_mean, HR_corr_err_mean, HRV_corr_err_mean
    else:
        return precision, recall, der


def get_peaks(window, prediction_peaks, threshold, dist, last_height, mask):

    """ Get peaks found in window """
    """ Inputs:
            window: np.array - waveform
            threshold: float
            dist: int - dist from last peak to window start
            last_height: float - last peak height
            prediction_peaks: np.array - growing predicted peaks
            mask: int - masking distance between consecutive peaks

        Outputs:
            window_peaks: np.array - indexes of the ordered peaks within the window
            updated_prediction_peaks: np.array - corrected growing predicted peaks
            dist: int - dist from last peak to window start
            last_height: float - last peak height"""
    
    peaks = np.array([])
    window_copy = window.copy()
    updated_prediction_peaks = prediction_peaks
    b = np.max(window_copy) > threshold

    while b:
        
        new_height = np.max(window_copy)
        peak = np.argmax(window_copy)

        if peak < int(mask - dist): # peak within the mask from the previous peak
            if new_height > last_height:
                updated_prediction_peaks = prediction_peaks[:-1]
                peaks = np.append(peaks, peak)
                major_index = min(peak+mask, len(window_copy)) # Mask
                window_copy[:major_index] = 0

            else:
                window_copy[:int(mask-dist)] = 0 # Mask

        else: # peak not within mask, add as usual
            updated_prediction_peaks = prediction_peaks
            peaks = np.append(peaks, peak)
            minor_index = max(peak-mask, 0) # Mask
            major_index = min(peak+mask, len(window_copy))
            window_copy[minor_index:major_index] = 0

        b = np.max(window_copy) > threshold

    # Sort peaks and get the distance to window end
    window_peaks = np.sort(peaks)
    if len(window_peaks) == 0:
        dist = len(window)
        last_height = 0
        
    else:
        dist = len(window) - window_peaks[-1]
        last_height = window[int(window_peaks[-1])]

    return window_peaks, updated_prediction_peaks, dist, last_height


def get_HRV(prediction_peaks, gt_peaks, fs):

    """ Get HRV prediction and correction from predicted peaks """
    """ Inputs:
            prediction_peaks: np.array
            gt_peaks: np.array
            fs: int

        Outputs:
            HRV_pred: float
            HRV_corr: float
            HRV_gt: float
            n_missed: int
            n_excess: int"""
    
    # print(prediction_peaks)

    # RR intervals
    RR_gt_intervals = np.diff(gt_peaks)

    if len(prediction_peaks) <= 3:
        HRV_pred = 0
        HRV_corr = 0
        n_missed = len(gt_peaks)
        n_excess = 0

    else:
        RR_pred_intervals = np.diff(prediction_peaks)
        RR_corr_intervals = np.copy(RR_pred_intervals) # Copy for correction

        # Correct abnormal short intervals (extra peaks)
        n_excess = 0
        median = np.median(RR_corr_intervals) # Reference
        expectation = np.round(RR_corr_intervals/median) # Expected values around median

        while np.any(expectation == 0): # extra peak somewhere
            extra_peak = np.argmin(RR_corr_intervals)
            if extra_peak == 0: # extra peak in position 1 -> 1st RR interval outlier
                RR_corr_intervals = np.concatenate(([RR_corr_intervals[0] + RR_corr_intervals[1]], RR_corr_intervals[2:])) # [RR1 + RR2, RR3, RR4, ...]

            elif extra_peak == len(expectation) - 1: # extra peak in position n - 1 -> last RR interval outlier
                RR_corr_intervals = np.concatenate((RR_corr_intervals[:-2], [RR_corr_intervals[-2] + RR_corr_intervals[-1]])) # [..., RRn-2, RRn-1 + RRn]

            else: # any other position
                possibilities = np.array([RR_corr_intervals[extra_peak - 1], RR_corr_intervals[extra_peak + 1]]) # Previous and next intervals
                choice = np.argmin(possibilities)

                if choice == 0:
                    RR_corr_intervals = np.concatenate((RR_corr_intervals[:extra_peak - 1], [RR_corr_intervals[extra_peak - 1] + RR_corr_intervals[extra_peak]], RR_corr_intervals[extra_peak + 1:]))
                else:
                    RR_corr_intervals = np.concatenate((RR_corr_intervals[:extra_peak], [RR_corr_intervals[extra_peak] + RR_corr_intervals[extra_peak + 1]], RR_corr_intervals[extra_peak + 2:]))

            n_excess += 1 # + 1 excess peak

            # Recompute median and expectation
            median = np.median(RR_corr_intervals)
            expectation = np.round(RR_corr_intervals/median)

        # Correct abnormal long intervals (missed peaks)
        n_missed = np.count_nonzero(expectation == 2) + np.count_nonzero(expectation == 3)*2 + np.count_nonzero(expectation == 4)*3 # If expectation = 2, 1 missed peak, etc
        RR_corr_intervals = RR_corr_intervals/expectation # same as introducing synthetic peaks at midpoints
        
        # TODO: other option, recurring correction like before
        # while np.any(expectation > 1): # missed peak somewhere
        #     missed_peak = np.argmax(RR_corr_intervals)

        #     if missed_peak == 0: # missed peak in first interval
        #         RR_corr_intervals = np.concatenate((np.array(RR_corr_intervals[0]/2, RR_corr_intervals[0]/2), RR_corr_intervals[1:])) # [RR1/2, RR1/2, RR2, RR3, ...]

        #     elif missed_peak == len(RR_corr_intervals) - 1: # missed peak in last interval
        #         RR_corr_intervals = np.concatenate((RR_corr_intervals[:-1], np.array([RR_corr_intervals[-1]/2, RR_corr_intervals[-1]/2]))) # [..., RRn-1, RRn/2, RRn/2]

        #     else: # missed peak in any other position
        #         RR_corr_intervals = np.concatenate((RR_corr_intervals[:missed_peak], np.array([RR_corr_intervals[missed_peak]/2, RR_corr_intervals[missed_peak]/2]), RR_corr_intervals[missed_peak + 1:])) # [..., RR/2, RR/2, ...]

        #     n_missed += 1 # + 1 missed peak

        #     # Recompute
        #     median = np.median(RR_corr_intervals)
        #     expectation = np.round(RR_corr_intervals/median)

        # Predictions
        RR_pred_intervals_ms = RR_pred_intervals*(1000/fs) # Predicted RR intervals in ms 
        RR_pred_intervals_ms_dif = np.diff(RR_pred_intervals_ms)
        HRV_pred = np.sqrt(np.mean(RR_pred_intervals_ms_dif**2)) # RMSSD

        # Corrections
        if len(RR_corr_intervals) < 2:
            HRV_pred = 0
            HRV_corr = 0
            n_missed = len(gt_peaks)
            n_excess = 0
        
        else:
            RR_pred_intervals_corr_ms = RR_corr_intervals*(1000/fs) # Corrected RR intervals in ms 
            RR_pred_intervals_corr_ms_dif = np.diff(RR_pred_intervals_corr_ms) 
            HRV_corr = np.sqrt(np.sum(RR_pred_intervals_corr_ms_dif**2)/len(RR_pred_intervals_corr_ms_dif) + n_missed) # RMSSD

    # GT
    RR_gt_intervals_ms = RR_gt_intervals*(1000/fs) # GT RR intervals in ms
    RR_gt_intervals_ms_dif = np.diff(RR_gt_intervals_ms)
    HRV_gt = np.sqrt(np.mean(RR_gt_intervals_ms_dif**2)) # RMSSD

    return HRV_pred, HRV_corr, HRV_gt, n_missed, n_excess


def get_HR(prediction_peaks, gt_peaks, n_missed, n_excess, fs, av_time):

    """ Get HR from predicted peaks """
    """ Inputs:
            last_pred_peaks: np.array
            last_gt_peaks: np.array
            n_missed: int
            n_excess: int
            fs: int

        Outputs:
            HR_pred: float
            HR_corr: float
            HR_gt: float"""
    
    # HR = peaks*60/time (in s)
    # time = (last_peak - first_peak)/fs

    
    HR_gt = len(gt_peaks)*(60/((gt_peaks[-1] - gt_peaks[0])/fs)) # n_peaks/1min

    if len(prediction_peaks) <= 3:
        HR_pred = 0
        HR_corr = 0
    else:
        # Prediction
        HR_pred = len(prediction_peaks)*(60/((prediction_peaks[-1] - prediction_peaks[0])/fs))
        # Correction
        HR_corr = (len(prediction_peaks) + n_missed - n_excess)*(60/((prediction_peaks[-1] - prediction_peaks[0])/fs)) 
    
    return HR_pred, HR_corr, HR_gt


def get_statistics(peak_predictions, gt_peaks, tolerance):

    """ Get statistics for the inference """
    """ Inputs:
            peak_predictions: np.array
            gt_peaks: np.array
            tolerance: int

        Outputs:
            tp: float - true positives
            fp: float - false positives
            fn: float - false negatives 
            precision: float
            recall: float 
            der: float"""
    
    tp = np.array([])
    fn = np.array([])
    total_pred = len(peak_predictions)
    total_peaks = len(gt_peaks)

    while len(peak_predictions) > 0 and len(gt_peaks) > 0:
        
        # Get the closest prediction to the next gt peak
        next_peak = gt_peaks[0]
        diff_array = np.abs(next_peak - peak_predictions)
        closest_pred = np.min(diff_array)
        index = np.argmin(diff_array)
        
        if closest_pred <= tolerance:
            tp = np.append(tp, next_peak)
            peak_predictions = np.delete(peak_predictions, index)
        else:
            fn = np.append(fn, next_peak)
    
        gt_peaks = np.delete(gt_peaks, 0)

    tp = len(tp)
    fn = len(fn)
    fp = total_pred - tp
    
    if tp == 0:
        precision = 0
        recall = 0
        der = 0
    else:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        der = (fp+fn)/total_peaks

    return tp, fp, fn, precision, recall, der


def save_HR_HRV_images(save_path, time, HR_trues, HR_preds, HR_corrs, HRV_trues, HRV_preds, HRV_corrs):

    """ Save HR and HRV predicted and corrected trends """
    """ Inputs:
            save_path: str - save_path
            HRV_preds: np.array
            HRV_corrs: np.array
            HRV_trues: np.array
            HR_preds: np.array
            HR_corrs: np.array
            HR_trues: np.array
            time: np.array
            
        Outputs:
            None"""
    
    # HRV
    plt.figure(dpi = 600)
    plt.plot(time, HRV_trues, linewidth = 2, color = '#2ca02c', label = 'GT')
    plt.plot(time, HRV_preds, linewidth = 2, color = '#1f77b4', linestyle = '-', label = 'Prediction')
    plt.plot(time, HRV_corrs, linewidth = 2, color = '#ff7f0e', linestyle = '--', label = 'Corrected Prediction')

    plt.ylabel('HRV (ms)', fontsize = 15)
    plt.xlabel('Time (s)', fontsize = 15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.savefig(os.path.join(save_path, 'HRV.pdf'), bbox_inches = 'tight')
    plt.close()

    # HR
    plt.figure(dpi = 600)
    plt.plot(time, HR_trues, linewidth = 2, color = '#2ca02c', label = 'GT')
    plt.plot(time, HR_preds, linewidth = 2, color = '#1f77b4', linestyle = '-', label = 'Prediction')
    plt.plot(time, HR_corrs, linewidth = 2, color = '#ff7f0e', linestyle = '--', label = 'Corrected Prediction')
    
    plt.ylabel('HR (bpm)', fontsize = 15)
    plt.xlabel('Time (s)', fontsize = 15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.savefig(os.path.join(save_path, 'HR.pdf'), bbox_inches = 'tight')
    plt.close()


def get_errors(HR_preds, HR_corrs, HR_trues, HRV_preds, HRV_corrs, HRV_trues):

    """ Get HR and HRV absolute errors """
    """ Inputs:
            HRV_preds: np.array
            HRV_corrs: np.array
            HRV_trues: np.array
            HR_preds: np.array
            HR_corrs: np.array
            HR_trues: np.array
            
        Outputs:
            HR_mean: float
            HR_std: float
            HRV_median: float
            HRV_absdev: float
            HR_corr_mean: float
            HR_corr_std: float
            HRV_corr_median: float
            HRV_corr_absdev: float"""
    
    # Naive prediction
    # HR
    err_HR = np.abs(HR_preds - HR_trues)
    HR_err_mean = np.mean(err_HR)
    HR_err_std = np.std(err_HR)
    # HRV
    err_HRV = np.abs(HRV_preds - HRV_trues)
    HRV_err_mean = np.mean(err_HRV)
    HRV_err_std = np.std(err_HRV)

    # Corrected prediction
    # HR
    err_HR_corr = np.abs(HR_corrs - HR_trues)
    HR_corr_err_mean = np.mean(err_HR_corr)
    HR_corr_err_std = np.std(err_HR_corr)
    # HRV
    err_HRV_corr = np.abs(HRV_corrs - HRV_trues)
    HRV_corr_err_mean = np.mean(err_HRV_corr)
    HRV_corr_err_std = np.std(err_HRV_corr)

    return HR_err_mean, HR_err_std, HRV_err_mean, HRV_err_std, HR_corr_err_mean, HR_corr_err_std, HRV_corr_err_mean, HRV_corr_err_std