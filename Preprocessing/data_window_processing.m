% Copyright (C) 2024 ETH Zurich. All rights reserved.   
% Author: Carlos Santos, ETH Zurich           

% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
% You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.   
% SPDX-License-Identifier: Apache-2.0


%% clear

clear;
clc;

%% ============ Processing start ============

% Folder and subject ID
data_folder = "Data";

% TODO: Adapt subject_id, bin_file_name and recording_id to the recording you want to process
subject_id = "S1";
bin_file_name = 'Data_20241206_174157.bin';
recording_id = "R1";

% Save location
save_dir = fullfile("..", data_folder, "Processed", subject_id, recording_id);
if ~exist(save_dir, 'dir') 
    mkdir(save_dir);
end

% process .bin file
path_to_bin = fullfile("..", data_folder, "Raw", subject_id, bin_file_name);
ExGData = convert_data(path_to_bin, 'uV', 's');
data = ExGData.Data;
recording_state = ExGData.Trigger;

% Sampling and downsampling frequencies
fs = 500; % Hz
down_fs = 250; % Hz

% Electrodes
LARM_ELECTRODE = 8;
RARM_ELECTRODE = 7;
LEAR_ELECTRODE = 2;
REAR_ELECTRODE = 1;


% ============ Define filters ============

% I) Notch filter
N_notch = 6;
f0 = 50;
q = 50;

notch = fdesign.notch('N,F0,Q', N_notch, f0, q, fs);
notch_filter = design(notch, 'butter');
%filterAnalyzer(notch_filter)

% II) [0.5 - 30] Hz bandpass
N_band_1 = 2;   % Order
Fc1 = 0.5;  % First Cutoff Frequency
Fc2_1 = 30;   % Second Cutoff Frequency

bandpass_1  = fdesign.bandpass('N,F3dB1,F3dB2', N_band_1, Fc1, Fc2_1, fs);
bandpass_filter_1 = design(bandpass_1, 'butter');
%filterAnalyzer(bandpass_filter_chest)

% % III) [0.5 - 45] Hz bandpass
% N_band_5 = 2;   % Order
% Fc1 = 0.5;  % First Cutoff Frequency
% Fc2_5 = 45;   % Second Cutoff Frequency
% 
% % Construct an FDESIGN object and call its BUTTER method.
% bandpass_5  = fdesign.bandpass('N,F3dB1,F3dB2', N_band_5, Fc1, Fc2_5, fs);
% bandpass_filter_5 = design(bandpass_5, 'butter');
% %filterAnalyzer(bandpass_filter_chest)

% ============ Arm ECG filtering ============

transient_response = 2000; % avoid filter transient response
avoid_seconds = 5; % seconds to erase on each side of the signal

recording = data(avoid_seconds*fs:end - avoid_seconds*fs, :); % Erase n seconds on both sides of each recording

% Step 0: Divide into arm channels
arm_recording = recording(:, min(RARM_ELECTRODE, LARM_ELECTRODE):max(RARM_ELECTRODE, LARM_ELECTRODE)); % Ch7: RA electrode, Ch8: LA electrode

% Step I: Notch for 50Hz noise
arm_recording_notch = filtfilt(notch_filter.sosMatrix, notch_filter.ScaleValues, arm_recording);

% Step II: Bandpass filter [0.5 - 30] Hz
arm_recording_bandpass = filtfilt(bandpass_filter_1.sosMatrix, bandpass_filter_1.ScaleValues, arm_recording_notch); % filter

% Chest Lead I = LA-RA
if (LARM_ELECTRODE > RARM_ELECTRODE)
    chest_LeadI = arm_recording_bandpass(:, 2) - arm_recording_bandpass(:, 1); % Chest Lead I = Larm(Ch8) - Rarm(Ch7)
else
    chest_LeadI = arm_recording_bandpass(:, 1) - arm_recording_bandpass(:, 2); % Chest Lead I = Larm(Ch7) - Rarm(Ch8)
end

chest_LeadI = chest_LeadI(transient_response + 1:end); % Avoid the filters transient response
chest_LeadI = normalize(chest_LeadI); % Normalize
chest_LeadI = chest_LeadI(1:2:end); % Downsample to 250 Hz

% ============ R peaks ============
chest_LeadI_der = -diff(chest_LeadI); % Look at the -differential of the Lead I

[~, chest_peaks] = findpeaks(chest_LeadI_der, "MinPeakHeight", 0.7, "MinPeakDistance", 80); % Best parameters found

% Step IV: Get trail of 1s in accordance with peaks

chest_LeadI_ones = zeros(size(chest_LeadI)); % Define vector of 0s

for j = 1:length(chest_peaks)
    chest_peaks(j) = get_max(chest_LeadI(chest_peaks(j) - 5:chest_peaks(j)+5), chest_peaks(j));
    chest_LeadI_ones(chest_peaks(j)-1:chest_peaks(j)+1) = ones(1, 3); % Assign [1, 1, 1] to peak location
end

% Check - Plot Lead signal
k = figure();
plot(chest_LeadI, 'b'); 
hold on;
xline(chest_peaks, 'r', 'LineWidth', 1.5); % Peak locations
uiwait(k)

% TODO: Manually correct any mismatching peaks

%% ============ Reinitialize filters ============

% I) Notch filter
N_notch = 6;
f0 = 50;
q = 50;

notch = fdesign.notch('N,F0,Q', N_notch, f0, q, fs);
notch_filter = design(notch, 'butter');
%filterAnalyzer(notch_filter)

% II) [0.5 - 30] Hz bandpass
N_band_1 = 2;   % Order
Fc1 = 0.5;  % First Cutoff Frequency
Fc2_1 = 30;   % Second Cutoff Frequency

bandpass_1  = fdesign.bandpass('N,F3dB1,F3dB2', N_band_1, Fc1, Fc2_1, fs);
bandpass_filter_1 = design(bandpass_1, 'butter');
%filterAnalyzer(bandpass_filter_chest)

% III) [0.5 - 45] Hz bandpass
N_band_5 = 2;   % Order
Fc1 = 0.5;  % First Cutoff Frequency
Fc2_5 = 45;   % Second Cutoff Frequency

% Construct an FDESIGN object and call its BUTTER method.
bandpass_5  = fdesign.bandpass('N,F3dB1,F3dB2', N_band_5, Fc1, Fc2_5, fs);
bandpass_filter_5 = design(bandpass_5, 'butter');
%filterAnalyzer(bandpass_filter_chest)

% ============ Whole biopotential filtering ============

% Step I: Notch for 50Hz noise
recording_notched = filter(notch_filter, recording);
recording_notched_zerophase = filtfilt(notch_filter.sosMatrix, notch_filter.ScaleValues, recording);

% Step II: Bandpass filter [0.5 - 30] Hz
recording_f1 = filter(bandpass_filter_1, recording_notched); % filter
recording_f1 = recording_f1(transient_response + 1:end, :); % Avoid the filters transient response
% recording_bandpass_zerophase = filtfilt(bandpass_filter_1.sosMatrix, bandpass_filter_1.ScaleValues, recording_notched_zerophase); % filter
% recording_bandpass_zerophase = recording_bandpass_zerophase(transient_response + 1:end, :); % Avoid the filters transient response

% Step III: Bandpass filter [0.5 - 45] Hz
recording_f6 = filter(bandpass_filter_1, recording_notched); % filter
recording_f6 = recording_f6(transient_response + 1:end, :); % Avoid the filters transient response

% Filter check
% figure()
% plot(recording_f1(1:5000, 2), 'r');
% hold on;
% plot(recording_bandpass_zerophase(1:5000, 2), 'b');


% ============ Ear biopotential filtering ============

window_size = 1000; % Sliding window size, 2 seconds (2*500)
overlap = 900; % Sliding overlap (fs = 500; overlap = 1.8s)

slice_count = 0;
slice_name = recording_id + "_" + num2str(slice_count) + ".mat";
start_index_ecg = 1;
start_index_ear = 1;

% Both signals have been filtered and the same transient offset has been applied
% Take corresponding window frames, normalize and save
while start_index_ear + window_size - 1 < length(recording_f1)
    
    % Update slice counter and name
    slice_count = slice_count + 1;
    slice_name = recording_id + "_" + num2str(slice_count) + ".mat";
    
    % normalize and downsample
    LeadI_chest = chest_LeadI(start_index_ecg:start_index_ecg + int32(window_size/2) - 1); % 1:500, 51:550
    LeadI_chest_ones = chest_LeadI_ones(start_index_ecg:start_index_ecg + int32(window_size/2) - 1);
    % f1 and f6 filtered - transient response already taken into account, not normalized or downsampled
    f1_window = recording_f1(start_index_ear:start_index_ear + window_size - 1, :); % 1:1000, 101: 1100
    f6_window = recording_f6(start_index_ear:start_index_ear + window_size - 1, :);
    
    % Save window
    data_save(save_dir, slice_name, f1_window, f6_window, LeadI_chest, LeadI_chest_ones, LEAR_ELECTRODE, REAR_ELECTRODE, fs, down_fs);
    
    % Update indexes
    start_index_ecg = start_index_ecg + int32((window_size - overlap)/2); % 1, 51, 101
    start_index_ear = start_index_ear + window_size - overlap; % 1, 101, 201
    
end

% ============ Save gt ecg trace and peaks ============
save_name = 'gt';
save(fullfile(save_dir, save_name), 'chest_LeadI', 'chest_peaks');

%% filter vs filtfilt Check

% avoid_seconds = 5;
% recording = data(avoid_seconds*fs:end - avoid_seconds*fs, :); % Erase n seconds on both sides of each recording
% 
% % Step 0: Divide into channels
% arm_recording = recording(:, 5:6); % Ch5: RA electrode, Ch6: LA electrode
% 
% arm_recording_window = arm_recording(1:1500, :);
% 
% window_notch_zerophase = filtfilt(notch_filter.sosMatrix, notch_filter.ScaleValues, arm_recording_window);
% window_bandpass_zerophase = filtfilt(bandpass_filter_1.sosMatrix, bandpass_filter_1.ScaleValues, window_notch_zerophase); % filter
% chest_LeadI_zerophase = window_bandpass_zerophase(:, 2) - window_bandpass_zerophase(:, 1); % Chest Lead I = Larm(Ch6) - Rarm(Ch5)
% chest_LeadI_zerophase = chest_LeadI_zerophase(transient_response + 1:end); % Avoid the filters transient response
% chest_LeadI_zerophase = normalize(chest_LeadI_zerophase); % Normalize
% chest_LeadI_zerophase = chest_LeadI_zerophase(1:2:end); % Downsample to 250 Hz
% 
% window_notch_phase = filter(notch_filter, arm_recording_window);
% window_bandpass_phase = filter(bandpass_filter_1, window_notch_phase); % filter
% chest_LeadI_phase = window_bandpass_phase(:, 2) - window_bandpass_phase(:, 1); % Chest Lead I = Larm(Ch6) - Rarm(Ch5)
% chest_LeadI_phase = chest_LeadI_phase(transient_response + 1:end); % Avoid the filters transient response
% chest_LeadI_phase = normalize(chest_LeadI_phase); % Normalize
% chest_LeadI_phase = chest_LeadI_phase(1:2:end); % Downsample to 250 Hz
% 
% figure()
% plot(chest_LeadI_zerophase, 'r');
% hold on;
% plot(chest_LeadI_phase, 'b');
% 
% %% Old processing vs new processing check
% 
% LeadI_chest_ones == LeadI_chest_ones_old;
% 
% figure()
% plot(LeadI_chest_old, 'r');
% hold on
% plot(LeadI_chest, '--b');


