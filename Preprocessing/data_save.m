% Copyright (C) 2024 ETH Zurich. All rights reserved.   
% Author: Carlos Santos, ETH Zurich           

% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
% You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.   
% SPDX-License-Identifier: Apache-2.0


function data_save(save_dir, slice_name, f1_window, f6_window, LeadI_chest, LeadI_chest_ones, lear_electrode, rear_electrode, fs, down_fs)
    
    %% function: data_save
        % saves to .mat files the biopotential + ecg traces already processed
        % Inputs:
            % save_dir: directory where files are saved
            % slice_name: data window name
            % f1_window: filtered (notch + bandpass f1) biopotential window signal
            % f6_window: filtered (notch + bandpass f6) biopotential window signal
            % chest_LeadI
            % chest_LeadI_ones
            % lear_electrode
            % rear_electrode
            % fs
            % down_fs

    %% Normalize and downsample      
    f1_window = normalize(f1_window);
    f1_window_down = downsample(f1_window, fs, down_fs); % downsample to 250 Hz - 500 samples

    f6_window = normalize(f6_window);
    f6_window_down = downsample(f6_window, fs, down_fs); % downsample to 250 Hz - 500 samples

    %% Declare features
    % In-ear
    % L
    L_inear_f1 = f1_window_down(:, lear_electrode)';
    L_inear_f6 = f6_window_down(:, lear_electrode)';
    % R
    R_inear_f1 = f1_window_down(:, rear_electrode)';
    R_inear_f6 = f6_window_down(:, rear_electrode)';
    

    %% Save
    save(fullfile(save_dir, slice_name), 'LeadI_chest', 'LeadI_chest_ones', 'L_inear_f1', 'L_inear_f6', 'R_inear_f1', 'R_inear_f6');

end