% Copyright (C) 2024 ETH Zurich. All rights reserved.   
% Author: Carlos Santos, ETH Zurich           

% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
% You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.   
% SPDX-License-Identifier: Apache-2.0


function downsampled_array = downsample(original_array, fs, new_fs)
% Keep every second value in the input signal, assuming we are downsampling
% by 2
    % Args:
    %   signal: 1D array, the input signal
    %
    % Returns:
    %   new_signal: 1D array, the signal with every second value
    jump = fs/new_fs;
    downsampled_array = original_array(1:jump:end, :);

end