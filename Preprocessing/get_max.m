% Copyright (C) 2024 ETH Zurich. All rights reserved.   
% Author: Carlos Santos, ETH Zurich           

% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
% You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and limitations under the License.   
% SPDX-License-Identifier: Apache-2.0


function [real_max] = get_max(signal, aprox_max)

    max_value = max(signal);
    max_location = find(signal == max_value);
    diff = max_location - 6;
    real_max = aprox_max + diff;

end


