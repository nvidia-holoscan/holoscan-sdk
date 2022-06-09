# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create library
add_library(tensor_probe_lib SHARED
  tensor_probe.cpp
  tensor_probe.hpp
)
target_link_libraries(tensor_probe_lib
  PUBLIC
    GXF::std
    yaml-cpp
)

# Create extension
add_library(tensor_probe SHARED
  tensor_probe_ext.cpp
)
target_link_libraries(tensor_probe
  PUBLIC tensor_probe_lib
)

