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
add_library(dummy_lib SHARED
  dummy.cpp
  dummy.hpp
)
target_link_libraries(dummy_lib
  PUBLIC
    CUDA::cudart
    GXF::std
    yaml-cpp
)

# Create extension
add_library(dummy SHARED
  dummy_ext.cpp
)
target_link_libraries(dummy
  PUBLIC dummy_lib
)
