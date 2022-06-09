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

cmake_minimum_required(VERSION 3.22)
project(RunGXE)

# Add our custom cmake scripts to the module path for
# include() and find_package(GXF)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
include(define_search_dir_for)
include(create_gxe_application)

# Find CUDA
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
define_search_dir_for(CUDA "The directory containing the CUDA install")
find_package(CUDAToolkit 11.4 REQUIRED)

# Find yaml-cpp
define_search_dir_for(yaml-cpp "The directory containing the yaml-cpp cmake config")
find_package(yaml-cpp 0.6.3 CONFIG REQUIRED)

# Find GXF
define_search_dir_for(GXF "The directory containing the GXF artifacts")
find_package(GXF 2.4 MODULE REQUIRED COMPONENTS core std gxe)

# Create a dummy extension
add_subdirectory(dummy_extension)

# Create minimal example
create_gxe_application(
  NAME RunGXE
  YAML rungxe.yaml
  EXTENSIONS
    GXF::std
    dummy
)
