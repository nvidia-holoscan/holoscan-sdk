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
project(clara-holoscan-sdk LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Add our custom cmake scripts to the module path for
# include() and find_package(GXF)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(define_search_dir_for)
include(create_gxe_application)

# Find AJA ntv2
define_search_dir_for(ajantv2 "The directory containing the AJA NTV2 SDK cmake config")
find_package(ajantv2 16.2 REQUIRED)

# Find CUDA: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit 11.6 REQUIRED)

# Find glad
define_search_dir_for(glad "The directory containing the glad cmake config")
find_package(glad 0.1 CONFIG REQUIRED)
# Add Gl/gl.h interface
target_include_directories(glad::glad
  INTERFACE
    cmake/patches/glad/include # GL/gl.h
)

# Find glfw3
define_search_dir_for(glfw3 "The directory containing the glfw3 cmake config")
find_package(glfw3 3.2 CONFIG REQUIRED)

# Find GXF
define_search_dir_for(GXF "The directory containing the GXF artifacts")
find_package(GXF 2.4.1 MODULE REQUIRED
  COMPONENTS
    core
    cuda
    gxe
    multimedia
    serialization
    std
)

# Find nanovg
define_search_dir_for(nanovg "The directory containing the nanovg cmake config")
find_package(nanovg CONFIG REQUIRED)

# Find OpenGL: https://cmake.org/cmake/help/latest/module/FindOpenGL.html
# find_package(OpenGL MODULE REQUIRED) # Using glad + patch instead for GL/gl.h

# Find TensorRT
define_search_dir_for(TensorRT "The directory containing the tensort install path")
find_package(TensorRT 8.2 MODULE REQUIRED)

# Find yaml-cpp
define_search_dir_for(yaml-cpp "The directory containing the yaml-cpp cmake config")
find_package(yaml-cpp 0.6.3 CONFIG REQUIRED)

# Targets
add_subdirectory(./gxf_extensions)
add_subdirectory(./test/extensions)
add_subdirectory(./apps)
