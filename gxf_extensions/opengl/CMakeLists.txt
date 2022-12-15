# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create library
add_library(opengl_renderer_lib SHARED
  opengl_renderer.cpp
  opengl_renderer.hpp
)
target_link_libraries(opengl_renderer_lib
  PUBLIC
    CUDA::cudart
    glad::glad
    glfw
    GXF::multimedia
    GXF::std
    # OpenGL::GL # Using glad + patch instead for GL/gl.h
    yaml-cpp
)

# Create extension
add_library(opengl_renderer SHARED
  opengl_ext.cpp
)
target_link_libraries(opengl_renderer
  PUBLIC opengl_renderer_lib
)
# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(opengl_renderer)
