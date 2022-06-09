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
add_library(visualizer_tool_tracking_lib SHARED
  frame_data.hpp
  instrument_label.cpp
  instrument_label.hpp
  instrument_tip.cpp
  instrument_tip.hpp
  opengl_utils.cpp
  opengl_utils.hpp
  overlay_img_vis.cpp
  overlay_img_vis.hpp
  video_frame.cpp
  video_frame.hpp
  visualizer.cpp
  visualizer.hpp
)
target_link_libraries(visualizer_tool_tracking_lib
  PUBLIC
    CUDA::cudart
    glad::glad
    glfw
    GXF::std
    # OpenGL::GL # Using glad + patch instead for GL/gl.h
    nanovg
    yaml-cpp
)

# Create extension
add_library(visualizer_tool_tracking SHARED
  visualizer_ext.cpp
)
target_link_libraries(visualizer_tool_tracking
  PUBLIC visualizer_tool_tracking_lib
)

# Copy resources
file(COPY fonts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
