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

# Set RPATH to $ORIGIN so that an extension library such as
# './gxf_extensions/custom_lstm_inference/libcustom_lstm_inference.so'
# can load 'libcustom_lstm_inference_lib.so' from the same directory.
set(CMAKE_INSTALL_RPATH \$ORIGIN)

add_subdirectory(aja)
add_subdirectory(custom_lstm_inference)
add_subdirectory(format_converter)

add_subdirectory(opengl)
add_subdirectory(probe)

add_subdirectory(segmentation_postprocessor)
add_subdirectory(segmentation_visualizer)
add_subdirectory(stream_playback)

add_subdirectory(tensor_rt)
add_subdirectory(v4l2)

add_subdirectory(visualizer_tool_tracking)

add_subdirectory(holoviz_viewer)
add_subdirectory(bayer_demosaic)

if(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP)
    add_subdirectory(emergent)
endif()

add_subdirectory(mocks)

add_subdirectory(sample)
