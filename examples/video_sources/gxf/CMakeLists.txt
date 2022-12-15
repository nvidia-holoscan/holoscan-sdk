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

create_gxe_application(
  NAME aja_capture
  YAML aja_capture.yaml
  EXTENSIONS
    GXF::std
    GXF::cuda
    GXF::multimedia
    aja_source
    holoviz
)

create_gxe_application(
  NAME v4l2_camera
  YAML v4l2_camera.yaml
  EXTENSIONS
    GXF::std
    GXF::multimedia
    opengl_renderer
    v4l2_source
)

create_gxe_application(
  NAME video_replayer
  YAML video_replayer.yaml
  EXTENSIONS
    GXF::std
    GXF::cuda
    GXF::multimedia
    GXF::serialization
    format_converter
    stream_playback
    holoviz
)

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-apps
)