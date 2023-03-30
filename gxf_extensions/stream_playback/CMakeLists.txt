# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_library(gxf_stream_playback_lib SHARED
  video_stream_serializer.cpp
  video_stream_serializer.hpp
)
target_link_libraries(gxf_stream_playback_lib
  PUBLIC
    GXF::serialization
    yaml-cpp
)

# Create extension
add_library(gxf_stream_playback SHARED
  stream_playback_ext.cpp
)
target_link_libraries(gxf_stream_playback
  PUBLIC
    gxf_stream_playback_lib
)
# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(gxf_stream_playback)
