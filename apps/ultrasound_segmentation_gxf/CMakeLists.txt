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

list(APPEND US_APP_COMMON_EXTENSIONS
  GXF::std
  GXF::cuda
  GXF::multimedia
  GXF::serialization
  segmentation_postprocessor
  segmentation_visualizer
  format_converter
  tensor_rt
)

create_gxe_application(
  NAME segmentation_aja
  YAML segmentation_aja.yaml
  EXTENSIONS
    ${US_APP_COMMON_EXTENSIONS}
    aja_source
)

create_gxe_application(
  NAME segmentation_mock
  YAML segmentation_mock.yaml
  EXTENSIONS
    ${US_APP_COMMON_EXTENSIONS}
    mocks
)

create_gxe_application(
  NAME segmentation_replayer
  YAML segmentation_replayer.yaml
  EXTENSIONS
    ${US_APP_COMMON_EXTENSIONS}
    stream_playback
)
# Create a CMake target for the segmentation replayer
add_custom_target(segmentation_replayer ALL)

# Download the associated dataset if needed
if(HOLOSCAN_DOWNLOAD_DATASETS)
  add_dependencies(segmentation_replayer ultrasound_data)
endif()