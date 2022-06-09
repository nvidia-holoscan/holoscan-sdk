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

list(APPEND ENDOSCOPY_APP_COMMON_EXTENSIONS
  GXF::std
  GXF::cuda
  GXF::multimedia
  GXF::serialization
  custom_lstm_inference
  format_converter
  visualizer_tool_tracking
)

create_gxe_application(
  NAME tracking_aja
  YAML tracking_aja.yaml
  EXTENSIONS
    ${ENDOSCOPY_APP_COMMON_EXTENSIONS}
    aja_source
)

create_gxe_application(
  NAME tracking_mock
  YAML tracking_mock.yaml
  EXTENSIONS
    ${ENDOSCOPY_APP_COMMON_EXTENSIONS}
    mocks
)

create_gxe_application(
  NAME tracking_replayer
  YAML tracking_replayer.yaml
  EXTENSIONS
    ${ENDOSCOPY_APP_COMMON_EXTENSIONS}
    stream_playback
)
