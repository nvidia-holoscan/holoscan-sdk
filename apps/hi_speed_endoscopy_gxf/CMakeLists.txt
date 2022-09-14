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

list(APPEND APP_COMMON_EXTENSIONS
  GXF::std
  GXF::multimedia
  GXF::serialization
)

create_gxe_application(
  NAME hi_speed_endoscopy
  YAML hi_speed_endoscopy.yaml
  EXTENSIONS
    ${APP_COMMON_EXTENSIONS}
    GXF::cuda
    holoviz_viewer
    bayer_demosaic
    stream_playback
    format_converter
    emergent_source
)
