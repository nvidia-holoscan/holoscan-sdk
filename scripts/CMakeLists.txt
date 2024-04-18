# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install useful scripts for developers
install(
  FILES
    convert_gxf_entities_to_images.py
    convert_gxf_entities_to_video.py
    convert_video_to_gxf_entities.py
    download_ngc_data
    generate_extension_uuids.py
    graph_surgeon.py
    gxf_entity_codec.py
    video_validation.py
  DESTINATION "bin"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  COMPONENT "holoscan-core"
)

install(
  FILES README.md
  DESTINATION "bin"
  COMPONENT "holoscan-core"
)