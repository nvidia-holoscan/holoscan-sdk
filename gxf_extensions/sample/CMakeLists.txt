# Copyright (c) 2022, NVIDIA CORPORATION.
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
add_library(ping_tx SHARED
    ping_tx.hpp
    ping_tx.cpp
)
target_link_libraries(ping_tx
    PUBLIC
    GXF::std
    yaml-cpp
)

# Create library
add_library(ping_rx SHARED
    ping_rx.hpp
    ping_rx.cpp
)
target_link_libraries(ping_rx
    PUBLIC
    GXF::std
    yaml-cpp
)

# Create extension
add_library(holoscan_sample SHARED
    sample.cpp
)
target_link_libraries(holoscan_sample
    PUBLIC
    GXF::std
    ping_tx
    ping_rx
    yaml-cpp
)
# Install GXF extension as a component 'holoscan-embedded-gxf_extensions'
install_gxf_extension(holoscan_sample)
