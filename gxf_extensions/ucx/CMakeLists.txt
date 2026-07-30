# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create library
add_library(gxf_ucx_holoscan_lib SHARED
  ucx_holoscan_component_serializer.cpp
)
target_link_libraries(gxf_ucx_holoscan_lib
  PUBLIC
    GXF::std
    GXF::multimedia
    GXF::serialization
    holoscan::core  # needed for ucx_holoscan_component_serializer.cpp
  PRIVATE
    ucx::ucp
    GXF::ucx
)

# Create extension
add_library(gxf_ucx_holoscan SHARED
  ucx_holoscan_ext.cpp
)
target_compile_definitions(gxf_ucx_holoscan
  PRIVATE HOLOSCAN_BUILD_VERSION=${HOLOSCAN_BUILD_VERSION}
)
target_link_libraries(gxf_ucx_holoscan
  PUBLIC  gxf_ucx_holoscan_lib
  PRIVATE holoscan_security_flags
)

# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(gxf_ucx_holoscan)
