# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install useful scripts for developers
install(
  FILES
    convert_gxf_entities_to_images.py
    convert_gxf_entities_to_video.py
    convert_video_to_gxf_entities.py
    download_ngc_data
    generate_extension_uuids.py
    get_cmake_cuda_archs.py
    graph_surgeon.py
    gxf_entity_codec.py
    video_validation.py
    ctest_time_comparison.py
    test_pattern_validation.py
    scheduler_overhead_benchmark.py
  DESTINATION "bin"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  COMPONENT "holoscan-core"
)

install(
  FILES README.md
  DESTINATION "bin"
  COMPONENT "holoscan-core"
)
