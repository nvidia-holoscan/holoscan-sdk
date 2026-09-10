# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(vision_pipeline vision_pipeline.cpp)
holoscan_configure_cxx20_target(vision_pipeline PRIVATE)
target_link_libraries(vision_pipeline PRIVATE holoscan::vision CUDA::cudart)

if(BUILD_TESTING)
    add_test(NAME VISION_PIPELINE_EXAMPLE COMMAND vision_pipeline)
    set_tests_properties(VISION_PIPELINE_EXAMPLE
        PROPERTIES
            LABELS "vision;examples"
            PASS_REGULAR_EXPRESSION "vision_pipeline: image and detection branches passed"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )
endif()
