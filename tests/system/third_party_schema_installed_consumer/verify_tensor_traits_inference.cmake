# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_TEST_BUILD_DIR OR HOLOSCAN_TEST_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_BUILD_DIR is required")
endif()
if(NOT DEFINED INFERRED_DESCRIPTOR_TRAITS_HEADER OR
   INFERRED_DESCRIPTOR_TRAITS_HEADER STREQUAL "")
    message(FATAL_ERROR "INFERRED_DESCRIPTOR_TRAITS_HEADER is required")
endif()

# A direct Tensor field selects descriptor traits without a redundant caller-authored mode flag.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${HOLOSCAN_TEST_BUILD_DIR}"
        --target third_party_tensor_schema_inferred_descriptor_traits
    RESULT_VARIABLE descriptor_root_build_result
)
if(NOT descriptor_root_build_result EQUAL 0)
    message(FATAL_ERROR "Descriptor traits were not inferred for a direct Tensor root")
endif()
file(READ "${INFERRED_DESCRIPTOR_TRAITS_HEADER}" inferred_descriptor_traits)
if(NOT inferred_descriptor_traits MATCHES
   "static constexpr bool descriptor_tensor = true;")
    message(FATAL_ERROR "Generated direct Tensor root did not select descriptor traits")
endif()

# An unrelated Tensor type in the BFBS inventory does not make the selected root descriptor-backed.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${HOLOSCAN_TEST_BUILD_DIR}"
        --target third_party_selected_tensor_schema_ordinary_traits
    RESULT_VARIABLE selected_root_build_result
)
if(NOT selected_root_build_result EQUAL 0)
    message(FATAL_ERROR "Ordinary traits rejected a root that does not reach Tensor")
endif()
