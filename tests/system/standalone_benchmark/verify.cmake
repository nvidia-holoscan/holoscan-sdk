# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT PROJECT_NAME STREQUAL "holoscan_benchmarks" OR
   NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    return()
endif()

function(holoscan_verify_standalone_benchmark_targets)
    if(DEFINED holoscan_SOURCE_DIR)
        message(FATAL_ERROR
            "find_package(holoscan) unexpectedly defined holoscan_SOURCE_DIR="
            "'${holoscan_SOURCE_DIR}'"
        )
    endif()
    if(NOT TARGET holoscan::core)
        message(FATAL_ERROR "Standalone benchmark configuration did not import holoscan::core")
    endif()
    get_target_property(holoscan_standalone_core_imported holoscan::core IMPORTED)
    if(NOT holoscan_standalone_core_imported)
        message(FATAL_ERROR "Standalone benchmark holoscan::core target is not imported")
    endif()
    if(TARGET core)
        message(FATAL_ERROR "Standalone benchmark configuration unexpectedly defines target core")
    endif()

    foreach(holoscan_standalone_public_target IN ITEMS
            channel_interprocess_benchmark
            channel_pod_benchmark
            channel_schema_benchmark
            channel_tensor_benchmark
            core_hello_benchmark
            holomq_pubsub_benchmarks
            holomq_transport_benchmark)
        if(NOT TARGET "${holoscan_standalone_public_target}")
            message(FATAL_ERROR
                "Standalone benchmark target '${holoscan_standalone_public_target}' was not configured"
            )
        endif()
    endforeach()

    foreach(holoscan_standalone_private_target IN ITEMS
            temporal_benchmark
            endpoint_catalog_benchmark
            holomq_cuda_sync_benchmarks
            runtime_dispatcher_benchmark
            temporal_contract_benchmark
            time_benchmark)
        if(TARGET "${holoscan_standalone_private_target}")
            message(FATAL_ERROR
                "Source-private benchmark target '${holoscan_standalone_private_target}' "
                "was configured from the installed package"
            )
        endif()
    endforeach()
endfunction()

cmake_language(DEFER
    DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL holoscan_verify_standalone_benchmark_targets
)
