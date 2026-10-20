# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

option(HOLOSCAN_ENABLE_COVERAGE "Instrument C/C++ with gcov (Linux/GCC, single-config)" OFF)
if(HOLOSCAN_ENABLE_COVERAGE)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR
       NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "HOLOSCAN_ENABLE_COVERAGE requires Linux and GNU C/C++ compilers")
    endif()
    get_property(_holoscan_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_holoscan_multi_config)
        message(FATAL_ERROR "HOLOSCAN_ENABLE_COVERAGE requires a single-config generator")
    endif()

    # Apply to this build and fetched sources, not installed consumers or CUDA compilation.
    # Directory options follow configuration flags: -Og overrides Release's optimization
    # without replacing its NDEBUG or other definitions. Target/source options may follow.
    # Atomic updates avoid torn counters when the runtime and HoloMQ execute concurrently.
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:-g;-Og;--coverage;-fprofile-abs-path;-fprofile-update=atomic>"
    )
    function(_holoscan_link_coverage_runtime directory)
        # Includes fetched source targets, but not imported providers or aliases.
        get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(type "${target}" TYPE)
            if(type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY|STATIC_LIBRARY|OBJECT_LIBRARY)$")
                # Equivalent to PRIVATE linkage, without mixing keyword/plain
                # target_link_libraries signatures in third-party projects.
                set_property(TARGET "${target}" APPEND PROPERTY LINK_LIBRARIES gcov)
                if(type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
                    # Archives/objects cannot resolve the runtime until their consumer links.
                    set_property(TARGET "${target}" APPEND PROPERTY
                        INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:gcov>")
                endif()
            endif()
        endforeach()
        get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(subdirectory IN LISTS subdirectories)
            _holoscan_link_coverage_runtime("${subdirectory}")
        endforeach()
    endfunction()

    # Apply once SDK and fetched targets exist; shared libraries keep gcov private.
    cmake_language(DEFER CALL _holoscan_link_coverage_runtime "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
