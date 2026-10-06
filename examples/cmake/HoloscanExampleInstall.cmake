# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holoscan_configure_example_install target)
    set(one_value_args RUNTIME_DESTINATION)
    cmake_parse_arguments(HCEI "" "${one_value_args}" "" ${ARGN})
    if(HCEI_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_configure_example_install received unknown arguments: "
            "${HCEI_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT HCEI_RUNTIME_DESTINATION)
        file(RELATIVE_PATH HCEI_EXAMPLE_RELATIVE_DIR
            "${HOLOSCAN_EXAMPLES_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}")
        set(HCEI_RUNTIME_DESTINATION
            "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/${HCEI_EXAMPLE_RELATIVE_DIR}")
    endif()
    # Installed C++ binaries mirror their source directory under holoscan/examples:
    #
    #   <prefix>/<libdir>/holoscan/examples/<example path>/cpp/<binary>
    #
    # The SDK libraries live in <prefix>/<libdir>. Compute the path from the
    # runtime destination so each ../ corresponds to one layout component and
    # nested examples remain relocatable without LD_LIBRARY_PATH.
    file(RELATIVE_PATH HCEI_LIBDIR_RELATIVE_PATH
        "/${HCEI_RUNTIME_DESTINATION}"
        "/${CMAKE_INSTALL_LIBDIR}")
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN/${HCEI_LIBDIR_RELATIVE_PATH}")
endfunction()

# Register one runnable Python example for both the SDK build and installed-source consumers.
function(holoscan_add_python_example)
    set(one_value_args NAME SCRIPT TEST_NAME TIMEOUT)
    set(multi_value_args LABELS)
    cmake_parse_arguments(HPE "" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(HPE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_add_python_example received unknown arguments: ${HPE_UNPARSED_ARGUMENTS}")
    endif()
    foreach(required_argument IN ITEMS NAME SCRIPT TEST_NAME)
        if(NOT HPE_${required_argument})
            message(FATAL_ERROR "holoscan_add_python_example requires ${required_argument}")
        endif()
    endforeach()
    if(NOT HPE_TIMEOUT)
        set(HPE_TIMEOUT 30)
    endif()

    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    if(NOT DEFINED HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR)
        find_package(holoscan REQUIRED CONFIG)
        if(NOT holoscan_PYTHON_PACKAGE_DIR)
            message(FATAL_ERROR
                "The selected Holoscan SDK package does not provide Python bindings")
        endif()
        set(HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR "${holoscan_PYTHON_PACKAGE_DIR}")
        set(HOLOSCAN_EXAMPLES_IN_SDK_BUILD OFF)
    endif()

    if(NOT HOLOSCAN_EXAMPLES_IN_SDK_BUILD)
        if(NOT EXISTS "${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}/holoscan/__init__.py")
            message(FATAL_ERROR
                "The installed Holoscan Python package is missing from "
                "${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}")
        endif()
        file(GLOB core_modules LIST_DIRECTORIES FALSE
            "${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}/holoscan/_core*.so"
            "${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}/holoscan/_core*.pyd"
            "${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}/holoscan/_core*.dylib")
        if(NOT core_modules)
            message(FATAL_ERROR
                "The installed Holoscan Python package does not include its Core extension")
        endif()
    endif()

    set(example_source "${CMAKE_CURRENT_SOURCE_DIR}/${HPE_SCRIPT}")
    add_custom_target(python_${HPE_NAME} ALL SOURCES "${example_source}")
    if(HOLOSCAN_EXAMPLES_IN_SDK_BUILD)
        set_property(GLOBAL APPEND PROPERTY
            HOLOSCAN_PYTHON_EXAMPLE_TARGETS python_${HPE_NAME})
    endif()

    if(BUILD_TESTING)
        # Execute the source file directly so build-tree and installed-source consumers exercise
        # the same public import path.
        add_test(NAME ${HPE_TEST_NAME}
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}"
                "PYTHONUNBUFFERED=1"
                "${Python3_EXECUTABLE}" "${example_source}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
        set_tests_properties(${HPE_TEST_NAME} PROPERTIES
            FAIL_REGULAR_EXPRESSION "Fatal Python error"
            LABELS "${HPE_LABELS}"
            TIMEOUT ${HPE_TIMEOUT}
        )
    endif()

    if(HOLOSCAN_EXAMPLES_IN_SDK_BUILD AND HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
        file(RELATIVE_PATH example_relative_dir
            "${HOLOSCAN_EXAMPLES_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
        install(FILES "${example_source}"
            DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/${example_relative_dir}"
            COMPONENT holoscan-core-examples)
    endif()
endfunction()
