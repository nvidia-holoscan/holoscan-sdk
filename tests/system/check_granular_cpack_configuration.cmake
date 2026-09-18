# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_CPACK_CONFIG OR NOT EXISTS "${HOLOSCAN_CPACK_CONFIG}")
    message(FATAL_ERROR "HOLOSCAN_CPACK_CONFIG must name the generated CPack configuration")
endif()

include("${HOLOSCAN_CPACK_CONFIG}")

if(NOT CPACK_COMPONENTS_GROUPING STREQUAL "ONE_PER_GROUP")
    message(FATAL_ERROR "The configured default must use granular package groups")
endif()
if(NOT EXISTS "${CPACK_PROJECT_CONFIG_FILE}")
    message(FATAL_ERROR "Missing CPack project configuration: ${CPACK_PROJECT_CONFIG_FILE}")
endif()

if("holoscan-inference" IN_LIST CPACK_COMPONENTS_ALL)
    foreach(tensorrt_runtime_package IN ITEMS
            libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10)
        string(FIND "${CPACK_DEBIAN_PACKAGE_RECOMMENDS}" "${tensorrt_runtime_package}"
            monolithic_recommendation_index)
        if(monolithic_recommendation_index EQUAL -1)
            message(FATAL_ERROR
                "The monolithic package lacks the ${tensorrt_runtime_package} recommendation: "
                "${CPACK_DEBIAN_PACKAGE_RECOMMENDS}")
        endif()
    endforeach()
endif()

if(NOT "${CPACK_COMPONENT_HOLOSCAN-CORE_GROUP}" STREQUAL "core")
    message(FATAL_ERROR "holoscan-core must belong to the core package group")
endif()
if("holoscan-vision" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-VISION_GROUP}" STREQUAL "vision")
    message(FATAL_ERROR "holoscan-vision must belong to the vision package group")
endif()
if("holoscan-vision-examples" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-VISION-EXAMPLES_GROUP}" STREQUAL "vision")
    message(FATAL_ERROR "holoscan-vision-examples must belong to the vision package group")
endif()
if("holoscan-vision-python_libs" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-VISION-PYTHON_LIBS_GROUP}" STREQUAL "vision")
    message(FATAL_ERROR "Vision Python bindings must belong to the vision archive group")
endif()
if("holoscan-inference" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-INFERENCE_GROUP}" STREQUAL "inference")
    message(FATAL_ERROR "holoscan-inference must belong to the inference package group")
endif()
if("holoscan-inference-examples" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-INFERENCE-EXAMPLES_GROUP}" STREQUAL "inference")
    message(FATAL_ERROR "holoscan-inference-examples must belong to the inference package group")
endif()
if("holoscan-visualization" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-VISUALIZATION_GROUP}" STREQUAL "visualization")
    message(FATAL_ERROR "holoscan-visualization must belong to the visualization package group")
endif()
if("holoscan-visualization-examples" IN_LIST CPACK_COMPONENTS_ALL
        AND NOT "${CPACK_COMPONENT_HOLOSCAN-VISUALIZATION-EXAMPLES_GROUP}" STREQUAL "visualization")
    message(FATAL_ERROR "holoscan-visualization-examples must belong to the visualization package group")
endif()

set(CPACK_GENERATOR DEB)
include("${CPACK_PROJECT_CONFIG_FILE}")
if(NOT CPACK_COMPONENTS_GROUPING STREQUAL "ONE_PER_GROUP")
    message(FATAL_ERROR "Granular CPack must emit one artifact per component group")
endif()
if(NOT CPACK_DEB_COMPONENT_INSTALL OR NOT CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS)
    message(FATAL_ERROR "Granular Debian component packaging and dependencies must be enabled")
endif()
if(NOT CPACK_ARCHIVE_COMPONENT_INSTALL)
    message(FATAL_ERROR "Granular archive component packaging must be enabled")
endif()
foreach(group IN ITEMS CORE VISION INFERENCE VISUALIZATION)
    if(NOT DEFINED CPACK_ARCHIVE_${group}_FILE_NAME
            OR "${CPACK_ARCHIVE_${group}_FILE_NAME}" STREQUAL "")
        message(FATAL_ERROR "Granular archive name is missing for ${group}")
    endif()
endforeach()

# Reload the complete inventory so each mode must independently exclude Python bindings.
foreach(monolithic IN ITEMS OFF ON)
    include("${HOLOSCAN_CPACK_CONFIG}")
    set(HOLOSCAN_CPACK_MONOLITHIC_PACKAGE ${monolithic})
    set(CPACK_GENERATOR DEB)
    include("${CPACK_PROJECT_CONFIG_FILE}")
    foreach(component IN LISTS CPACK_COMPONENTS_ALL)
        if(component MATCHES "-python_libs$")
            message(FATAL_ERROR
                "Debian packages must omit ${component} (monolithic=${monolithic})")
        endif()
    endforeach()
endforeach()
