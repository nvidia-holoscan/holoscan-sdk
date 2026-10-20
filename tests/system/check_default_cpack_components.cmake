# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_CPACK_CONFIG OR NOT EXISTS "${HOLOSCAN_CPACK_CONFIG}")
    message(FATAL_ERROR "HOLOSCAN_CPACK_CONFIG must name the generated CPack configuration")
endif()

include("${HOLOSCAN_CPACK_CONFIG}")

set(_holoscan_external_install_components iceoryx2-lib dev bin)
foreach(component IN LISTS CPACK_COMPONENTS_ALL)
    list(FIND _holoscan_external_install_components "${component}" external_component_index)
    if(NOT component MATCHES "^holoscan(-|$)" AND external_component_index EQUAL -1)
        message(FATAL_ERROR
            "The default SDK package has an unowned install component: ${component}")
    endif()
endforeach()

foreach(component IN LISTS CPACK_COMPONENTS_ALL)
    if(component MATCHES "-examples$")
        message(FATAL_ERROR "The default SDK package must not include examples: ${component}")
    endif()
endforeach()
if(HOLOSCAN_BUILD_EXAMPLES AND NOT "holoscan-core-examples" IN_LIST CPACK_HOLOSCAN_EXAMPLE_COMPONENTS)
    message(FATAL_ERROR "Enabled examples must be recorded for the CPack opt-in")
endif()

foreach(required_component IN ITEMS
        holoscan-core
        holoscan-holomq
        holoscan-sensor-io
        holoscan-sensor-io-python_libs)
    list(FIND CPACK_COMPONENTS_ALL "${required_component}" component_index)
    if(component_index EQUAL -1)
        message(FATAL_ERROR "The default SDK package must include ${required_component}")
    endif()
endforeach()

if(HOLOSCAN_BUILD_TESTING_LIBRARY)
    if(NOT "holoscan-testing" IN_LIST CPACK_COMPONENTS_ALL)
        message(FATAL_ERROR "The enabled testing library must have its own install component")
    endif()
elseif("holoscan-testing" IN_LIST CPACK_COMPONENTS_ALL
        OR "holoscan-testing-examples" IN_LIST CPACK_HOLOSCAN_EXAMPLE_COMPONENTS)
    message(FATAL_ERROR "A testing-disabled SDK must not package testing support")
endif()

foreach(removed_sdk_component IN ITEMS holomq holoscan-examples)
    list(FIND CPACK_COMPONENTS_ALL "${removed_sdk_component}" component_index)
    if(NOT component_index EQUAL -1)
        message(FATAL_ERROR
            "The default SDK package must not use legacy SDK component ${removed_sdk_component}")
    endif()
endforeach()

list(FIND CPACK_COMPONENTS_ALL "holoscan-graph-artifacts" graph_artifact_component_index)
if(NOT graph_artifact_component_index EQUAL -1)
    message(FATAL_ERROR
        "The default SDK package must not execute the optional graph-artifact component")
endif()

if(HOLOSCAN_BUILD_INFERENCE)
    list(FIND CPACK_COMPONENTS_ALL "holoscan-inference" inference_component_index)
    if(inference_component_index EQUAL -1)
        message(FATAL_ERROR
            "The default SDK package must include the enabled inference component")
    endif()
endif()
