# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(sensor_daemon)
add_subdirectory(recorder)

add_executable(external_client_relay external_client_relay.cpp)
target_compile_features(external_client_relay PRIVATE cxx_std_20)
target_link_libraries(external_client_relay PRIVATE holoscan::core)
holoscan_configure_example_install(external_client_relay)

if(HOLOSCAN_BUILD_PYTHON_EXAMPLES)
    set(_HOLOSCAN_EXTERNAL_CLIENT_PYTHON_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/sensor_daemon/python/sensor_daemon.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/recorder/python/recorder.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/external_client_relay.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/run_clients.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/test_run_clients.py"
    )
    add_custom_target(python_external_clients ALL
        SOURCES ${_HOLOSCAN_EXTERNAL_CLIENT_PYTHON_FILES})
    if(HOLOSCAN_EXAMPLES_IN_SDK_BUILD)
        set_property(GLOBAL APPEND PROPERTY
            HOLOSCAN_PYTHON_EXAMPLE_TARGETS python_external_clients)
    endif()

    if(BUILD_TESTING)
        add_test(NAME EXAMPLE_PYTHON_EXTERNAL_CLIENTS_RUNNER_TEST
            COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/python/test_run_clients.py"
        )
        set_tests_properties(EXAMPLE_PYTHON_EXTERNAL_CLIENTS_RUNNER_TEST
            PROPERTIES
                LABELS "examples;python;channel"
                TIMEOUT 30
        )

        foreach(_graph IN ITEMS cpp python)
            if(_graph STREQUAL "cpp")
                set(_graph_command $<TARGET_FILE:external_client_relay>)
            else()
                set(_graph_command
                    "${CMAKE_CURRENT_SOURCE_DIR}/python/external_client_relay.py")
            endif()
            string(TOUPPER "${_graph}" _graph_upper)
            set(_test_name
                "EXAMPLE_PYTHON_EXTERNAL_CLIENTS_${_graph_upper}_GRAPH_TEST")
            # REQ: PY-CHANNEL-001
            add_test(NAME ${_test_name}
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}"
                    "PYTHONUNBUFFERED=1"
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/python/run_clients.py"
                    --graph ${_graph_command}
                    --sensor
                        "${CMAKE_CURRENT_SOURCE_DIR}/sensor_daemon/python/sensor_daemon.py"
                    --recorder
                        "${CMAKE_CURRENT_SOURCE_DIR}/recorder/python/recorder.py"
            )
            set_tests_properties(${_test_name}
                PROPERTIES
                    PASS_REGULAR_EXPRESSION
                        "Python external clients round trip complete"
                    LABELS "examples;python;channel;external-topic;interprocess"
                    RUN_SERIAL TRUE
                    TIMEOUT 60
            )
        endforeach()

        # REQ: PY-CHANNEL-006
        add_test(NAME EXAMPLE_PYTHON_EXTERNAL_CLIENTS_CPP_GRAPH_ASYNCIO_TEST
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}"
                "PYTHONUNBUFFERED=1"
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/python/run_clients.py"
                --graph $<TARGET_FILE:external_client_relay>
                --sensor
                    "${CMAKE_CURRENT_SOURCE_DIR}/sensor_daemon/python/sensor_daemon.py"
                --recorder
                    "${CMAKE_CURRENT_SOURCE_DIR}/recorder/python/recorder.py"
                --async-recorder
        )
        set_tests_properties(EXAMPLE_PYTHON_EXTERNAL_CLIENTS_CPP_GRAPH_ASYNCIO_TEST
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "Python external clients round trip complete"
                LABELS "examples;python;channel;asyncio;external-topic;interprocess"
                RUN_SERIAL TRUE
                TIMEOUT 60
        )
    endif()
endif()

install(TARGETS external_client_relay
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients"
    COMPONENT holoscan-core-examples
)
if(HOLOSCAN_BUILD_PYTHON_EXAMPLES
   AND HOLOSCAN_EXAMPLES_IN_SDK_BUILD
   AND HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/python/external_client_relay.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/run_clients.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/test_run_clients.py"
        DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients/python"
        COMPONENT holoscan-core-examples
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/sensor_daemon/python/sensor_daemon.py"
        DESTINATION
            "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients/sensor_daemon/python"
        COMPONENT holoscan-core-examples
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/recorder/python/recorder.py"
        DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients/recorder/python"
        COMPONENT holoscan-core-examples
    )
endif()
