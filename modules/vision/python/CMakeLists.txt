# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(_vision_package "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan_vision")
set(_vision_operators_package "${_vision_package}/operators")
set(_vision_install "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan_vision")
file(MAKE_DIRECTORY "${_vision_package}" "${_vision_operators_package}")
set(_vision_package_files __init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _vision_package_files __init__.pyi py.typed)
endif()
foreach(package_file IN LISTS _vision_package_files)
    configure_file("holoscan_vision/${package_file}" "${_vision_package}/${package_file}" COPYONLY)
endforeach()
configure_file(
    "holoscan_vision/operators/__init__.py"
    "${_vision_operators_package}/__init__.py"
    COPYONLY)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    configure_file(
        "holoscan_vision/operators/__init__.pyi"
        "${_vision_operators_package}/__init__.pyi"
        COPYONLY)
endif()

nanobind_add_module(holoscan_vision_python NB_STATIC STABLE_ABI _vision.cpp)
target_link_libraries(holoscan_vision_python PRIVATE holoscan::vision)
file(RELATIVE_PATH _vision_install_lib_relative "/${_vision_install}" "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_vision_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_vision_package}"
    OUTPUT_NAME _vision
    BUILD_RPATH "\$ORIGIN/../../../lib"
    INSTALL_RPATH "\$ORIGIN/${_vision_install_lib_relative}"
)

add_dependencies(holoscan-python holoscan_vision_python)

if(HOLOSCAN_BUILD_PYTHON_STUBS)
    set(_vision_generated_operator_stub
        "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/holoscan_vision/operators/__init__.pyi")
    holoscan_add_nanobind_stubs(
        TARGET holoscan-vision-python-stubs
        ALL
        MODULE holoscan_vision._vision
        OUTPUT "${_vision_package}/_vision.pyi"
        EXTENSION_TARGETS holoscan_core_python holoscan_vision_python
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
        GENERATOR_SCRIPT "${_holoscan_nanobind_stubgen}"
    )
    add_dependencies(holoscan-python holoscan-vision-python-stubs)
    holoscan_add_operator_stubs(
        TARGET generate-holoscan-vision-python-stubs
        OUTPUT "${_vision_generated_operator_stub}"
        PROVIDER_MODULES holoscan_vision
        REGISTRY_PREFIX holoscan.vision.
        MODULE_DOC "Registered Holoscan Vision Operator proxies."
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
        EXTENSION_TARGETS holoscan_core_python holoscan_vision_python
        DEPENDS holoscan-vision-python-stubs
        COMMITTED_STUB "${CMAKE_CURRENT_SOURCE_DIR}/holoscan_vision/operators/__init__.pyi"
        CHECK_TARGET check-holoscan-vision-python-stubs
        COMMENT "Generating registry-backed Holoscan Vision Operator stubs"
    )
endif()

# Keep Python out of the native Debian payload while retaining source/archive installs.
install(TARGETS holoscan_vision_python
    LIBRARY DESTINATION "${_vision_install}" COMPONENT holoscan-vision-python_libs)
set(_vision_install_files "${_vision_package}/__init__.py")
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _vision_install_files
        "${_vision_package}/__init__.pyi"
        "${_vision_package}/_vision.pyi"
        "${_vision_package}/py.typed")
endif()
install(FILES ${_vision_install_files}
    DESTINATION "${_vision_install}" COMPONENT holoscan-vision-python_libs)
install(FILES "${_vision_operators_package}/__init__.py"
    DESTINATION "${_vision_install}/operators" COMPONENT holoscan-vision-python_libs)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    install(FILES "${_vision_operators_package}/__init__.pyi"
        DESTINATION "${_vision_install}/operators" COMPONENT holoscan-vision-python_libs)
endif()

# Native installs need the metadata that registers holoscan.vision.
holoscan_install_python_module_metadata(MODULE vision)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_executable(holoscan_vision_python_plan tests/vision_plan.cpp)
    holoscan_configure_cxx20_target(holoscan_vision_python_plan PRIVATE)
    target_link_libraries(holoscan_vision_python_plan PRIVATE holoscan::vision)
    add_dependencies(holoscan-python holoscan_vision_python_plan)

    add_test(NAME python-vision.bindings
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py")
    set_tests_properties(python-vision.bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
        LABELS "python;vision;unit")

    add_test(NAME python-vision.operators
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vision.py")
    set_tests_properties(python-vision.operators PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT};HOLOSCAN_VISION_PLAN_HELPER=$<TARGET_FILE:holoscan_vision_python_plan>"
        LABELS "python;vision;unit")

    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        add_test(NAME python-vision.stub_surface
            COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stubs.py")
        add_test(NAME python-vision.type_completeness
            COMMAND "${Python3_EXECUTABLE}"
                "${HOLOSCAN_PYTHON_TOOLS_DIR}/check_type_completeness.py"
                holoscan_vision --minimum 100)
        add_test(NAME python-vision.stubtest
            COMMAND "${Python3_EXECUTABLE}" -m mypy.stubtest holoscan_vision
                --allowlist "${CMAKE_CURRENT_SOURCE_DIR}/stubtest_allowlist.txt"
                --ignore-unused-allowlist)
        set_tests_properties(
            python-vision.stub_surface
            python-vision.type_completeness
            python-vision.stubtest
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
                WORKING_DIRECTORY "${holoscan_SOURCE_DIR}"
                LABELS "python;vision;unit")
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
