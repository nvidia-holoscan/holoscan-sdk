# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(_inference_package "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan_inference")
set(_inference_install "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan_inference")
file(MAKE_DIRECTORY "${_inference_package}/operators")
foreach(package_file IN ITEMS __init__.py operators/__init__.py)
    configure_file("holoscan_inference/${package_file}"
                   "${_inference_package}/${package_file}" COPYONLY)
endforeach()
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    foreach(package_file IN ITEMS __init__.pyi py.typed operators/__init__.pyi)
        configure_file("holoscan_inference/${package_file}"
                       "${_inference_package}/${package_file}" COPYONLY)
    endforeach()
endif()

nanobind_add_module(holoscan_inference_python NB_STATIC STABLE_ABI _inference.cpp)
target_link_libraries(holoscan_inference_python PRIVATE holoscan::op_inference)
file(RELATIVE_PATH _inference_install_lib_relative
    "/${_inference_install}" "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_inference_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_inference_package}"
    OUTPUT_NAME _inference
    BUILD_RPATH "\$ORIGIN/../../../../lib"
    INSTALL_RPATH "\$ORIGIN/${_inference_install_lib_relative}"
)

add_dependencies(holoscan-python holoscan_inference_python)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    holoscan_add_nanobind_stubs(
        TARGET holoscan-inference-python-stubs
        ALL
        MODULE holoscan_inference._inference
        OUTPUT "${_inference_package}/_inference.pyi"
        EXTENSION_TARGETS holoscan_core_python holoscan_inference_python
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
        GENERATOR_SCRIPT "${_holoscan_nanobind_stubgen}"
    )
    add_dependencies(holoscan-python holoscan-inference-python-stubs)
endif()

install(TARGETS holoscan_inference_python
    LIBRARY DESTINATION "${_inference_install}"
    COMPONENT holoscan-inference-python_libs)
install(FILES
    "${_inference_package}/__init__.py"
    DESTINATION "${_inference_install}"
    COMPONENT holoscan-inference-python_libs)
install(FILES
    "${_inference_package}/operators/__init__.py"
    DESTINATION "${_inference_install}/operators"
    COMPONENT holoscan-inference-python_libs)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    install(FILES
        "${_inference_package}/__init__.pyi"
        "${_inference_package}/_inference.pyi"
        "${_inference_package}/py.typed"
        DESTINATION "${_inference_install}"
        COMPONENT holoscan-inference-python_libs)
    install(FILES
        "${_inference_package}/operators/__init__.pyi"
        DESTINATION "${_inference_install}/operators"
        COMPONENT holoscan-inference-python_libs)
endif()

# Native installs need the metadata that registers holoscan.inference.
holoscan_install_python_module_metadata(MODULE inference)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_executable(holoscan_inference_python_plan tests/inference_plan.cpp)
    holoscan_configure_cxx20_target(holoscan_inference_python_plan PRIVATE)
    target_link_libraries(holoscan_inference_python_plan PRIVATE holoscan::op_inference)
    add_dependencies(holoscan-python holoscan_inference_python_plan)

    add_test(NAME python-inference.bindings
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py")
    add_test(NAME python-inference.operators
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inference.py")
    set_tests_properties(python-inference.bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
        LABELS "python;inference;unit")
    set_tests_properties(python-inference.operators PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT};HOLOSCAN_INFERENCE_PLAN_HELPER=$<TARGET_FILE:holoscan_inference_python_plan>"
        LABELS "python;inference;unit")

    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        add_test(NAME python-inference.type_completeness
            COMMAND "${Python3_EXECUTABLE}"
                "${HOLOSCAN_PYTHON_TOOLS_DIR}/check_type_completeness.py"
                holoscan_inference --minimum 100)
        add_test(NAME python-inference.stubtest
            COMMAND "${Python3_EXECUTABLE}" -m mypy.stubtest holoscan_inference.operators)
        set_tests_properties(
            python-inference.type_completeness
            python-inference.stubtest
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
                WORKING_DIRECTORY "${holoscan_SOURCE_DIR}"
                LABELS "python;inference;unit")
    endif()

    add_test(NAME python-inference.installed_component
        COMMAND "${CMAKE_COMMAND}"
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/installed-component"
            "-DHOLOSCAN_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
            "-DHOLOSCAN_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
            "-DHOLOSCAN_PYTHON_VERSION=${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
            "-DHOLOSCAN_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DHOLOSCAN_BUILD_PYTHON_STUBS=${HOLOSCAN_BUILD_PYTHON_STUBS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_installed_component.cmake")
    set_tests_properties(python-inference.installed_component PROPERTIES
        LABELS "python;inference;installed-package;package-boundary")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
