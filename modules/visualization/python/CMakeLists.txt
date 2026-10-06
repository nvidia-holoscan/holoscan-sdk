# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(_visualization_package "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan_visualization")
set(_visualization_install "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan_visualization")
file(MAKE_DIRECTORY "${_visualization_package}/operators")
foreach(package_file IN ITEMS __init__.py operators/__init__.py)
    configure_file("holoscan_visualization/${package_file}"
                   "${_visualization_package}/${package_file}" COPYONLY)
endforeach()
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    foreach(package_file IN ITEMS __init__.pyi py.typed operators/__init__.pyi)
        configure_file("holoscan_visualization/${package_file}"
                       "${_visualization_package}/${package_file}" COPYONLY)
    endforeach()
endif()

nanobind_add_module(holoscan_visualization_python NB_STATIC STABLE_ABI _visualization.cpp)
target_link_libraries(holoscan_visualization_python PRIVATE holoscan::visualization::op_holoviz)
file(RELATIVE_PATH _visualization_install_lib_relative
    "/${_visualization_install}" "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_visualization_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_visualization_package}"
    OUTPUT_NAME _visualization
    BUILD_RPATH "\$ORIGIN/../../../../lib"
    INSTALL_RPATH "\$ORIGIN/${_visualization_install_lib_relative}"
)

add_dependencies(holoscan-python holoscan_visualization_python)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    holoscan_add_nanobind_stubs(
        TARGET holoscan-visualization-python-stubs
        ALL
        MODULE holoscan_visualization._visualization
        OUTPUT "${_visualization_package}/_visualization.pyi"
        EXTENSION_TARGETS holoscan_core_python holoscan_visualization_python
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
        GENERATOR_SCRIPT "${_holoscan_nanobind_stubgen}"
        DEPENDS
            holoscan-python-init
            ${HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES}
    )
    add_dependencies(holoscan-python holoscan-visualization-python-stubs)
endif()

# Keep Python out of the native Debian payload while retaining source/archive installs.
install(TARGETS holoscan_visualization_python
    LIBRARY DESTINATION "${_visualization_install}"
    COMPONENT holoscan-visualization-python_libs)
install(FILES "${_visualization_package}/__init__.py"
    DESTINATION "${_visualization_install}"
    COMPONENT holoscan-visualization-python_libs)
install(FILES "${_visualization_package}/operators/__init__.py"
    DESTINATION "${_visualization_install}/operators"
    COMPONENT holoscan-visualization-python_libs)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    install(FILES
        "${_visualization_package}/__init__.pyi"
        "${_visualization_package}/_visualization.pyi"
        "${_visualization_package}/py.typed"
        DESTINATION "${_visualization_install}"
        COMPONENT holoscan-visualization-python_libs)
    install(FILES "${_visualization_package}/operators/__init__.pyi"
        DESTINATION "${_visualization_install}/operators"
        COMPONENT holoscan-visualization-python_libs)
endif()

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_executable(holoscan_visualization_python_plan tests/visualization_plan.cpp)
    holoscan_configure_cxx20_target(holoscan_visualization_python_plan PRIVATE)
    target_link_libraries(holoscan_visualization_python_plan
        PRIVATE holoscan::visualization::op_holoviz)
    add_dependencies(holoscan-python holoscan_visualization_python_plan)

    add_test(NAME python-visualization.bindings
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py")
    add_test(NAME python-visualization.operators
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_visualization.py")
    set_tests_properties(python-visualization.bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
        LABELS "python;visualization;unit")
    set_tests_properties(python-visualization.operators PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT};HOLOSCAN_VISUALIZATION_PLAN_HELPER=$<TARGET_FILE:holoscan_visualization_python_plan>"
        LABELS "python;visualization;unit")

    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        add_test(NAME python-visualization.type_completeness
            COMMAND "${Python3_EXECUTABLE}"
                "${holoscan_SOURCE_DIR}/python/check_type_completeness.py"
                holoscan_visualization --minimum 100)
        add_test(NAME python-visualization.stubtest
            COMMAND "${Python3_EXECUTABLE}" -m mypy.stubtest holoscan_visualization)
        set_tests_properties(
            python-visualization.type_completeness
            python-visualization.stubtest
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
                WORKING_DIRECTORY "${holoscan_SOURCE_DIR}"
                LABELS "python;visualization;unit")
    endif()

    add_test(NAME python-visualization.installed_component
        COMMAND "${CMAKE_COMMAND}"
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/installed-component"
            "-DHOLOSCAN_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
            "-DHOLOSCAN_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
            "-DHOLOSCAN_PYTHON_VERSION=${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
            "-DHOLOSCAN_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DHOLOSCAN_BUILD_PYTHON_STUBS=${HOLOSCAN_BUILD_PYTHON_STUBS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_installed_component.cmake")
    set_tests_properties(python-visualization.installed_component PROPERTIES
        LABELS "python;visualization;installed-package;package-boundary")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
