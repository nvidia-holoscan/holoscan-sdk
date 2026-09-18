# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(_vision_package "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan_vision")
set(_vision_install "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan_vision")
file(MAKE_DIRECTORY "${_vision_package}")
set(_vision_package_files __init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _vision_package_files __init__.pyi py.typed)
endif()
foreach(package_file IN LISTS _vision_package_files)
    configure_file("holoscan_vision/${package_file}" "${_vision_package}/${package_file}" COPYONLY)
endforeach()

nanobind_add_module(holoscan_vision_python NB_STATIC STABLE_ABI _vision.cpp)
target_link_libraries(holoscan_vision_python PRIVATE holoscan::vision)
file(RELATIVE_PATH _vision_install_lib_relative "/${_vision_install}" "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_vision_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_vision_package}"
    OUTPUT_NAME _vision
    BUILD_RPATH "\$ORIGIN/../../../lib"
    INSTALL_RPATH "\$ORIGIN/${_vision_install_lib_relative}"
)

if(HOLOSCAN_BUILD_PYTHON_STUBS)
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
endif()
add_dependencies(holoscan-python holoscan_vision_python)

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

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_test(NAME python-vision.bindings
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py")
    set_tests_properties(python-vision.bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
        LABELS "python;vision;unit")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
