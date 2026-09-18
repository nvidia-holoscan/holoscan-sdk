# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")
include(HoloscanStubGeneration)
include(HoloscanPythonWheel)

# Use the generator from the SDK's pinned nanobind checkout.
find_file(_holoscan_nanobind_stubgen stubgen.py PATHS "${NB_DIR}/src" "${NB_DIR}"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)

add_custom_target(holoscan-python ALL)

set(HOLOSCAN_PYTHON_MODULE_BINARY_DIR "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}")

set(HOLOSCAN_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(HOLOSCAN_PYTHON_CORE_INSTALL_DIR "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan/core")

set(HOLOSCAN_PYTHON_INIT_OUTPUT "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/__init__.py")
add_custom_command(
    OUTPUT "${HOLOSCAN_PYTHON_INIT_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
            "${HOLOSCAN_PYTHON_INIT_OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    COMMENT "Copying holoscan Python package __init__.py"
)
add_custom_target(holoscan-python-init DEPENDS "${HOLOSCAN_PYTHON_INIT_OUTPUT}")
add_dependencies(holoscan-python holoscan-python-init)

set(_holoscan_python_copied_files _module_discovery.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _holoscan_python_copied_files __init__.pyi py.typed core/__init__.pyi)
endif()
foreach(package_file IN LISTS _holoscan_python_copied_files)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${package_file}"
        "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${package_file}" COPYONLY)
endforeach()

add_subdirectory(core)

# Vision's native target is configured earlier; its bindings need nanobind and Python first.
if(TARGET holoscan::vision)
    add_subdirectory("${holoscan_SOURCE_DIR}/modules/vision/python"
        "${holoscan_BINARY_DIR}/modules/vision/python")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
# wheel.cmake stays self-contained for the packaging tests, so its payload list
# is declared separately from the copy list above. Fail closed if they drift.
foreach(copied IN LISTS _holoscan_python_copied_files)
    if(NOT copied IN_LIST _holoscan_python_files)
        message(FATAL_ERROR
            "'${copied}' is copied into the build tree but missing from "
            "_holoscan_python_files in cmake/wheel.cmake; the wheel would omit it")
    endif()
endforeach()
foreach(path IN LISTS _holoscan_python_files)
    get_filename_component(directory "${path}" DIRECTORY)
    install(FILES "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${path}"
        DESTINATION "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan/${directory}"
        COMPONENT holoscan-python_libs)
endforeach()
install(TARGETS holoscan_core_python
    LIBRARY DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}" COMPONENT holoscan-python_libs)
