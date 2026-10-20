# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nanobind_add_module(holoscan_testing_python STABLE_ABI harness.cpp adapters.cpp registered_adapters.cpp)
target_link_libraries(holoscan_testing_python PRIVATE holoscan::testing holoscan::params)
target_include_directories(holoscan_testing_python PRIVATE
    "${HOLOSCAN_PYTHON_BINDING_SERVICES_INCLUDE_DIR}")
target_compile_definitions(holoscan_testing_python PRIVATE
    HOLOSCAN_PYTHON_SERVICES_VERSION="${HOLOSCAN_BUILD_VERSION}")
file(RELATIVE_PATH _testing_libdir "/${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}/testing"
    "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_testing_python PROPERTIES
    OUTPUT_NAME _testing
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/testing"
    BUILD_RPATH "\$ORIGIN/../../../../lib"
    INSTALL_RPATH "\$ORIGIN/${_testing_libdir}")
configure_file(__init__.py "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/testing/__init__.py" COPYONLY)
add_dependencies(holoscan-python holoscan_testing_python)
install(TARGETS holoscan_testing_python
    LIBRARY DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}/testing"
    COMPONENT holoscan-testing-python_libs)
install(FILES __init__.py DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}/testing"
    COMPONENT holoscan-testing-python_libs)

if(HOLOSCAN_BUILD_PYTHON_STUBS)
    foreach(_testing_file IN ITEMS __init__.pyi py.typed)
        configure_file("${_testing_file}"
            "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/testing/${_testing_file}" COPYONLY)
        install(FILES "${_testing_file}"
            DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}/testing"
            COMPONENT holoscan-testing-python_libs)
    endforeach()
endif()
