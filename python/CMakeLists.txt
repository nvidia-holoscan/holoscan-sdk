# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# create a folder to store the compiled libraries
set(CMAKE_PYBIND11_MODULE_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/holoscan)
file(MAKE_DIRECTORY ${CMAKE_PYBIND11_MODULE_OUT_DIR})

# custom target to encapsulate all
add_custom_target(holoscan-python ALL)

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(holoscan_pybind11_module CMAKE_PYBIND11_MODULE_NAME)

    pybind11_add_module(holoscan_${CMAKE_PYBIND11_MODULE_NAME}
        MODULE
        ${ARGN}
    )
    add_dependencies(holoscan-python holoscan_${CMAKE_PYBIND11_MODULE_NAME})

    target_link_libraries(holoscan_${CMAKE_PYBIND11_MODULE_NAME}
        PRIVATE
        ${HOLOSCAN_PACKAGE_NAME}
    )

    # Sets the rpath of the module
    file(RELATIVE_PATH install_lib_relative_path
         ${CMAKE_CURRENT_LIST_DIR}/pybind11/${CMAKE_PYBIND11_MODULE_NAME}/__init__.py
         ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR} )
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our install tree (same layout as src)
        "\$ORIGIN/../lib" # in our python wheel"
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET holoscan_${CMAKE_PYBIND11_MODULE_NAME}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # make submodule folder
    file(MAKE_DIRECTORY ${CMAKE_PYBIND11_MODULE_OUT_DIR}/${CMAKE_PYBIND11_MODULE_NAME})

    # custom target to ensure the module's __init__.py file is copied
    set(CMAKE_SUBMODULE_INIT_FILE
        ${CMAKE_CURRENT_LIST_DIR}/pybind11/${CMAKE_PYBIND11_MODULE_NAME}/__init__.py
    )
    set(CMAKE_SUBMODULE_OUT_DIR ${CMAKE_PYBIND11_MODULE_OUT_DIR}/${CMAKE_PYBIND11_MODULE_NAME})
    add_custom_target(holoscan-${CMAKE_PYBIND11_MODULE_NAME}-init
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SUBMODULE_INIT_FILE}" "${CMAKE_SUBMODULE_OUT_DIR}/"
        DEPENDS "${CMAKE_SUBMODULE_INIT_FILE}"
    )
    add_dependencies(holoscan_${CMAKE_PYBIND11_MODULE_NAME}
        holoscan-${CMAKE_PYBIND11_MODULE_NAME}-init
    )

    # Note: OUTPUT_NAME filename (_holoscan_core) must match the module name in the PYBIND11_MODULE macro
    set_target_properties(holoscan_${CMAKE_PYBIND11_MODULE_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
        OUTPUT_NAME holoscan/${CMAKE_PYBIND11_MODULE_NAME}/_${CMAKE_PYBIND11_MODULE_NAME}
    )

endfunction()

# custom target for top-level __init__.py file is copied
set(CMAKE_PYBIND11_PRIMARY_INIT_FILE ${CMAKE_CURRENT_LIST_DIR}/pybind11/__init__.py)
add_custom_target(holoscan-python-pyinit
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}" "${CMAKE_PYBIND11_MODULE_OUT_DIR}/"
    DEPENDS "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}"
)
add_dependencies(holoscan-python holoscan-python-pyinit)

# custom target for top-level config.py file
set(CMAKE_PYBIND11_CONFIG_FILE ${CMAKE_CURRENT_LIST_DIR}/pybind11/config.py)
add_custom_target(holoscan-python-config
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_PYBIND11_CONFIG_FILE}" "${CMAKE_PYBIND11_MODULE_OUT_DIR}/"
    DEPENDS "${CMAKE_PYBIND11_CONFIG_FILE}"
)
add_dependencies(holoscan-python holoscan-python-config)

######################
# Add pybind11 modules
######################

holoscan_pybind11_module(
    core
    pybind11/core/core.cpp
    pybind11/core/core.hpp
    pybind11/core/core_pydoc.hpp
    pybind11/core/dl_converter.cpp
    pybind11/core/dl_converter.hpp
    pybind11/core/trampolines.cpp
    pybind11/core/trampolines.hpp
    pybind11/kwarg_handling.cpp
    pybind11/kwarg_handling.hpp
    pybind11/macros.hpp
)

holoscan_pybind11_module(
    gxf
    pybind11/core/dl_converter.cpp
    pybind11/core/dl_converter.hpp
    pybind11/core/trampolines.cpp
    pybind11/core/trampolines.hpp
    pybind11/macros.hpp
    pybind11/gxf/gxf.cpp
    pybind11/gxf/gxf.hpp
    pybind11/gxf/gxf_pydoc.hpp
)

holoscan_pybind11_module(
    logger
    pybind11/macros.hpp
    pybind11/logger/logger.cpp
    pybind11/logger/logger_pydoc.hpp
)

holoscan_pybind11_module(
    conditions
    pybind11/macros.hpp
    pybind11/conditions/conditions.cpp
    pybind11/conditions/conditions_pydoc.hpp
)

holoscan_pybind11_module(
    resources
    pybind11/macros.hpp
    pybind11/resources/resources.cpp
    pybind11/resources/resources_pydoc.hpp
)

holoscan_pybind11_module(
    graphs
    pybind11/macros.hpp
    pybind11/graphs/graphs.cpp
    pybind11/graphs/graphs_pydoc.hpp
)

holoscan_pybind11_module(
    operators
    pybind11/macros.hpp
    pybind11/operators/operators.cpp
    pybind11/operators/operators_pydoc.hpp
)

holoscan_pybind11_module(
    executors
    pybind11/macros.hpp
    pybind11/executors/executors.cpp
    pybind11/executors/executors_pydoc.hpp
)

# For convenience, we currently copy the tests folder to build/python/lib as well

# custom target for Python unit test files
set(CMAKE_PYBIND11_TEST_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests)
add_custom_target(holoscan-python-tests
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_PYBIND11_TEST_DIRECTORY}" "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/tests"
    DEPENDS "${CMAKE_PYBIND11_CONFIG_FILE}"
)
add_dependencies(holoscan-python holoscan-python-tests)

install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib"
    DESTINATION python
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-python_libs"
    PATTERN "__pycache__" EXCLUDE
)

if(HOLOSCAN_BUILD_TESTS)
    add_test(NAME python-api-tests
        COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/tests
        # either have to run from this working directory or set PYTHONPATH
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
endif()
