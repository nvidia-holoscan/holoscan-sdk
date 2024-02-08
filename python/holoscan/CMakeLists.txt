# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set(HOLOSCAN_PYTHON_MODULE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../${HOLOSCAN_INSTALL_LIB_DIR}/holoscan)
file(MAKE_DIRECTORY ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR})

# Utility function to generate pybind11 modules
function(holoscan_pybind11_module pybind11_module_name)
    set(module_target ${pybind11_module_name}_python)
    pybind11_add_module(${module_target} MODULE ${ARGN})

    add_dependencies(holoscan-python ${module_target})

    target_link_libraries(${module_target}
        PRIVATE holoscan::core
    )

    # Sets the rpath of the module
    file(RELATIVE_PATH module_path # here to module
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}
    )
    set(module_binary_dir ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${module_path})
    file(RELATIVE_PATH install_lib_relative_path # module to install lib
        ${module_binary_dir}
        ${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
    )
    file(RELATIVE_PATH module_path_to_here # module to here
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}
    )
    set(wheel_lib_relative_path ${module_path_to_here}/${HOLOSCAN_INSTALL_LIB_DIR})
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our C++ install tree
        "\$ORIGIN/${wheel_lib_relative_path}"   # in our python wheel
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET ${module_target}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # make submodule folder
    file(MAKE_DIRECTORY ${module_binary_dir})

    # custom target to ensure the module's __init__.py file is copied
    set(module_init_file ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
    set(module_init_target ${module_target}_init)
    add_custom_target(${module_init_target}
        COMMAND ${CMAKE_COMMAND} -E copy "${module_init_file}" "${module_binary_dir}/"
        DEPENDS "${module_init_file}"
        BYPRODUCTS "${module_binary_dir}/__init__.py"
    )
    add_dependencies(${module_target} ${module_init_target})

    # Notes:
    #   - OUTPUT_NAME filename (ex: _core) must match the module name in the PYBIND11_MODULE macro
    #   - fvisibility=hidden is recommended: https://pybind11.readthedocs.io/en/stable/faq.html
    #   - other options as in: https://pybind11.readthedocs.io/en/stable/cmake/index.html
    set_target_properties(${module_target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${module_binary_dir}
        OUTPUT_NAME _${pybind11_module_name}
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endfunction()

# custom target for top-level __init__.py file is copied
set(CMAKE_PYBIND11_PRIMARY_INIT_FILE ${CMAKE_CURRENT_LIST_DIR}/__init__.py)
add_custom_target(holoscan-python-pyinit
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}" "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/"
    DEPENDS "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}"
)
add_dependencies(holoscan-python holoscan-python-pyinit)

# copy Holoscan Python CLI module
set(HOLOSCAN_PYTHON_CLI_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cli)
add_custom_target(holoscan-python-cli
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLOSCAN_PYTHON_CLI_MODULE_PATH}" "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/"
    DEPENDS "${HOLOSCAN_PYTHON_CLI_MODULE_PATH}"
)
add_dependencies(holoscan-python holoscan-python-cli)

# submodules
add_subdirectory(cli)
add_subdirectory(conditions)
add_subdirectory(core)
add_subdirectory(executors)
add_subdirectory(graphs)
add_subdirectory(gxf)
add_subdirectory(logger)
add_subdirectory(network_contexts)
add_subdirectory(operators)
add_subdirectory(resources)
add_subdirectory(schedulers)

# install
install(
    DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}"
    DESTINATION python/${HOLOSCAN_INSTALL_LIB_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-python_libs"
    PATTERN "__pycache__" EXCLUDE
)
