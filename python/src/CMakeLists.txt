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

# Utility function to generate pybind11 modules
function(holoscan_pybind11_module CMAKE_PYBIND11_MODULE_NAME)
    set(module_name holoscan_${CMAKE_PYBIND11_MODULE_NAME}_python)
    pybind11_add_module(${module_name} MODULE ${ARGN})

    add_dependencies(holoscan-python ${module_name})

    target_link_libraries(${module_name}
        PRIVATE holoscan::core
    )

    # Sets the rpath of the module
    file(RELATIVE_PATH install_lib_relative_path
         ${CMAKE_CURRENT_LIST_DIR}/${CMAKE_PYBIND11_MODULE_NAME}/__init__.py
         ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR} )
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our install tree (same layout as src)
        "\$ORIGIN/../lib" # in our python wheel"
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET ${module_name}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # make submodule folder
    file(MAKE_DIRECTORY ${CMAKE_PYBIND11_MODULE_OUT_DIR}/${CMAKE_PYBIND11_MODULE_NAME})

    # custom target to ensure the module's __init__.py file is copied
    set(CMAKE_SUBMODULE_INIT_FILE
        ${CMAKE_CURRENT_LIST_DIR}/${CMAKE_PYBIND11_MODULE_NAME}/__init__.py
    )
    set(CMAKE_SUBMODULE_OUT_DIR ${CMAKE_PYBIND11_MODULE_OUT_DIR}/${CMAKE_PYBIND11_MODULE_NAME})
    add_custom_target(holoscan-${CMAKE_PYBIND11_MODULE_NAME}-init
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SUBMODULE_INIT_FILE}" "${CMAKE_SUBMODULE_OUT_DIR}/"
        DEPENDS "${CMAKE_SUBMODULE_INIT_FILE}"
    )
    add_dependencies(${module_name}
        holoscan-${CMAKE_PYBIND11_MODULE_NAME}-init
    )

    # Note: OUTPUT_NAME filename (_holoscan_core) must match the module name in the PYBIND11_MODULE macro
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../${CMAKE_INSTALL_LIBDIR}
        OUTPUT_NAME holoscan/${CMAKE_PYBIND11_MODULE_NAME}/_${CMAKE_PYBIND11_MODULE_NAME}
    )

endfunction()

# custom target for top-level __init__.py file is copied
set(CMAKE_PYBIND11_PRIMARY_INIT_FILE ${CMAKE_CURRENT_LIST_DIR}/__init__.py)
add_custom_target(holoscan-python-pyinit
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}" "${CMAKE_PYBIND11_MODULE_OUT_DIR}/"
    DEPENDS "${CMAKE_PYBIND11_PRIMARY_INIT_FILE}"
)
add_dependencies(holoscan-python holoscan-python-pyinit)

######################
# Add pybind11 modules
######################

holoscan_pybind11_module(
    core
    core/core.cpp
    core/core.hpp
    core/core_pydoc.hpp
    core/dl_converter.cpp
    core/dl_converter.hpp
    core/trampolines.cpp
    core/trampolines.hpp
    kwarg_handling.cpp
    kwarg_handling.hpp
    macros.hpp
)

holoscan_pybind11_module(
    gxf
    core/dl_converter.cpp
    core/dl_converter.hpp
    core/trampolines.cpp
    core/trampolines.hpp
    macros.hpp
    gxf/gxf.cpp
    gxf/gxf.hpp
    gxf/gxf_pydoc.hpp
)

holoscan_pybind11_module(
    logger
    macros.hpp
    logger/logger.cpp
    logger/logger_pydoc.hpp
)

holoscan_pybind11_module(
    conditions
    macros.hpp
    conditions/conditions.cpp
    conditions/conditions_pydoc.hpp
)

holoscan_pybind11_module(
    resources
    macros.hpp
    resources/resources.cpp
    resources/resources_pydoc.hpp
)

holoscan_pybind11_module(
    graphs
    macros.hpp
    graphs/graphs.cpp
    graphs/graphs_pydoc.hpp
)

holoscan_pybind11_module(
    operators
    macros.hpp
    operators/operators.cpp
    operators/operators_pydoc.hpp
)
target_link_libraries(holoscan_operators_python
    PUBLIC
    holoscan::ops::aja
    holoscan::ops::bayer_demosaic
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::multiai_inference
    holoscan::ops::multiai_postprocessor
    holoscan::ops::segmentation_postprocessor
    holoscan::ops::tensor_rt
    holoscan::ops::video_stream_recorder
    holoscan::ops::video_stream_replayer
)

# custom target to copy operators/_native.py file
set(CMAKE_PYBIND11_NATIVE_OPS_FILE ${CMAKE_CURRENT_LIST_DIR}/operators/_native.py)
add_custom_target(holoscan-python-native-ops
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_PYBIND11_NATIVE_OPS_FILE}" "${CMAKE_PYBIND11_MODULE_OUT_DIR}/operators/_native.py"
    DEPENDS "${CMAKE_PYBIND11_NATIVE_OPS_FILE}"
)
add_dependencies(holoscan-python holoscan-python-native-ops)

holoscan_pybind11_module(
    executors
    macros.hpp
    executors/executors.cpp
    executors/executors_pydoc.hpp
)

install(DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/../lib"
    DESTINATION python
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-python_libs"
    PATTERN "__pycache__" EXCLUDE
)
