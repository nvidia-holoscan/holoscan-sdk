# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(v4l2_camera_passthrough_op INTERFACE
    v4l2_camera_passthrough_op.hpp
)
add_library(holoscan::ops::v4l2_camera_passthrough ALIAS v4l2_camera_passthrough_op)
target_link_libraries(v4l2_camera_passthrough_op INTERFACE holoscan::core)
target_include_directories(v4l2_camera_passthrough_op INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_PYTHON_EXAMPLES)
    # Build Python bindings for V4L2CameraPassthroughOp
    # Refer to holoscan_pybind11_module for standard Pybind11 wrapper, not used
    # here due to examples path requirements.
    find_package(pybind11 REQUIRED)

    pybind11_add_module(v4l2_camera_passthrough_python MODULE
        v4l2_camera_passthrough_op_pybind11.cpp
    )

    target_link_libraries(v4l2_camera_passthrough_python
        PRIVATE
            holoscan::core
            holoscan::pybind11
    )

    # Sets the rpath of the module
    set(HOLOSCAN_PYTHON_MODULE_BINARY_DIR ${CMAKE_BINARY_DIR}/python/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan)
    set(module_binary_dir ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/operators/v4l2_camera_passthrough)
    file(RELATIVE_PATH install_lib_relative_path
        ${module_binary_dir}
        ${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
    )
    list(APPEND _rpath "\$ORIGIN/${install_lib_relative_path}")
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET v4l2_camera_passthrough_python
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # custom target to ensure the module's __init__.py file is copied
    add_custom_target(v4l2_camera_passthrough_python_init
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" "${module_binary_dir}/"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
        BYPRODUCTS "${module_binary_dir}/__init__.py"
    )
    add_dependencies(v4l2_camera_passthrough_python v4l2_camera_passthrough_python_init)

    set_target_properties(v4l2_camera_passthrough_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${module_binary_dir}
        OUTPUT_NAME _v4l2_camera_passthrough
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    add_dependencies(holoscan-python v4l2_camera_passthrough_python)
endif()

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)

    # Get relative folder path for the app
    get_filename_component(parent_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${parent_dir})

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera_passthrough_op.hpp"
        DESTINATION "${app_relative_dest_path}"
        COMPONENT holoscan-examples
    )
endif()