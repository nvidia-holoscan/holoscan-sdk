#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
project(holoscan_inference_toolkit
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan Inference toolkit"
    LANGUAGES CXX CUDA
    )

include(CMakePackageConfigHelpers)
set(HOLOSCAN_INFERENCE_INSTALL_COMPONENT holoscan-inference)
# The Python bindings are configured later from the sibling python/holoscan
# subtree. Publish the module-owned component name to their common parent so
# every Inference payload remains in the same independently installable unit.
set(HOLOSCAN_INFERENCE_INSTALL_COMPONENT
    "${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}" PARENT_SCOPE)

add_subdirectory(src)
add_subdirectory(operators)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

set(_holoscan_inference_package_dir
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-inference-package")
set(_holoscan_inference_cmake_install_dir
    "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}")
file(MAKE_DIRECTORY "${_holoscan_inference_package_dir}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${_holoscan_inference_package_dir}/${HOLOSCAN_PACKAGE_NAME}-inference-config.cmake"
    INSTALL_DESTINATION "${_holoscan_inference_cmake_install_dir}"
)
write_basic_package_version_file(
    "${_holoscan_inference_package_dir}/${HOLOSCAN_PACKAGE_NAME}-inference-config-version.cmake"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    FILE ${HOLOSCAN_PACKAGE_NAME}-inference-targets.cmake
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    DESTINATION "${_holoscan_inference_cmake_install_dir}"
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
)
install(FILES
    "${_holoscan_inference_package_dir}/${HOLOSCAN_PACKAGE_NAME}-inference-config.cmake"
    "${_holoscan_inference_package_dir}/${HOLOSCAN_PACKAGE_NAME}-inference-config-version.cmake"
    DESTINATION "${_holoscan_inference_cmake_install_dir}"
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
)

if(NOT BUILD_SHARED_LIBS)
    # Static consumers receive yaml-cpp through an SDK-private imported target. Keep the archive
    # with holoscan-inference so the component has no public yaml-cpp package requirement.
    install(FILES $<TARGET_FILE:yaml-cpp::yaml-cpp>
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
        RENAME
            ${CMAKE_STATIC_LIBRARY_PREFIX}holoscan_inference_yaml_private${CMAKE_STATIC_LIBRARY_SUFFIX}
        COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
    )
endif()

unset(_holoscan_inference_cmake_install_dir)
unset(_holoscan_inference_package_dir)
