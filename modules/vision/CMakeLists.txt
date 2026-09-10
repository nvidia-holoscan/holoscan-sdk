# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CVCUDAVersion.cmake")

# CV-CUDA's package configuration consumes the nvcv_types target without discovering it.
find_package(nvcv_types
    ${HOLOSCAN_VISION_CVCUDA_VERSION_MIN}...<${HOLOSCAN_VISION_CVCUDA_VERSION_MAX}
    CONFIG REQUIRED
)
find_package(cvcuda
    ${HOLOSCAN_VISION_CVCUDA_VERSION_MIN}...<${HOLOSCAN_VISION_CVCUDA_VERSION_MAX}
    CONFIG REQUIRED
)

add_library(vision
    src/convert_to_op.cpp
    src/copy_make_border_op.cpp
    src/custom_crop_op.cpp
    src/detail/cvcuda_tensor.cpp
    src/non_maximum_suppression_op.cpp
    src/normalize_op.cpp
    src/reformat_op.cpp
    src/remap_op.cpp
    src/resize_op.cpp
)
add_library(holoscan::vision ALIAS vision)

holoscan_configure_cxx20_target(vision PUBLIC)
target_include_directories(vision
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(vision
    PUBLIC
        holoscan::core
    PRIVATE
        cvcuda
        nvcv_types
)
target_compile_definitions(vision
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_VISION_STATIC_DEFINE>
)
generate_export_header(vision
    BASE_NAME HOLOSCAN_VISION
    EXPORT_MACRO_NAME HOLOSCAN_VISION_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_VISION_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_VISION_DEPRECATED
    STATIC_DEFINE HOLOSCAN_VISION_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_VISION_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/vision/holoscan_vision_export.h"
)
set_target_properties(vision PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME vision
    OUTPUT_NAME holoscan_vision
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(vision)

set(_holoscan_vision_package_dir
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-vision-package"
)
set(_holoscan_vision_cmake_install_dir
    "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan-vision"
)
file(MAKE_DIRECTORY "${_holoscan_vision_package_dir}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${_holoscan_vision_package_dir}/holoscan-vision-config.cmake"
    INSTALL_DESTINATION "${_holoscan_vision_cmake_install_dir}"
)
write_basic_package_version_file(
    "${_holoscan_vision_package_dir}/holoscan-vision-config-version.cmake"
    COMPATIBILITY SameMajorVersion
)

install(TARGETS vision
    EXPORT holoscan-vision-targets
    ARCHIVE DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}" COMPONENT holoscan-vision
    LIBRARY DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}" COMPONENT holoscan-vision
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT holoscan-vision
)
install(DIRECTORY include/holoscan/vision
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan"
    COMPONENT holoscan-vision
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/vision/holoscan_vision_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/vision"
    COMPONENT holoscan-vision
)
install(EXPORT holoscan-vision-targets
    FILE holoscan-vision-targets.cmake
    NAMESPACE holoscan::
    DESTINATION "${_holoscan_vision_cmake_install_dir}"
    COMPONENT holoscan-vision
)
install(FILES
    "${_holoscan_vision_package_dir}/holoscan-vision-config.cmake"
    "${_holoscan_vision_package_dir}/holoscan-vision-config-version.cmake"
    DESTINATION "${_holoscan_vision_cmake_install_dir}"
    COMPONENT holoscan-vision
)

unset(_holoscan_vision_cmake_install_dir)
unset(_holoscan_vision_package_dir)

if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples/vision_pipeline/cpp)
endif()

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
