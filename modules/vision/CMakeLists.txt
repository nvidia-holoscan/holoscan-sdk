# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(holoscan_vision
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan Vision module"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CVCUDAVersion.cmake")
# The SDK package config also needs these bounds for static Vision consumers.
if(NOT holoscan_vision_IS_TOP_LEVEL)
    set(HOLOSCAN_VISION_CVCUDA_VERSION_MIN "${HOLOSCAN_VISION_CVCUDA_VERSION_MIN}" PARENT_SCOPE)
    set(HOLOSCAN_VISION_CVCUDA_VERSION_MAX "${HOLOSCAN_VISION_CVCUDA_VERSION_MAX}" PARENT_SCOPE)
endif()

set(HOLOSCAN_VISION_EXAMPLES_INSTALL_COMPONENT holoscan-vision-examples)
if(WIN32)
    set(HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/holoscan/examples")
else()
    set(HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/holoscan/examples")
endif()

# CV-CUDA's package configuration consumes the nvcv_types target without discovering it.
find_package(nvcv_types
    ${HOLOSCAN_VISION_CVCUDA_VERSION_MIN}...<${HOLOSCAN_VISION_CVCUDA_VERSION_MAX}
    CONFIG REQUIRED
)
find_package(cvcuda
    ${HOLOSCAN_VISION_CVCUDA_VERSION_MIN}...<${HOLOSCAN_VISION_CVCUDA_VERSION_MAX}
    CONFIG REQUIRED
)
# BayerDemosaicOp requires NPP's CFA-to-RGB(A) kernels. Match NPP linkage to Vision;
# static consumers link the NPP archives when linking their final application.
if(BUILD_SHARED_LIBS)
    set(_holoscan_vision_npp_target CUDA::nppicc)
else()
    set(_holoscan_vision_npp_target CUDA::nppicc_static)
endif()
if(NOT TARGET ${_holoscan_vision_npp_target})
    find_package(CUDAToolkit REQUIRED)
endif()
if(NOT TARGET ${_holoscan_vision_npp_target})
    message(FATAL_ERROR
        "BayerDemosaicOp requires ${_holoscan_vision_npp_target}; install the NPP development files, "
        "including its static archives for BUILD_SHARED_LIBS=OFF."
    )
endif()

# The CV-CUDA tensor and image-metadata helpers are compiled once for the library and for the unit
# tests that exercise them directly (the library keeps them hidden), with the library's flags.
add_library(vision_detail_objects OBJECT
    src/detail/cvcuda_tensor.cpp
    src/detail/image_metadata.cpp
)
holoscan_configure_cxx20_target(vision_detail_objects PRIVATE)
target_include_directories(vision_detail_objects
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)
target_link_libraries(vision_detail_objects
    PUBLIC
        holoscan::core
        holoscan::sensor_io
        cvcuda
        nvcv_types
        CUDA::cudart
)
target_compile_definitions(vision_detail_objects
    PRIVATE vision_EXPORTS
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_VISION_STATIC_DEFINE>
)
set_target_properties(vision_detail_objects PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)

add_library(vision
    src/bayer_demosaic_op.cpp
    src/convert_to_op.cpp
    src/copy_make_border_op.cpp
    src/custom_crop_op.cpp
    $<TARGET_OBJECTS:vision_detail_objects>
    src/non_maximum_suppression_op.cpp
    src/normalize_op.cpp
    src/operator_catalog.cpp
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
        holoscan::sensor_io
    PRIVATE
        cvcuda
        nvcv_types
        ${_holoscan_vision_npp_target}
        # Keep the runtime after NPP's archives for downstream static links.
        CUDA::cudart
)
unset(_holoscan_vision_npp_target)
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

# The CV-CUDA project publishes Debian archives as release assets, not through an APT repository.
# Bundle the exact shared objects selected for this build with the shared Vision payload so its
# Debian package remains installable from an APT repository containing only Holoscan packages.
if(BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_holoscan_vision_cvcuda_install_dir "${HOLOSCAN_INSTALL_LIB_DIR}/holoscan/cvcuda")
    set_property(TARGET vision APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/holoscan/cvcuda")
    include(HoloscanImportedInstall)
    foreach(_holoscan_vision_cvcuda_target IN ITEMS cvcuda nvcv_types)
        holoscan_imported_install_helper("${_holoscan_vision_cvcuda_target}"
            "${CMAKE_SHARED_LIBRARY_PREFIX}${_holoscan_vision_cvcuda_target}${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${_holoscan_vision_cvcuda_install_dir}" holoscan-vision)
    endforeach()
    unset(_holoscan_vision_cvcuda_target)
    unset(_holoscan_vision_cvcuda_install_dir)
endif()

set(_holoscan_vision_package_dir
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-vision-package"
)
set(_holoscan_vision_cmake_install_dir
    "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
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
