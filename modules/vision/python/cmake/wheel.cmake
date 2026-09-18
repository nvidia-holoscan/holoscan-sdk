# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CV-CUDA is a private dependency supplied by the cvcuda-cu13 wheel rather than
# bundled here, so only Vision's own runtime is owned by this wheel. Core and its
# shared runtime remain in the companion Core wheel.
# Ship typing only as a complete set, matching Core's posture.
set(_vision_wheel_python_files __init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _vision_wheel_python_files __init__.pyi py.typed _vision.pyi)
endif()
# Providing wheels unpack as siblings in site-packages, so every runpath entry walks
# up to that root and back down into the distribution that owns the library. Core owns
# holoscan/lib; auditwheel stages a distribution's repaired libraries in
# <distribution>.libs, which is where cvcuda-cu13 keeps CV-CUDA.
set(_vision_siblings_from_lib "$ORIGIN/../..")
set(_vision_siblings_from_package "$ORIGIN/..")
holoscan_add_python_wheel_component(
    MODULE vision
    PACKAGE holoscan_vision
    PYTHON_DIRECTORY "${_vision_package}"
    PYTHON_FILES ${_vision_wheel_python_files}
    EXTENSION_TARGET holoscan_vision_python
    NATIVE_TARGETS vision
    LIBRARY_RUNPATH
        "${_vision_siblings_from_lib}/holoscan/lib"
        "${_vision_siblings_from_lib}/cvcuda_cu13.libs"
    EXTENSION_RUNPATH "${_vision_siblings_from_package}/holoscan/lib")

# Retarget Vision's CV-CUDA linkage onto the cvcuda-cu13 wheel installed alongside
# this package. That wheel is auditwheel-repaired, so its libraries carry
# content-derived SONAMEs recorded per architecture while building the wheel
# builder image.
set(_vision_cvcuda_manifest "$ENV{HOLOSCAN_CVCUDA_WHEEL_SONAMES}")
if(NOT _vision_cvcuda_manifest)
    set(_vision_cvcuda_manifest "/opt/nvidia/holoscan/cvcuda-wheel-sonames.txt")
endif()
string(CONFIGURE [=[
    set(manifest "@_vision_cvcuda_manifest@")
    if(NOT EXISTS "${manifest}")
      message(FATAL_ERROR
        "Wheel staging requires the CV-CUDA SONAME manifest at ${manifest}; "
        "build the wheel in the wheels-builder image")
    endif()
    find_program(_holoscan_patchelf patchelf REQUIRED)
    # Staging renames each library to its SONAME, so patch that name rather than
    # the development symlink, which the wheel does not carry.
    set(library
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/holoscan_vision/lib/$<TARGET_SONAME_FILE_NAME:vision>")
    file(STRINGS "${manifest}" entries)
    foreach(entry IN LISTS entries)
      string(REPLACE " " ";" fields "${entry}")
      list(GET fields 0 stem)
      list(GET fields 1 soname)
      execute_process(
        COMMAND "${_holoscan_patchelf}" --replace-needed "${stem}.so.0" "${soname}" "${library}"
        RESULT_VARIABLE result ERROR_VARIABLE error)
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "Wheel patchelf failed for ${stem}: ${error}")
      endif()
    endforeach()
  ]=] _vision_cvcuda_code @ONLY)
install(CODE "${_vision_cvcuda_code}"
    COMPONENT holoscan-wheel-vision EXCLUDE_FROM_ALL)
