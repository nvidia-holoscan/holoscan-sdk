# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CV-CUDA and NPP are supplied by the cvcuda-cu13 and nvidia-npp wheels, so only
# Vision's own runtime is owned by this wheel. Core and its shared runtime remain
# in the companion Core wheel.
# Ship typing only as a complete set, matching Core's posture.
set(_vision_wheel_python_files
    __init__.py
    operators/__init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _vision_wheel_python_files
        __init__.pyi
        operators/__init__.pyi
        py.typed
        _vision.pyi)
endif()
# Providing wheels unpack as siblings in site-packages, so every search-path entry walks
# up to that root and back down into the distribution that owns the library. Core owns
# holoscan/lib; auditwheel stages a distribution's repaired libraries in
# <distribution>.libs, which is where cvcuda-cu13 keeps CV-CUDA. NPP uses
# nvidia/cu13/lib instead.
set(_vision_siblings_from_lib "$ORIGIN/../..")
set(_vision_siblings_from_package "$ORIGIN/..")
set(_vision_library_search_paths
    "${_vision_siblings_from_lib}/holoscan/lib"
    "${_vision_siblings_from_lib}/cvcuda_cu13.libs"
    "${_vision_siblings_from_lib}/nvidia/cu13/lib")
string(JOIN ":" _vision_library_rpath "$ORIGIN" ${_vision_library_search_paths})
holoscan_add_python_wheel_component(
    MODULE vision
    PACKAGE holoscan_vision
    PYTHON_DIRECTORY "${_vision_package}"
    PYTHON_FILES ${_vision_wheel_python_files}
    EXTENSION_TARGET holoscan_vision_python
    NATIVE_TARGETS vision
    LIBRARY_RUNPATH ${_vision_library_search_paths}
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
    # CV-CUDA's provider uses these hashed SONAMEs. Replacing DT_NEEDED with
    # the same plain name keeps ELF version-need filenames consistent with
    # the loaded library's identity, and excludes the system's unhashed copy.
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
    # NPP preserves its SONAMEs, so no --replace-needed is required. Use RPATH
    # rather than RUNPATH to prefer the wheel over CUDA's LD_LIBRARY_PATH.
    # Keep DT_NEEDED as SONAMEs: replacing them with $ORIGIN paths also changes
    # .gnu.version_r filenames, which glibc's version lookup does not expand.
    # That rewrite triggers a loader assertion with patchelf 0.18.0 and 0.19.1.
    # libnppicc has its own RUNPATH, which prevents our RPATH from controlling
    # its indirect libnppc dependency. Make libnppc direct so both are resolved
    # from Vision's wheel-relative paths before visiting NPP's dependencies.
    execute_process(
      COMMAND "${_holoscan_patchelf}" --force-rpath
        --set-rpath "@_vision_library_rpath@" --add-needed libnppc.so.13 "${library}"
      RESULT_VARIABLE result ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Vision wheel NPP loader patch failed: ${error}")
    endif()
    message(STATUS "Vision wheel: prefer nvidia/cu13/lib for NPP and its core dependency")
  ]=] _vision_runtime_code @ONLY)
install(CODE "${_vision_runtime_code}"
    COMPONENT holoscan-wheel-vision EXCLUDE_FROM_ALL)
