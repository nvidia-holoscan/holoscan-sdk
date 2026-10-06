# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Declare wheel-layout install rules; scikit-build-core later assembles the wheel.
# This call registers holoscan-wheel-<MODULE>, excluded from ordinary SDK installs.
# It does not compile targets, generate typing files, or create a wheel archive.
#
# Required arguments:
#   MODULE: unique lowercase component key (for example, core).
#   PACKAGE: top-level Python package directory (for example, holoscan).
#   PYTHON_DIRECTORY: directory containing the completed Python/typing payload.
#   EXTENSION_TARGET: existing native extension target to copy into the package.
# Optional arguments:
#   PYTHON_FILES: paths relative to PYTHON_DIRECTORY, retained relative to PACKAGE.
#   EXTENSION_DIRECTORY: package-relative extension directory; defaults to PACKAGE.
#   NATIVE_TARGETS: owned runtime targets; shared libraries are copied by SONAME,
#     or by filename for IMPORTED_NO_SONAME targets. Static/interface targets add
#     no payload. This does not enable packaging a statically linked SDK.
#   NATIVE_FILES: absolute paths to additional owned, non-driver runtime files.
#   LIBRARY_RUNPATH: entries appended to the default $ORIGIN for copied libraries.
#   EXTENSION_RUNPATH: entries appended to the extension's relative path to lib/.
#
# The install prefix is the wheel's Python import root: PACKAGE contains Python
# files and the extension; PACKAGE/lib contains owned runtimes. Payloads must
# exist before installation. patchelf changes only the wheel copies. See
# python/packaging/README.md for the layout and shared-SDK packaging requirement.
function(holoscan_add_python_wheel_component)
  set(options "")
  set(one_value_keywords MODULE PACKAGE PYTHON_DIRECTORY EXTENSION_TARGET EXTENSION_DIRECTORY)
  set(multi_value_keywords PYTHON_FILES NATIVE_TARGETS NATIVE_FILES LIBRARY_RUNPATH EXTENSION_RUNPATH)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${one_value_keywords}"
    "${multi_value_keywords}")
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid Python wheel component arguments")
  endif()
  foreach(key IN ITEMS MODULE PACKAGE PYTHON_DIRECTORY EXTENSION_TARGET)
    if(NOT ARG_${key})
      message(FATAL_ERROR "Python wheel component requires ${key}")
    endif()
  endforeach()
  if(NOT ARG_MODULE MATCHES "^[a-z][a-z0-9_]*$"
      OR NOT ARG_PACKAGE MATCHES "^[a-zA-Z_][a-zA-Z0-9_]*$")
    message(FATAL_ERROR "Invalid wheel module or package name")
  endif()
  set(component "holoscan-wheel-${ARG_MODULE}")
  get_property(components GLOBAL PROPERTY HOLOSCAN_PYTHON_WHEEL_COMPONENTS)
  if(component IN_LIST components)
    message(FATAL_ERROR "Duplicate wheel component: ${component}")
  endif()
  list(APPEND components "${component}")
  set_property(GLOBAL PROPERTY HOLOSCAN_PYTHON_WHEEL_COMPONENTS "${components}")
  set(HOLOSCAN_PYTHON_WHEEL_COMPONENTS "${components}" CACHE INTERNAL
    "Wheel components available in this native build" FORCE)

  foreach(path IN LISTS ARG_PYTHON_FILES)
    if(IS_ABSOLUTE "${path}" OR path MATCHES "(^|/)\\.\\.?(/|$)")
      message(FATAL_ERROR "Python wheel files must be package-relative: ${path}")
    endif()
    get_filename_component(directory "${path}" DIRECTORY)
    install(FILES "${ARG_PYTHON_DIRECTORY}/${path}"
      DESTINATION "${ARG_PACKAGE}/${directory}" COMPONENT "${component}" EXCLUDE_FROM_ALL)
  endforeach()
  if(IS_ABSOLUTE "${ARG_EXTENSION_DIRECTORY}"
      OR ARG_EXTENSION_DIRECTORY MATCHES "(^|/)\\.\\.?(/|$)")
    message(FATAL_ERROR "Wheel extension directory must be package-relative")
  endif()
  set(extension_directory "${ARG_PACKAGE}")
  if(ARG_EXTENSION_DIRECTORY)
    string(APPEND extension_directory "/${ARG_EXTENSION_DIRECTORY}")
  endif()
  _holoscan_add_python_wheel_extension(
    "${component}" "${ARG_PACKAGE}" "${ARG_EXTENSION_TARGET}" "${ARG_EXTENSION_DIRECTORY}")
  file(RELATIVE_PATH relative_lib "/${extension_directory}" "/${ARG_PACKAGE}/lib")
  set(library_runpath "$ORIGIN" ${ARG_LIBRARY_RUNPATH})
  list(JOIN library_runpath ":" library_runpath)
  if(ARG_EXTENSION_RUNPATH)
    set(extension_runpath "$ORIGIN/${relative_lib}" ${ARG_EXTENSION_RUNPATH})
    list(JOIN extension_runpath ":" extension_runpath)
    _holoscan_set_wheel_runpath("${component}"
      "${extension_directory}/$<TARGET_FILE_NAME:${ARG_EXTENSION_TARGET}>" "${extension_runpath}")
  endif()

  foreach(target IN LISTS ARG_NATIVE_TARGETS)
    get_target_property(type "${target}" TYPE)
    if(type STREQUAL "SHARED_LIBRARY")
      get_target_property(no_soname "${target}" IMPORTED_NO_SONAME)
      if(no_soname)
        set(name "$<TARGET_FILE_NAME:${target}>")
      else()
        set(name "$<TARGET_SONAME_FILE_NAME:${target}>")
      endif()
      install(FILES "$<TARGET_FILE:${target}>" DESTINATION "${ARG_PACKAGE}/lib"
        RENAME "${name}" COMPONENT "${component}" EXCLUDE_FROM_ALL)
      _holoscan_set_wheel_runpath("${component}" "${ARG_PACKAGE}/lib/${name}" "${library_runpath}")
    elseif(NOT type STREQUAL "STATIC_LIBRARY" AND NOT type STREQUAL "INTERFACE_LIBRARY")
      message(FATAL_ERROR "Unsupported wheel runtime target type: ${type}")
    endif()
  endforeach()
  foreach(path IN LISTS ARG_NATIVE_FILES)
    get_filename_component(name "${path}" NAME)
    if(NOT IS_ABSOLUTE "${path}" OR name MATCHES "^libcuda\\.so")
      message(FATAL_ERROR "Wheel runtime files must be absolute, non-driver library paths")
    endif()
    install(FILES "${path}" DESTINATION "${ARG_PACKAGE}/lib"
      COMPONENT "${component}" EXCLUDE_FROM_ALL)
    _holoscan_set_wheel_runpath("${component}" "${ARG_PACKAGE}/lib/${name}" "${library_runpath}")
  endforeach()
endfunction()

function(_holoscan_add_python_wheel_extension component package target directory)
  if(IS_ABSOLUTE "${directory}" OR directory MATCHES "(^|/)\\.\\.?(/|$)")
    message(FATAL_ERROR "Wheel extension directory must be package-relative")
  endif()
  set(destination "${package}")
  if(directory)
    string(APPEND destination "/${directory}")
  endif()
  install(FILES "$<TARGET_FILE:${target}>"
    DESTINATION "${destination}" COMPONENT "${component}" EXCLUDE_FROM_ALL)
  file(RELATIVE_PATH relative_lib "/${destination}" "/${package}/lib")
  set(runpath "$ORIGIN/${relative_lib}")
  _holoscan_set_wheel_runpath("${component}"
    "${destination}/$<TARGET_FILE_NAME:${target}>" "${runpath}")
endfunction()

function(_holoscan_set_wheel_runpath component path runpath)
  string(CONFIGURE [=[
    find_program(_holoscan_patchelf patchelf REQUIRED)
    execute_process(
      COMMAND "${_holoscan_patchelf}" --set-rpath "@runpath@"
        "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/@path@"
      RESULT_VARIABLE result ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Wheel patchelf failed: ${error}")
    endif()
  ]=] code @ONLY)
  install(CODE "${code}" COMPONENT "${component}" EXCLUDE_FROM_ALL)
endfunction()
