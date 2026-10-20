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
  set_property(GLOBAL PROPERTY "HOLOSCAN_PYTHON_WHEEL_PACKAGE_${ARG_MODULE}" "${ARG_PACKAGE}")
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
  _holoscan_add_python_wheel_extension("${component}" "${ARG_PACKAGE}"
    "${ARG_EXTENSION_TARGET}" "${ARG_EXTENSION_DIRECTORY}" ${ARG_EXTENSION_RUNPATH})
  set(library_runpath "$ORIGIN" ${ARG_LIBRARY_RUNPATH})
  list(JOIN library_runpath ":" library_runpath)

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

# Add another native extension, such as a generated schema codec, to a wheel component that
# holoscan_add_python_wheel_component already declared.
#
# Required arguments:
#   MODULE: the value passed to holoscan_add_python_wheel_component; the extension joins that
#     component's PACKAGE.
#   TARGET: existing native extension target to copy into the package.
# Optional arguments:
#   DIRECTORY: package-relative directory for the extension; defaults to PACKAGE.
#   RUNPATH: entries appended to the extension's relative path to PACKAGE/lib.
function(holoscan_add_python_wheel_extension)
  set(options "")
  set(one_value_keywords MODULE TARGET DIRECTORY)
  set(multi_value_keywords RUNPATH)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${one_value_keywords}"
    "${multi_value_keywords}")
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid Python wheel extension arguments")
  endif()
  foreach(key IN ITEMS MODULE TARGET)
    if(NOT ARG_${key})
      message(FATAL_ERROR "Python wheel extension requires ${key}")
    endif()
  endforeach()
  set(component "holoscan-wheel-${ARG_MODULE}")
  get_property(components GLOBAL PROPERTY HOLOSCAN_PYTHON_WHEEL_COMPONENTS)
  if(NOT component IN_LIST components)
    message(FATAL_ERROR
      "Declare ${component} with holoscan_add_python_wheel_component before adding extensions")
  endif()
  if(NOT TARGET "${ARG_TARGET}")
    message(FATAL_ERROR "Unknown Python wheel extension target: ${ARG_TARGET}")
  endif()
  get_property(package GLOBAL PROPERTY "HOLOSCAN_PYTHON_WHEEL_PACKAGE_${ARG_MODULE}")
  _holoscan_add_python_wheel_extension(
    "${component}" "${package}" "${ARG_TARGET}" "${ARG_DIRECTORY}" ${ARG_RUNPATH})
endfunction()

# Install <target> in <package>/<directory> and set its RUNPATH once: its relative path to
# <package>/lib, then any further entries.
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
  set(runpath "$ORIGIN/${relative_lib}" ${ARGN})
  list(JOIN runpath ":" runpath)
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

function(_holoscan_write_if_different path content)
  if(EXISTS "${path}")
    file(READ "${path}" existing)
    if(existing STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE "${path}" "${content}")
endfunction()

# Install the minimal holoscan_<MODULE>_cu13-<PROJECT_VERSION>.dist-info through which
# Core exposes holoscan_<MODULE> as holoscan.<MODULE> in an ordinary CMake install.
# Wheels generate this metadata from their pyproject; native installs do not run the
# wheel backend. The distribution matches the module wheel, holoscan-<MODULE>-cu13:
# importlib.metadata reads entry points from only the first distribution of a given
# name on the path, so a co-installed wheel does not register a second provider.
# The files join holoscan-<MODULE>-python_libs under HOLOSCAN_PYTHON_INSTALL_DIR.
# The build tree gets the same metadata in HOLOSCAN_PYTHON_MODULE_ROOT, next to the
# provider package, so build-tree imports and tests resolve holoscan.<MODULE> as an
# install does. That copy omits the version from its directory name, so a version
# change rewrites it in place instead of leaving a stale distribution behind.
#
# Required arguments:
#   MODULE: public module name under holoscan (for example, vision).
# Optional arguments:
#   REQUIRES: Requires-Dist entries for METADATA.
function(holoscan_install_python_module_metadata)
  set(options "")
  set(one_value_keywords MODULE)
  set(multi_value_keywords REQUIRES)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${one_value_keywords}"
    "${multi_value_keywords}")
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid Python module metadata arguments")
  endif()
  if(NOT ARG_MODULE MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid Python module name: ${ARG_MODULE}")
  endif()
  if(NOT PROJECT_VERSION MATCHES "^[A-Za-z0-9.+!]+$")
    message(FATAL_ERROR "Python module metadata requires PROJECT_VERSION")
  endif()
  foreach(variable IN ITEMS HOLOSCAN_PYTHON_INSTALL_DIR HOLOSCAN_PYTHON_MODULE_ROOT)
    if(NOT ${variable})
      message(FATAL_ERROR "Python module metadata requires ${variable}")
    endif()
  endforeach()
  get_property(modules GLOBAL PROPERTY HOLOSCAN_PYTHON_MODULE_METADATA)
  if(ARG_MODULE IN_LIST modules)
    message(FATAL_ERROR "Duplicate Python module metadata: holoscan.${ARG_MODULE}")
  endif()
  set_property(GLOBAL APPEND PROPERTY HOLOSCAN_PYTHON_MODULE_METADATA "${ARG_MODULE}")

  string(REPLACE "_" "-" distribution "holoscan-${ARG_MODULE}-cu13")
  set(stem "holoscan_${ARG_MODULE}_cu13")
  string(TOLOWER "${stem}-${PROJECT_VERSION}.dist-info" dist_info)
  string(CONCAT metadata "Metadata-Version: 2.1\nName: ${distribution}\n"
    "Version: ${PROJECT_VERSION}\nRequires-Python: >=3.12\n")
  foreach(requirement IN LISTS ARG_REQUIRES)
    string(APPEND metadata "Requires-Dist: ${requirement}\n")
  endforeach()
  set(output "${HOLOSCAN_PYTHON_MODULE_ROOT}/${stem}.dist-info")
  _holoscan_write_if_different("${output}/METADATA" "${metadata}")
  _holoscan_write_if_different("${output}/entry_points.txt"
    "[holoscan.modules]\n${ARG_MODULE} = holoscan_${ARG_MODULE}\n")
  install(FILES "${output}/METADATA" "${output}/entry_points.txt"
    DESTINATION "${HOLOSCAN_PYTHON_INSTALL_DIR}/${dist_info}"
    COMPONENT "holoscan-${ARG_MODULE}-python_libs")
endfunction()
