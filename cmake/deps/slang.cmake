# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Slang is a build-time toolchain dependency only: HoloViz compiles its built-in
# shaders to SPIR-V with slangc while building, embeds the results into generated
# headers, and links no Slang runtime library. This recipe resolves a slangc
# executable and records its path in the HOLOVIZ_SLANGC_EXECUTABLE global property.
#
# The recipe finds; it never acquires. The SDK build container installs the pinned
# upstream prebuilt release under /opt/slang and puts its bin/ on the PATH
# (public/Dockerfile, SLANG_VERSION). Resolution, first match wins:
#   1. HOLOVIZ_SLANGC (explicit executable) or slang_ROOT (explicit prefix).
#      These state intent, so a missing, non-runnable or wrong-version slangc is
#      a configure error.
#   2. An ambient slangc from CMAKE_PREFIX_PATH <prefix>/bin or the PATH. It is
#      used when it runs and reports the pinned version; otherwise configure fails
#      and names the rejected candidate.
# Whatever this recipe records must report exactly ${HOLOSCAN_SLANG_VERSION}: the
# generated SPIR-V and reflection are part of the build's reproducibility. Keep the
# pin in lockstep with SLANG_VERSION in public/Dockerfile.

set(HOLOSCAN_SLANG_VERSION 2026.14)
set(HOLOVIZ_SLANGC "" CACHE FILEPATH
  "slangc executable used to compile HoloViz shaders (empty: search CMAKE_PREFIX_PATH and PATH)")

get_property(_holoviz_slangc_recorded GLOBAL PROPERTY HOLOVIZ_SLANGC_EXECUTABLE)
if(_holoviz_slangc_recorded)
  unset(_holoviz_slangc_recorded)
  return()
endif()
unset(_holoviz_slangc_recorded)

# Run `slangc -v` to prove the candidate is executable on this host and to read
# its version. On success OUT_OK is TRUE; otherwise OUT_REASON explains why.
function(_holoviz_slang_probe_slangc slangc out_ok out_reason)
  execute_process(
    COMMAND "${slangc}" -v
    RESULT_VARIABLE _probe_result
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    TIMEOUT 30
  )
  if(NOT _probe_result EQUAL 0)
    set(${out_ok} FALSE PARENT_SCOPE)
    set(${out_reason}
      "${slangc} could not be executed (exit status: ${_probe_result})${_probe_stderr}"
      PARENT_SCOPE)
    return()
  endif()
  # slangc prints the bare version (e.g. "2026.14") on stdout; older builds used stderr.
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _probe_version "${_probe_stdout}\n${_probe_stderr}")
  if(NOT _probe_version STREQUAL HOLOSCAN_SLANG_VERSION)
    set(${out_ok} FALSE PARENT_SCOPE)
    set(${out_reason}
      "${slangc} reports Slang ${_probe_version}; HoloViz pins ${HOLOSCAN_SLANG_VERSION}"
      PARENT_SCOPE)
    return()
  endif()
  set(${out_ok} TRUE PARENT_SCOPE)
  set(${out_reason} "" PARENT_SCOPE)
endfunction()

# 1. Explicit intent: an exact executable, or an install prefix.
if(HOLOVIZ_SLANGC)
  _holoviz_slang_probe_slangc("${HOLOVIZ_SLANGC}" _holoviz_slangc_ok _holoviz_slangc_reason)
  if(NOT _holoviz_slangc_ok)
    message(FATAL_ERROR "HOLOVIZ_SLANGC is not usable: ${_holoviz_slangc_reason}")
  endif()
  set_property(GLOBAL PROPERTY HOLOVIZ_SLANGC_EXECUTABLE "${HOLOVIZ_SLANGC}")
  unset(_holoviz_slangc_ok)
  unset(_holoviz_slangc_reason)
  return()
endif()

if(slang_ROOT)
  # A bare find_program does not consult <Package>_ROOT; search the prefix explicitly and
  # nowhere else so a stale ambient toolchain cannot shadow the requested one.
  unset(_holoviz_slangc_found CACHE)
  find_program(_holoviz_slangc_found slangc
    HINTS
        "${slang_ROOT}/bin"
        "${slang_ROOT}"
    NO_DEFAULT_PATH
  )
  if(NOT _holoviz_slangc_found)
    message(FATAL_ERROR
      "slang_ROOT=${slang_ROOT} does not contain a slangc executable (looked in bin/ and the root)")
  endif()
  _holoviz_slang_probe_slangc("${_holoviz_slangc_found}" _holoviz_slangc_ok _holoviz_slangc_reason)
  if(NOT _holoviz_slangc_ok)
    message(FATAL_ERROR "slangc from slang_ROOT is not usable: ${_holoviz_slangc_reason}")
  endif()
  set_property(GLOBAL PROPERTY HOLOVIZ_SLANGC_EXECUTABLE "${_holoviz_slangc_found}")
  unset(_holoviz_slangc_found CACHE)
  unset(_holoviz_slangc_ok)
  unset(_holoviz_slangc_reason)
  return()
endif()

# 2. Ambient toolchain: CMAKE_PREFIX_PATH <prefix>/bin directories, then the PATH.
# A scratch result variable keeps HOLOVIZ_SLANGC user-owned and re-runs the search
# on every configure instead of pinning whatever an earlier configure found.
unset(_holoviz_slangc_found CACHE)
find_program(_holoviz_slangc_found slangc)
if(_holoviz_slangc_found)
  _holoviz_slang_probe_slangc("${_holoviz_slangc_found}" _holoviz_slangc_ok _holoviz_slangc_reason)
  set(_holoviz_slangc_detail "found ${_holoviz_slangc_found} but ${_holoviz_slangc_reason}")
else()
  set(_holoviz_slangc_ok FALSE)
  set(_holoviz_slangc_detail "none found on CMAKE_PREFIX_PATH or PATH")
endif()
if(NOT _holoviz_slangc_ok)
  message(FATAL_ERROR
    "HoloViz needs slangc ${HOLOSCAN_SLANG_VERSION} to compile its shaders: "
    "${_holoviz_slangc_detail}. The SDK build container installs it under /opt/slang "
    "(public/Dockerfile); elsewhere set HOLOVIZ_SLANGC=<slangc> or slang_ROOT=<prefix>.")
endif()
message(STATUS "HoloViz: using slangc ${HOLOSCAN_SLANG_VERSION} from ${_holoviz_slangc_found}")
set_property(GLOBAL PROPERTY HOLOVIZ_SLANGC_EXECUTABLE "${_holoviz_slangc_found}")
unset(_holoviz_slangc_found CACHE)
unset(_holoviz_slangc_ok)
unset(_holoviz_slangc_reason)
unset(_holoviz_slangc_detail)
