# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MESSAGE_CONTEXT "python_deps")

find_package(Python3 3.12 REQUIRED COMPONENTS
  Interpreter
  Development.Module
  Development.SABIModule
)

# nanobind reads FindPython-style variables from its helper
# functions even when callers use FindPython3. Mirror the discovered Python3
# values so nanobind receives the include directory and ABI metadata when the
# helper functions are later called from this directory.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
set(Python_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
set(Python_INTERPRETER_ID "${Python3_INTERPRETER_ID}")
set(Python_SOABI "${Python3_SOABI}")
set(Python_SOSABI "${Python3_SOSABI}")
set(Python_VERSION "${Python3_VERSION}")

# Resolve the versionless Python::SABIModule target so nanobind builds a stable-ABI
# (limited-API / abi3) extension. Python 3.12 is both nanobind's minimum for STABLE_ABI and the
# Holoscan SDK 5.x wheel ABI floor. The produced extension and wheel use these identities:
#   - extension -> _core.abi3.so
#   - wheel     -> cp312-abi3-<platform>
# The bare "_core.abi3.so" name (no <plat>, no cpXY) is just CPython's limited-API import-suffix
# convention -- the binary is still platform/arch-specific native code. abi3 only removes the
# Python-version axis; the platform and the 3.12 minimum live in the wheel tag (cp312-abi3-<platform>),
# not the filename, so wheels remain per OS/arch.
# Python_EXECUTABLE (set above from the Python3 find) pins this to the same interpreter.
find_package(Python 3.12 REQUIRED COMPONENTS
  Interpreter
  Development.Module
  Development.SABIModule
)

# During dependency acquisition, avoid finding incomplete package config files
# from the active build tree before the dependency's imported targets have been
# generated. Restore the caller's setting after Python dependency acquisition.
set(_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}")
list(APPEND CMAKE_IGNORE_PREFIX_PATH "${holoscan_BINARY_DIR}")

rapids_cpm_init()

set(_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")

holoscan_add_dependency(nanobind COMPONENT holoscan-dependencies)

set(CMAKE_CXX_CLANG_TIDY "${_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY}")
unset(_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY)

set(CMAKE_IGNORE_PREFIX_PATH "${_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH}")
unset(_HOLOSCAN_PYTHON_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
