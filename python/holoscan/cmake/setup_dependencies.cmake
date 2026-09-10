# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MESSAGE_CONTEXT "python_deps")

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind 2.12 still reads FindPython-style variables from its helper
# functions even when callers use FindPython3. Mirror the discovered Python3
# values so nanobind receives the include directory and ABI metadata when the
# helper functions are later called from this directory.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
set(Python_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
set(Python_INTERPRETER_ID "${Python3_INTERPRETER_ID}")
set(Python_SOABI "${Python3_SOABI}")
set(Python_SOSABI "${Python3_SOSABI}")
set(Python_VERSION "${Python3_VERSION}")

# On Python >= 3.12, additionally resolve the versionless Python::SABIModule target so nanobind
# builds a stable-ABI (limited-API / abi3) extension. nanobind's STABLE_ABI flag is version-gated by
# nanobind itself, so it stays unconditional on nanobind_add_module; this find is the only
# version-dependent piece. Effect on the produced extension and the wheel's Python/ABI tag:
#   - Python 3.10  -> _core.cpython-310-<plat>.so  (cp310 wheel)
#   - Python 3.11  -> _core.cpython-311-<plat>.so  (cp311 wheel)
#   - Python 3.12+ -> _core.abi3.so                (cp312-abi3 wheel; loads on 3.12/3.13/3.14+)
# The bare "_core.abi3.so" name (no <plat>, no cpXY) is just CPython's limited-API import-suffix
# convention -- the binary is still platform/arch-specific native code. abi3 only removes the
# Python-version axis; the platform and the 3.12 minimum live in the wheel tag (cp312-abi3-<platform>),
# not the filename, so wheels remain per OS/arch.
# Python_EXECUTABLE (set above from the Python3 find) pins this to the same interpreter.
if(Python3_VERSION VERSION_GREATER_EQUAL 3.12)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
endif()

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
