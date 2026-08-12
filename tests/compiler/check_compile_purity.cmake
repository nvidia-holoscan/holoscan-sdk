# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_SOURCE_DIR)
  message(FATAL_ERROR "HOLOSCAN_SOURCE_DIR is required")
endif()

# compile() and immutable plan inspection are pure cold operations. Scan the
# complete compiler-owned implementation closure for APIs that would create
# threads, mutate scheduling/affinity, or write files. artifact_emitter.cpp is
# deliberately excluded: it implements the separately invoked explicit cold
# publication API and is not reachable from compile().
file(GLOB_RECURSE _compiler_sources LIST_DIRECTORIES FALSE
  "${HOLOSCAN_SOURCE_DIR}/src/compiler/*.cpp"
  "${HOLOSCAN_SOURCE_DIR}/src/compiler/*.hpp"
)
list(FILTER _compiler_sources EXCLUDE REGEX "/artifact_emitter\\.cpp$")
list(APPEND _compiler_sources
  "${HOLOSCAN_SOURCE_DIR}/src/core/execution_plan.cpp"
)

set(_forbidden_tokens
  "std::thread"
  "std::jthread"
  "std::async"
  "pthread_create"
  "CreateThread"
  "_beginthread"
  "sched_setaffinity"
  "pthread_setaffinity_np"
  "SetThreadAffinityMask"
  "SetProcessAffinityMask"
  "sched_setscheduler"
  "pthread_setschedparam"
  "setpriority"
  "SetThreadPriority"
  "SetPriorityClass"
  "std::ofstream"
  "std::fstream"
  "std::filesystem::create_"
  "std::filesystem::remove"
  "std::filesystem::rename"
  "std::filesystem::copy"
  "std::filesystem::permissions"
  "std::filesystem::resize_file"
  "std::filesystem::current_path("
)

foreach(_source IN LISTS _compiler_sources)
  file(READ "${_source}" _contents)
  foreach(_token IN LISTS _forbidden_tokens)
    string(FIND "${_contents}" "${_token}" _position)
    if(NOT _position EQUAL -1)
      file(RELATIVE_PATH _relative "${HOLOSCAN_SOURCE_DIR}" "${_source}")
      message(FATAL_ERROR
        "compile purity violation: ${_relative} contains forbidden token '${_token}'"
      )
    endif()
  endforeach()
endforeach()
