# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_PUBLIC_SOURCE_DIR OR
   "${HOLOSCAN_PUBLIC_SOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR "HOLOSCAN_PUBLIC_SOURCE_DIR is required")
endif()

file(GLOB experimental_entries LIST_DIRECTORIES true
  "${HOLOSCAN_PUBLIC_SOURCE_DIR}/examples/experimental/*")
set(experimental_count 0)
foreach(entry IN LISTS experimental_entries)
  if(NOT IS_DIRECTORY "${entry}")
    continue()
  endif()
  math(EXPR experimental_count "${experimental_count} + 1")
  set(readme "${entry}/README.md")
  if(NOT EXISTS "${readme}")
    message(FATAL_ERROR "Experimental example is missing README.md: ${entry}")
  endif()
  file(READ "${readme}" opening LIMIT 512)
  string(FIND "${opening}" "> **Developer preview" developer_preview)
  string(FIND "${opening}" "> **Experimental" experimental)
  if((developer_preview LESS 0 OR developer_preview GREATER 256) AND
     (experimental LESS 0 OR experimental GREATER 256))
    message(FATAL_ERROR
      "Experimental example README does not open with a maturity banner: ${readme}")
  endif()
endforeach()
if(experimental_count EQUAL 0)
  message(FATAL_ERROR "No experimental examples were inspected")
endif()

set(essential_readme
  "${HOLOSCAN_PUBLIC_SOURCE_DIR}/examples/essential/config_file/README.md")
if(NOT EXISTS "${essential_readme}")
  message(FATAL_ERROR "Essential example README is missing: ${essential_readme}")
endif()
file(READ "${essential_readme}" essential_opening LIMIT 512)
string(FIND "${essential_opening}" "> **Essential module.**" essential_banner)
if(essential_banner LESS 0 OR essential_banner GREATER 256)
  message(FATAL_ERROR
    "Essential example README does not open with its module-status banner: ${essential_readme}")
endif()
