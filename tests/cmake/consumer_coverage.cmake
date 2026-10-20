# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Declare packaging/relocation consumer objects ineligible for coverage collection. The tests
# still execute, and instrumented SDK libraries still write counters in the main build tree.
# A directory-local marker avoids a shared manifest write race between parallel fixtures.
function(holoscan_exclude_consumer_coverage build_root consumer_build_dir)
    if(NOT IS_ABSOLUTE "${build_root}" OR NOT IS_ABSOLUTE "${consumer_build_dir}")
        message(FATAL_ERROR "Consumer coverage directories must be absolute")
    endif()
    file(REAL_PATH "${build_root}" coverage_root)
    file(REAL_PATH "${consumer_build_dir}" consumer_dir)
    file(RELATIVE_PATH consumer_relative "${coverage_root}" "${consumer_dir}")
    if(consumer_relative STREQUAL "" OR consumer_relative STREQUAL "."
            OR consumer_relative MATCHES "^\\.\\.(/|$)" OR IS_ABSOLUTE "${consumer_relative}")
        message(FATAL_ERROR "Consumer coverage exclusion must be strictly inside the SDK build")
    endif()
    file(MAKE_DIRECTORY "${consumer_dir}")
    file(WRITE "${consumer_dir}/.holoscan-coverage-exclude"
        "# Exclude this temporary installed-consumer build from gcovr collection.\n"
        "# Packaging/relocation tests still run; main SDK library coverage remains included.\n")
endfunction()
