# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prove the SDK's install rules name their sources in the SDK, not in whatever project happens to
# be the top of the build.
#
# Configuring as a subproject does not exercise this. An install rule written against
# CMAKE_SOURCE_DIR configures cleanly, because CMAKE_SOURCE_DIR is a valid directory either way; it
# is simply the parent's. The rule only misbehaves at install time, and then either fails on a
# missing directory or installs nothing at all. The generated script is the first artifact where
# the two roots are distinguishable, so it is what gets inspected here.

foreach(required_variable IN ITEMS
        HOLOSCAN_INSTALL_SCRIPT
        HOLOSCAN_SOURCE_DIR
        HOLOSCAN_SUBPROJECT_SOURCE_DIR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

if(NOT EXISTS "${HOLOSCAN_INSTALL_SCRIPT}")
    message(FATAL_ERROR "No generated install script at ${HOLOSCAN_INSTALL_SCRIPT}")
endif()
file(READ "${HOLOSCAN_INSTALL_SCRIPT}" holoscan_install_script_contents)

# The fixture contributes no headers, libraries, or data of its own, so every install source that
# resolves inside it is an SDK rule that reached for the parent root. Matching the parent directory
# rather than any one component keeps this sensitive to the next rule written the same way.
string(REGEX REPLACE "([][+.*()^$?|\\\\])" "\\\\\\1"
    holoscan_subproject_source_pattern "${HOLOSCAN_SUBPROJECT_SOURCE_DIR}")
string(REGEX MATCHALL "\"${holoscan_subproject_source_pattern}/[^\"]*\""
    holoscan_leaked_install_sources "${holoscan_install_script_contents}")
if(holoscan_leaked_install_sources)
    list(REMOVE_DUPLICATES holoscan_leaked_install_sources)
    string(REPLACE ";" "\n  " holoscan_leaked_install_sources_text
        "${holoscan_leaked_install_sources}")
    message(FATAL_ERROR
        "Holoscan install rules name sources inside the parent project instead of the SDK source "
        "tree. Use holoscan_SOURCE_DIR rather than CMAKE_SOURCE_DIR:\n  "
        "${holoscan_leaked_install_sources_text}"
    )
endif()

# State the expectation positively as well, so the check fails if the rule disappears rather than
# quietly passing on an install script that no longer carries the headers at all. Sensor I/O is
# named because it installs a directory of its own for a separately exported library, so a wrong
# root there omits that library's entire public interface rather than one stray file.
set(holoscan_sensor_io_install_source "${HOLOSCAN_SOURCE_DIR}/include/holoscan/sensor_io")
string(FIND "${holoscan_install_script_contents}" "\"${holoscan_sensor_io_install_source}\""
    holoscan_sensor_io_install_source_position)
if(holoscan_sensor_io_install_source_position EQUAL -1)
    message(FATAL_ERROR
        "Generated install script does not install the sensor I/O headers from "
        "${holoscan_sensor_io_install_source}"
    )
endif()
