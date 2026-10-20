# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Keep SDK logging thresholds private to each caller target so multi-config generators select the
# intended build policy without exporting that policy to installed consumers. Symbolic levels are
# resolved by logger.hpp in each caller translation unit.
function(holoscan_apply_logging_policy target_name)
    target_compile_definitions(${target_name} PRIVATE
        $<$<CONFIG:Debug>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_TRACE>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_DEBUG>
        $<$<CONFIG:MinSizeRel>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_INFO>
    )
endfunction()
