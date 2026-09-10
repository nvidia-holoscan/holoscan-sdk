/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/logger.hpp>

namespace holomq {

#if defined(HOLOMQ_TRANSPORT_IOX2)
void sync_iox2_log_level(LogLevel level) noexcept;
bool install_iox2_log_bridge() noexcept;
bool transport_logging_enabled() noexcept;
#endif

}  // namespace holomq
