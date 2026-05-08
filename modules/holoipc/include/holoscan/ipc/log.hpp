/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Logging macros for the holoscan::ipc library.
// Forwards to the Holoscan logger (HOLOSCAN_LOG_*) so IPC logs use the same backend.
// Use fmt-style format strings, e.g.:
//   HOLOSCAN_IPC_LOG_ERROR("{}: DomainParticipant is null", __FUNCTION__);
//   HOLOSCAN_IPC_LOG_DEBUG("key={} type={}", key, static_cast<int>(message_type));

#ifndef HOLOSCAN_IPC_LOG_HPP
#define HOLOSCAN_IPC_LOG_HPP

// Ensure Holoscan log level is defined so logger.hpp can be included.
// Override before including this header if needed: 0=TRACE … 5=CRITICAL (see holoscan logger).
#ifndef HOLOSCAN_LOG_ACTIVE_LEVEL
#define HOLOSCAN_LOG_ACTIVE_LEVEL 2  // INFO when Holoscan did not define a level first
#endif

#include <holoscan/logger/logger.hpp>

/// @brief IPC log macros that forward to the Holoscan logger (fmt-style format).
/// The first argument of each macro must be a **string literal** so it can be concatenated with
/// the "[holoscan::ipc] " prefix at preprocess time. Example:
///   HOLOSCAN_IPC_LOG_ERROR("{}: DomainParticipant is null", __FUNCTION__);
#define HOLOSCAN_IPC_LOG_TRACE(...) HOLOSCAN_LOG_TRACE("[holoscan::ipc] " __VA_ARGS__)
#define HOLOSCAN_IPC_LOG_DEBUG(...) HOLOSCAN_LOG_DEBUG("[holoscan::ipc] " __VA_ARGS__)
#define HOLOSCAN_IPC_LOG_INFO(...) HOLOSCAN_LOG_INFO("[holoscan::ipc] " __VA_ARGS__)
#define HOLOSCAN_IPC_LOG_WARN(...) HOLOSCAN_LOG_WARN("[holoscan::ipc] " __VA_ARGS__)
#define HOLOSCAN_IPC_LOG_ERROR(...) HOLOSCAN_LOG_ERROR("[holoscan::ipc] " __VA_ARGS__)
#define HOLOSCAN_IPC_LOG_CRITICAL(...) HOLOSCAN_LOG_CRITICAL("[holoscan::ipc] " __VA_ARGS__)

#endif  // HOLOSCAN_IPC_LOG_HPP
