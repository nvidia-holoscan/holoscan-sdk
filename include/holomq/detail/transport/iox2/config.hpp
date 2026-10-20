/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/limits.hpp>

// MSVC may resolve iceoryx2's quoted "iox2/config.hpp" through a parent include directory
// to this wrapper. Load the dependency through its include path before that can happen.
#include <iox2/config.hpp>

namespace holomq::detail::transport {

using holomq::k_lossless_flow;
using holomq::k_max_in_flight;
using holomq::k_publisher_history_depth;
using holomq::k_subscriber_queue_depth;

}  // namespace holomq::detail::transport
