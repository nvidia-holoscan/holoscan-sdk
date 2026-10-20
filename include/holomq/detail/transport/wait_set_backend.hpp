/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <holomq/detail/transport/iox2/wait_set_backend.hpp>
#else
#error "Unknown HOLOMQ_TRANSPORT backend"
#error "Define HOLOMQ_TRANSPORT_IOX2"
#endif

namespace holomq::detail::transport {

class Runtime;

using WaitSetBackend = Iox2WaitSetBackend;

[[nodiscard]] auto create_wait_set_backend(Runtime& runtime)
    -> holomq::expected<WaitSetBackend, holomq::Error>;

}  // namespace holomq::detail::transport
