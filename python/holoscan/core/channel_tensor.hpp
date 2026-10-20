/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <vector>

#include <holoscan/authoring/channel_registry.hpp>
#include <holoscan/channel/channel.hpp>

namespace holoscan::python::detail {

namespace nb = nanobind;

[[nodiscard]] nb::object channel_tensor_loan(const authoring::RegisteredPayloadType& payload,
                                             authoring::OpaqueChannelPublisher& publisher,
                                             std::mutex& mutex,
                                             const std::vector<std::int64_t>& shape,
                                             nb::handle dtype, nb::handle strides);

[[nodiscard]] std::uint32_t emit_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelPublisher& publisher,
    std::mutex& mutex, nb::handle value, EmitOptions options,
    std::optional<std::chrono::nanoseconds> retry_timeout,
    std::optional<std::chrono::steady_clock::time_point> retry_deadline);

[[nodiscard]] std::uint32_t emit_captured_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelPublisher& publisher,
    std::mutex& mutex, nb::handle value, EmitOptions options);

[[nodiscard]] expected<authoring::ForeignTensorSample, Error> try_receive_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelSubscriber& subscriber,
    std::mutex& mutex);

[[nodiscard]] expected<authoring::ForeignTensorSample, Error> receive_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelSubscriber& subscriber,
    std::mutex& mutex, std::chrono::steady_clock::time_point deadline,
    const std::stop_token& token);

void init_channel_tensor(nb::module_& module);

}  // namespace holoscan::python::detail
