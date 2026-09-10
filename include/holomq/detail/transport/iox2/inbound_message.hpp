/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "iox2/sample.hpp"
#include "iox2/service_type.hpp"

#include <utility>

namespace holomq::detail::transport {

/// Received message for the iceoryx2 transport backend; released when destroyed.
template <typename Payload>
class InboundMessage {
 public:
  InboundMessage() = default;
  InboundMessage(InboundMessage&&) noexcept = default;
  InboundMessage& operator=(InboundMessage&&) noexcept = default;
  ~InboundMessage() = default;

  InboundMessage(const InboundMessage&) = delete;
  InboundMessage& operator=(const InboundMessage&) = delete;

  [[nodiscard]] const Payload& payload() const noexcept { return inner_.payload(); }

 private:
  template <typename P>
  friend class SubscriberPort;

  using IoxSample = iox2::Sample<iox2::ServiceType::Ipc, Payload, void>;

  explicit InboundMessage(IoxSample inner) noexcept : inner_(std::move(inner)) {}

  IoxSample inner_;
};

}  // namespace holomq::detail::transport
