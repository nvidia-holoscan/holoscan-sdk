/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/detail/transport/backend.hpp>

#include <utility>

namespace holomq {

template <typename Payload>
class Subscriber;

/// Move-only inbound message from \ref Subscriber::try_take or \ref Subscriber::wait_and_take.
///
/// Obtain only from subscriber take APIs (not default-constructible). Application-facing
/// \c holomq::InboundMessage; distinct from \c holomq::detail::transport::InboundMessage.
///
/// \note The transport sample hold is released when this object is destroyed. For \ref ByteSlice,
/// \ref payload() is exactly the published frame.
/// @warning When the payload contains a \ref BufferHandle, call \ref Context::import_buffer while
/// this message is still alive. Copying the handle does not extend the transport sample hold.
template <typename Payload>
class InboundMessage {
 public:
  InboundMessage(const InboundMessage&) = delete;
  InboundMessage& operator=(const InboundMessage&) = delete;

  InboundMessage(InboundMessage&&) noexcept = default;
  InboundMessage& operator=(InboundMessage&&) noexcept = default;
  ~InboundMessage() = default;

  [[nodiscard]] ConstPayloadView<Payload> payload() const noexcept { return inner_.payload(); }

 private:
  friend class Subscriber<Payload>;

  InboundMessage() = delete;

  explicit InboundMessage(detail::transport::InboundMessage<Payload> inner) noexcept
      : inner_(std::move(inner)) {}

  detail::transport::InboundMessage<Payload> inner_;
};

}  // namespace holomq
