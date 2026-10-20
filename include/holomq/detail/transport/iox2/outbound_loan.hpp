/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/detail/transport/iox2/iox_payload.hpp>

#include <iox2/sample_mut_uninit.hpp>
#include <iox2/service_type.hpp>

#include <utility>

namespace holomq::detail::transport {

template <typename Payload>
class PublisherPort;

/// Loaned outbound message storage for the iceoryx2 transport backend.
template <typename Payload>
class OutboundLoan {
 public:
  OutboundLoan() = default;
  OutboundLoan(OutboundLoan&&) noexcept = default;
  OutboundLoan& operator=(OutboundLoan&&) noexcept = default;
  ~OutboundLoan() = default;

  OutboundLoan(const OutboundLoan&) = delete;
  OutboundLoan& operator=(const OutboundLoan&) = delete;

  [[nodiscard]] holomq::MutablePayloadView<Payload> payload() noexcept {
    if constexpr (holomq::kIsByteSlice<Payload>) {
      return as_bytes(inner_.payload_mut());
    } else {
      return inner_.payload_mut();
    }
  }

  [[nodiscard]] holomq::ConstPayloadView<Payload> payload() const noexcept {
    if constexpr (holomq::kIsByteSlice<Payload>) {
      return as_bytes(inner_.payload());
    } else {
      return inner_.payload();
    }
  }

  /// Return the payload start, which the backend reports again when it releases the sample.
  [[nodiscard]] const void* release_key() const noexcept {
    if constexpr (holomq::kIsByteSlice<Payload>) {
      return static_cast<const void*>(payload().data());
    } else {
      return static_cast<const void*>(&payload());
    }
  }

 private:
  friend class PublisherPort<Payload>;

  using IoxSampleMutUninit =
      iox2::SampleMutUninit<iox2::ServiceType::Ipc, IoxPayload<Payload>, void>;

  explicit OutboundLoan(IoxSampleMutUninit inner) noexcept : inner_(std::move(inner)) {}

  [[nodiscard]] IoxSampleMutUninit release() && noexcept { return std::move(inner_); }

  IoxSampleMutUninit inner_;
};

}  // namespace holomq::detail::transport
