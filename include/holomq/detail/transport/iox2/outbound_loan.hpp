/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

  [[nodiscard]] Payload& payload() noexcept { return inner_.payload_mut(); }

  [[nodiscard]] const Payload& payload() const noexcept { return inner_.payload(); }

  [[nodiscard]] const void* release_key() const noexcept {
    return static_cast<const void*>(&payload());
  }

 private:
  friend class PublisherPort<Payload>;

  using IoxSampleMutUninit = iox2::SampleMutUninit<iox2::ServiceType::Ipc, Payload, void>;

  explicit OutboundLoan(IoxSampleMutUninit inner) noexcept : inner_(std::move(inner)) {}

  [[nodiscard]] IoxSampleMutUninit release() && noexcept { return std::move(inner_); }

  IoxSampleMutUninit inner_;
};

}  // namespace holomq::detail::transport
