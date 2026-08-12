/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {
/** @brief Publisher endpoint for payload descriptor control-plane samples. */
class ChannelDescriptorPublisher {
 public:
  HOLOSCAN_CORE_EXPORT ChannelDescriptorPublisher(ChannelDescriptorPublisher&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ChannelDescriptorPublisher& operator=(
      ChannelDescriptorPublisher&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~ChannelDescriptorPublisher();

  ChannelDescriptorPublisher(const ChannelDescriptorPublisher&) = delete;
  ChannelDescriptorPublisher& operator=(const ChannelDescriptorPublisher&) = delete;

  /** @brief Publish a channel descriptor through the payload descriptor endpoint.
   * @param descriptor Channel descriptor to convert and send.
   * @return Recipient count or Error for a closed publisher, unavailable loan, or send failure. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> publish(
      const ChannelSampleDescriptor& descriptor);

  /// @brief Publish a public payload descriptor.
  /// @param payload_descriptor Descriptor-backed payload reference to send.
  /// @return Recipient count after accepted publication, or a typed error.
  /// @retval ErrorCode::kInvalidArgument The descriptor selects the inline-payload kind, which
  /// this reference-only endpoint cannot transport.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> publish(
      const PayloadDescriptor& payload_descriptor);

  /// @brief Publish a sample envelope through the reference-only endpoint.
  /// @param envelope Envelope whose payload descriptor is transported.
  /// @return Recipient count after accepted publication, or a typed error.
  /// @retval ErrorCode::kInvalidArgument The envelope selects the inline-payload kind.
  /// @note Temporal fields are not carried by this low-level endpoint; use typed `emit_sample()`
  /// APIs when temporal metadata is required.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> publish(
      const SampleEnvelope& envelope);

  /** @brief Refresh backend connection state.
   * @return Empty success or Error when closed or refresh fails. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> update_connections() noexcept;

  /** @brief Return whether at least one subscriber is connected.
   * @return True when connected; false otherwise. This function is noexcept. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool has_subscribers() const noexcept;

 private:
  friend class ChannelRuntime;

  class Impl;

  HOLOSCAN_CORE_EXPORT explicit ChannelDescriptorPublisher(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

/// Subscriber endpoint for payload descriptor control-plane samples.
class ChannelDescriptorSubscriber {
 public:
  HOLOSCAN_CORE_EXPORT ChannelDescriptorSubscriber(ChannelDescriptorSubscriber&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ChannelDescriptorSubscriber& operator=(
      ChannelDescriptorSubscriber&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~ChannelDescriptorSubscriber();

  ChannelDescriptorSubscriber(const ChannelDescriptorSubscriber&) = delete;
  ChannelDescriptorSubscriber& operator=(const ChannelDescriptorSubscriber&) = delete;

  /// Return whether the transport reports a pending sample without consuming it.
  ///
  /// @return The readiness state, or an error when the backend cannot query it.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<bool, Error> has_samples() const;

  /// Receive the next descriptor.
  ///
  /// @return The next descriptor, or a typed error.
  /// @retval ErrorCode::kInvalidArgument The received record selects the inline-payload kind.
  /// @retval ErrorCode::kNotReady No descriptor is queued.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelSampleDescriptor, Error> receive();

  /// Receive the next payload descriptor.
  ///
  /// @return The next payload reference, or a typed error.
  /// @retval ErrorCode::kInvalidArgument The received record selects the inline-payload kind.
  /// @retval ErrorCode::kNotReady No payload descriptor is queued.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<PayloadDescriptor, Error>
  receive_payload_descriptor();

  /// Receive the next sample envelope.
  ///
  /// @return The next envelope, or a typed error.
  /// @retval ErrorCode::kInvalidArgument The received record selects the inline-payload kind.
  /// @retval ErrorCode::kNotReady No envelope is queued.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<SampleEnvelope, Error> receive_envelope();

 private:
  friend class ChannelRuntime;

  class Impl;

  HOLOSCAN_CORE_EXPORT explicit ChannelDescriptorSubscriber(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan
