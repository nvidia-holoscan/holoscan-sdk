/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <semaphore>
#include <string_view>

#include <holomq/context.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>

#include "channel_runtime_impl.hpp"

namespace holoscan::channel_internal {

// Source-tree-only access for preserving inherited transport-level tests and constructing the
// subscriber peer used to verify catalog Tensor-loan publication. This header is not installed.
struct TensorPublisherTestAccess {
  struct RequiredPublisherWaitBarrier {
    std::binary_semaphore reached{0};
    std::binary_semaphore resume{0};
    std::binary_semaphore refreshed{0};
    std::binary_semaphore resume_after_refresh{0};
    bool pause_after_refresh{};
  };

  struct ReceiveProbeBarrier {
    std::binary_semaphore reached{0};
    std::binary_semaphore resume{0};

    void arm() noexcept { armed.store(true, std::memory_order_release); }
    void disarm() noexcept { armed.store(false, std::memory_order_release); }

    [[nodiscard]] bool pause_after_empty_data_probe() noexcept {
      if (!armed.exchange(false, std::memory_order_acq_rel)) {
        return false;
      }
      reached.release();
      resume.acquire();
      return true;
    }

   private:
    std::atomic<bool> armed{true};
  };

  [[nodiscard]] static expected<ManualTensorPublisher, Error> create(
      ChannelRuntime& runtime, const std::string_view channel_name,
      const ChannelOptions& options = {}) {
    return runtime.create_tensor_publisher(channel_name, options);
  }

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<ChannelSubscriber<Tensor>, Error>
  create_catalog_subscriber(ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::unique_ptr<InlinePayloadSubscriber>,
                                                     Error>
  create_catalog_clock_health_subscriber(ChannelRuntime& runtime,
                                         ChannelEndpointCatalog::EndpointView endpoint);

  /// Pause one required-peer Tensor subscriber after construction and before connection refresh.
  HOLOSCAN_CORE_EXPORT static void pause_next_required_publisher_wait_for_test(
      RequiredPublisherWaitBarrier& barrier) noexcept;

  /// Cold-attach a one-shot data-probe barrier to the next Tensor subscriber construction attempt
  /// on this thread.
  HOLOSCAN_CORE_EXPORT static void attach_receive_probe_barrier_to_next_subscriber_for_test(
      ReceiveProbeBarrier& barrier) noexcept;

  [[nodiscard]] static expected<std::uint32_t, Error> emit_end_of_stream(
      ManualTensorPublisher& publisher, SampleMetadata metadata) noexcept {
    return publisher.emit_end_of_stream(metadata);
  }

  static void fail_next_manual_sidecar_with_bad_alloc(ChannelRuntime& runtime) noexcept {
    if (runtime.impl_) {
      runtime.impl_->fail_next_manual_sidecar_with_bad_alloc_for_test();
    }
  }

  static void fail_next_manual_sidecar_with_length_error(ChannelRuntime& runtime) noexcept {
    if (runtime.impl_) {
      runtime.impl_->fail_next_manual_sidecar_with_length_error_for_test();
    }
  }

  static void fail_next_manual_backend_with_exception(ChannelRuntime& runtime) noexcept {
    if (runtime.impl_) {
      runtime.impl_->fail_next_manual_backend_with_exception_for_test();
    }
  }

  // Raw transport access is restricted to source-tree tests that must inject records which the
  // public Channel publisher correctly rejects before transport.
  [[nodiscard]] static holomq::Context* transport_context(ChannelRuntime& runtime) noexcept {
    return runtime.impl_ ? &runtime.impl_->context_ : nullptr;
  }
};

[[nodiscard]] inline std::optional<
    std::reference_wrapper<TensorPublisherTestAccess::ReceiveProbeBarrier>>&
next_tensor_subscriber_receive_probe_barrier_for_test() noexcept {
  static thread_local std::optional<
      std::reference_wrapper<TensorPublisherTestAccess::ReceiveProbeBarrier>>
      barrier;
  return barrier;
}

[[nodiscard]] inline TensorPublisherTestAccess::ReceiveProbeBarrier*
take_next_tensor_subscriber_receive_probe_barrier_for_test() noexcept {
  auto& pending = next_tensor_subscriber_receive_probe_barrier_for_test();
  if (!pending.has_value()) {
    return nullptr;
  }
  auto* const barrier = std::addressof(pending->get());
  pending.reset();
  return barrier;
}

}  // namespace holoscan::channel_internal
