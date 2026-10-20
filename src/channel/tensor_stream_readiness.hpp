/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <optional>

#include <cuda_runtime_api.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "core/domain/tensor_access.hpp"
#include "scoped_cuda_device.hpp"

namespace holoscan::channel_internal {

// Keep producer identity beside its borrowed handle. Host waits pin only the enqueue,
// never the blocking event wait; owner quiescence can therefore drain without deadlock.
class TensorStreamReadiness {
 public:
  void set(cudaStream_t stream,
           const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime = {}) noexcept {
    const std::lock_guard lock{mutex_};
    if (stream == nullptr) {
      stream_ = nullptr;
      lifetime_.reset();
      active_ = false;
      return;
    }
    stream_ = stream;
    lifetime_ = lifetime;
    active_ = true;
  }
  void clear() noexcept {
    const std::lock_guard lock{mutex_};
    stream_ = nullptr;
    lifetime_.reset();
    active_ = false;
  }
  [[nodiscard]] cudaStream_t stream() const noexcept {
    const std::lock_guard lock{mutex_};
    return lifetime_ != nullptr && lifetime_->state() == Lifetime::State::kRetired ? nullptr
                                                                                   : stream_;
  }
  [[nodiscard]] std::shared_ptr<const holomq::cuda::StreamLifetime> lifetime() const noexcept {
    const std::lock_guard lock{mutex_};
    return lifetime_;
  }
  [[nodiscard]] expected<void, Error> consumer(cudaStream_t stream) const {
    return synchronize(stream);
  }
  [[nodiscard]] expected<void, Error> host() const { return synchronize(std::nullopt); }

 private:
  using Lifetime = holomq::cuda::StreamLifetime;
  // Keep event creation, lifetime pins, enqueue, and teardown in one ordered wait transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  expected<void, Error> synchronize(std::optional<cudaStream_t> consumer) const {
    cudaStream_t producer{};
    std::shared_ptr<const Lifetime> lifetime;
    {
      const std::lock_guard lock{mutex_};
      if (!active_) {
        return {};
      }
      producer = stream_;
      lifetime = lifetime_;
    }
    if (lifetime != nullptr && lifetime->state() == Lifetime::State::kRetired) {
      return {};
    }
    if (lifetime != nullptr && lifetime->state() != Lifetime::State::kOpen) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    const auto& target = consumer ? detail::TensorConsumerScope::lifetime(*consumer)
                                  : std::shared_ptr<const Lifetime>{};
    if (consumer && producer == *consumer && lifetime == target) {
      return {};
    }
    // An escaped Tensor can be waited on with a different device current. Events must
    // belong to the producer's device; restore the caller's selection after the wait.
    const ScopedCUDADevice selected{
        lifetime != nullptr ? std::optional<std::int32_t>{lifetime->device()} : std::nullopt};
    if (!selected.active()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    cudaEvent_t event{};
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto producer_pin = lifetime != nullptr ? lifetime->try_pin() : Lifetime::PinResult{};
    auto consumer_pin = target != nullptr ? target->try_pin() : Lifetime::PinResult{};
    cudaError_t result = cudaSuccess;
    if (lifetime != nullptr && !producer_pin.pin) {
      result = producer_pin.state == Lifetime::State::kRetired ? cudaSuccess
                                                               : cudaErrorInvalidResourceHandle;
    } else if (target != nullptr && !consumer_pin.pin) {
      result = cudaErrorInvalidResourceHandle;
    } else {
      result = cudaEventRecord(event, producer);
      if (result == cudaSuccess && consumer) {
        result = cudaStreamWaitEvent(*consumer, event, 0U);
      }
    }
    producer_pin.pin.reset();
    consumer_pin.pin.reset();
    if (result == cudaSuccess && !consumer) {
      result = cudaEventSynchronize(event);
    }
    static_cast<void>(cudaEventDestroy(event));
    return result == cudaSuccess ? expected<void, Error>{}
                                 : make_unexpected(Error{ErrorCode::kClosed});
  }
  mutable std::mutex mutex_;
  bool active_{};
  cudaStream_t stream_{};
  std::shared_ptr<const Lifetime> lifetime_;
};

}  // namespace holoscan::channel_internal
