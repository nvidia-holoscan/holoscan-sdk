/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <holoscan/testing/operator_test_harness.hpp>

namespace holoscan::testing::test {

struct ManagedTensorProbe {
  std::mutex mutex;
  std::condition_variable changed;
  NotificationSender sender;
  std::size_t arms{};
  std::size_t restart_allocations{};
  bool stale_guard_rejected{};
  bool delay{};
  bool delay_entered{};
  bool delay_released{};
  bool delay_expired{};

  bool wait_delay() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds{5}, [&] { return delay_entered; });
  }
  void release_delay() {
    const std::lock_guard lock(mutex);
    delay_released = true;
    changed.notify_all();
  }
  static void delay_callback(void* state) {
    auto& probe = *static_cast<ManagedTensorProbe*>(state);
    std::unique_lock lock(probe.mutex);
    probe.delay_entered = true;
    probe.changed.notify_all();
    probe.delay_expired = !probe.changed.wait_for(
        lock, std::chrono::seconds{5}, [&] { return probe.delay_released; });
  }

  NotificationSender wait_sender(std::size_t generation) {
    std::unique_lock lock(mutex);
    if (!changed.wait_for(lock, std::chrono::seconds{5}, [&] { return arms >= generation; }))
      return {};
    return sender;
  }
};

template <PublishedPayload T>
class ManagedTensorSource final : public Operator<> {
 public:
  explicit ManagedTensorSource(std::shared_ptr<ManagedTensorProbe> probe, T metadata = {},
                               bool stale_writer = false, bool lifecycle_rate = false,
                               std::uint64_t calls = 2U, bool device = false)
      : probe_(std::move(probe)),
        metadata_(std::move(metadata)),
        stale_writer_(stale_writer),
        lifecycle_rate_(lifecycle_rate),
        calls_(calls),
        device_(device) {}
  void setup(OperatorSpec& spec) override {
    auto& output = spec.output(out, "out");
    if (lifecycle_rate_)
      output.lifecycle_nominal_rate(LifecycleStage::kArm).max_nominal_rate(Rate::hz(120U));
    const TensorOutputSpec layout{
        .representation = {.memory_kind = device_ ? MemoryKind::kCudaDevice : MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8, 1},
                           .rank = 1U},
        .bounds = tensor_bounds(4U),
        .storage = TensorOutputStorage::kRuntimePool};
    if constexpr (std::same_as<T, Tensor>) {
      output.produces_tensor(layout);
    } else {
      using Traits = DescriptorTensorSchemaNativeTraits<T>;
      if constexpr (Traits::tensor_count == 1U) {
        output.produces_tensor(layout);
      } else {
        for (std::size_t index = 0; index < Traits::tensor_count; ++index)
          output
              .tensor_field(TensorFieldSelector<T>{Traits::tensor_field_ids[index],
                                                   static_cast<std::uint16_t>(index)})
              .produces_tensor(layout);
      }
    }
    spec.notification_source(event_, "step").capacity(1U).sender_reference_capacity(2U);
    spec.lifecycle()
        .stage(LifecycleStage::kAllocate, &ManagedTensorSource::allocate)
        .stage(LifecycleStage::kArm, &ManagedTensorSource::arm);
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnNotified{.event = event_}).complete_after(ComputeCalls{.limit = calls_});
    return c;
  }
  LifecycleStatus allocate(LifecycleContext& c) noexcept {
    if (c.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      const std::lock_guard lock(probe_->mutex);
      ++probe_->restart_allocations;
    }
    return LifecycleStatus::kOk;
  }
  LifecycleStatus arm(LifecycleContext& c) noexcept {
    if (lifecycle_rate_) {
      const auto bound = c.bind_nominal_rate(out, Rate::hz(60U));
      if (!bound) {
        c.report_error(bound.error());
        return LifecycleStatus::kFatalFailure;
      }
    }
    auto sender = c.notification_sender(event_);
    if (!sender) {
      c.report_error(sender.error());
      return LifecycleStatus::kFatalFailure;
    }
    const std::lock_guard lock(probe_->mutex);
    probe_->sender = std::move(*sender);
    ++probe_->arms;
    probe_->changed.notify_all();
    return LifecycleStatus::kOk;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    const std::array<std::int64_t, 1> shape{4};
    if constexpr (std::same_as<T, Tensor>) {
      if (stale_guard_) {
        auto committed = std::move(*stale_guard_).commit();
        const bool rejected = !committed && (committed.error().code == ErrorCode::kClosed ||
                                             committed.error().code == ErrorCode::kInvalidArgument);
        stale_guard_.reset();
        stale_loan_.reset();
        {
          const std::lock_guard lock(probe_->mutex);
          probe_->stale_guard_rejected = rejected;
        }
        if (!rejected)
          return make_unexpected(Error{ErrorCode::kFailure, "stale writer was not revoked"});
      }
      auto loan = out.allocate_tensor(shape);
      if (!loan)
        return make_unexpected(loan.error());
      auto filled =
          device_
              ? loan->write_and_commit(
                    context.cuda_stream(),
                    [&](const auto& write) -> expected<void, Error> {
                      const auto stream = context.cuda_stream().get();
                      if (probe_->delay && next_ == 0U &&
                          cudaLaunchHostFunc(stream,
                                             &ManagedTensorProbe::delay_callback,
                                             probe_.get()) != cudaSuccess)
                        return make_unexpected(
                            Error{ErrorCode::kFailure, "cannot enqueue readiness gate"});
                      const auto status = cudaMemsetAsync(write.data(),
                                                          static_cast<int>(11 * (next_ + 1U)),
                                                          write.byte_size(),
                                                          stream);
                      if (status != cudaSuccess)
                        return make_unexpected(
                            Error{ErrorCode::kFailure, cudaGetErrorString(status)});
                      return {};
                    })
              : loan->write_host_and_commit([&](const auto& write) {
                  std::memset(write.data(), static_cast<int>(11 * (next_ + 1U)), write.byte_size());
                });
      if (!filled)
        return filled;
      auto emitted = out.emit(std::move(*loan));
      if (!emitted)
        return emitted;
      if (stale_writer_ && next_ == 0U) {
        auto extra = out.allocate_tensor(shape);
        if (!extra)
          return make_unexpected(extra.error());
        stale_loan_.emplace(std::move(*extra));
        auto writer = stale_loan_->write_host();
        if (!writer)
          return make_unexpected(writer.error());
        stale_guard_.emplace(std::move(*writer));
      }
    } else {
      using Traits = DescriptorTensorSchemaNativeTraits<T>;
      if constexpr (Traits::tensor_count == 1U) {
        auto loan = out.allocate_tensor(shape);
        if (!loan)
          return make_unexpected(loan.error());
        auto filled = loan->write_host_and_commit([&](const auto& write) {
          std::memset(write.data(), 11 * (next_ + 1U), write.byte_size());
        });
        if (!filled)
          return filled;
        auto emitted = out.emit_tensor(std::move(*loan), metadata_);
        if (!emitted)
          return emitted;
      } else {
        std::array<TensorFieldLoanRequest<T>, Traits::tensor_count> requests;
        for (std::size_t i = 0; i < requests.size(); ++i)
          requests[i] = {.field = {Traits::tensor_field_ids[i], static_cast<std::uint16_t>(i)},
                         .tensor = {.shape = shape, .dtype = DLDataType{kDLUInt, 8, 1}}};
        auto loans = out.allocate_tensors(requests);
        if (!loans)
          return make_unexpected(loans.error());
        for (std::size_t i = 0; i < requests.size(); ++i) {
          auto filled = loans->write_host_and_commit(requests[i].field, [&](const auto& write) {
            std::memset(write.data(), 11 * (next_ + i + 1U), write.byte_size());
          });
          if (!filled)
            return filled;
        }
        auto emitted = out.emit_tensors(std::move(*loans), metadata_);
        if (!emitted)
          return emitted;
      }
    }
    ++next_;
    return {};
  }
  Output<T> out;

 private:
  std::shared_ptr<ManagedTensorProbe> probe_;
  T metadata_;
  bool stale_writer_;
  bool lifecycle_rate_;
  std::uint64_t calls_;
  bool device_;
  NotificationSource event_;
  std::size_t next_{};
  std::optional<TensorOutputLoan> stale_loan_;
  std::optional<TensorOutputWriteGuard> stale_guard_;
};

inline LifecycleTransactionOptions tensor_restart_capacity() {
  return {.max_pending_requests = 8U,
          .max_pending_transactions = 8U,
          .max_receipts = 16U,
          .max_fenced_operators = 16U,
          .max_candidate_generations = 16U,
          .max_events = 128U,
          .max_audit_records = 32U,
          .max_quiesce_records = 64U,
          .max_rollback_records = 64U};
}

inline expected<void, Error> post_tensor(const NotificationSender& sender) {
  auto result = sender.post();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!result &&
         (result.error().code == ErrorCode::kNotReady ||
          result.error().code == ErrorCode::kBackpressured) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
    result = sender.post();
  }
  return result;
}

}  // namespace holoscan::testing::test
