/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace holoscan::test {

// Check accepted plan geometry on the cold path, including in Release and benchmark builds.
inline void require_retirement_ordered_acknowledgment(
    const ExecutionPartitionView& sink, const ExecutionPartitionView& acknowledgment) {
  if (sink.id() != acknowledgment.id() ||
      sink.execution_lane() != acknowledgment.execution_lane()) {
    throw RuntimeError{ErrorCode::kInvalidArgument,
                       "Retirement-ordered acknowledgment must share its sink's serial partition"};
  }
}

// Test-only bounded pause after the sink publishes its acknowledgment but before compute returns.
// The external test releases it; no callback waits for another activation. Timed runs omit it.
class AcknowledgmentRetirementGate {
 public:
  expected<void, Error> hold(std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    std::unique_lock lock(mutex_);
    held_ = true;
    ready_.notify_all();
    if (!ready_.wait_for(lock, timeout, [&] { return released_; })) {
      return make_unexpected(
          Error{ErrorCode::kTimeout,
                "Retirement acknowledgment gate timed out waiting for test release"});
    }
    return {};
  }
  bool wait_until_held() {
    std::unique_lock lock(mutex_);
    return ready_.wait_for(lock, std::chrono::seconds{5}, [&] { return held_; });
  }
  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    ready_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  bool held_{};
  bool released_{};
};

// One-in-flight harness control, not a general publication retry mechanism. The preceding data
// sink MUST share this operator's serial partition: its selected payload then retires before this
// callback grants the next source credit. Only a scalar sequence crosses this extra control edge.
// Validate each sink/ack pair with require_retirement_ordered_acknowledgment() before running.
template <typename Probe>
class RetirementOrderedAcknowledgment final : public Operator<> {
 public:
  RetirementOrderedAcknowledgment(std::shared_ptr<Probe> probe, std::size_t index)
      : probe_(std::move(probe)), index_(index) {}
  void setup(OperatorSpec& spec) override { spec.input(input).queue_depth(1); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, input.receive());
    if (sample.data != probe_->acknowledged[index_].load(std::memory_order_relaxed) + 1)
      return make_unexpected(Error{ErrorCode::kProtocolError});
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->acknowledged[index_].store(sample.data, std::memory_order_release);
    }
    probe_->ready.notify_all();
    return {};
  }
  Input<std::uint64_t> input{"input"};

 private:
  std::shared_ptr<Probe> probe_;
  std::size_t index_;
};

}  // namespace holoscan::test
