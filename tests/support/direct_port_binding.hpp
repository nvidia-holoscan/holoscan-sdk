/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "runtime/port_binding.hpp"

namespace holoscan::test {

/// Bounded in-process Runtime binding used only by direct execution tests.
///
/// This fixture enters through the same private `PortBinding` seam as a
/// production adapter. It is not installed, does not add test queues to
/// `Input<T>`/`Output<T>`, and grants no public standalone-Operator execution
/// API.
class DirectPortBinding final : public detail::PortBinding {
 public:
  DirectPortBinding(std::size_t capacity, detail::RuntimeEvent restart_event);

  DirectPortBinding(const DirectPortBinding&) = delete;
  DirectPortBinding& operator=(const DirectPortBinding&) = delete;
  DirectPortBinding(DirectPortBinding&&) = delete;
  DirectPortBinding& operator=(DirectPortBinding&&) = delete;
  ~DirectPortBinding() override = default;

  [[nodiscard]] expected<void, Error> push(detail::RuntimeEvent event) noexcept;
  void close() noexcept { closed_ = true; }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent> destination) noexcept override;
  [[nodiscard]] SourceId source() const noexcept override { return restart_event_.source; }
  void cancel() noexcept override;
  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override;
  [[nodiscard]] std::size_t capacity() const noexcept override { return storage_.size(); }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  std::vector<detail::RuntimeEvent> storage_;
  std::size_t head_{};
  std::size_t tail_{};
  std::size_t size_{};
  bool closed_{};
  detail::RuntimeEvent restart_event_;
};

/// Create one finite direct adapter per immutable-plan readiness source.
class DirectPortBindingFactory final : public detail::RuntimePortBindingFactory {
 public:
  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override;
};

/// Attach the bounded direct adapter without changing public plan or port API.
///
/// `activation_count` selects the production-shaped capacity-one direct source
/// that publishes its next witness only after retirement.
[[nodiscard]] ExecutionPlan enable_direct_runtime(const ExecutionPlan& plan,
                                                  std::uint32_t activation_count = 1U);

}  // namespace holoscan::test
