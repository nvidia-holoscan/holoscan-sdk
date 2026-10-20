/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "direct_port_binding.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "runtime/runtime_test_access.hpp"

namespace holoscan::test {

DirectPortBinding::DirectPortBinding(std::size_t capacity, detail::RuntimeEvent restart_event)
    : storage_(capacity), restart_event_(restart_event) {}

expected<void, Error> DirectPortBinding::push(detail::RuntimeEvent event) noexcept {
  if (closed_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (storage_.empty() || size_ == storage_.size()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  storage_[tail_] = std::move(event);
  tail_ = (tail_ + 1U) % storage_.size();
  ++size_;
  return {};
}

expected<std::size_t, Error> DirectPortBinding::drain(
    std::span<detail::RuntimeEvent> destination) noexcept {
  const std::size_t count = std::min(size_, destination.size());
  for (std::size_t index = 0U; index < count; ++index) {
    destination[index] = std::move(storage_[head_]);
    head_ = (head_ + 1U) % storage_.size();
  }
  size_ -= count;
  return count;
}

void DirectPortBinding::cancel() noexcept {
  head_ = 0U;
  tail_ = 0U;
  size_ = 0U;
  closed_ = true;
}

expected<std::uint64_t, Error> DirectPortBinding::advance_source_generation() noexcept {
  if (!closed_ || size_ != 0U || !restart_event_.source) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (restart_event_.source_generation == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  ++restart_event_.source_generation;
  restart_event_.source_sequence = 1U;
  head_ = 0U;
  tail_ = 0U;
  // A lifecycle restart advances identity but does not manufacture a new
  // physical witness. This finite direct fixture remains closed; periodic
  // clock tests cover a new post-restart activation explicitly.
  return restart_event_.source_generation;
}

expected<std::unique_ptr<detail::PortBinding>, Error> DirectPortBindingFactory::bind(
    const detail::RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept {
  if (request.plan_generation == 0U || request.run_generation == 0U || !request.source ||
      !request.source_clock.valid() || request.source_clock != request.reference_clock ||
      request.source_to_reference_conversion_id != 0U || request.capacity == 0U ||
      start_coordinate.clock_id != request.reference_clock ||
      request.initial_source_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const detail::RuntimeEvent initial{
        .kind = request.kind,
        .source = request.source,
        .port = request.port,
        .source_decision_time = start_coordinate,
        .reference_time = start_coordinate,
        .source_generation = request.source_generation,
        .source_sequence = request.initial_source_sequence + 1U,
    };
    auto binding = std::make_unique<DirectPortBinding>(request.capacity, initial);
    auto pushed = binding->push(initial);
    if (!pushed) {
      return make_unexpected(std::move(pushed.error()));
    }
    binding->close();
    return std::unique_ptr<detail::PortBinding>{std::move(binding)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
  }
#endif
}

ExecutionPlan enable_direct_runtime(const ExecutionPlan& plan, std::uint32_t activation_count) {
  if (activation_count != 1U) {
    return detail::RuntimeTestPlanAccess::enable_direct_count(plan, activation_count);
  }
  return detail::RuntimeTestPlanAccess::with_port_binding_factory(
      plan, std::make_shared<DirectPortBindingFactory>());
}

}  // namespace holoscan::test
