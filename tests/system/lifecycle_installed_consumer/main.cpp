/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace {

class InstalledLifecycleOperator final : public holoscan::Operator<> {
 public:
  explicit InstalledLifecycleOperator(bool producer) : producer_(producer) {}

  void setup(holoscan::OperatorSpec& spec) override {
    auto& lifecycle = spec.lifecycle();
    lifecycle.stage(holoscan::LifecycleStage::kConfigure, &InstalledLifecycleOperator::configure)
        .optional(holoscan::LifecycleStage::kStart, &InstalledLifecycleOperator::on_start);
    if (producer_) {
      lifecycle.produces(holoscan::LifecycleStage::kConfigure,
                         holoscan::LifecycleDependencyKey{"installed.configured"});
    } else {
      lifecycle.requires_dependency(holoscan::LifecycleStage::kConfigure,
                                    holoscan::LifecycleDependencyKey{"installed.configured"});
    }
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

 private:
  holoscan::LifecycleStatus configure(holoscan::LifecycleContext&) noexcept {
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept {
    return holoscan::LifecycleStatus::kOk;
  }

  bool producer_{};
};

static_assert(std::is_standard_layout_v<holoscan::LifecycleRestartOptions>);
static_assert(std::is_trivially_copyable_v<holoscan::LifecycleRestartReceipt>);
static_assert(sizeof(holoscan::LifecycleContext) == sizeof(void*));
static_assert(sizeof(holoscan::LifecycleControlHandle) == sizeof(void*));
static_assert(std::is_copy_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_copy_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_move_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_move_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_move_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_move_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_destructible_v<holoscan::LifecycleControlHandle>);

[[nodiscard]] bool exercise_closed_lifecycle_control() {
  const holoscan::LifecycleControlHandle original;
  holoscan::LifecycleControlHandle copied{original};
  holoscan::LifecycleControlHandle moved{std::move(copied)};
  copied = original;
  moved = std::move(copied);

  const auto restricted = moved.restrict({});
  const auto requested = moved.request_restart({});
  const auto inspected = moved.inspect({});
  const auto waited = moved.await({}, std::chrono::nanoseconds{0});
  return !original && !copied && !moved && !restricted && !requested && !inspected && !waited &&
         restricted.error().code == holoscan::ErrorCode::kClosed &&
         requested.error().code == holoscan::ErrorCode::kClosed &&
         inspected.error().code == holoscan::ErrorCode::kClosed &&
         waited.error().code == holoscan::ErrorCode::kClosed;
}

}  // namespace

// REQ: LIFECYCLE-002
// REQ: LIFECYCLE-003
// REQ: LIFECYCLE-005
// REQ: LIFECYCLE-007
// REQ: RUNTIME-008
int main() {
  const auto cancellation = &holoscan::LifecycleContext::cancellation;
  const auto params_generation = &holoscan::LifecycleContext::params_generation;
  const auto transition_phase = &holoscan::LifecycleContext::transition_phase;
  const auto notification_sender = &holoscan::LifecycleContext::notification_sender;
  const auto restrict_control = &holoscan::LifecycleControlHandle::restrict;
  const auto request_restart = &holoscan::LifecycleControlHandle::request_restart;
  const auto inspect_restart = &holoscan::LifecycleControlHandle::inspect;
  const auto await_restart = &holoscan::LifecycleControlHandle::await;
  if (cancellation == nullptr || params_generation == nullptr || transition_phase == nullptr ||
      notification_sender == nullptr || restrict_control == nullptr || request_restart == nullptr ||
      inspect_restart == nullptr || await_restart == nullptr) {
    return 1;
  }

  if (!exercise_closed_lifecycle_control()) {
    return 2;
  }

  const holoscan::LifecycleRestartOptions defaults;
  if (defaults.from_stage != holoscan::LifecycleStage::kConfigure) {
    return 3;
  }

  holoscan::Graph graph{"installed-lifecycle"};
  const auto producer = graph.op<InstalledLifecycleOperator>("producer", true);
  const auto consumer = graph.op<InstalledLifecycleOperator>("consumer", false);
  graph.lifecycle()
      .before(producer, consumer, holoscan::LifecycleStage::kConfigure)
      .allow_self_restart(producer)
      .allow_external_restart(consumer)
      .allow_restart_request(producer, consumer);

  const holoscan::GraphView view = graph.view();
  if (view.lifecycle_dependencies().size() != 1U ||
      view.lifecycle_restart_permissions().size() != 3U) {
    return 4;
  }
  return 0;
}
