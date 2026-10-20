/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/catalog_pod_test_types.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "run_session_channel_execution_test_support.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_channel_binding_schema_tensor_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

enum class OptionalPayloadScenario { kUnwired, kSilent, kSelected };

// Share the same selector and topology across payload families; only sample construction and
// inspection differ. Tensor backing is owned by the existing host-only release-counting fixture.
// Used by RunSessionTest.OptionalTensorPreservesAbsenceAndConnectedPayload.
template <typename T>
struct OptionalPayloadProbe {
  static inline std::atomic_uint32_t observations{};
  static inline std::atomic_size_t read_allocations{};
  static inline std::atomic<const void*> backing{};
  static inline std::shared_ptr<std::atomic<std::uint32_t>> releases;

  static void reset() {
    observations.store(0U);
    read_allocations.store(0U);
    backing = nullptr;
    releases.reset();
  }

  static expected<T, Error> make_sample() {
    if constexpr (std::is_same_v<T, schema::StringT>) {
      T value;
      value.data = "optional-value";
      return value;
    } else if constexpr (std::is_same_v<T, channel_test::CatalogProcessPod>) {
      return T{.frame = 17U, .value = -23};
    } else {
      auto owned =
          detail::runtime_channel_binding_schema_tensor_test::make_owned_tensor_sample(73U);
      if (!owned) {
        return make_unexpected(std::move(owned).error());
      }
      backing = owned->data;
      releases = owned->releases;
      if constexpr (std::is_same_v<T, Tensor>) {
        return std::move(owned->tensor);
      } else {
        T value;
        value.height = 1;
        value.width = 1;
        value.encoding = schema::ImageEncoding_CUSTOM;
        value.data = std::make_shared<Tensor>(std::move(owned->tensor));
        return value;
      }
    }
  }

  static bool matches(const T& value) {
    if constexpr (std::is_same_v<T, schema::StringT>) {
      return value.data == "optional-value";
    } else if constexpr (std::is_same_v<T, channel_test::CatalogProcessPod>) {
      return value.frame == 17U && value.value == -23;
    } else if constexpr (std::is_same_v<T, Tensor>) {
      return value.data() == backing.load() &&
             *static_cast<const std::uint64_t*>(value.data()) == 73U;
    } else {
      return value.height == 1 && value.width == 1 &&
             value.encoding == schema::ImageEncoding_CUSTOM && value.data &&
             value.data->data() == backing.load() &&
             *static_cast<const std::uint64_t*>(value.data->data()) == 73U;
    }
  }
};

// Emit a payload and/or independent driver for the optional payload scenario matrix.
// Used by RunSessionTest.OptionalGeneratedSchemaPreservesAbsenceAndConnectedPayload.
template <typename T>
class OptionalPayloadSource final : public Operator<> {
 public:
  explicit OptionalPayloadSource(bool publish, bool driven) : publish_(publish), driven_(driven) {}
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    if (driven_) {
      spec.output(driver, "driver");
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (publish_ && emitted_) {
      return {};
    }
    const auto capture = context.activation_time();
    const SampleMetadata metadata{.capture_timestamp_ns = capture.timestamp_ns,
                                  .source_clock_id = capture.clock_id,
                                  .frame_id = 17U};
    if (publish_) {
      auto value = OptionalPayloadProbe<T>::make_sample();
      if (!value) {
        return make_unexpected(std::move(value).error());
      }
      auto emitted = output.emit(Sample<T>{std::move(*value), metadata});
      if (!emitted) {
        return emitted;
      }
      emitted_ = true;
    }
    // Exact selects against this independent driver's capture coordinate, not the later
    // scheduling-clock instant at which the consumer happens to execute.
    return driven_ ? driver.emit(Sample<std::uint32_t>{1U, metadata}) : expected<void, Error>{};
  }
  Output<T> output;
  Output<std::uint32_t> driver;

 private:
  bool publish_;
  bool driven_;
  bool emitted_{};
};

// Check selection, accessors, and payload ownership for each optional-input scenario.
// Used by RunSessionTest.OptionalDescriptorSchemaPreservesAbsenceAndConnectedPayload.
template <typename T>
class OptionalPayloadSink final : public Operator<> {
 public:
  explicit OptionalPayloadSink(OptionalPayloadScenario scenario, bool driven)
      : scenario_(scenario), driven_(driven) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U).optional();
    if (driven_) {
      spec.input(driver, "driver").queue_depth(4U);
    }
  }
  Contract contract() const override {
    Contract result;
    if (driven_) {
      result.trigger(OnEach{driver});
    } else {
      result.trigger(OnClock{.period = 1ms});
    }
    result.input(input, Exact{}).on_missing(input, SafeDefault{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    const auto selection = input.selection();
    auto batch = input.selected_batch();
    if (batch || batch.error().code != ErrorCode::kInputSelectionCardinalityMismatch) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (selection.state == InputState::kSelected) {
      auto sample = input.receive();
      auto value = input.receive_data();
      if (scenario_ != OptionalPayloadScenario::kSelected ||
          selection.reason != SelectionReason::kNone || !sample || !value ||
          !OptionalPayloadProbe<T>::matches(sample->data) ||
          !OptionalPayloadProbe<T>::matches(*value) || sample->metadata.frame_id != 17U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      OptionalPayloadProbe<T>::observations.fetch_add(1U);
      context.request_completion();
      return {};
    }
    if (selection.state != InputState::kSafeDefault ||
        selection.reason != (scenario_ == OptionalPayloadScenario::kUnwired
                                 ? SelectionReason::kNotConnected
                                 : SelectionReason::kNeverSeen)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    bool empty = false;
    std::size_t count = 0U;
    {
      const test::ScopedAllocationCounter allocations;
      auto sample = input.receive();
      auto value = input.receive_data();
      auto absent_batch = input.selected_batch();
      empty = !sample && sample.error().code == ErrorCode::kInputEmpty && !value &&
              value.error().code == ErrorCode::kInputEmpty && !absent_batch &&
              absent_batch.error().code == ErrorCode::kInputSelectionCardinalityMismatch;
      count = allocations.count();
    }
    OptionalPayloadProbe<T>::read_allocations.fetch_add(count);
    if (!empty) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (scenario_ != OptionalPayloadScenario::kSelected) {
      if (OptionalPayloadProbe<T>::observations.fetch_add(1U) + 1U == 8U) {
        context.request_completion();
      }
    } else {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return {};
  }
  Input<T> input;
  Input<std::uint32_t> driver;

 private:
  OptionalPayloadScenario scenario_;
  bool driven_;
};

// Build the payload/driver topology used by
// RunSessionTest.OptionalDeclaredPODPreservesAbsenceAndConnectedPayload.
// Graph: source.output --(if wired)--> sink.input; source.driver --(if driven)--> sink.driver.
// Without either edge, only the clock-driven sink is present.
template <typename T>
ExecutionPlan optional_payload_plan(OptionalPayloadScenario scenario, bool driven = false) {
  Graph graph{"optional-payload"};
  const auto sink = graph.op<OptionalPayloadSink<T>>("sink", scenario, driven);
  if (scenario != OptionalPayloadScenario::kUnwired || driven) {
    const auto source = graph.op<OptionalPayloadSource<T>>(
        "source", scenario == OptionalPayloadScenario::kSelected, driven);
    if (scenario != OptionalPayloadScenario::kUnwired) {
      graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 4U});
    }
    if (driven) {
      graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 4U});
    }
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  return compile(graph);
}

// Run the absence/silence/delivery matrix and check retained ownership in
// RunSessionTest.OptionalTensorPreservesAbsenceAndConnectedPayload.
template <typename T>
void check_optional_payload_sessions() {
  for (const auto scenario : {OptionalPayloadScenario::kUnwired,
                              OptionalPayloadScenario::kSilent,
                              OptionalPayloadScenario::kSelected}) {
    SCOPED_TRACE(static_cast<unsigned>(scenario));
    OptionalPayloadProbe<T>::reset();
    const auto plan = optional_payload_plan<T>(scenario, true);
    ASSERT_TRUE(plan.ok()) << plan.json();
    {
      RunSession session = run_async(plan);
      session.wait();
      EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
      EXPECT_TRUE(session.diagnostics().empty());
    }
    EXPECT_EQ(OptionalPayloadProbe<T>::observations.load(),
              scenario == OptionalPayloadScenario::kSelected ? 1U : 8U);
    EXPECT_EQ(OptionalPayloadProbe<T>::read_allocations.load(), 0U);
    if constexpr (std::is_same_v<T, Tensor> || std::is_same_v<T, schema::ImageT>) {
      if (scenario == OptionalPayloadScenario::kSelected) {
        ASSERT_NE(OptionalPayloadProbe<T>::releases, nullptr);
        EXPECT_EQ(OptionalPayloadProbe<T>::releases->load(), 1U);
      } else {
        EXPECT_EQ(OptionalPayloadProbe<T>::releases, nullptr);
      }
    }
  }
}

// REQ: RUNTIME-055
// REQ: COMPILE-080
// REQ: COMPILE-087
// REQ: COMPILE-091
// REQ: COMPILE-098
TEST(RunSessionTest, OptionalGeneratedSchemaPreservesAbsenceAndConnectedPayload) {
  // Key setup: run StringT through unwired, connected-silent, and selected scenarios.
  // Key assertion: absence stays allocation-free; connected samples retain their values and
  // ownership.
  // Graph:
  //   source.driver --------------------> sink.driver
  //   source.output --(silent/selected)--> sink.input [optional; unwired case omits this edge]
  check_optional_payload_sessions<schema::StringT>();
}

// REQ: RUNTIME-055
// REQ: COMPILE-080
// REQ: COMPILE-087
// REQ: COMPILE-091
// REQ: COMPILE-098
TEST(RunSessionTest, OptionalTensorPreservesAbsenceAndConnectedPayload) {
  // Key setup: run Tensor through unwired, connected-silent, and selected scenarios.
  // Key assertion: absence stays allocation-free; connected samples retain their values and
  // ownership.
  // Graph:
  //   source.driver --------------------> sink.driver
  //   source.output --(silent/selected)--> sink.input [optional; unwired case omits this edge]
  check_optional_payload_sessions<Tensor>();
}

// REQ: RUNTIME-055
// REQ: COMPILE-080
// REQ: COMPILE-087
// REQ: COMPILE-091
// REQ: COMPILE-098
TEST(RunSessionTest, OptionalDescriptorSchemaPreservesAbsenceAndConnectedPayload) {
  // Key setup: run ImageT with a Tensor descriptor through unwired, connected-silent, and
  // selected scenarios.
  // Key assertion: absence stays allocation-free; connected samples retain their values and
  // ownership.
  // Graph:
  //   source.driver --------------------> sink.driver
  //   source.output --(silent/selected)--> sink.input [optional; unwired case omits this edge]
  check_optional_payload_sessions<schema::ImageT>();
}

// REQ: RUNTIME-055
// REQ: COMPILE-080
// REQ: COMPILE-087
// REQ: COMPILE-091
// REQ: COMPILE-098
TEST(RunSessionTest, OptionalDeclaredPODPreservesAbsenceAndConnectedPayload) {
  // Key setup: run declared raw POD through unwired, connected-silent, and selected scenarios.
  // Key assertion: absence stays allocation-free; connected samples retain their values and
  // ownership.
  // Graph:
  //   source.driver --------------------> sink.driver
  //   source.output --(silent/selected)--> sink.input [optional; unwired case omits this edge]
  check_optional_payload_sessions<channel_test::CatalogProcessPod>();
}

// Inspect resources and generation/phase/cardinality precedence for
// RuntimeChannelBinding.OptionalPayloadFamiliesPreserveResourcesAndAccessorPrecedence.
template <typename T>
void check_optional_payload_resources_and_guards() {
  for (const bool wired : {false, true}) {
    SCOPED_TRACE(wired);
    const auto plan = optional_payload_plan<T>(wired ? OptionalPayloadScenario::kSilent
                                                     : OptionalPayloadScenario::kUnwired);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                               RunGenerationId{77U});
    ASSERT_TRUE(exec) << exec.error().message;
    std::vector<detail::RuntimeChannelPortReplayRecord> ports;
    for (const auto& instance : (*exec)->operators()) {
      for (std::size_t index = 0U;
           index < detail::OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
           ++index) {
        const auto replay = detail::OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
        const auto planned = std::ranges::find_if((*exec)->plan().ports, [&](const auto& port) {
          return port.owner == instance.vertex && port.direction == replay.direction &&
                 port.local_name == replay.name;
        });
        ASSERT_NE(planned, (*exec)->plan().ports.end());
        ports.push_back({.port = planned->id,
                         .owner = instance.vertex,
                         .direction = replay.direction,
                         .declaration = const_cast<void*>(replay.declaration),
                         .type_ops = replay.runtime_type_ops});
      }
    }
    const auto replay = std::ranges::find(
        ports, PortDirection::kInput, &detail::RuntimeChannelPortReplayRecord::direction);
    ASSERT_NE(replay, ports.end());
    auto& input = *static_cast<Input<T>*>(replay->declaration);
    auto session = detail::RuntimeChannelSession::create((*exec)->plan(), ports);
    ASSERT_TRUE(session) << session.error().message;
    const auto resources =
        detail::RuntimeChannelSessionTestAccess::input_resource_state(**session, replay->port);
    ASSERT_TRUE(resources);
    EXPECT_EQ(resources->subscriber_count, wired ? 1U : 0U);
    EXPECT_EQ(resources->claim_pool_present, wired);
    EXPECT_EQ(resources->payload_slot_count == 0U, !wired);
    const auto pump = detail::RuntimeChannelSessionTestAccess::pump_index_state(**session);
    ASSERT_TRUE(pump);
    if (!wired) {
      const auto& state = (*exec)->plan();
      EXPECT_TRUE(state.payload_edges.empty());
      EXPECT_TRUE(state.channel_services.empty());
      EXPECT_TRUE(state.channel_subscribers.empty());
      EXPECT_TRUE(state.process_local_channel_services.empty());
      EXPECT_TRUE(state.process_local_channel_subscribers.empty());
      EXPECT_TRUE(state.cross_process_channel_services.empty());
      EXPECT_TRUE(state.cross_process_channel_subscribers.empty());
      EXPECT_TRUE(state.generic_output_loans.empty());
      EXPECT_TRUE(state.tensor_allocations.empty());
      EXPECT_TRUE(state.tensor_field_allocations.empty());
      EXPECT_EQ(pump->wait_handle_count, 0U);
      EXPECT_EQ(pump->drain_subscriber_count, 0U);
      EXPECT_FALSE(pump->background_thread_required);
    }
    const auto check_errors = [&](ErrorCode single, ErrorCode batch) {
      const auto sample = input.receive();
      const auto value = input.receive_data();
      const auto selected_batch = input.selected_batch();
      return !sample && sample.error().code == single && !value && value.error().code == single &&
             !selected_batch && selected_batch.error().code == batch;
    };
    EXPECT_TRUE(check_errors(ErrorCode::kStaleEndpoint, ErrorCode::kStaleEndpoint));
    const std::array not_selected{detail::RuntimeChannelActivationSelection{
        .port = replay->port,
        .selection = {.state = InputState::kNotSelected,
                      .reason = SelectionReason::kNotSelectedForActivation},
        .samples = {},
    }};
    ASSERT_TRUE((*session)->prepare_activation(replay->owner, 1U, not_selected));
    EXPECT_TRUE(check_errors(ErrorCode::kInputNotSelected, ErrorCode::kInputNotSelected));
    ASSERT_TRUE((*session)->retire_activation(replay->owner, 1U, false));
    const std::array absent{detail::RuntimeChannelActivationSelection{
        .port = replay->port,
        .selection = {.state = InputState::kSafeDefault,
                      .reason =
                          wired ? SelectionReason::kNeverSeen : SelectionReason::kNotConnected},
        .samples = {},
    }};
    bool correct = true;
    std::size_t allocation_count = 0U;
    {
      const test::ScopedAllocationCounter allocations;
      for (std::uint64_t generation = 2U; generation <= 129U; ++generation) {
        if (!(*session)->prepare_activation(replay->owner, generation, absent)) {
          correct = false;
          break;
        }
        correct =
            correct && input.selection().state == InputState::kSafeDefault &&
            input.selection().reason == absent.front().selection.reason &&
            check_errors(ErrorCode::kInputEmpty, ErrorCode::kInputSelectionCardinalityMismatch);
        if (!(*session)->retire_activation(replay->owner, generation, true)) {
          correct = false;
          break;
        }
        correct = correct && check_errors(ErrorCode::kStaleEndpoint, ErrorCode::kStaleEndpoint);
      }
      allocation_count = allocations.count();
    }
    EXPECT_TRUE(correct);
    EXPECT_EQ(allocation_count, 0U);
    const auto after =
        detail::RuntimeChannelSessionTestAccess::input_resource_state(**session, replay->port);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->subscriber_count, resources->subscriber_count);
    EXPECT_EQ(after->payload_slot_count, resources->payload_slot_count);
    EXPECT_EQ(after->claim_pool_present, resources->claim_pool_present);
  }
}

// REQ: RUNTIME-055
// REQ: COMPILE-080
// REQ: COMPILE-087
// REQ: COMPILE-091
// REQ: COMPILE-098
// REQ: OPERATOR-020
TEST(RuntimeChannelBinding, OptionalPayloadFamiliesPreserveResourcesAndAccessorPrecedence) {
  // Key setup: exercise four payload families with unwired and connected-silent single-sample
  // inputs.
  // Key assertion: unwired ports own no payload/transport resources; phase and cardinality errors
  // persist.
  // Graph:
  //   source.output --(wired case only)--> sink.input [optional]
  //   clock drives sink; the unwired case has only sink
  check_optional_payload_resources_and_guards<schema::StringT>();
  check_optional_payload_resources_and_guards<Tensor>();
  check_optional_payload_resources_and_guards<schema::ImageT>();
  check_optional_payload_resources_and_guards<channel_test::CatalogProcessPod>();
}

// Export the sink inputs at the inner boundary for
// RunSessionTest.OptionalNestedAliasesDistinguishUnwiredSilentAndSelectedInputs.
// Graph aliases: leaf.{first,second,side} ==> sink.{first,second,optional}.
class OptionalAlignedLeaf final : public Graph {
 public:
  Input<std::uint32_t> first{"first"};
  Input<std::uint32_t> second{"second"};
  Input<std::uint32_t> side{"side"};
  [[nodiscard]] ClockKey timeline() const { return timeline_; }

 protected:
  void compose() override {
    timeline_ = import_clock("timeline");
    set_default_clock(timeline_);
    auto sink = op<test::RuntimeAlignedChannelSinkOperator>("sink");
    export_input(first, sink->first);
    export_input(second, sink->second);
    export_input(side, sink->optional);
  }

 private:
  ClockKey timeline_;
};

// Add a second exported-input boundary for
// RunSessionTest.OptionalNestedAliasesDistinguishUnwiredSilentAndSelectedInputs.
// Graph aliases: branch.{first,second,side} ==> leaf.{first,second,side}.
class OptionalAlignedBranch final : public Graph {
 public:
  Input<std::uint32_t> first{"first"};
  Input<std::uint32_t> second{"second"};
  Input<std::uint32_t> side{"side"};
  [[nodiscard]] ClockKey timeline() const { return timeline_; }

 protected:
  void compose() override {
    timeline_ = import_clock("timeline");
    set_default_clock(timeline_);
    auto leaf = embed<OptionalAlignedLeaf>("leaf");
    bind_clock(leaf, leaf->timeline(), timeline_);
    export_input(first, leaf->first);
    export_input(second, leaf->second);
    export_input(side, leaf->side);
  }

 private:
  ClockKey timeline_;
};

// Toggle declaration permission and request one restart to probe replay and lifecycle behavior.
// Used by RunSessionTest.OptionalDeclarationDriftRejectsBeforeCallbacks.
class OptionalReplayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& input = spec.input(side, "side");
    if (permission.load()) {
      input.optional();
    }
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).input(side, Exact{}).on_missing(side, SafeDefault{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext& context) override {
    if (side.selection().state != InputState::kSafeDefault ||
        side.selection().reason != SelectionReason::kNotConnected || side.receive() ||
        side.selected_batch()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto ordinal = callbacks.fetch_add(1U);
    if (ordinal == 0U) {
      auto receipt = context.request_restart();
      if (!receipt) {
        return make_unexpected(std::move(receipt).error());
      }
    } else {
      context.request_completion();
    }
    return {};
  }

  Input<std::uint32_t> side;
  static inline std::atomic_bool permission{true};
  static inline std::atomic_uint32_t callbacks{};
};

// Build the self-restartable owner used by
// RunSessionTest.OptionalAbsenceSurvivesLifecycleRestartAndFreshSessions.
// Graph: [unwired] target.side; target is clock-driven and can restart itself.
ExecutionPlan optional_replay_plan() {
  Graph graph{"optional-replay"};
  const auto target = graph.op<OptionalReplayOperator>("target");
  graph.lifecycle().allow_self_restart(target);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{.max_pending_requests = 4U,
                                                               .max_pending_transactions = 4U,
                                                               .max_receipts = 8U,
                                                               .max_fenced_operators = 8U,
                                                               .max_candidate_generations = 8U,
                                                               .max_events = 64U,
                                                               .max_audit_records = 16U,
                                                               .max_quiesce_records = 16U,
                                                               .max_rollback_records = 32U};
  return compile(graph, std::move(options));
}

// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalDeclarationDriftRejectsBeforeCallbacks) {
  // Key setup: compile an optional declaration, then replay it as required.
  // Key assertion: reconstruction reports a port mismatch before any callback runs.
  // Graph:
  //   [unwired]  target.side [optional at compile time; required at replay]
  OptionalReplayOperator::permission.store(true);
  OptionalReplayOperator::callbacks.store(0U);
  const auto plan = optional_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  OptionalReplayOperator::permission.store(false);
  detail::RuntimeReplayMismatchDetail mismatch;
  auto changed = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{75U}, {}, &mismatch);
  OptionalReplayOperator::permission.store(true);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().message, "RUNTIME_PORT_BINDING_UNAVAILABLE");
  EXPECT_TRUE(mismatch.present);
  EXPECT_EQ(mismatch.category, detail::RuntimeReplayMismatchCategory::kPorts);
  EXPECT_NE(mismatch.expected, mismatch.observed);
  EXPECT_EQ(OptionalReplayOperator::callbacks.load(), 0U);
}

// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalAbsenceSurvivesLifecycleRestartAndFreshSessions) {
  // Key setup: restart the same unwired operator within each of two fresh sessions.
  // Key assertion: each session completes after two callbacks with safe-default absence preserved.
  // Graph:
  //   [unwired]  target.side [optional; target requests its own restart]
  OptionalReplayOperator::permission.store(true);
  const auto plan = optional_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  for (unsigned run = 0U; run < 2U; ++run) {
    SCOPED_TRACE(run);
    OptionalReplayOperator::callbacks.store(0U);
    RunSession session = run_async(plan);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(OptionalReplayOperator::callbacks.load(), 2U);
  }
}

// REQ: RUNTIME-055
TEST(RuntimeChannelBinding, OptionalAbsenceSupportsFullAndOwnerResetWithoutPayloadStorage) {
  // Key setup: cycle an unwired binding through 128 activations, resets, and restart commit.
  // Key assertion: activation is allocation-free and resets require no payload edges or
  // subscribers.
  // Graph:
  //   [unwired]  target.side [optional]
  const auto plan = optional_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{76U});
  ASSERT_TRUE(exec) << exec.error().message;
  ASSERT_EQ((*exec)->operators().size(), 1U);
  const auto& instance = (*exec)->operators().front();
  const auto replay = detail::OperatorSpecCompilerAccess::port(*instance.replay_spec, 0U);
  const auto input = plan.input_selections().front().input;
  const std::array ports{detail::RuntimeChannelPortReplayRecord{
      .port = input,
      .owner = instance.vertex,
      .direction = PortDirection::kInput,
      .declaration = const_cast<void*>(replay.declaration),
      .type_ops = replay.runtime_type_ops,
  }};
  auto session = detail::RuntimeChannelSession::create((*exec)->plan(), ports);
  ASSERT_TRUE(session) << session.error().message;
  const std::array selection{detail::RuntimeChannelActivationSelection{
      .port = input,
      .selection = {.state = InputState::kSafeDefault, .reason = SelectionReason::kNotConnected},
      .samples = {},
  }};
  bool succeeded = true;
  std::size_t allocation_count = 0U;
  {
    const test::ScopedAllocationCounter allocations;
    for (std::uint64_t generation = 1U; generation <= 128U; ++generation) {
      if (!(*session)->prepare_activation(instance.vertex, generation, selection) ||
          !(*session)->retire_activation(instance.vertex, generation, true)) {
        succeeded = false;
        break;
      }
    }
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(succeeded);
  EXPECT_EQ(allocation_count, 0U);
  EXPECT_TRUE((*session)->reset());
  EXPECT_TRUE((*session)->reset(instance.vertex));
  const std::array affected{instance.vertex};
  EXPECT_TRUE((*session)->commit_restart(instance.vertex, affected));
  auto retired = (*session)->retire_terminal_input_generation(instance.vertex);
  ASSERT_TRUE(retired) << retired.error().message;
  EXPECT_TRUE(*retired);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.channel_subscribers().empty());
}

// Both groups have optional inputs, but each worker reconstructs only its own declaration.
// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalUnwiredInputsMaterializeOnlyForTheSelectedWorker) {
  // Key setup: put two unwired optional operators in separate worker process groups.
  // Key assertion: each worker invokes only its own operator and observes kNotConnected.
  // Graph:
  //   first worker:  [unwired]  a-optional.optional
  //   second worker: [unwired]  z-optional.optional
  Graph graph{"runtime-optional-worker"};
  const auto first = graph.op<test::RuntimeOptionalSafeDefaultOperator>("a-optional");
  const auto second = graph.op<test::RuntimeOptionalSafeDefaultOperator>("z-optional");
  graph.partition("optional-first").add(first);
  graph.partition("optional-second").add(second);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("optional-worker-clock"));
  constexpr std::array first_members{detail::GraphObjectSelector{.graph_path = "a-optional"}};
  constexpr std::array second_members{detail::GraphObjectSelector{.graph_path = "z-optional"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("first",
                         detail::ProcessPlacement{.members = first_members,
                                                  .launch_group = "optional-first-process"})
      .bind_process_role("second",
                         detail::ProcessPlacement{.members = second_members,
                                                  .launch_group = "optional-second-process"});
  const auto plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_EQ(state->input_selections.size(), 2U);
  for (const auto& selection : state->input_selections) {
    EXPECT_TRUE(selection.unwired_safe_default);
  }
  EXPECT_TRUE(state->payload_edges.empty());
  EXPECT_TRUE(state->channel_services.empty());
  EXPECT_TRUE(state->process_local_channel_services.empty());
  EXPECT_TRUE(state->cross_process_channel_services.empty());
  EXPECT_TRUE(state->channel_subscribers.empty());
  EXPECT_TRUE(state->process_local_channel_subscribers.empty());
  EXPECT_TRUE(state->cross_process_channel_subscribers.empty());
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
  ASSERT_EQ(roles.size(), 2U);
  for (const auto& role : roles) {
    SCOPED_TRACE(role.role);
    ASSERT_EQ(role.members.size(), 1U);
    const auto group = detail::process_group_for(*state, role.members.front());
    ASSERT_TRUE(group);
    test::RuntimeOptionalSafeDefaultOperator::compute_count.store(0U);
    RunSession session = detail::run_async_worker(plan, group);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::compute_count.load(), 1U);
    EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::observed_state.load(),
              InputState::kSafeDefault);
    EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::observed_reason.load(),
              SelectionReason::kNotConnected);
    EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::scalar_error.load(),
              ErrorCode::kInputEmpty);
    EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::batch_error.load(),
              ErrorCode::kInputSelectionCardinalityMismatch);
  }
}

// The owning worker combines real aligned drivers with two absent sides; the other worker
// reconstructs neither its producer nor its typed optional declarations.
// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalWorkerIsolationPreservesConnectedDriversAndUnwiredSides) {
  // Key setup: separate an aligned source/sink pair from an independent optional operator.
  // Key assertion: each worker executes only its own graph members and preserves optional absence.
  // Graph:
  //   mixed worker: mixed-source.first  --> mixed-sink.first
  //                 mixed-source.second --> mixed-sink.second
  //                 [unwired] mixed-sink.optional, mixed-sink.additional_optional
  //   other worker: [unwired] other.optional
  Graph graph{"runtime-optional-mixed-worker"};
  const auto source = graph.op<test::RuntimeAlignedChannelSourceOperator>("mixed-source");
  const auto sink = graph.op<test::RuntimeAlignedChannelSinkOperator>("mixed-sink");
  const auto other = graph.op<test::RuntimeOptionalSafeDefaultOperator>("other");
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 8U});
  graph.partition("mixed").add(source).add(sink);
  graph.partition("other").add(other);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("mixed-worker-clock"));
  constexpr std::array mixed_members{detail::GraphObjectSelector{.graph_path = "mixed-source"},
                                     detail::GraphObjectSelector{.graph_path = "mixed-sink"}};
  constexpr std::array other_members{detail::GraphObjectSelector{.graph_path = "other"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role(
          "mixed",
          detail::ProcessPlacement{.members = mixed_members, .launch_group = "mixed-process"})
      .bind_process_role(
          "other",
          detail::ProcessPlacement{.members = other_members, .launch_group = "other-process"});
  const auto plan = detail::compile_with_process_placement(graph, {}, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->payload_edges.size(), 2U);
  EXPECT_TRUE(state->cross_process_channel_services.empty());
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
  ASSERT_EQ(roles.size(), 2U);
  for (const auto& role : roles) {
    SCOPED_TRACE(role.role);
    const auto group = detail::process_group_for(*state, role.members.front());
    ASSERT_TRUE(group);
    test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
    test::RuntimeOptionalSafeDefaultOperator::compute_count.store(0U);
    RunSession session = detail::run_async_worker(plan, group);
    if (role.role == "mixed") {
      const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
      session.request_stop();
      session.wait();
      EXPECT_TRUE(observed);
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::compute_count.load(), 0U);
      EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
                InputState::kSafeDefault);
      EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
                SelectionReason::kNotConnected);
    } else {
      session.wait();
      EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
      EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::compute_count.load(), 1U);
    }
    EXPECT_TRUE(session.diagnostics().empty());
  }
}

// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalComputeSideRemainsUnselectedUntilPredicateFires) {
  // Key setup: evaluate five driver samples before the predicate permits compute.
  // Key assertion: the optional side stays unselected during evaluation and safe-defaults only in
  // compute.
  // Graph:
  //   source.driver --> sink.driver [predicate trigger]
  //   [unwired]        sink.side [optional compute input]
  using namespace test::run_session_channel_execution;
  reset_predicate_retirement_probe();
  const auto plan = make_predicate_retirement_plan(true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::size_t predicate_phases = 0U;
  for (const auto& aggregate : plan.missing_aggregates()) {
    if (aggregate.inputs.empty()) {
      continue;
    }
    ++predicate_phases;
    ASSERT_EQ(aggregate.inputs.size(), 1U);
    if (aggregate.phase == MissingSelectionPhase::kPredicateCompute) {
      EXPECT_EQ(aggregate.inputs.front().selector, InputSelectorKind::kExact);
      EXPECT_EQ(aggregate.inputs.front().missing, TemporalRecordMissingKind::kSafeDefault);
    } else {
      EXPECT_EQ(aggregate.phase, MissingSelectionPhase::kPredicateEvaluation);
    }
  }
  EXPECT_EQ(predicate_phases, 2U);
  RunSession session = run_async(plan);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.source_count, 5U);
  EXPECT_EQ(snapshot.evaluation_count, 5U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.computed_driver, 5U);
  EXPECT_EQ(snapshot.optional_not_selected_count, 5U);
  EXPECT_EQ(snapshot.optional_state, InputState::kSafeDefault);
  EXPECT_EQ(snapshot.optional_reason, SelectionReason::kNotConnected);
  EXPECT_EQ(snapshot.optional_scalar_error, ErrorCode::kInputEmpty);
  EXPECT_EQ(snapshot.optional_batch_error, ErrorCode::kInputSelectionCardinalityMismatch);
}

// A degraded evaluation input cannot prematurely select the optional compute-only side.
// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalComputeSidePreservesAnEarlierDegradedPredicatePhase) {
  // Key setup: connect a silent evaluation input while leaving the compute side unwired.
  // Key assertion: the degraded callback runs without selecting the later compute-only side.
  // Graph:
  //   source.driver --> sink.driver
  //   source.silent --> sink.evaluation_side [connected but silent]
  //   [unwired]        sink.side [optional compute input]
  using namespace test::run_session_channel_execution;
  reset_predicate_retirement_probe();
  const auto plan = make_predicate_retirement_plan(true, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.degraded_callback_count, 1U);
  EXPECT_EQ(snapshot.optional_not_selected_count, 0U);
  EXPECT_EQ(snapshot.evaluation_count, 0U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.optional_state, InputState::kNotSelected);
  EXPECT_EQ(snapshot.optional_reason, SelectionReason::kNotSelectedForActivation);
  EXPECT_EQ(snapshot.optional_scalar_error, ErrorCode::kInputNotSelected);
  EXPECT_EQ(snapshot.optional_batch_error, ErrorCode::kInputNotSelected);
}

// An optional side remains compute-only for both successful and timed-out windows.
// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalWindowSidePreservesSuccessfulAndDegradedWindowSelection) {
  // Key setup: run successor-closed and timeout-closed windows with an unwired compute side.
  // Key assertion: the side stays safely absent while the window is selected or safely defaulted.
  // Graph:
  //   source.anchor   --> sink.anchor
  //   source.windowed --> sink.windowed
  //   [unwired]           sink.side [optional]
  for (const auto script : {test::RuntimeWindowChannelScript::kSuccessorClose,
                            test::RuntimeWindowChannelScript::kTimeoutClose}) {
    SCOPED_TRACE(static_cast<unsigned>(script));
    test::RuntimeWindowChannelProbeState::reset(script);
    test::RuntimeWindowChannelSinkOperator::optional_checks.store(0U);
    const auto plan = test::make_runtime_window_channel_plan(false, true);
    ASSERT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
    session.request_stop();
    session.wait();
    EXPECT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(test::RuntimeWindowChannelSinkOperator::optional_checks.load(), 1U);
    EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(),
              script == test::RuntimeWindowChannelScript::kSuccessorClose
                  ? InputState::kSelected
                  : InputState::kSafeDefault);
  }
}

// Every exported boundary resolves to the real canonical payload edge. An omitted side
// retains only operator-owned absence; the two required members still form the activation.
// REQ: RUNTIME-055
TEST(RunSessionTest, OptionalNestedAliasesDistinguishUnwiredSilentAndSelectedInputs) {
  // Key setup: route aligned inputs through two exported graph boundaries in three side-input
  // cases.
  // Key assertion: canonical side selection distinguishes kNotConnected, kNeverSeen, and a real
  // sample.
  // Graph:
  //   source.first  --> branch.first  ==> leaf.first  ==> sink.first
  //   source.second --> branch.second ==> leaf.second ==> sink.second
  //   source.optional --(if wired)--> branch.side ==> leaf.side ==> sink.optional
  //   [unwired] sink.additional_optional; ==> denotes an exported input alias
  for (unsigned scenario = 0U; scenario < 3U; ++scenario) {
    SCOPED_TRACE(scenario);
    test::RuntimeAlignedChannelProbeState::reset(
        scenario == 2U ? test::RuntimeAlignedChannelScript::kSideInputReference
                       : test::RuntimeAlignedChannelScript::kExactBoundary);
    Graph graph{"optional-alias-runtime"};
    const auto source = graph.op<test::RuntimeAlignedChannelSourceOperator>("source");
    const auto branch = graph.embed<OptionalAlignedBranch>("branch");
    const auto clock = graph.add_clock<RealtimeClock>("clock");
    graph.set_default_clock(clock);
    graph.bind_clock(branch, branch->timeline(), clock);
    graph.add_flow(source->first, branch->first, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->second, branch->second, ConnectionOptions{.queue_depth = 8U});
    if (scenario != 0U) {
      graph.add_flow(source->optional, branch->side, ConnectionOptions{.queue_depth = 8U});
    }
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
    session.request_stop();
    session.wait();
    EXPECT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
              scenario == 2U ? InputState::kSelected : InputState::kSafeDefault);
    EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
              scenario == 0U   ? SelectionReason::kNotConnected
              : scenario == 1U ? SelectionReason::kNeverSeen
                               : SelectionReason::kNone);
    if (scenario == 2U) {
      EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional.load(), 3U);
    }
  }
}

}  // namespace
}  // namespace holoscan
