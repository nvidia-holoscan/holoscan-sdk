/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/time/synthetic_clock.hpp>
#include "common/allocation_counter.hpp"
#include "runtime/runtime_fixture_observation.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail {
namespace {
using namespace std::chrono_literals;

struct ObservationGate {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered{};
  bool released{};
  void enter() {
    std::unique_lock lock(mutex);
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return released; });
  }
  bool await() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return entered; });
  }
  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};

struct ObservationProbe {
  ObservationGate source;
  ObservationGate collector;
  std::atomic<std::int32_t> captured{};
};

struct ReleaseObservationGates {
  std::shared_ptr<ObservationProbe> probe;
  ~ReleaseObservationGates() {
    probe->source.release();
    probe->collector.release();
  }
};

class ObservationSource final : public Operator<> {
 public:
  explicit ObservationSource(std::shared_ptr<ObservationProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }
  Contract contract() const override {
    Contract c;
    c.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return c;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    Sample<std::int32_t> sample{42, {}};
    sample.metadata.capture_timestamp_ns = 900000000;
    sample.metadata.source_clock_id = context.clock().id();
    auto emitted = out.emit(std::move(sample));
    probe_->source.enter();
    return emitted;
  }
  Output<std::int32_t> out;

 private:
  std::shared_ptr<ObservationProbe> probe_;
};

class ObservationOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnClock{.period = 10ms})
        .input(in, LatestWithin{.max_age = 100ms})
        .on_missing(in, Skip{});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return make_unexpected(value.error());
    return out.emit(std::move(*value));
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;
};

class ObservationCollector final : public Operator<> {
 public:
  explicit ObservationCollector(std::shared_ptr<ObservationProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }
  expected<void, Error> compute(ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return make_unexpected(value.error());
    probe_->captured.store(value->data);
    probe_->collector.enter();
    return {};
  }
  Input<std::int32_t> in;

 private:
  std::shared_ptr<ObservationProbe> probe_;
};

ExecutionPlan observation_plan(const std::shared_ptr<ObservationProbe>& probe,
                               const std::shared_ptr<SyntheticClock>& clock) {
  Graph graph{"observation"};
  const auto source = graph.op<ObservationSource>("source", probe);
  const auto tested = graph.op<ObservationOperator>("tested");
  const auto collector = graph.op<ObservationCollector>("collector", probe);
  graph.add_flow(source->out, tested->in);
  graph.add_flow(tested->out, collector->in);
  graph.set_default_clock(graph.bind_shared_clock("clock", clock));
  graph.partition("fixture").include_all();
  return compile(graph);
}

template <class Predicate>
bool await_observation(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

VertexId observed_vertex(const ExecutionPlan& plan, std::string_view name) {
  for (const auto& row : plan.graph().vertices())
    if (row.name == name)
      return row.id;
  return {};
}

bool has_retired(const RuntimeFixtureObservation& observation, VertexId owner) {
  return std::any_of(
      observation.records().begin(), observation.records().end(), [&](const auto& record) {
        return record.kind == FixtureObservationKind::kCandidateFinished && record.owner == owner &&
               record.admitted && record.outcome == FixtureCandidateOutcome::kRetired;
      });
}

class ObservedChainSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out").max_emits_per_compute(1U); }
  Contract contract() const override {
    Contract c;
    c.trigger(OnClock{.period = 1ms});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override { return out.emit(std::int32_t{42}); }
  Output<std::int32_t> out;
};

class ObservedChainRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(2U);
    spec.output(out, "out").max_emits_per_compute(1U);
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnEach{in});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return make_unexpected(value.error());
    return out.emit(std::move(*value));
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;
};

class ObservedChainSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(in, "in").queue_depth(2U); }
  Contract contract() const override {
    Contract c;
    c.trigger(OnEach{in});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto value = in.receive();
    return value ? expected<void, Error>{} : make_unexpected(value.error());
  }
  Input<std::int32_t> in;
};

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, RetiredCallbackSurvivesLaterCausalFrontierFailure) {
  Graph graph{"observed-chain"};
  const auto source = graph.op<ObservedChainSource>("source");
  const auto first = graph.op<ObservedChainRelay>("first");
  const auto second = graph.op<ObservedChainRelay>("second");
  const auto sink = graph.op<ObservedChainSink>("sink");
  const ConnectionOptions options{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 2U};
  graph.add_flow(source->out, first->in, options);
  graph.add_flow(first->out, second->in, options);
  graph.add_flow(second->out, sink->in, options);
  graph.set_default_clock(graph.bind_shared_clock("clock", std::make_shared<SyntheticClock>(0)));
  graph.partition("fixture").include_all();
  auto control = std::make_shared<RuntimeDirectContinuationTestControl>();
  control->force_causal_freeze_failure();
  const auto plan =
      RuntimeTestPlanAccess::with_direct_continuation_test_control(compile(graph), control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto observation = std::make_shared<RuntimeFixtureObservation>(32U, plan.ports().size(), 32U);
  auto session = start_operator_test_run(plan, observation);
  const bool cleanup_recorded = control->wait_until_causal_freeze_failure_cleanup();
  if (!cleanup_recorded)
    session.request_stop();
  session.wait();
  ASSERT_TRUE(cleanup_recorded);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(observation->invalid());
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().service, "partition_causal_frontier_freeze");
  std::size_t admitted{};
  for (const auto& record : observation->records()) {
    if (record.kind != FixtureObservationKind::kCandidateFinished || !record.admitted)
      continue;
    ++admitted;
    EXPECT_EQ(record.owner, observed_vertex(plan, "source"));
    EXPECT_EQ(record.outcome, FixtureCandidateOutcome::kRetired);
    ASSERT_EQ(record.publications.size(), 1U);
    EXPECT_EQ(record.publications.front().sequence, 1U);
  }
  EXPECT_EQ(admitted, 1U);
  const auto cleanup = control->causal_freeze_failure_observation();
  EXPECT_EQ(cleanup.pending_after_cancel, 0U);
  EXPECT_EQ(cleanup.outstanding_after_cancel, 0U);
  EXPECT_TRUE(cleanup.retired_all);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, DisabledScopeDoesNotInvokeRecordFactory) {
  static_assert(std::is_empty_v<FixtureActivationObservation<false>>);
  std::size_t calls{};
  {
    FixtureActivationObservation<false> scope{nullptr, [&]() noexcept {
                                                ++calls;
                                                return FixtureObservationRecord{};
                                              }};
    scope.generation(7U);
    scope.select_inputs({});
    scope.admitted(true);
    scope.finish(FixtureCandidateOutcome::kRetired);
  }
  EXPECT_EQ(calls, 0U);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, ObservedScopeBuildsOnceAndPreservesEarlyFailure) {
  RuntimeFixtureObservation observation{2U, 0U, 0U};
  std::size_t calls{};
  {
    FixtureActivationObservation<true> scope{
        &observation, [&]() noexcept {
          ++calls;
          return FixtureObservationRecord{.owner = VertexId{3U}};
        }};
    scope.generation(7U);
    scope.admitted(true);
  }
  ASSERT_EQ(calls, 1U);
  ASSERT_EQ(observation.records().size(), 1U);
  EXPECT_EQ(observation.records()[0].owner, VertexId{3U});
  EXPECT_EQ(observation.records()[0].activation_generation, 7U);
  EXPECT_TRUE(observation.records()[0].admitted);
  EXPECT_EQ(observation.records()[0].outcome, FixtureCandidateOutcome::kFailed);
  EXPECT_FALSE(observation.invalid());
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, RejectedOutsideEmitPreservesPrefixAndLaterActivation) {
  RuntimeFixtureObservation observation{4U, 0U, 2U};
  ASSERT_TRUE(observation.append(FixtureObservationRecord{.owner = VertexId{3U}}, {}, {}));
  const auto prefix = observation.records();
  observation.publication(
      VertexId{3U},
      7U,
      FixtureObservedPublication{
          .output = PortId{9U}, .sequence = 11U, .error = ErrorCode::kInvalidArgument});
  ASSERT_FALSE(observation.invalid());
  ASSERT_EQ(prefix.size(), 1U);
  EXPECT_EQ(prefix[0].owner, VertexId{3U});
  ASSERT_EQ(observation.records().size(), 2U);
  const auto& rejected = observation.records()[1];
  EXPECT_EQ(rejected.kind, FixtureObservationKind::kPublicationRejectedOutsideActivation);
  EXPECT_FALSE(rejected.admitted);
  ASSERT_EQ(rejected.publications.size(), 1U);
  EXPECT_EQ(rejected.publications[0].error, ErrorCode::kInvalidArgument);
  EXPECT_FALSE(rejected.publications[0].sequence_consumed);

  observation.begin_activation(
      FixtureObservationRecord{.owner = VertexId{3U}, .activation_generation = 8U});
  observation.publication(
      VertexId{3U},
      8U,
      FixtureObservedPublication{.output = PortId{9U}, .sequence = 11U, .sequence_consumed = true});
  observation.finish_activation(true, FixtureCandidateOutcome::kRetired);
  ASSERT_EQ(observation.records().size(), 3U);
  EXPECT_EQ(observation.records()[2].kind, FixtureObservationKind::kCandidateFinished);
  EXPECT_EQ(observation.records()[2].outcome, FixtureCandidateOutcome::kRetired);
  EXPECT_FALSE(observation.invalid());
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, OutsideSuccessOrConsumedAuthorityInvalidatesWithoutErasingPrefix) {
  for (const auto publication : {FixtureObservedPublication{.output = PortId{9U}, .sequence = 11U},
                                 FixtureObservedPublication{.output = PortId{9U},
                                                            .sequence = 11U,
                                                            .error = ErrorCode::kFailure,
                                                            .sequence_consumed = true}}) {
    RuntimeFixtureObservation observation{2U, 0U, 1U};
    ASSERT_TRUE(observation.append(FixtureObservationRecord{.owner = VertexId{3U}}, {}, {}));
    observation.publication(VertexId{3U}, 7U, publication);
    EXPECT_TRUE(observation.invalid());
    ASSERT_EQ(observation.records().size(), 1U);
    EXPECT_EQ(observation.records()[0].owner, VertexId{3U});
  }
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, ChannelOwnsObservationThroughRejectedInactiveEmit) {
  runtime_channel_binding_test::SessionFixture fixture;
  auto channels = RuntimeChannelSession::create(fixture.plan, fixture.replay);
  ASSERT_TRUE(channels);
  auto observation = std::make_shared<RuntimeFixtureObservation>(4U, 0U, 2U);
  const std::weak_ptr<RuntimeFixtureObservation> weak = observation;
  ASSERT_TRUE((*channels)->observe_fixture_publications(observation));
  observation.reset();
  ASSERT_FALSE(weak.expired());

  // No activation is live. The binding still invokes the observed thunk, so its raw pointer
  // must be backed by the channel itself after the caller releases its strong observation owner.
  const auto emitted = fixture.source.output.emit(std::int32_t{7});
  ASSERT_FALSE(emitted);
  {
    const auto retained = weak.lock();
    ASSERT_NE(retained, nullptr);
    EXPECT_FALSE(retained->invalid());
    ASSERT_EQ(retained->records().size(), 1U);
    EXPECT_EQ(retained->records()[0].kind,
              FixtureObservationKind::kPublicationRejectedOutsideActivation);
    ASSERT_EQ(retained->records()[0].publications.size(), 1U);
    EXPECT_EQ(retained->records()[0].publications[0].error, emitted.error().code);
  }
  channels->reset();
  EXPECT_TRUE(weak.expired());
}

class ObservationUnsupportedTensorSource final : public Operator<> {
 public:
  explicit ObservationUnsupportedTensorSource(std::shared_ptr<std::atomic_uint32_t> computes)
      : computes_(std::move(computes)) {}
  void setup(OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(
            TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                .dtype = DLDataType{kDLUInt, 8U, 1U},
                                                .rank = 1U,
                                                .strides = TensorStridePolicy::kContiguousOnly},
                             .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 1U},
                             .storage = TensorOutputStorage::kRuntimePool});
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    ++*computes_;
    return {};
  }
  Output<Tensor> out;

 private:
  std::shared_ptr<std::atomic_uint32_t> computes_;
};

class ObservationUnsupportedTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<Tensor> in;
};

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, UnsupportedPayloadHasRegisteredPortDiagnosticBeforeCompute) {
  auto computes = std::make_shared<std::atomic_uint32_t>(0U);
  Graph graph{"unsupported-observation"};
  const auto source = graph.op<ObservationUnsupportedTensorSource>("source", computes);
  const auto sink = graph.op<ObservationUnsupportedTensorSink>("sink");
  graph.add_flow(source->out, sink->in);
  graph.set_default_clock(graph.add_clock<SyntheticClock>("clock", 0));
  graph.partition("fixture").include_all();
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto observation = std::make_shared<RuntimeFixtureObservation>(8U, 1U, 8U);
  auto session = start_operator_test_run(plan, observation);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(computes->load(), 0U);
  const auto diagnostics = session.diagnostics();
  const auto found = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
    return row.token == "RUNTIME_FIXTURE_OBSERVATION_UNSUPPORTED";
  });
  ASSERT_NE(found, diagnostics.end());
  EXPECT_NE(found->id, DiagnosticId{});
  EXPECT_EQ(found->source.vertex, observed_vertex(plan, "source"));
  EXPECT_NE(found->source.port, PortId{});
  EXPECT_TRUE(observation->invalid());
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, OnlyCommittedImmutableRecordsAreReadable) {
  RuntimeFixtureObservation observation{2U, 2U, 2U};
  EXPECT_TRUE(observation.records().empty());
  const std::array inputs{FixtureObservedInput{
      .input = PortId{7U}, .selection = {InputState::kAbsent, SelectionReason::kTooStale}}};
  const std::array publications{FixtureObservedPublication{.output = PortId{9U}, .sequence = 11U}};
  EXPECT_TRUE(
      observation.append(FixtureObservationRecord{.owner = VertexId{3U}}, inputs, publications));
  const auto first = observation.records();
  ASSERT_EQ(first.size(), 1U);
  EXPECT_EQ(first[0].owner, VertexId{3U});
  ASSERT_EQ(first[0].inputs.size(), 1U);
  EXPECT_EQ(first[0].inputs[0].selection.reason, SelectionReason::kTooStale);
  ASSERT_EQ(first[0].publications.size(), 1U);
  EXPECT_EQ(first[0].publications[0].sequence, 11U);
  EXPECT_TRUE(observation.append(FixtureObservationRecord{.owner = VertexId{4U}}, {}, {}));
  EXPECT_EQ(first.size(), 1U);
  EXPECT_EQ(first[0].owner, VertexId{3U});
  EXPECT_EQ(observation.records().size(), 2U);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, OverflowInvalidatesEvidenceWithoutPublishingPartialRecord) {
  RuntimeFixtureObservation observation{1U, 1U, 1U};
  ASSERT_TRUE(observation.append({}, {}, {}));
  EXPECT_FALSE(observation.append({}, {}, {}));
  EXPECT_TRUE(observation.invalid());
  EXPECT_EQ(observation.records().size(), 1U);
  RuntimeFixtureObservation details{2U, 1U, 1U};
  const std::array publications{FixtureObservedPublication{}, FixtureObservedPublication{}};
  EXPECT_FALSE(details.append({}, {}, publications));
  EXPECT_TRUE(details.invalid());
  EXPECT_TRUE(details.records().empty());
  EXPECT_FALSE(details.append({}, {}, {}));
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, AcquireReadSeesAllFieldsAndDetails) {
  constexpr std::size_t count = 1024U;
  RuntimeFixtureObservation observation{count, 1U, count};
  std::atomic<bool> writer_ok{true};
  std::jthread writer{[&] {
    for (std::size_t i = 0; i < count; ++i) {
      const std::array publications{FixtureObservedPublication{
          .output = PortId{7U}, .sequence = static_cast<std::uint64_t>(i + 1U)}};
      if (!observation.append(FixtureObservationRecord{.owner = VertexId{3U}}, {}, publications)) {
        writer_ok.store(false);
        break;
      }
    }
  }};
  std::size_t checked{};
  while (checked != count && writer_ok.load()) {
    const auto records = observation.records();
    for (; checked < records.size(); ++checked) {
      ASSERT_EQ(records[checked].owner, VertexId{3U});
      ASSERT_EQ(records[checked].publications.size(), 1U);
      EXPECT_EQ(records[checked].publications[0].sequence, checked + 1U);
    }
    std::this_thread::yield();
  }
  writer.join();
  EXPECT_TRUE(writer_ok.load());
  EXPECT_EQ(checked, count);
  EXPECT_FALSE(observation.invalid());
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, PublicationArenaIsBoundedAcrossRecordsWithoutHotAllocation) {
  RuntimeFixtureObservation observation{4U, 1U, 1U};
  const std::array publications{FixtureObservedPublication{.output = PortId{7U}, .sequence = 1U}};
  bool first{}, second{};
  std::size_t allocations{};
  {
    const holoscan::test::ScopedAllocationCounter counter;
    first = observation.append({}, {}, publications);
    second = observation.append({}, {}, publications);
    allocations = counter.count();
  }
  EXPECT_TRUE(first);
  EXPECT_FALSE(second);
  EXPECT_TRUE(observation.invalid());
  EXPECT_EQ(observation.records().size(), 1U);
  EXPECT_EQ(allocations, 0U);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, PublicationMustMatchTheActiveOwnerAndGeneration) {
  RuntimeFixtureObservation observation{4U, 1U, 4U};
  ASSERT_TRUE(observation.append(FixtureObservationRecord{.owner = VertexId{1U}}, {}, {}));
  observation.begin_activation(
      FixtureObservationRecord{.owner = VertexId{3U}, .activation_generation = 7U});
  observation.publication(VertexId{4U}, 7U, FixtureObservedPublication{});
  EXPECT_TRUE(observation.invalid());
  ASSERT_EQ(observation.records().size(), 1U);
  EXPECT_EQ(observation.records()[0].owner, VertexId{1U});
  RuntimeFixtureObservation other{4U, 1U, 4U};
  ASSERT_TRUE(other.append(FixtureObservationRecord{.owner = VertexId{1U}}, {}, {}));
  other.begin_activation(
      FixtureObservationRecord{.owner = VertexId{3U}, .activation_generation = 7U});
  // Even a native rejection within an active scope must authenticate its generation.
  other.publication(
      VertexId{3U}, 8U, FixtureObservedPublication{.error = ErrorCode::kInvalidArgument});
  EXPECT_TRUE(other.invalid());
  ASSERT_EQ(other.records().size(), 1U);
  EXPECT_EQ(other.records()[0].owner, VertexId{1U});
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, HeldSourceAndCollectorCannotPublishRetirementEarly) {
  auto probe = std::make_shared<ObservationProbe>();
  auto clock = std::make_shared<SyntheticClock>(0);
  const auto plan = observation_plan(probe, clock);
  ASSERT_TRUE(plan.ok());
  auto observation = std::make_shared<RuntimeFixtureObservation>(64U, 4U, 16U);
  auto session = start_operator_test_run(plan, observation);
  const ReleaseObservationGates release{probe};
  ASSERT_TRUE(probe->source.await());
  EXPECT_FALSE(has_retired(*observation, observed_vertex(plan, "source")));
  probe->source.release();
  ASSERT_TRUE(await_observation([&] {
    const auto records = observation->records();
    return has_retired(*observation, observed_vertex(plan, "source")) &&
           std::any_of(
               records.begin(),
               records.end(),
               [](const auto& r) { return r.kind == FixtureObservationKind::kInputIngested; }) &&
           std::any_of(records.begin(), records.end(), [&](const auto& r) {
             return r.owner == observed_vertex(plan, "tested") &&
                    r.outcome == FixtureCandidateOutcome::kSuppressed;
           });
  }));
  clock->advance_to(1000000000);
  ASSERT_TRUE(probe->collector.await());
  EXPECT_EQ(probe->captured.load(), 42);
  EXPECT_FALSE(has_retired(*observation, observed_vertex(plan, "collector")));
  probe->collector.release();
  ASSERT_TRUE(await_observation(
      [&] { return has_retired(*observation, observed_vertex(plan, "collector")); }));
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_FALSE(observation->invalid());
  bool joined{};
  for (const auto& source : observation->records()) {
    if (source.owner != observed_vertex(plan, "tested") || !source.admitted)
      continue;
    ASSERT_EQ(source.publications.size(), 1U);
    EXPECT_EQ(source.publications.front().error, ErrorCode::kSuccess);
    EXPECT_TRUE(source.publications.front().sequence_consumed);
    for (const auto& sink : observation->records()) {
      if (sink.owner != observed_vertex(plan, "collector") || !sink.admitted)
        continue;
      ASSERT_EQ(sink.inputs.size(), 1U);
      joined = sink.inputs.front().sample.source_sequence == source.publications.front().sequence;
    }
  }
  EXPECT_TRUE(joined);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, FailedRetirementKeepsReceiptWithoutSuccessEvidence) {
  auto probe = std::make_shared<ObservationProbe>();
  probe->source.release();
  probe->collector.release();
  auto clock = std::make_shared<SyntheticClock>(0);
  const auto plan = RuntimeTestPlanAccess::fail_next_channel_activation_retirement_for_test(
      observation_plan(probe, clock));
  ASSERT_TRUE(plan.ok());
  auto observation = std::make_shared<RuntimeFixtureObservation>(64U, 4U, 16U);
  auto session = start_operator_test_run(plan, observation);
  ASSERT_TRUE(await_observation([&] { return !session.is_running(); }));
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_FALSE(has_retired(*observation, observed_vertex(plan, "source")));
  bool accepted_but_failed{};
  for (const auto& record : observation->records()) {
    if (record.owner == observed_vertex(plan, "source") && record.admitted &&
        record.outcome == FixtureCandidateOutcome::kFailed && record.publications.size() == 1U)
      accepted_but_failed = record.publications.front().error == ErrorCode::kSuccess;
  }
  EXPECT_TRUE(accepted_but_failed);
}

// REQ: RUNTIME-062
TEST(RuntimeFixtureObservation, EachSessionHasFreshRunIdentityAndStorage) {
  auto probe = std::make_shared<ObservationProbe>();
  probe->source.release();
  probe->collector.release();
  const auto plan = observation_plan(probe, std::make_shared<SyntheticClock>(0));
  ASSERT_TRUE(plan.ok());
  auto first = std::make_shared<RuntimeFixtureObservation>(64U, 4U, 16U);
  auto second = std::make_shared<RuntimeFixtureObservation>(64U, 4U, 16U);
  for (const auto& observation : {first, second}) {
    auto session = start_operator_test_run(plan, observation);
    ASSERT_TRUE(await_observation(
        [&] { return has_retired(*observation, observed_vertex(plan, "source")); }));
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
  }
  EXPECT_EQ(first->plan_generation(), second->plan_generation());
  EXPECT_NE(first->run_generation(), second->run_generation());
  EXPECT_NE(first->records().data(), second->records().data());
  EXPECT_FALSE(first->bind(first->plan_generation(), second->run_generation()));
  EXPECT_TRUE(first->invalid());
  EXPECT_FALSE(second->invalid());
}

}  // namespace
}  // namespace holoscan::detail
