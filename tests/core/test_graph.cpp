/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "core/graph_compiler_access.hpp"

#include "graph_test_support.hpp"

namespace holoscan {

// GraphArrayPodA and GraphArrayPodB share the same ChannelTypeIdentity but differ in
// fixed-array element counts, exercising POD layout conflict detection.
// Representative test: PodFixedArraysRetainElementCountsAndConflictByLayout.
struct GraphArrayPodA {
  std::array<std::uint8_t, 2> values{};
  std::array<std::uint8_t, 2> remainder{};
};

struct GraphArrayPodB {
  std::array<std::uint8_t, 3> values{};
  std::array<std::uint8_t, 1> remainder{};
};

template <>
struct ChannelTypeIdentity<GraphArrayPodA> {
  static constexpr std::string_view name = "nvidia.holoscan.test.GraphArrayPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelTypeIdentity<GraphArrayPodB> : ChannelTypeIdentity<GraphArrayPodA> {};

template <>
struct ChannelPodLayout<GraphArrayPodA> {
  static constexpr auto fields = pod_fields(
      pod_field("values", &GraphArrayPodA::values, offsetof(GraphArrayPodA, values)),
      pod_field("remainder", &GraphArrayPodA::remainder, offsetof(GraphArrayPodA, remainder)));
};

template <>
struct ChannelPodLayout<GraphArrayPodB> {
  static constexpr auto fields = pod_fields(
      pod_field("values", &GraphArrayPodB::values, offsetof(GraphArrayPodB, values)),
      pod_field("remainder", &GraphArrayPodB::remainder, offsetof(GraphArrayPodB, remainder)));
};

// POD with an implicit padding gap between fields (uint8_t followed by uint32_t).
// Verifies that the graph rejects POD layouts whose declared fields do not cover all bytes.
// Representative test: PodCoverageAndGraphWideIdentityCatalogFailClosed.
struct GraphGapPod {
  std::uint8_t first{};
  std::uint32_t second{};
};

template <>
struct ChannelTypeIdentity<GraphGapPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.GraphGapPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<GraphGapPod> {
  static constexpr auto fields =
      pod_fields(pod_field("first", &GraphGapPod::first, offsetof(GraphGapPod, first)),
                 pod_field("second", &GraphGapPod::second, offsetof(GraphGapPod, second)));
};

// POD with implicit tail padding (uint32_t followed by uint8_t, leaving 3 pad bytes).
// Exercises the same coverage-rejection path as GraphGapPod.
// Representative test: PodCoverageAndGraphWideIdentityCatalogFailClosed.
struct GraphTailPod {
  std::uint32_t first{};
  std::uint8_t second{};
};

template <>
struct ChannelTypeIdentity<GraphTailPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.GraphTailPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<GraphTailPod> {
  static constexpr auto fields =
      pod_fields(pod_field("first", &GraphTailPod::first, offsetof(GraphTailPod, first)),
                 pod_field("second", &GraphTailPod::second, offsetof(GraphTailPod, second)));
};

// Alias/compatibility family for POD type-identity tests:
//   AliasA  — canonical type (identity: GraphPodAlias v1, single "value" field).
//   AliasB  — same identity as AliasA but different field layout (conflict on import).
//   CompatibleB — same identity AND layout as AliasA (accepted downstream).
//   DifferentB  — completely distinct identity; causes a port-type-mismatch on add_flow.
// Representative test: DistinctCppPodTypesInteroperateOnlyByCompleteContract.
struct GraphPodAliasB {
  std::uint16_t first{};
  std::uint16_t second{};
};

struct GraphPodCompatibleB {
  std::uint32_t value{};
};

struct GraphPodDifferentB {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<GraphPodAliasB> : ChannelTypeIdentity<GraphPodAliasA> {};

template <>
struct ChannelTypeIdentity<GraphPodCompatibleB> : ChannelTypeIdentity<GraphPodAliasA> {};

template <>
struct ChannelTypeIdentity<GraphPodDifferentB> {
  static constexpr std::string_view name = "nvidia.holoscan.test.GraphPodDifferent";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<GraphPodAliasB> {
  static constexpr auto fields =
      pod_fields(pod_field("first", &GraphPodAliasB::first, offsetof(GraphPodAliasB, first)),
                 pod_field("second", &GraphPodAliasB::second, offsetof(GraphPodAliasB, second)));
};

template <>
struct ChannelPodLayout<GraphPodCompatibleB> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &GraphPodCompatibleB::value, offsetof(GraphPodCompatibleB, value)));
};

template <>
struct ChannelPodLayout<GraphPodDifferentB> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &GraphPodDifferentB::value, offsetof(GraphPodDifferentB, value)));
};

namespace {

using namespace test::graph_support;

// Parameter struct for ParamOp and CountingParamOp.
// A value of 11 causes ParamOp to declare an extra "extra" output, changing the port shape
// and enabling ParameterReplacementRemapsPortsAtomicallyAndFlowHandlesResolveAgain to
// probe the port-shape-change rejection.
struct GraphParams {
  std::uint32_t value{};
};

[[nodiscard]] bool graph_even_value(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

struct ValidatedGraphParams {
  std::uint32_t bounded{2U};
  std::uint32_t even{2U};
};

struct Utf8ValidatedGraphParams {
  std::uint32_t even{2U};
};

constexpr std::string_view kLongUtf8ValidatorId{
    "x"
    "éééééééé"
    "éééééééé"
    "éééééééé"
    "éééééééé"
    "éééééééé"
    "éééééééé"};
static_assert(kLongUtf8ValidatorId.size() == 97U);

// Two distinct local enum types sharing the same underlying type.
// EnumChangingParamOp switches output between them based on params, so that
// ParameterReplacementPreservesExactLocalPortType can assert that a type change
// is rejected atomically without mutating the accepted graph.
enum class GraphEnumA : std::uint8_t {
  kValue,
};

enum class GraphEnumB : std::uint8_t {
  kValue,
};

template <typename T>
concept ImportableGraphTopic =
    requires(Graph& graph) { graph.template import_topic<T>("holomq://compile-time-check"); };

static_assert(!ImportableGraphTopic<GraphEnumA>);

// Two distinct schema identities used to verify that changing schema, capture-time, or
// attach options produces a different canonical graph hash.
// Representative test: ExternalTopicProvenanceParticipatesInCanonicalHash.
constexpr SchemaIdentity kBoundarySchemaA{
    .canonical_name = "nvidia.holoscan.test.BoundarySchemaA",
    .version = 1,
    .schema_sha256 = {std::byte{0x11}},
    .id = SchemaId{.value = {std::byte{0x21}}},
};

constexpr SchemaIdentity kBoundarySchemaB{
    .canonical_name = "nvidia.holoscan.test.BoundarySchemaB",
    .version = 1,
    .schema_sha256 = {std::byte{0x12}},
    .id = SchemaId{.value = {std::byte{0x22}}},
};

HOLOSCAN_PARAM_SCHEMA(GraphParams, holoscan::param_schema("nvidia.holoscan.test.GraphParams", 1),
                      holoscan::field<&GraphParams::value>("value"));

HOLOSCAN_PARAM_SCHEMA(ValidatedGraphParams,
                      holoscan::param_schema("nvidia.holoscan.test.ValidatedGraphParams", 1),
                      holoscan::field<&ValidatedGraphParams::bounded>("bounded").range(1U, 4U),
                      holoscan::field<&ValidatedGraphParams::even>("even")
                          .validator<graph_even_value>("test.graph.even/v1"));

HOLOSCAN_PARAM_SCHEMA(Utf8ValidatedGraphParams,
                      holoscan::param_schema("nvidia.holoscan.test.Utf8ValidatedGraphParams", 1),
                      holoscan::field<&Utf8ValidatedGraphParams::even>("even")
                          .validator<graph_even_value>(kLongUtf8ValidatorId));

// Accepts GraphPodCompatibleB, which shares type identity and field layout with
// GraphPodAliasA; verifies that a fully compatible type is accepted at add_flow.
// Representative test: DistinctCppPodTypesInteroperateOnlyByCompleteContract.
class CompatiblePodTargetOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<GraphPodCompatibleB> input;
};

// Accepts GraphPodAliasB, which shares type identity with GraphPodAliasA but uses a
// conflicting field layout; exercises the catalog-conflict rejection path.
// Representative test: DistinctCppPodTypesInteroperateOnlyByCompleteContract.
class IncompatiblePodTargetOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<GraphPodAliasB> input;
};

// Accepts GraphPodDifferentB, which has a completely distinct type identity from
// GraphPodAliasA, provoking a GRAPH_PORT_TYPE_MISMATCH error at add_flow.
// Representative test: DistinctCppPodTypesInteroperateOnlyByCompleteContract.
class DifferentPodTargetOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<GraphPodDifferentB> input;
};

// Combines a non-inline emit policy (kCopyIfNeeded) with a non-zero max_inline_bytes
// on a Tensor output, which is an invalid configuration.
// Representative test: RejectsUnicodeEdgeWhitespaceAndInvalidClosedEnumValues.
class InvalidInlineOptionsOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    PayloadOptions options;
    options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
    options.max_inline_bytes = 64;
    spec.output(output, "output").payload_options(options);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Tracks live_count via atomic RAII to verify that a failed compose() destroys all
// operators allocated into the partial draft before returning.
// Representative test: FailedCompositionDiscardsItsPartialDraftBeforeReturning.
class CountingOp : public Operator<> {
 public:
  CountingOp() { ++live_count; }
  ~CountingOp() override { --live_count; }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  static inline std::atomic<int> live_count{};
};

// Counts constructor calls to verify compose() is not entered before propagating its
// terminal failure to a factory that the caller requests after the error.
// Representative test: LazyComposeFinishesBeforeRequestedFactoryOrBoundaryConstruction.
class SideEffectOp : public Operator<> {
 public:
  SideEffectOp() { ++constructor_calls; }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  static inline std::atomic<int> constructor_calls{};
};

// Declares a ReadableSource (deployment-selected wake) and a SpinSource (busy-poll wake).
// Used to verify readiness snapshot provenance and wake-requirement propagation.
// Representative test: ReadinessSnapshotCarriesStableProvenanceAndWakeRequirement.
class ReadinessOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.readable_source(readable, "readable");
    spec.spin_source(spin, "spin");
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  ReadableSource readable;
  SpinSource spin;
};

// Declares a NotificationSource and a DoorbellSource with explicit capacity,
// sender-reference capacity, and named clock key overrides per source.
// Representative test: CountedReadinessSnapshotRetainsCapacityAndClockProvenance.
class CountedReadinessOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification, "notification")
        .capacity(4)
        .sender_reference_capacity(3)
        .clock(ClockKey::named("notification_clock"));
    spec.doorbell_source(doorbell, "doorbell").capacity(5).clock(ClockKey::named("doorbell_clock"));
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  NotificationSource notification;
  DoorbellSource doorbell;
};

// Combines a POD output with a ReadableSource to exercise the multi-port allocation
// rollback path in CompoundMutationFaultsRollbackEveryLogicalCommit.
class PodReadinessOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    spec.readable_source(readable, "readable");
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<GraphPodPayload> output;
  ReadableSource readable;
};

// Param-driven operator whose port shape depends on the parameter value:
// params.value == 11 causes an additional "extra" output to be declared,
// changing the port shape and triggering GRAPH_PARAMETER_PORT_SHAPE_CHANGED.
// Representative test: ParameterReplacementRemapsPortsAtomicallyAndFlowHandlesResolveAgain.
class ParamOp : public Operator<GraphParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
    if (params().value == 11) {
      spec.output(extra, "extra");
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  Output<std::uint32_t> extra;
};

// Declares the derived output before its followed input so Graph capture must canonicalize the
// complete owner-local declaration set rather than depend on setup declaration order.
class OutputFirstRateOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").follows(input, RateRatio{1U, 4U});
    spec.input(input, "input").expected_rate(Rate::hz(240U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
  Input<std::uint32_t> input;
};

// Rate metadata is authored through the same declarations in several equivalent forms. The
// graph snapshot must retain normalized values and canonical declaration IDs rather than the
// source spelling or setup order.
class RateIdentityOp : public Operator<> {
 public:
  explicit RateIdentityOp(bool output_first = false, bool noncanonical = false,
                          bool changed_rate = false, bool explicit_depth = false)
      : output_first_(output_first),
        noncanonical_(noncanonical),
        changed_rate_(changed_rate),
        explicit_depth_(explicit_depth) {}

  void setup(OperatorSpec& spec) override {
    const Rate input_rate =
        changed_rate_ ? Rate::hz(121U) : (noncanonical_ ? Rate::hz(240U, 2U) : Rate::hz(120U));
    const RateRatio follows_ratio = noncanonical_ ? RateRatio{2U, 8U} : RateRatio{1U, 4U};
    if (output_first_) {
      spec.output(output, "output").follows(input, follows_ratio);
    }
    auto& input_spec = spec.input(input, "input").expected_rate(input_rate);
    if (explicit_depth_) {
      input_spec.queue_depth(3U);
    }
    if (!output_first_) {
      spec.output(output, "output").follows(input, follows_ratio);
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;

 private:
  bool output_first_{};
  bool noncanonical_{};
  bool changed_rate_{};
  bool explicit_depth_{};
};

// Reusable embedded graph for verifying that a follows declaration is canonicalized to the
// child-local PortId even though the authoring reference is a C++ declaration pointer.
class EmbeddedRateGraph : public Graph {
 protected:
  void compose() override { (void)op<RateIdentityOp>("rate"); }
};

// Deliberately follows a declaration outside this Operator so Graph capture can reject the raw
// authoring reference before any pointer escapes into an immutable snapshot.
class ForeignRateReferenceOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output").follows(*foreign);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  static inline const Input<std::uint32_t>* foreign{};
};

// Counts constructor calls to verify that parameter replacement constructs exactly one
// candidate instance rather than building tentative copies.
// Representative test: ParameterReplacementConstructsExactlyOneCandidate.
class CountingParamOp : public Operator<GraphParams> {
 public:
  CountingParamOp() { ++constructor_calls; }

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  static inline std::atomic<int> constructor_calls{};
};

// Holds bounded and custom-validated parameters for redacted validation-diagnostic checks.
// Representative test: ParameterReplacementReportsBoundedValidationIdentityWithoutValues.
class ValidatedParamOp : public Operator<ValidatedGraphParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Holds a parameter whose long Unicode validator identity exercises UTF-8-safe truncation.
// Representative test: ParameterValidationIdentityTruncationPreservesUtf8.
class Utf8ValidatedParamOp : public Operator<Utf8ValidatedGraphParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Switches output port type between GraphEnumA (params.value == 0) and GraphEnumB
// (params.value != 0) to probe port-shape-change rejection during replacement.
// Representative test: ParameterReplacementPreservesExactLocalPortType.
class EnumChangingParamOp : public Operator<GraphParams> {
 public:
  void setup(OperatorSpec& spec) override {
    if (params().value == 0U) {
      spec.output(enum_a, "output");
    } else {
      spec.output(enum_b, "output");
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<GraphEnumA> enum_a;
  Output<GraphEnumB> enum_b;
};

// Single-input operator accepting GraphEnumA; paired with EnumChangingParamOp to detect
// when parameter replacement would silently change the connected port's C++ type.
// Representative test: ParameterReplacementPreservesExactLocalPortType.
class EnumATargetOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<GraphEnumA> input;
};

// Counts compose() invocations; used to assert exactly-once lazy composition and that
// all subsequent view() calls share the result without re-entering compose().
// Representative test: LazyCompositionRunsExactlyOnceAndSnapshotsAreIndependent.
class LazyGraph final : public Graph {
 public:
  using Graph::Graph;
  int compose_calls{};

 protected:
  void compose() override {
    ++compose_calls;
    (void)op<PassOp>("lazy");
  }
};

// Supplies one embedded Operator so include_all authoring can prove that it retains a late
// expansion directive instead of capturing only the root members present at the call site.
class SinglePassGraph final : public Graph {
 protected:
  void compose() override { (void)op<PassOp>("embedded"); }
};

// Attempts to carry a child-local include_all directive across an embed boundary. Only the
// compile root may own that directive, so parent publication must reject the child atomically.
class IncludeAllChildGraph final : public Graph {
 protected:
  void compose() override { partition("child-all").include_all(); }
};

// Attempts to carry compile-root automatic-remainder intent across an embed boundary.
class AutomaticRemainderChildGraph final : public Graph {
 protected:
  void compose() override { auto_partition_remaining(); }
};

// Supplies an ordinary child-local manual partition. A root include_all directive cannot coexist
// with that second composed declaration even though neither child directive is include_all.
class ManualPartitionChildGraph final : public Graph {
 protected:
  void compose() override {
    auto op = this->op<PassOp>("embedded");
    partition("child-manual").add(op);
  }
};

// Calls view() from within compose(), triggering the reentrancy guard.
// Verifies that GRAPH_COMPOSE_REENTRANT is the sticky terminal failure for all retries.
// Representative test: ReentrantObservationFailsAndTheFailureIsSticky.
class ReentrantGraph final : public Graph {
 public:
  using Graph::Graph;

 protected:
  void compose() override { (void)view(); }
};

// Creates one CountingOp then throws std::runtime_error to exercise partial-draft discard.
// Representative test: FailedCompositionDiscardsItsPartialDraftBeforeReturning.
class PartialFailureGraph final : public Graph {
 protected:
  void compose() override {
    (void)op<CountingOp>("partial");
    throw std::runtime_error{"compose failure detail"};
  }
};

// Throws a 200-char detail string with an embedded newline at position 17 to verify that
// compose exception context is truncated to 160 chars and newlines are replaced with '?'.
// Representative test: ExceptionContextIsSanitizedBoundedAndSticky.
class LongContextFailureGraph final : public Graph {
 protected:
  void compose() override {
    std::string detail(200, 'x');
    detail[17] = '\n';
    throw std::runtime_error{detail};
  }
};

// Registers the same operator name twice to trigger GRAPH_COMPOSE_FAILED with a cause.
// Representative test: TypedAndUnknownComposeFailuresUseStableStickyTokens.
class TypedFailureGraph final : public Graph {
 protected:
  void compose() override {
    (void)op<PassOp>("duplicate");
    (void)op<PassOp>("duplicate");
  }
};

// Throws a non-exception value (integer literal 7) from compose() to exercise the
// GRAPH_COMPOSE_UNKNOWN_EXCEPTION sticky failure path.
// Representative test: TypedAndUnknownComposeFailuresUseStableStickyTokens.
class UnknownFailureGraph final : public Graph {
 protected:
  void compose() override { throw 7; }
};

// Throws from compose() before any operator or boundary is requested.
// Ensures the terminal failure is visible to the caller before the requested factory runs.
// Representative test: LazyComposeFinishesBeforeRequestedFactoryOrBoundaryConstruction.
class FailBeforeFactoryGraph final : public Graph {
 protected:
  void compose() override { throw std::runtime_error{"fail before requested factory"}; }
};

// Stores an ImportedTopic handle that escapes compose() before compose() throws.
// Verifies that the handle is invalidated (graph_valid() == false) and does not dangle.
// Representative test: FailedCompositionInvalidatesExternalHandlesWithoutDanglingPointers.
class ExternalHandleFailureGraph final : public Graph {
 public:
  std::optional<ImportedTopic<std::uint32_t>> topic;

 protected:
  void compose() override {
    topic = import_topic<std::uint32_t>("holomq://partial");
    throw std::runtime_error{"discard boundary"};
  }
};

// Escapes an OperatorHandle, a FlowHandle, and a TensorOutputPortConfigurator* from a
// failing compose() to verify tombstoning of handles and storage retention for configurators.
// Representative test: FailedCompositionInvalidatesFlowAndRetainsConfiguratorStorage.
class EscapedDraftHandleFailureGraph final : public Graph {
 public:
  std::optional<OperatorHandle<PassOp>> operator_handle;
  std::optional<FlowHandle<std::uint32_t>> flow;
  TensorOutputPortConfigurator* tensor_configurator{};

 protected:
  void compose() override {
    auto source = op<PassOp>("source");
    operator_handle = source;
    auto target = op<PassOp>("target");
    flow = add_flow(source->output, target->input);
    auto tensor = op<TensorOp>("tensor");
    tensor_configurator = &configure(tensor->output);
    throw std::runtime_error{"discard draft handles"};
  }
};

// Blocks inside compose() until a shared_future is signaled to control concurrency tests.
// The compose_started promise lets tests wait until compose() is in the controlled wait.
// Representative test: ConcurrentObserversShareOneComposerAndWaitForTerminalSuccess.
class BlockingGraph final : public Graph {
 public:
  explicit BlockingGraph(std::shared_future<void> release) : release_(std::move(release)) {}

  std::promise<void> compose_started;
  std::atomic<int> compose_calls{};

 protected:
  void compose() override {
    ++compose_calls;
    compose_started.set_value();
    release_.wait();
    (void)op<PassOp>("only");
  }

 private:
  std::shared_future<void> release_;
};

// Blocks inside compose() then throws std::bad_alloc to simulate a diagnostic-allocation
// failure while a second thread is already waiting for the compose result.
// Representative test: DiagnosticAllocationFailureStillPublishesTerminalFailure.
class BlockingAllocationFailureGraph final : public Graph {
 public:
  explicit BlockingAllocationFailureGraph(std::shared_future<void> release)
      : release_(std::move(release)) {}

  std::promise<void> compose_started;

 protected:
  void compose() override {
    compose_started.set_value();
    release_.wait();
    throw std::bad_alloc{};
  }

 private:
  std::shared_future<void> release_;
};

// Clock that throws in its constructor; exercises the GRAPH_CLOCK_CONSTRUCTION_FAILED
// wrapper and verifies that the internal constructor detail is not leaked to callers.
// Representative test: ClockOwnershipConversionsAndFactoryFailuresAreExplicit.
class ThrowingClock final : public Clock {
 public:
  ThrowingClock() { throw std::runtime_error{"private clock construction detail"}; }

  [[nodiscard]] double time() const override { return 0.0; }
  [[nodiscard]] std::int64_t timestamp() const override { return 0; }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

// Tracks constructor_calls and live_count to verify that add_clock() constructs exactly
// one validation instance and immediately destroys it without retaining a live copy.
// Representative test: ClockFactoryValidationConstructsAndDiscardsOneInstance.
class CountingClock final : public Clock {
 public:
  CountingClock() {
    ++constructor_calls;
    ++live_count;
  }
  ~CountingClock() override { --live_count; }

  [[nodiscard]] double time() const override { return 0.0; }
  [[nodiscard]] std::int64_t timestamp() const override { return 0; }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t) override { return {}; }

  static inline std::atomic<int> constructor_calls{};
  static inline std::atomic<int> live_count{};
};

// Builds a one-topic graph importing type T, snapshots it, and asserts that the external
// topic's scalar identity kind matches expected_kind.
// Representative test: ExternalScalarKindsCoverThePortableFamily.
template <typename T>
void expect_external_scalar_kind(ScalarIdentityKind expected_kind) {
  Graph graph;
  (void)graph.import_topic<T>("holomq://scalar-kind");
  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1);
  const auto& scalar = view.external_topics()[0].payload_identity.scalar;
  ASSERT_TRUE(scalar);
  if (scalar) {
    EXPECT_EQ(scalar->kind, expected_kind);
  }
}

// REQ: GRAPH-001
// REQ: GRAPH-003
static_assert(detail::GraphOwnedCapture<std::string>);
static_assert(detail::GraphOwnedCapture<std::string_view>);
static_assert(!detail::GraphOwnedCapture<int*>);
static_assert(!detail::GraphOwnedCapture<std::span<const int>>);
static_assert(!detail::GraphOwnedCapture<std::shared_ptr<int>>);
static_assert(!detail::GraphOwnedCapture<std::weak_ptr<int>>);
static_assert(!detail::GraphOwnedCapture<std::unique_ptr<int>>);
static_assert(!detail::GraphOwnedCapture<std::reference_wrapper<int>>);
static_assert(!detail::GraphOwnedCapture<std::initializer_list<int>>);
static_assert(!detail::GraphOwnedCapture<std::function<void()>>);
static_assert(!detail::GraphOwnedCapture<std::ranges::ref_view<std::string>>);

// Move-only type used in a static_assert to confirm that move-only types are not
// GraphOwnedCapture; types that cannot be cheaply copied must not be captured by value.
struct MoveOnlyCapture {
  MoveOnlyCapture() = default;
  MoveOnlyCapture(const MoveOnlyCapture&) = delete;
  MoveOnlyCapture(MoveOnlyCapture&&) = default;
};
static_assert(!detail::GraphOwnedCapture<MoveOnlyCapture>);

// Concepts used below to assert that Graph does not expose GXF-style lifecycle methods
// (compile, run, run_async, start, stop, restart, request_restart, or scheduler).
// Graph is an authoring-only object; execution primitives live in separate types.
template <typename T>
concept HasCompileMember = requires(T& value) { value.compile(); };
template <typename T>
concept HasRunMember = requires(T& value) { value.run(); };
template <typename T>
concept HasRunAsyncMember = requires(T& value) { value.run_async(); };
template <typename T>
concept HasStartMember = requires { &T::start; } || requires(T& value) { value.start(); } ||
                         requires(T& value) { value.start(0); };
template <typename T>
concept HasStopMember = requires { &T::stop; } || requires(T& value) { value.stop(); } ||
                        requires(T& value) { value.stop(0); };
template <typename T>
concept HasRestartMember = requires { &T::restart; } || requires(T& value) { value.restart(); } ||
                           requires(T& value) { value.restart(0); };
template <typename T>
concept HasRequestRestartMember = requires { &T::request_restart; } || requires(T& value) {
  value.request_restart();
} || requires(T& value) { value.request_restart(0); };
template <typename T>
concept HasSchedulerMember =
    requires { &T::scheduler; } || requires(T& value) { value.scheduler(); };
template <typename T>
concept HasPartitionMember = requires(T& value) { value.partition("partition"); };
template <typename Source, typename Target>
concept HasTypedFlow = requires(Graph& graph, const Source& source, const Target& target) {
  graph.add_flow(source, target);
};
static_assert(!HasCompileMember<Graph>);
static_assert(!HasRunMember<Graph>);
static_assert(!HasRunAsyncMember<Graph>);
static_assert(!HasStartMember<Graph>);
static_assert(!HasStopMember<Graph>);
static_assert(!HasRestartMember<Graph>);
static_assert(!HasRequestRestartMember<Graph>);
static_assert(!HasSchedulerMember<Graph>);
static_assert(HasPartitionMember<Graph>);
static_assert(HasTypedFlow<Output<std::uint32_t>, Input<std::uint32_t>>);
static_assert(!HasTypedFlow<Output<std::uint32_t>, Input<std::uint64_t>>);
static_assert(HasTypedFlow<Output<GraphPodAliasA>, Input<GraphPodCompatibleB>>);
static_assert(HasTypedFlow<Output<GraphPodAliasA>, Input<GraphPodAliasB>>);
static_assert(HasTypedFlow<Output<GraphPodAliasA>, Input<GraphPodDifferentB>>);
// REQ: GRAPH-002
// REQ: GRAPH-009
TEST(GraphComposition, LazyCompositionRunsExactlyOnceAndSnapshotsAreIndependent) {
  LazyGraph graph{"root"};

  const GraphView first = graph.view();
  const GraphView second = graph.view();

  EXPECT_EQ(graph.compose_calls, 1);
  ASSERT_EQ(first.vertices().size(), 1);
  EXPECT_EQ(first.vertices()[0].name, "lazy");
  EXPECT_EQ(first.generation(), second.generation());

  (void)graph.op<PassOp>("later");
  const GraphView third = graph.view();
  EXPECT_EQ(first.vertices().size(), 1);
  EXPECT_EQ(third.vertices().size(), 2);
  EXPECT_GT(third.generation(), first.generation());
}

// REQ: GRAPH-002
TEST(GraphComposition, ReentrantObservationFailsAndTheFailureIsSticky) {
  ReentrantGraph graph{"reentrant"};
  const Error first = graph_error_from([&] { (void)graph.view(); });
  const Error second = graph_error_from([&] { (void)graph.view(); });
  EXPECT_EQ(first.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(first.message, "GRAPH_COMPOSE_REENTRANT");
  EXPECT_EQ(second.message, first.message);
}

// REQ: GRAPH-002
TEST(GraphComposition, FailedCompositionDiscardsItsPartialDraftBeforeReturning) {
  // Key setup: CountingOp::live_count is zeroed so any leaked operator is detectable.
  CountingOp::live_count = 0;
  PartialFailureGraph graph;
  const Error first = graph_error_from([&] { (void)graph.view(); });
  EXPECT_EQ(first.message, "GRAPH_COMPOSE_EXCEPTION");
  // Key assertion: live_count must be zero both times — the partial draft was fully destroyed.
  EXPECT_EQ(CountingOp::live_count.load(), 0);
  const Error second = graph_error_from([&] { (void)graph.view(); });
  EXPECT_EQ(second.message, first.message);
  EXPECT_EQ(CountingOp::live_count.load(), 0);

  const auto first_diagnostic = detail::GraphCompilerAccess::composition_diagnostic(graph);
  const auto second_diagnostic = detail::GraphCompilerAccess::composition_diagnostic(graph);
  if (!first_diagnostic || !second_diagnostic) {
    FAIL() << "terminal composition diagnostics were not retained";
    return;
  }
  EXPECT_EQ(first_diagnostic->primary.code, ErrorCode::kFailure);
  EXPECT_EQ(first_diagnostic->primary.message, "GRAPH_COMPOSE_EXCEPTION");
  EXPECT_FALSE(first_diagnostic->cause);
  EXPECT_EQ(first_diagnostic->context, "compose failure detail");
  EXPECT_LE(first_diagnostic->context.size(), 160U);
  EXPECT_EQ(second_diagnostic->primary.code, first_diagnostic->primary.code);
  EXPECT_EQ(second_diagnostic->primary.message, first_diagnostic->primary.message);
  EXPECT_EQ(second_diagnostic->context, first_diagnostic->context);
}

// REQ: GRAPH-002
TEST(GraphComposition, TypedAndUnknownComposeFailuresUseStableStickyTokens) {
  TypedFailureGraph typed;
  const Error typed_first = graph_error_from([&] { (void)typed.view(); });
  const Error typed_second = graph_error_from([&] { (void)typed.view(); });
  EXPECT_EQ(typed_first.code, ErrorCode::kFailure);
  EXPECT_EQ(typed_first.message, "GRAPH_COMPOSE_FAILED");
  EXPECT_EQ(typed_second.message, typed_first.message);
  const auto typed_diagnostic = detail::GraphCompilerAccess::composition_diagnostic(typed);
  const auto typed_replay = detail::GraphCompilerAccess::composition_diagnostic(typed);
  if (!typed_diagnostic || !typed_replay || !typed_diagnostic->cause || !typed_replay->cause) {
    FAIL() << "typed composition cause was not retained";
    return;
  }
  EXPECT_EQ(typed_diagnostic->primary.code, ErrorCode::kFailure);
  EXPECT_EQ(typed_diagnostic->primary.message, "GRAPH_COMPOSE_FAILED");
  EXPECT_EQ(typed_diagnostic->cause->code, ErrorCode::kDuplicateName);
  EXPECT_EQ(typed_diagnostic->cause->message, "GRAPH_OPERATOR_NAME_DUPLICATE");
  EXPECT_EQ(typed_replay->primary.code, typed_diagnostic->primary.code);
  EXPECT_EQ(typed_replay->primary.message, typed_diagnostic->primary.message);
  EXPECT_EQ(typed_replay->cause->code, typed_diagnostic->cause->code);
  EXPECT_EQ(typed_replay->cause->message, typed_diagnostic->cause->message);
  EXPECT_TRUE(typed_diagnostic->context.empty());

  UnknownFailureGraph unknown;
  const Error unknown_error = graph_error_from([&] { (void)unknown.view(); });
  EXPECT_EQ(unknown_error.code, ErrorCode::kFailure);
  EXPECT_EQ(unknown_error.message, "GRAPH_COMPOSE_UNKNOWN_EXCEPTION");
}

// REQ: GRAPH-002
TEST(GraphComposition, ExceptionContextIsSanitizedBoundedAndSticky) {
  LongContextFailureGraph graph;
  const Error first = graph_error_from([&] { (void)graph.view(); });
  const Error second = graph_error_from([&] { (void)graph.view(); });
  EXPECT_EQ(first.code, ErrorCode::kFailure);
  EXPECT_EQ(first.message, "GRAPH_COMPOSE_EXCEPTION");
  EXPECT_EQ(second.code, first.code);
  EXPECT_EQ(second.message, first.message);

  const auto diagnostic = detail::GraphCompilerAccess::composition_diagnostic(graph);
  const auto replay = detail::GraphCompilerAccess::composition_diagnostic(graph);
  if (!diagnostic || !replay) {
    FAIL() << "bounded composition context was not retained";
    return;
  }
  EXPECT_EQ(diagnostic->context.size(), 160U);
  EXPECT_EQ(diagnostic->context[17], '?');
  EXPECT_EQ(replay->context, diagnostic->context);
  EXPECT_FALSE(diagnostic->cause);
}

// REQ: GRAPH-002
// REQ: GRAPH-003
TEST(GraphComposition, LazyComposeFinishesBeforeRequestedFactoryOrBoundaryConstruction) {
  SideEffectOp::constructor_calls = 0;
  FailBeforeFactoryGraph operator_graph;
  const Error operator_error =
      graph_error_from([&] { (void)operator_graph.op<SideEffectOp>("never"); });
  EXPECT_EQ(operator_error.message, "GRAPH_COMPOSE_EXCEPTION");
  EXPECT_EQ(SideEffectOp::constructor_calls.load(), 0);

  FailBeforeFactoryGraph boundary_graph;
  const Error boundary_error =
      graph_error_from([&] { (void)boundary_graph.import_topic<std::uint32_t>("holomq://never"); });
  EXPECT_EQ(boundary_error.message, "GRAPH_COMPOSE_EXCEPTION");
}

// REQ: GRAPH-002
// REQ: GRAPH-011
TEST(GraphComposition, FailedCompositionInvalidatesExternalHandlesWithoutDanglingPointers) {
  ExternalHandleFailureGraph graph;
  EXPECT_EQ(graph_error_from([&] { (void)graph.view(); }).message, "GRAPH_COMPOSE_EXCEPTION");
  ASSERT_TRUE(graph.topic.has_value());
  // GoogleTest assertions establish this invariant, but clang-tidy does not model the macro.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_FALSE(graph.topic.value().graph_valid());
  EXPECT_THROW((void)graph.topic.value().output(), RuntimeError);
  // NOLINTEND(bugprone-unchecked-optional-access)
}

// REQ: GRAPH-002
// REQ: GRAPH-011
TEST(GraphComposition, FailedCompositionInvalidatesFlowAndRetainsConfiguratorStorage) {
  EscapedDraftHandleFailureGraph graph;
  const Error failure = graph_error_from([&] { (void)graph.view(); });
  EXPECT_EQ(failure.message, "GRAPH_COMPOSE_EXCEPTION");
  if (!graph.flow) {
    FAIL() << "failed composition did not retain the escaped flow tombstone";
    return;
  }
  EXPECT_FALSE(graph.flow->graph_valid());
  if (!graph.operator_handle) {
    FAIL() << "failed composition did not retain the escaped Operator handle";
    return;
  }
  const auto& operator_handle = *graph.operator_handle;
  EXPECT_FALSE(operator_handle.graph_valid());
  const Error handle_error = graph_error_from([&] { (void)operator_handle->input; });
  EXPECT_EQ(handle_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(handle_error.message, "GRAPH_HANDLE_INVALID");
  ASSERT_NE(graph.tensor_configurator, nullptr);
  EXPECT_THROW(graph.tensor_configurator->max_emits_per_compute(1), RuntimeError);
}

// REQ: GRAPH-002
TEST(GraphComposition, ConcurrentObserversShareOneComposerAndWaitForTerminalSuccess) {
  // Key setup: BlockingGraph holds compose() in a controlled wait until release fires;
  //            two async threads call view() concurrently while compose() is in-flight.
  std::promise<void> release;
  BlockingGraph graph{release.get_future().share()};
  auto started = graph.compose_started.get_future();
  auto first = std::async(std::launch::async, [&] { return graph.view(); });
  if (started.wait_for(std::chrono::seconds{1}) != std::future_status::ready) {
    release.set_value();
    FAIL() << "composer did not enter the controlled wait";
    return;
  }
  auto second = std::async(std::launch::async, [&] { return graph.view(); });

  EXPECT_EQ(first.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);
  release.set_value();
  EXPECT_EQ(first.get().vertices().size(), 1);
  EXPECT_EQ(second.get().vertices().size(), 1);
  // Key assertion: compose_calls == 1 confirms exactly one compose() ran despite two callers.
  EXPECT_EQ(graph.compose_calls.load(), 1);
}

// REQ: GRAPH-002
TEST(GraphComposition, DiagnosticAllocationFailureStillPublishesTerminalFailure) {
  // Key setup: fail_composition_diagnostic_capture() pre-arms allocation failure so that
  //            the diagnostic-capture step inside compose() itself throws bad_alloc.
  std::promise<void> release;
  BlockingAllocationFailureGraph graph{release.get_future().share()};
  detail::GraphCompilerAccess::fail_composition_diagnostic_capture(graph);
  auto started = graph.compose_started.get_future();
  const auto observe = [&graph] {
    try {
      (void)graph.view();
    } catch (const RuntimeError& error) {
      return make_error(error);
    }
    return Error{ErrorCode::kSuccess};
  };

  auto first = std::async(std::launch::async, observe);
  if (started.wait_for(std::chrono::seconds{1}) != std::future_status::ready) {
    release.set_value();
    FAIL() << "composer did not enter the controlled diagnostic wait";
    return;
  }
  auto second = std::async(std::launch::async, observe);
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds{10}), std::future_status::timeout);

  release.set_value();
  ASSERT_EQ(first.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  ASSERT_EQ(second.wait_for(std::chrono::seconds{1}), std::future_status::ready);
  const Error first_error = first.get();
  const Error second_error = second.get();
  // Key assertion: even when diagnostic capture fails, a kFailure with empty message is
  //                published so that waiting observers are unblocked with a terminal status.
  EXPECT_EQ(first_error.code, ErrorCode::kFailure);
  EXPECT_TRUE(first_error.message.empty());
  EXPECT_EQ(second_error.code, first_error.code);
  EXPECT_EQ(second_error.message, first_error.message);

  const auto diagnostic = detail::GraphCompilerAccess::composition_diagnostic(graph);
  ASSERT_TRUE(diagnostic);
  if (diagnostic) {
    EXPECT_EQ(diagnostic->primary.code, ErrorCode::kFailure);
    EXPECT_TRUE(diagnostic->primary.message.empty());
    EXPECT_FALSE(diagnostic->cause);
    EXPECT_TRUE(diagnostic->context.empty());
  }
  EXPECT_EQ(observe().code, ErrorCode::kFailure);
}

// REQ: GRAPH-003
// REQ: GRAPH-004
TEST(GraphComposition, OperatorCaptureIsOwnedAndHandlesDoNotInventPlanIdentity) {
  Graph graph{"root"};
  std::string source = "owned";
  auto handle = graph.op<PassOp>("pass", std::string_view{source});
  source.assign("changed");

  ASSERT_TRUE(handle);
  ASSERT_NE(handle.operator->().operator->(), nullptr);
  EXPECT_EQ(handle->label_, "owned");
  EXPECT_EQ(handle.name(), "pass");
  EXPECT_FALSE(handle.ref().valid());
  EXPECT_FALSE(handle.plan_bound());
  EXPECT_EQ(handle.id(), 0);
  EXPECT_EQ(handle.plan_generation(), 0);
}

// REQ: GRAPH-004
TEST(GraphComposition, OperatorHandleChainedAccessFailsClosedAfterGraphRetirement) {
  std::optional<OperatorHandle<PassOp>> handle;
  {
    Graph graph{"root"};
    handle = graph.op<PassOp>("pass", std::string{"owned"});
    EXPECT_EQ(handle.value()->label_, "owned");
  }

  ASSERT_TRUE(handle.has_value());
  EXPECT_FALSE(handle->graph_valid());
  const Error error = graph_error_from([&] { (void)handle.value()->input; });
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_HANDLE_INVALID");
}

// REQ: GRAPH-006
//   source(PassOp) --[output→input]--> target(PassOp)
TEST(GraphComposition, TypedFlowRetainsSeparateSetupAndGraphOverrideLayers) {
  Graph graph{"root"};
  auto source = graph.op<PassOp>("source");
  auto target = graph.op<PassOp>("target");
  auto flow = graph.add_flow(
      source->output,
      target->input,
      ConnectionOptions{
          .queue_policy = QueuePolicy::kDropOldest, .queue_depth = 2, .topic_name = "authored"});
  graph.configure(target->input).queue_depth(1).diagnostic_alias("sink");
  graph.configure(flow).queue_depth(1).topic_name("override");

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1);
  EXPECT_EQ(view.edges()[0].add_flow_options.queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(view.edges()[0].add_flow_options.queue_depth, 2);
  EXPECT_EQ(view.edges()[0].graph_override.queue_depth, 1);
  EXPECT_EQ(view.edges()[0].graph_override.topic_name, "override");
  ASSERT_EQ(view.ports().size(), 4);
  const auto target_port = view.edges()[0].target;
  const auto found = std::find_if(view.ports().begin(), view.ports().end(), [&](const auto& port) {
    return port.id == target_port;
  });
  ASSERT_NE(found, view.ports().end());
  EXPECT_EQ(found->setup.queue_depth, 2);
  EXPECT_EQ(found->graph_override.queue_depth, 1);
  EXPECT_EQ(found->graph_override.diagnostic_alias, "sink");
}

// REQ: GRAPH-033
TEST(GraphComposition, RateCaptureCanonicalizesOutputFirstFollowsAndPreservesDepthPresence) {
  Graph graph{"root"};
  (void)graph.op<OutputFirstRateOp>("rate");

  const GraphView view = graph.view();
  ASSERT_EQ(view.ports().size(), 2U);
  const auto input = std::ranges::find(view.ports(), std::string_view{"input"}, &PortRecord::name);
  const auto output =
      std::ranges::find(view.ports(), std::string_view{"output"}, &PortRecord::name);
  ASSERT_NE(input, view.ports().end());
  ASSERT_NE(output, view.ports().end());

  EXPECT_FALSE(input->setup.queue_depth.has_value());
  EXPECT_EQ(input->setup.rate_kind, PortRateKind::kExpectedInput);
  EXPECT_EQ(input->setup.declared_expected_rate, Rate::hz(240U));
  EXPECT_EQ(output->setup.rate_kind, PortRateKind::kFollowsInput);
  EXPECT_EQ(output->setup.follows_input, input->id);
  EXPECT_EQ(output->setup.follows_ratio, (RateRatio{1U, 4U}));
}

// REQ: GRAPH-009
// REQ: GRAPH-033
// Equivalent rate spellings and owner-local declaration order must produce one semantic graph
// identity. A changed rate or authored queue depth is semantic and must change that identity.
TEST(GraphComposition, RateMetadataCanonicalizesEquivalentFormsAndHashesSemanticChanges) {
  Graph normalized{"rate-identity"};
  (void)normalized.op<RateIdentityOp>("rate");

  Graph equivalent_spelling{"rate-identity"};
  (void)equivalent_spelling.op<RateIdentityOp>("rate", true, true);

  // Key assertion: reduced rates/ratios and declaration order are not part of the identity.
  EXPECT_EQ(normalized.view().identity().hash, equivalent_spelling.view().identity().hash);

  Graph changed_rate{"rate-identity"};
  (void)changed_rate.op<RateIdentityOp>("rate", false, false, true);
  EXPECT_NE(normalized.view().identity().hash, changed_rate.view().identity().hash);

  Graph changed_depth{"rate-identity"};
  (void)changed_depth.op<RateIdentityOp>("rate", false, false, false, true);
  EXPECT_NE(normalized.view().identity().hash, changed_depth.view().identity().hash);
}

// REQ: GRAPH-009
// REQ: GRAPH-033
// Embedded instances are keyed by decoded graph paths, while follows references are rewritten to
// canonical child-local IDs. Reversing instance declaration order must not perturb the hash.
TEST(GraphComposition, EmbeddedRateInstancesCanonicalizeFollowsReferencesAndOrder) {
  Graph first{"embedded-rate-identity"};
  (void)first.embed<EmbeddedRateGraph>("z");
  (void)first.embed<EmbeddedRateGraph>("a");

  Graph second{"embedded-rate-identity"};
  (void)second.embed<EmbeddedRateGraph>("a");
  (void)second.embed<EmbeddedRateGraph>("z");

  EXPECT_EQ(first.view().identity().hash, second.view().identity().hash);

  const GraphView view = first.view();
  const auto input = std::ranges::find(view.ports(), std::string_view{"input"}, &PortRecord::name);
  const auto output =
      std::ranges::find(view.ports(), std::string_view{"output"}, &PortRecord::name);
  ASSERT_NE(input, view.ports().end());
  ASSERT_NE(output, view.ports().end());
  ASSERT_TRUE(output->setup.follows_input.has_value());
  EXPECT_EQ(output->setup.follows_input, input->id);
}

// REQ: GRAPH-033
TEST(GraphComposition, RateCaptureRejectsForeignFollowsReferenceWithoutPublishingState) {
  Graph graph{"root"};
  Input<std::uint32_t> foreign{"foreign"};
  ForeignRateReferenceOp::foreign = &foreign;

  const Error error = graph_error_from([&] { (void)graph.op<ForeignRateReferenceOp>("bad"); });
  ForeignRateReferenceOp::foreign = nullptr;
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "PORT_RATE_REFERENCE_INVALID");
  EXPECT_TRUE(graph.view().ports().empty());
}

// REQ: GRAPH-027
// REQ: GRAPH-028
TEST(GraphComposition, DropOldestQueuePolicyRetainsAuthoredDepthAndIdentity) {
  static_assert(static_cast<std::uint8_t>(QueuePolicy::kBlocking) == 0U);
  static_assert(static_cast<std::uint8_t>(QueuePolicy::kDropOldest) == 1U);

  const auto populate_graph = [](Graph& graph, std::uint32_t queue_depth) {
    auto source = graph.op<PassOp>("source");
    auto target = graph.op<PassOp>("target");
    (void)graph.add_flow(source->output,
                         target->input,
                         ConnectionOptions{
                             .queue_policy = QueuePolicy::kDropOldest,
                             .queue_depth = queue_depth,
                         });
  };

  Graph depth_one{"root"};
  Graph depth_many{"root"};
  populate_graph(depth_one, 1U);
  populate_graph(depth_many, 4U);
  const GraphView depth_one_view = depth_one.view();
  const GraphView depth_many_view = depth_many.view();

  ASSERT_EQ(depth_one_view.edges().size(), 1U);
  EXPECT_EQ(depth_one_view.edges().front().add_flow_options.queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(depth_one_view.edges().front().add_flow_options.queue_depth, 1U);
  ASSERT_EQ(depth_many_view.edges().size(), 1U);
  EXPECT_EQ(depth_many_view.edges().front().add_flow_options.queue_policy,
            QueuePolicy::kDropOldest);
  EXPECT_EQ(depth_many_view.edges().front().add_flow_options.queue_depth, 4U);
  EXPECT_NE(depth_one_view.identity().hash, depth_many_view.identity().hash);

  auto source = depth_one.op<PassOp>("invalid-source");
  auto target = depth_one.op<PassOp>("invalid-target");
  // Value two is the first value outside the closed queue-policy vocabulary.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_THROW(
      (void)depth_one.add_flow(source->output,
                               target->input,
                               ConnectionOptions{.queue_policy = static_cast<QueuePolicy>(2U)}),
      RuntimeError);
}

// REQ: GRAPH-008
//   source(PassOp) --[output→input]--> target(PassOp)
TEST(GraphComposition, NonTensorTryConfigurePreservesPortAndFlowOverrides) {
  Graph graph{"root"};
  auto source = graph.op<PassOp>("source");
  auto target = graph.op<PassOp>("target");
  auto flow = graph.add_flow(source->output, target->input);

  auto input = graph.try_configure(target->input);
  auto output = graph.try_configure(source->output);
  auto configured_flow = graph.try_configure(flow);
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  ASSERT_TRUE(configured_flow);
  input.value().queue_depth(4).diagnostic_alias("input_override");
  output.value().max_emits_per_compute(3).late_joiner_history_depth(2).diagnostic_alias(
      "output_override");
  configured_flow.value()
      .queue_policy(QueuePolicy::kDropOldest)
      .queue_depth(5)
      .topic_name("flow_override")
      .max_inflight_samples(6);

  Graph foreign;
  const auto foreign_input = foreign.try_configure(target->input);
  const auto foreign_output = foreign.try_configure(source->output);
  const auto foreign_flow = foreign.try_configure(flow);
  ASSERT_FALSE(foreign_input);
  ASSERT_FALSE(foreign_output);
  ASSERT_FALSE(foreign_flow);
  EXPECT_EQ(foreign_input.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(foreign_output.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(foreign_flow.error().code, ErrorCode::kInvalidArgument);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1);
  const EdgeRecord& edge = view.edges()[0];
  EXPECT_EQ(edge.graph_override.queue_policy, QueuePolicy::kDropOldest);
  EXPECT_EQ(edge.graph_override.queue_depth, 5);
  EXPECT_EQ(edge.graph_override.topic_name, "flow_override");
  EXPECT_EQ(edge.graph_override.max_inflight_samples, 6);

  const auto source_port =
      std::find_if(view.ports().begin(), view.ports().end(), [&](const PortRecord& port) {
        return port.id == edge.source;
      });
  const auto target_port =
      std::find_if(view.ports().begin(), view.ports().end(), [&](const PortRecord& port) {
        return port.id == edge.target;
      });
  ASSERT_NE(source_port, view.ports().end());
  ASSERT_NE(target_port, view.ports().end());
  EXPECT_EQ(source_port->graph_override.max_emits_per_compute, 3);
  EXPECT_EQ(source_port->graph_override.late_joiner_history_depth, 2);
  EXPECT_EQ(source_port->graph_override.diagnostic_alias, "output_override");
  EXPECT_EQ(target_port->graph_override.queue_depth, 4);
  EXPECT_EQ(target_port->graph_override.diagnostic_alias, "input_override");
}

// REQ: GRAPH-006
// REQ: GRAPH-010
//   source(AliasA) --[output→input]--> compatible(CompatibleB)  [accepted]
//   source(AliasA) --X [GRAPH_PORT_TYPE_MISMATCH]--> different(DifferentB)
//   incompatible(AliasB)  [GRAPH_POD_IDENTITY_CONFLICT on op construction]
TEST(GraphComposition, DistinctCppPodTypesInteroperateOnlyByCompleteContract) {
  Graph graph{"root"};
  auto source = graph.op<PodSourceOp>("source");
  auto compatible = graph.op<CompatiblePodTargetOp>("compatible");
  auto flow = graph.add_flow(source->output, compatible->input);

  EXPECT_TRUE(flow.graph_valid());
  EXPECT_EQ(&flow.source(), &source->output);
  EXPECT_EQ(&flow.target(), &compatible->input);
  ASSERT_EQ(graph.view().edges().size(), 1);

  auto different = graph.op<DifferentPodTargetOp>("different");
  const Error mismatch =
      graph_error_from([&] { (void)graph.add_flow(source->output, different->input); });
  EXPECT_EQ(mismatch.message, "GRAPH_PORT_TYPE_MISMATCH");
  EXPECT_EQ(graph.view().edges().size(), 1);

  const Error catalog_conflict =
      graph_error_from([&] { (void)graph.op<IncompatiblePodTargetOp>("incompatible"); });
  EXPECT_EQ(catalog_conflict.message, "GRAPH_POD_IDENTITY_CONFLICT");
  EXPECT_EQ(graph.view().edges().size(), 1);
}

// REQ: GRAPH-005
//   source(ParamOp) --[output→input]--> target(ParamOp)
//   Both operators undergo parameter replacement; flow handles must re-resolve to new ports.
TEST(GraphComposition, ParameterReplacementRemapsPortsAtomicallyAndFlowHandlesResolveAgain) {
  Graph graph{"root"};
  auto source = graph.op<ParamOp>("source");
  auto target = graph.op<ParamOp>("target");
  auto flow = graph.add_flow(source->output, target->input);
  const auto* old_source = &flow.source();
  const auto* old_target = &flow.target();

  source.set_params(GraphParams{.value = 7});
  target.set_params(GraphParams{.value = 9});
  EXPECT_NE(&flow.source(), old_source);
  EXPECT_NE(&flow.target(), old_target);
  EXPECT_EQ(source->params().value, 7);
  EXPECT_EQ(target->params().value, 9);

  const GraphView before_failure = graph.view();
  EXPECT_THROW(source.set_params(GraphParams{.value = 11}), RuntimeError);
  EXPECT_EQ(source->params().value, 7);
  EXPECT_EQ(graph.view().generation(), before_failure.generation());
  EXPECT_NO_THROW((void)flow.source());
  EXPECT_NO_THROW((void)flow.target());
}

// REQ: GRAPH-005
TEST(GraphComposition, ParameterReplacementConstructsExactlyOneCandidate) {
  CountingParamOp::constructor_calls = 0;
  Graph graph;
  auto op = graph.op<CountingParamOp>("counted");
  const int before = CountingParamOp::constructor_calls.load();
  op.set_params(GraphParams{.value = 7});
  EXPECT_EQ(CountingParamOp::constructor_calls.load(), before + 1);
  EXPECT_EQ(op->params().value, 7);
}

// REQ: PARAM-005
TEST(GraphComposition, ParameterReplacementReportsBoundedValidationIdentityWithoutValues) {
  Graph graph;
  auto op = graph.op<ValidatedParamOp>("validated");

  const Error range_error =
      graph_error_from([&] { op.set_params(ValidatedGraphParams{.bounded = 9U, .even = 2U}); });
  EXPECT_EQ(range_error.code, ErrorCode::kInvalidArgument);
  EXPECT_NE(range_error.message.find("PARAM_VALIDATION_FAILED category=out_of_range"),
            std::string::npos);
  EXPECT_NE(range_error.message.find("field_name=bounded"), std::string::npos);
  EXPECT_EQ(range_error.message.find("value="), std::string::npos);

  const Error validator_error =
      graph_error_from([&] { op.set_params(ValidatedGraphParams{.bounded = 2U, .even = 3U}); });
  EXPECT_EQ(validator_error.code, ErrorCode::kInvalidArgument);
  EXPECT_NE(validator_error.message.find("category=validator_rejected"), std::string::npos);
  EXPECT_NE(validator_error.message.find("field_name=even"), std::string::npos);
  EXPECT_NE(validator_error.message.find("validator_id=test.graph.even/v1"), std::string::npos);
  EXPECT_EQ(validator_error.message.find("value="), std::string::npos);
  EXPECT_EQ(op->params().bounded, 2U);
  EXPECT_EQ(op->params().even, 2U);
}

// REQ: PARAM-005
TEST(GraphComposition, ParameterValidationIdentityTruncationPreservesUtf8) {
  Graph graph;
  auto op = graph.op<Utf8ValidatedParamOp>("validated_utf8");

  const Error validator_error =
      graph_error_from([&] { op.set_params(Utf8ValidatedGraphParams{.even = 3U}); });
  EXPECT_EQ(validator_error.code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(detail::valid_utf8(validator_error.message));
  EXPECT_NE(validator_error.message.find("validator_id=x"), std::string::npos);
  EXPECT_NE(validator_error.message.find("...;bytes=97;sha256="), std::string::npos);
  EXPECT_EQ(validator_error.message.find("value="), std::string::npos);
}

// REQ: GRAPH-005
// REQ: GRAPH-006
//   source(EnumChangingParamOp.enum_a) --[→input]--> target(EnumATargetOp)
//   set_params(value=1) would switch source to enum_b output; rejected to protect flow.
TEST(GraphComposition, ParameterReplacementPreservesExactLocalPortType) {
  Graph graph{"root"};
  auto source = graph.op<EnumChangingParamOp>("source");
  auto target = graph.op<EnumATargetOp>("target");
  auto flow = graph.add_flow(source->enum_a, target->input);

  const GraphView before = graph.view();
  ASSERT_EQ(before.edges().size(), 1);
  ASSERT_EQ(before.ports().size(), 2);
  const auto& scalar = before.ports()[0].payload_identity.scalar;
  ASSERT_TRUE(scalar);
  if (scalar) {
    EXPECT_EQ(scalar->kind, ScalarIdentityKind::kLocalEnum);
  }

  const Error mismatch = graph_error_from([&] { source.set_params(GraphParams{.value = 1}); });
  EXPECT_EQ(mismatch.code, ErrorCode::kIncompatible);
  EXPECT_EQ(mismatch.message, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED");
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(source->params().value, 0U);
  EXPECT_EQ(&flow.source(), &source->enum_a);
  EXPECT_EQ(&flow.target(), &target->input);
}

// REQ: GRAPH-006
//   a --[output→input]--> b --[output→input]--> c
//   c --X [GRAPH_INPUT_FANIN_UNSUPPORTED]--> b   (b already has a source)
//   c --X [GRAPH_DIRECT_CYCLE]--> a              (would close a→b→c→a loop)
//   a --X [GRAPH_CROSS_GRAPH_PORT]--> outsider   (ports from different Graph instances)
TEST(GraphComposition, RejectsFaninCyclesAndCrossGraphPorts) {
  Graph graph{"root"};
  auto a = graph.op<PassOp>("a");
  auto b = graph.op<PassOp>("b");
  auto c = graph.op<PassOp>("c");
  (void)graph.add_flow(a->output, b->input);
  const Error fanin = graph_error_from([&] { (void)graph.add_flow(c->output, b->input); });
  EXPECT_EQ(fanin.message, "GRAPH_INPUT_FANIN_UNSUPPORTED");
  (void)graph.add_flow(b->output, c->input);
  const Error cycle = graph_error_from([&] { (void)graph.add_flow(c->output, a->input); });
  EXPECT_EQ(cycle.message, "GRAPH_DIRECT_CYCLE");

  Graph other{"other"};
  auto outsider = other.op<PassOp>("outside");
  const Error cross_graph =
      graph_error_from([&] { (void)graph.add_flow(a->output, outsider->input); });
  EXPECT_EQ(cross_graph.message, "GRAPH_CROSS_GRAPH_PORT");
}

// REQ: GRAPH-007
//   public_input =[export_input alias]=> op(inside) =[export_output alias]=> public_output
//   Interface bindings are directional aliases, not data-copy queue edges.
TEST(GraphComposition, SameDirectionBindingsAreAliasesRatherThanQueueEdges) {
  Graph graph{"root"};
  auto op = graph.op<PassOp>("inside");
  const Input<std::uint32_t> public_input{"public_input"};
  const Output<std::uint32_t> public_output{"public_output"};

  graph.export_input(public_input, op->input);
  graph.export_output(public_output, op->output);

  const GraphView view = graph.view();
  EXPECT_TRUE(view.edges().empty());
  ASSERT_EQ(view.interface_bindings().size(), 2);
  EXPECT_EQ(view.interface_bindings()[0].direction, PortDirection::kInput);
  EXPECT_EQ(view.interface_bindings()[1].direction, PortDirection::kOutput);
}

// REQ: GRAPH-007
//   public_input --[add_flow]--> op(inside) --[add_flow]--> public_output
//   Order matters: internal-then-boundary direction must be respected; foreign ports rejected.
TEST(GraphComposition, RawInterfaceAliasesEnforceOrderAndGraphOwnership) {
  Graph graph{"root"};
  auto op = graph.op<PassOp>("inside");
  const Input<std::uint32_t> public_input{"public_input"};
  const Output<std::uint32_t> public_output{"public_output"};

  graph.add_flow(public_input, op->input);
  graph.add_flow(op->output, public_output);
  ASSERT_EQ(graph.view().interface_bindings().size(), 2);

  Graph wrong_order{"wrong_order"};
  auto wrong_order_op = wrong_order.op<PassOp>("inside");
  const Input<std::uint32_t> wrong_order_input{"input"};
  const Output<std::uint32_t> wrong_order_output{"output"};
  EXPECT_THROW(wrong_order.add_flow(wrong_order_op->input, wrong_order_input), RuntimeError);
  EXPECT_THROW(wrong_order.add_flow(wrong_order_output, wrong_order_op->output), RuntimeError);
  EXPECT_TRUE(wrong_order.view().interface_bindings().empty());

  Graph foreign{"foreign"};
  auto foreign_op = foreign.op<PassOp>("inside");
  EXPECT_THROW(foreign.add_flow(public_input, foreign_op->input), RuntimeError);
  EXPECT_THROW(foreign.add_flow(foreign_op->output, public_output), RuntimeError);
  EXPECT_TRUE(foreign.view().interface_bindings().empty());
}

// REQ: GRAPH-003
TEST(GraphComposition, AddOperatorForwardsToTypedOperatorAuthoring) {
  Graph graph;
  auto op = graph.add_operator<PassOp>("aliased", std::string{"capture"});

  EXPECT_EQ(op->label_, "capture");
  const GraphView view = graph.view();
  ASSERT_EQ(view.vertices().size(), 1);
  EXPECT_EQ(view.vertices()[0].name, "aliased");
}

// REQ: GRAPH-007
TEST(GraphComposition, InterfaceNamesAreUniqueAcrossBothDirections) {
  Graph graph{"root"};
  auto op = graph.op<PassOp>("inside");
  const Input<std::uint32_t> public_input{"shared"};
  const Output<std::uint32_t> public_output{"shared"};
  graph.export_input(public_input, op->input);
  const Error error = graph_error_from([&] { graph.export_output(public_output, op->output); });
  EXPECT_EQ(error.code, ErrorCode::kDuplicateName);
  EXPECT_EQ(error.message, "GRAPH_INTERFACE_NAME_DUPLICATE");
}

// REQ: GRAPH-010
// REQ: GRAPH-011
TEST(GraphComposition, InvalidBoundaryConsumesItsDeterministicDeclarationOrdinal) {
  Graph graph{"root"};
  TopicOptions invalid;
  invalid.endpoint_name = " invalid";
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://bad", invalid), RuntimeError);

  TopicOptions valid;
  valid.endpoint_name = "valid";
  valid.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1};
  auto topic = graph.import_topic<std::uint32_t>("holomq://valid", valid);
  ASSERT_TRUE(topic);

  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1);
  EXPECT_TRUE(view.external_topics()[0].imported);
  EXPECT_EQ(view.external_topics()[0].declaration_ordinal, 1);
  EXPECT_EQ(view.external_topics()[0].endpoint_name, "valid");
  ASSERT_TRUE(view.external_topics()[0].payload_identity.scalar);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(view.external_topics()[0].payload_identity.scalar.value().kind,
            ScalarIdentityKind::kUInt32);
}
// REQ: GRAPH-010
TEST(GraphComposition, PodCoverageAndGraphWideIdentityCatalogFailClosed) {
  // Malformed erased POD record --> imported boundary (rejected before publication)
  // Valid PodAliasA record ------> imported boundary --> catalog
  Graph coverage;
  // Key setup: inject pointer-free records that simulate untrusted artifacts; public typed
  //            authoring rejects these C++ contracts at compile time.
  auto gap_identity = detail::operator_spec_payload_identity<GraphPodPayload>();
  PodIdentityRecord gap_pod{
      .canonical_name = std::string{ChannelTypeIdentity<GraphGapPod>::name},
      .version = ChannelTypeIdentity<GraphGapPod>::version,
      .object_size = sizeof(GraphGapPod),
      .object_alignment = alignof(GraphGapPod),
  };
  gap_pod.fields = {
      PodFieldIdentityRecord{.name = "first",
                             .byte_offset = offsetof(GraphGapPod, first),
                             .scalar_kind = PodScalarKind::kUInt8,
                             .element_count = 1U},
      PodFieldIdentityRecord{.name = "second",
                             .byte_offset = offsetof(GraphGapPod, second),
                             .scalar_kind = PodScalarKind::kUInt32,
                             .element_count = 1U},
  };
  gap_identity.pod = std::move(gap_pod);
  auto gap = detail::GraphCompilerAccess::register_import_with_payload_identity<GraphPodPayload>(
      coverage, std::move(gap_identity));
  ASSERT_FALSE(gap);
  // Key assertion: runtime validation remains fail-closed for malformed records that do not pass
  //                through the public typed API.
  EXPECT_EQ(gap.error().message, "GRAPH_POD_LAYOUT_INVALID");

  auto tail_identity = detail::operator_spec_payload_identity<GraphPodPayload>();
  PodIdentityRecord tail_pod{
      .canonical_name = std::string{ChannelTypeIdentity<GraphTailPod>::name},
      .version = ChannelTypeIdentity<GraphTailPod>::version,
      .object_size = sizeof(GraphTailPod),
      .object_alignment = alignof(GraphTailPod),
  };
  tail_pod.fields = {
      PodFieldIdentityRecord{.name = "first",
                             .byte_offset = offsetof(GraphTailPod, first),
                             .scalar_kind = PodScalarKind::kUInt32,
                             .element_count = 1U},
      PodFieldIdentityRecord{.name = "second",
                             .byte_offset = offsetof(GraphTailPod, second),
                             .scalar_kind = PodScalarKind::kUInt8,
                             .element_count = 1U},
  };
  tail_identity.pod = std::move(tail_pod);
  auto tail = detail::GraphCompilerAccess::register_import_with_payload_identity<GraphPodPayload>(
      coverage, std::move(tail_identity));
  ASSERT_FALSE(tail);
  EXPECT_EQ(tail.error().message, "GRAPH_POD_LAYOUT_INVALID");

  Graph catalog;
  (void)catalog.import_topic<GraphPodAliasA>("holomq://pod-a");
  const Error conflict =
      graph_error_from([&] { (void)catalog.import_topic<GraphPodAliasB>("holomq://pod-b"); });
  EXPECT_EQ(conflict.code, ErrorCode::kIncompatible);
  EXPECT_EQ(conflict.message, "GRAPH_POD_IDENTITY_CONFLICT");
  EXPECT_NO_THROW((void)catalog.import_topic<GraphPodAliasA>("holomq://pod-a-again"));
}

// REQ: GRAPH-010
TEST(GraphComposition, PodFixedArraysRetainElementCountsAndConflictByLayout) {
  Graph graph;
  (void)graph.import_topic<GraphArrayPodA>("holomq://array-a");

  const GraphView accepted = graph.view();
  ASSERT_EQ(accepted.external_topics().size(), 1);
  const auto& pod = accepted.external_topics()[0].payload_identity.pod;
  if (!pod) {
    FAIL() << "fixed-array POD identity was not retained";
    return;
  }
  ASSERT_EQ(pod->fields.size(), 2);
  EXPECT_EQ(pod->fields[0].element_count, 2);
  EXPECT_EQ(pod->fields[1].element_count, 2);

  const Error conflict =
      graph_error_from([&] { (void)graph.import_topic<GraphArrayPodB>("holomq://array-b"); });
  EXPECT_EQ(conflict.code, ErrorCode::kIncompatible);
  EXPECT_EQ(conflict.message, "GRAPH_POD_IDENTITY_CONFLICT");
  EXPECT_EQ(graph.view().external_topics().size(), 1);
}

// REQ: GRAPH-012
TEST(GraphComposition, ExternalUrisAndGeneratedEndpointNamespaceAreCanonical) {
  Graph graph;
  EXPECT_EQ(
      graph_error_from([&] { (void)graph.import_topic<std::uint32_t>("other://topic"); }).message,
      "GRAPH_TOPIC_URI_INVALID");
  EXPECT_EQ(graph_error_from([&] { (void)graph.import_topic<std::uint32_t>("holomq://"); }).message,
            "GRAPH_TOPIC_URI_INVALID");

  TopicOptions reserved_v1;
  reserved_v1.endpoint_name = "__holoscan_auto_endpoint/v1/user";
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://v1", reserved_v1);
            }).message,
            "GRAPH_RESERVED_ENDPOINT_NAMESPACE");
  TopicOptions reserved_v2;
  reserved_v2.endpoint_name = "__holoscan_auto_endpoint/v2/user";
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://v2", reserved_v2);
            }).message,
            "GRAPH_RESERVED_ENDPOINT_NAMESPACE");
}

// REQ: GRAPH-012
TEST(GraphComposition, ExternalTopicProvenanceParticipatesInCanonicalHash) {
  TopicOptions options;
  options.schema = kBoundarySchemaA;
  options.capture_time = RequireCaptureTime{.clock = ClockKey::named("capture_clock")};
  options.attach = TopicAttachPolicy{.peer = PeerRequirement::kRequired,
                                     .max_wait = std::chrono::milliseconds{7}};

  const auto hash_for = [](const TopicOptions& authored) {
    Graph graph{"same"};
    (void)graph.import_topic<Tensor>("holomq://provenance", authored);
    return graph.view().identity().hash;
  };

  Graph graph{"same"};
  (void)graph.import_topic<Tensor>("holomq://provenance", options);
  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1);
  const ExternalTopicRecord& topic = view.external_topics()[0];
  EXPECT_FALSE(topic.endpoint_name);
  EXPECT_TRUE(topic.generated_name_requested);
  EXPECT_EQ(topic.attach.peer, PeerRequirement::kRequired);
  EXPECT_EQ(topic.attach.max_wait, std::chrono::milliseconds{7});
  ASSERT_TRUE(topic.capture_time);
  if (topic.capture_time) {
    EXPECT_EQ(topic.capture_time->clock, ClockKey::named("capture_clock"));
  }
  ASSERT_TRUE(topic.schema);
  if (topic.schema) {
    EXPECT_EQ(topic.schema->canonical_name, kBoundarySchemaA.canonical_name);
    EXPECT_EQ(topic.schema->version, kBoundarySchemaA.version);
  }

  TopicOptions changed_attach = options;
  changed_attach.attach.max_wait = std::chrono::milliseconds{8};
  TopicOptions changed_capture = options;
  changed_capture.capture_time = RequireCaptureTime{.clock = ClockKey::named("other_clock")};
  TopicOptions changed_schema = options;
  changed_schema.schema = kBoundarySchemaB;
  EXPECT_NE(hash_for(options), hash_for(changed_attach));
  EXPECT_NE(hash_for(options), hash_for(changed_capture));
  EXPECT_NE(hash_for(options), hash_for(changed_schema));
}
// REQ: GRAPH-010
TEST(GraphComposition, ExternalScalarKindsCoverThePortableFamily) {
  expect_external_scalar_kind<bool>(ScalarIdentityKind::kBoolean);
  expect_external_scalar_kind<std::uint8_t>(ScalarIdentityKind::kUInt8);
  expect_external_scalar_kind<std::int8_t>(ScalarIdentityKind::kInt8);
  expect_external_scalar_kind<std::uint16_t>(ScalarIdentityKind::kUInt16);
  expect_external_scalar_kind<std::int16_t>(ScalarIdentityKind::kInt16);
  expect_external_scalar_kind<std::uint32_t>(ScalarIdentityKind::kUInt32);
  expect_external_scalar_kind<std::int32_t>(ScalarIdentityKind::kInt32);
  expect_external_scalar_kind<std::uint64_t>(ScalarIdentityKind::kUInt64);
  expect_external_scalar_kind<std::int64_t>(ScalarIdentityKind::kInt64);
  expect_external_scalar_kind<float>(ScalarIdentityKind::kFloat32);
  expect_external_scalar_kind<double>(ScalarIdentityKind::kFloat64);
}

// REQ: GRAPH-012
// REQ: GRAPH-018
// REQ: GRAPH-031
TEST(GraphComposition, RejectsUnicodeEdgeWhitespaceAndInvalidClosedEnumValues) {
  Graph graph{"root"};
  TopicOptions options;
  options.endpoint_name = std::string_view{"\xE3\x80\x80name"};
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://unicode", options), RuntimeError);

  auto source = graph.op<PassOp>("source");
  auto target = graph.op<PassOp>("target");
  // Fabricated underlying values exercise the public fail-closed validation boundary.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_THROW(
      (void)graph.add_flow(source->output,
                           target->input,
                           ConnectionOptions{.queue_policy = static_cast<QueuePolicy>(0xFF)}),
      RuntimeError);
  EXPECT_THROW(graph.region("bad_profile").profile(static_cast<Profile>(0xFF)), RuntimeError);
  EXPECT_THROW(graph.lifecycle().before(source, target, static_cast<LifecycleStage>(0xFF)),
               RuntimeError);

  // Key setup: exercise the Tensor-spec enum validator through ports whose Operator setup
  // explicitly admitted Tensor configuration.
  auto tensor = graph.op<TensorPassOp>("tensor");
  const Error invalid_input_memory = graph_error_from([&] {
    graph.configure(tensor->input)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = static_cast<MemoryKind>(0xFF)}});
  });
  const Error invalid_output_memory = graph_error_from([&] {
    graph.configure(tensor->output)
        .produces_tensor(
            TensorOutputSpec{.representation = {.memory_kind = static_cast<MemoryKind>(0xFF)}});
  });
  // Key assertion: both directions reject the fabricated closed-enum value at the Tensor-spec
  // boundary with the exact diagnostic.
  EXPECT_EQ(invalid_input_memory.message, "GRAPH_TENSOR_ENUM_INVALID");
  EXPECT_EQ(invalid_output_memory.message, "GRAPH_TENSOR_ENUM_INVALID");
  const Error invalid_input_dtype_code = graph_error_from([&] {
    graph.configure(tensor->input)
        .expects_tensor(TensorInputSpec{
            .representation = {.dtype = DLDataType{static_cast<std::uint8_t>(0xFF), 8, 1}}});
  });
  const Error invalid_output_dtype_bits = graph_error_from([&] {
    graph.configure(tensor->output)
        .produces_tensor(TensorOutputSpec{.representation = {.dtype = DLDataType{kDLUInt, 0, 1}}});
  });
  const Error invalid_input_dtype_lanes = graph_error_from([&] {
    graph.configure(tensor->input)
        .expects_tensor(TensorInputSpec{.representation = {.dtype = DLDataType{kDLUInt, 8, 0}}});
  });
  // Key assertion: every malformed dtype reaches the dtype validator rather than an earlier
  // Tensor-carrier rejection.
  EXPECT_EQ(invalid_input_dtype_code.message, "GRAPH_TENSOR_DTYPE_INVALID");
  EXPECT_EQ(invalid_output_dtype_bits.message, "GRAPH_TENSOR_DTYPE_INVALID");
  EXPECT_EQ(invalid_input_dtype_lanes.message, "GRAPH_TENSOR_DTYPE_INVALID");
  EXPECT_THROW((void)graph.op<InvalidInlineOptionsOp>("invalid_inline_policy"), RuntimeError);
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
}
// REQ: GRAPH-014
TEST(GraphComposition, ClockHealthOrdinaryAndMonitoredContractsFailClosed) {
  Graph graph{"root"};
  (void)graph.import_topic<std::uint32_t>("holomq://ordinary");

  TopicOptions monitored;
  monitored.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = std::chrono::milliseconds{10},
  };
  (void)graph.import_topic<std::uint32_t>("holomq://monitored", monitored);

  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 2);
  EXPECT_EQ(view.external_topics()[0].clock_health.capability, ClockHealthCapability::kUnavailable);
  EXPECT_EQ(view.external_topics()[0].clock_health.requirement,
            ClockHealthRequirement::kNotRequired);
  EXPECT_FALSE(view.external_topics()[0].clock_health.max_report_age);
  EXPECT_FALSE(view.external_topics()[0].clock_health_authored);
  EXPECT_TRUE(view.external_topics()[1].clock_health_authored);

  EXPECT_THROW((void)graph.export_topic<std::uint32_t>("holomq://monitored_export", monitored),
               RuntimeError);

  TopicOptions incomplete;
  incomplete.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
  };
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://incomplete", incomplete),
               RuntimeError);

  TopicOptions zero_age = monitored;
  zero_age.clock_health->max_report_age = std::chrono::nanoseconds{0};
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://zero_age", zero_age),
               RuntimeError);

  TopicOptions negative_age = monitored;
  negative_age.clock_health->max_report_age = std::chrono::nanoseconds{-1};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://negative-age", negative_age);
            }).message,
            "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");

  const auto rejected = [&](std::string_view uri, ClockHealthOptions health) {
    TopicOptions options;
    options.clock_health = health;
    return graph_error_from([&] { (void)graph.import_topic<std::uint32_t>(uri, options); });
  };
  EXPECT_EQ(rejected("holomq://unavailable-required",
                     ClockHealthOptions{.capability = ClockHealthCapability::kUnavailable,
                                        .requirement = ClockHealthRequirement::kFreshHealthyReport,
                                        .max_report_age = std::chrono::milliseconds{1}})
                .message,
            "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");
  EXPECT_EQ(rejected("holomq://ordinary-age",
                     ClockHealthOptions{.capability = ClockHealthCapability::kUnavailable,
                                        .requirement = ClockHealthRequirement::kNotRequired,
                                        .max_report_age = std::chrono::milliseconds{1}})
                .message,
            "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");
  EXPECT_EQ(rejected("holomq://reports-not-required",
                     ClockHealthOptions{.capability = ClockHealthCapability::kStatusReports,
                                        .requirement = ClockHealthRequirement::kNotRequired})
                .message,
            "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");
  EXPECT_EQ(rejected("holomq://reports-not-required-age",
                     ClockHealthOptions{.capability = ClockHealthCapability::kStatusReports,
                                        .requirement = ClockHealthRequirement::kNotRequired,
                                        .max_report_age = std::chrono::milliseconds{1}})
                .message,
            "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");
  TopicOptions invalid_capability;
  // Fabricated values prove that future enum additions fail closed until explicitly admitted.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  invalid_capability.clock_health =
      ClockHealthOptions{.capability = static_cast<ClockHealthCapability>(0xFF),
                         .requirement = ClockHealthRequirement::kNotRequired};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://invalid-capability",
                                                      invalid_capability);
            }).message,
            "ENDPOINT_CLOCK_HEALTH_ENUM_INVALID");
  TopicOptions invalid_requirement;
  invalid_requirement.clock_health =
      ClockHealthOptions{.capability = ClockHealthCapability::kUnavailable,
                         .requirement = static_cast<ClockHealthRequirement>(0xFF)};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://invalid-requirement",
                                                      invalid_requirement);
            }).message,
            "ENDPOINT_CLOCK_HEALTH_ENUM_INVALID");
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
}

// REQ: GRAPH-013
TEST(GraphComposition, ExternalEndpointBoundsAreValidatedByPeerRole) {
  Graph graph{"root"};
  TopicOptions import_wrong;
  import_wrong.bounds.subscriber = SubscriberEndpointBounds{.queue_depth = 1};
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://import", import_wrong),
               RuntimeError);

  TopicOptions export_wrong;
  export_wrong.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1};
  EXPECT_THROW((void)graph.export_topic<std::uint32_t>("holomq://export", export_wrong),
               RuntimeError);
}

// REQ: GRAPH-013
TEST(GraphComposition, ExternalEndpointBoundsRejectZeroValues) {
  Graph graph{"root"};

  TopicOptions publisher_zero;
  publisher_zero.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 0};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<std::uint32_t>("holomq://publisher-zero", publisher_zero);
            }).message,
            "GRAPH_TOPIC_PUBLISHER_BOUND_INVALID");

  TopicOptions queue_zero;
  queue_zero.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 0, .max_borrowed_samples = 1};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.export_topic<std::uint32_t>("holomq://queue-zero", queue_zero);
            }).message,
            "GRAPH_TOPIC_SUBSCRIBER_BOUND_INVALID");

  TopicOptions borrowed_zero;
  borrowed_zero.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1, .max_borrowed_samples = 0};
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.export_topic<std::uint32_t>("holomq://borrowed-zero", borrowed_zero);
            }).message,
            "GRAPH_TOPIC_SUBSCRIBER_BOUND_INVALID");

  EXPECT_TRUE(graph.view().external_topics().empty());
}
// REQ: GRAPH-015
// REQ: GRAPH-016
// REQ: GRAPH-017
//   a(PassOp) and b(PassOp) grouped in region "critical" [bounded-memory, real-time]
//   lifecycle: a must start before b; clock "capture" set as default
TEST(GraphComposition, RegionsLifecycleAndClocksRemainPointerFreeSnapshotRecords) {
  Graph graph{"root"};
  auto a = graph.op<PassOp>("a");
  auto b = graph.op<PassOp>("b");
  graph.region("critical")
      .add(a)
      .add(b)
      .constrain(RegionConstraint::bounded_memory())
      .profile(Profile::kRealTime);
  graph.lifecycle().before(a, b, LifecycleStage::kStart).allow_restart_request(a, b);
  const ClockKey clock = graph.add_clock<ManualClock>("capture", std::int64_t{12});
  graph.set_default_clock(clock);

  const GraphView view = graph.view();
  ASSERT_EQ(view.regions().size(), 1);
  EXPECT_EQ(view.regions()[0].members.size(), 2);
  EXPECT_EQ(view.regions()[0].profile, Profile::kRealTime);
  ASSERT_EQ(view.lifecycle_dependencies().size(), 1);
  EXPECT_EQ(view.lifecycle_dependencies()[0].stage, LifecycleStage::kStart);
  ASSERT_EQ(view.lifecycle_restart_permissions().size(), 1);
  EXPECT_EQ(view.lifecycle_restart_permissions()[0].origin,
            LifecycleRestartOrigin::kOperatorRequest);
  ASSERT_EQ(view.clocks().size(), 1);
  EXPECT_TRUE(view.clocks()[0].is_default);
}

// REQ: GRAPH-009
// REQ: GRAPH-015
//   first: region{b, a} (insertion order reversed from second)
//   second: region{a, b}
//   Members are sorted before hashing; canonical hash must be equal across both graphs.
TEST(GraphComposition, RegionsAreOrderInvariantAndProfileMayBeAuthoredOnlyOnce) {
  Graph first;
  auto first_a = first.op<PassOp>("a");
  auto first_b = first.op<PassOp>("b");
  first.region("region")
      .add(first_b)
      .add(first_a)
      .constrain(RegionConstraint::latency_budget(std::chrono::milliseconds{2}))
      .constrain(RegionConstraint::static_topology())
      .constrain(RegionConstraint::bounded_memory())
      .optimize(OptimizationHint::kCudaGraphFusion)
      .profile(Profile::kStandard);
  const GraphView first_view = first.view();
  ASSERT_EQ(first_view.regions().size(), 1);
  ASSERT_EQ(first_view.regions()[0].members.size(), 2);
  EXPECT_LT(first_view.regions()[0].members[0].value, first_view.regions()[0].members[1].value);

  Graph second;
  auto second_a = second.op<PassOp>("a");
  auto second_b = second.op<PassOp>("b");
  second.region("region")
      .add(second_a)
      .add(second_b)
      .constrain(RegionConstraint::bounded_memory())
      .constrain(RegionConstraint::static_topology())
      .constrain(RegionConstraint::latency_budget(std::chrono::milliseconds{2}))
      .optimize(OptimizationHint::kCudaGraphFusion)
      .profile(Profile::kStandard);
  const GraphView second_view = second.view();
  EXPECT_EQ(first_view.identity().hash, second_view.identity().hash);
  ASSERT_EQ(first_view.regions()[0].constraints.size(), 3);
  EXPECT_LT(static_cast<std::uint8_t>(first_view.regions()[0].constraints[0].kind()),
            static_cast<std::uint8_t>(first_view.regions()[0].constraints[1].kind()));
  EXPECT_LT(static_cast<std::uint8_t>(first_view.regions()[0].constraints[1].kind()),
            static_cast<std::uint8_t>(first_view.regions()[0].constraints[2].kind()));

  auto duplicate = first.region("duplicate-profile");
  duplicate.profile(Profile::kStandard);
  const Error error = graph_error_from([&] { duplicate.profile(Profile::kResearch); });
  EXPECT_EQ(error.code, ErrorCode::kDuplicateName);
  EXPECT_EQ(error.message, "GRAPH_REGION_PROFILE_DUPLICATE");
}

TEST(GraphComposition, PartitionsRemainDeepOwnedCanonicalSnapshotRecords) {
  Graph first{"root"};
  auto first_a = first.op<PassOp>("a");
  auto first_b = first.op<PassOp>("b");
  auto first_partition = first.partition(std::string{"serial"});
  first_partition.add(first_b).add(first_a);

  const GraphView retained = first.view();
  ASSERT_EQ(retained.partitions().size(), 1);
  EXPECT_TRUE(retained.partitions()[0].id);
  EXPECT_EQ(retained.partitions()[0].graph, retained.graphs()[0].id);
  EXPECT_EQ(retained.partitions()[0].name, "serial");
  ASSERT_EQ(retained.partitions()[0].members.size(), 2);

  (void)first.op<PassOp>("later");
  EXPECT_EQ(retained.partitions()[0].name, "serial");
  EXPECT_EQ(retained.partitions()[0].members.size(), 2);

  Graph second{"root"};
  auto second_a = second.op<PassOp>("a");
  auto second_b = second.op<PassOp>("b");
  second.partition("serial").add(second_a).add(second_b);
  EXPECT_EQ(retained.identity().hash, second.view().identity().hash);
}

// REQ: GRAPH-032
TEST(GraphComposition, PartitionVariadicAddCommitsTheCompleteBatchAtomically) {
  Graph graph{"root"};
  auto first = graph.op<PassOp>("first");
  auto second = graph.op<PassOp>("second");
  auto third = graph.op<PassOp>("third");
  auto partition = graph.partition("serial");

  partition.add(first, second, third);
  const GraphView accepted = graph.view();
  ASSERT_EQ(accepted.partitions().size(), 1);
  ASSERT_EQ(accepted.partitions()[0].members.size(), 3);

  Graph foreign_graph{"foreign"};
  auto foreign = foreign_graph.op<PassOp>("foreign");
  const Error foreign_error = graph_error_from([&] { partition.add(first, foreign); });
  EXPECT_EQ(foreign_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(foreign_error.message, "GRAPH_PARTITION_MEMBER_FOREIGN");
  EXPECT_EQ(graph.view().identity(), accepted.identity());

  auto duplicate_partition = graph.partition("duplicates");
  const GraphView before_existing_duplicate = graph.view();
  EXPECT_EQ(graph_error_from([&] { duplicate_partition.add(first, second); }).message,
            "GRAPH_PARTITION_MEMBER_DUPLICATE");
  EXPECT_EQ(graph.view().identity(), before_existing_duplicate.identity());

  Graph batch_duplicate_graph{"batch-duplicate"};
  auto batch_first = batch_duplicate_graph.op<PassOp>("first");
  auto batch_second = batch_duplicate_graph.op<PassOp>("second");
  auto batch_partition = batch_duplicate_graph.partition("serial");
  const GraphView before_batch_duplicate = batch_duplicate_graph.view();
  EXPECT_EQ(graph_error_from([&] { batch_partition.add(batch_first, batch_first); }).message,
            "GRAPH_PARTITION_MEMBER_DUPLICATE");
  EXPECT_EQ(batch_duplicate_graph.view().identity(), before_batch_duplicate.identity());
  EXPECT_TRUE(batch_duplicate_graph.view().partitions()[0].members.empty());

  const GraphView before_allocation_failure = batch_duplicate_graph.view();
  detail::GraphCompilerAccess::fail_mutation_after(batch_duplicate_graph, 0);
  EXPECT_THROW(batch_partition.add(batch_first, batch_second), std::bad_alloc);
  EXPECT_EQ(batch_duplicate_graph.view().identity(), before_allocation_failure.identity());
  EXPECT_TRUE(batch_duplicate_graph.view().partitions()[0].members.empty());
  EXPECT_NO_THROW(batch_partition.add(batch_first, batch_second));

  auto stale = [] {
    auto owner = std::make_unique<Graph>("temporary");
    auto op = owner->op<PassOp>("op");
    auto authored = owner->partition("partition");
    return std::pair{std::move(authored), std::move(op)};
  }();
  EXPECT_EQ(graph_error_from([&] { stale.first.add(stale.second, stale.second); }).message,
            "GRAPH_PARTITION_HANDLE_INVALID");
}

// REQ: GRAPH-032
TEST(GraphComposition, AutomaticRemainderDirectiveIsIdempotentOrderIndependentAndHashed) {
  Graph first{"root"};
  auto first_a = first.op<PassOp>("a");
  (void)first.op<PassOp>("b");
  first.partition("anchor").add(first_a);
  first.auto_partition_remaining();
  const GraphView once = first.view();
  EXPECT_TRUE(once.automatic_partition_remaining());
  first.auto_partition_remaining();
  EXPECT_EQ(first.view().identity(), once.identity());

  const auto first_snapshot = detail::GraphCompilerAccess::canonical_snapshot(first);
  EXPECT_TRUE(first_snapshot->automatic_partition_remaining);

  Graph second{"root"};
  second.auto_partition_remaining();
  auto second_a = second.op<PassOp>("a");
  (void)second.op<PassOp>("b");
  second.partition("anchor").add(second_a);
  EXPECT_EQ(first.view().identity().hash, second.view().identity().hash);

  Graph manual{"root"};
  auto manual_a = manual.op<PassOp>("a");
  (void)manual.op<PassOp>("b");
  manual.partition("anchor").add(manual_a);
  EXPECT_FALSE(manual.view().automatic_partition_remaining());
  EXPECT_NE(first.view().identity().hash, manual.view().identity().hash);
}

// REQ: GRAPH-032
TEST(GraphComposition, IncludeAllRetainsLateRootExpansionMetadataAndCanonicalIdentity) {
  Graph first{"root"};
  auto all = first.partition("application");
  all.include_all().include_all();
  (void)first.op<PassOp>("root-op");
  (void)first.embed<SinglePassGraph>("child");

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(first);
  ASSERT_EQ(snapshot->partitions.size(), 1);
  EXPECT_TRUE(snapshot->partitions[0].include_all);
  EXPECT_TRUE(snapshot->partitions[0].members.empty());
  EXPECT_EQ(std::ranges::count(snapshot->vertices, VertexKind::kOperator, &VertexRecord::kind), 2);

  Graph second{"root"};
  (void)second.op<PassOp>("root-op");
  (void)second.embed<SinglePassGraph>("child");
  second.partition("application").include_all();
  EXPECT_EQ(first.view().identity().hash, second.view().identity().hash);
}

// REQ: GRAPH-032
TEST(GraphComposition, IncludeAllAndAutomaticRemainderConflictsAreTransactional) {
  Graph explicit_members{"explicit"};
  auto op = explicit_members.op<PassOp>("op");
  auto explicit_partition = explicit_members.partition("application");
  explicit_partition.add(op);
  const GraphView before_explicit_conflict = explicit_members.view();
  EXPECT_EQ(graph_error_from([&] { explicit_partition.include_all(); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(explicit_members.view().identity(), before_explicit_conflict.identity());

  Graph include_first{"include-first"};
  auto include_first_op = include_first.op<PassOp>("op");
  auto include_first_partition = include_first.partition("application");
  include_first_partition.include_all();
  const GraphView before_member_conflict = include_first.view();
  EXPECT_EQ(graph_error_from([&] { include_first_partition.add(include_first_op); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(include_first.view().identity(), before_member_conflict.identity());
  EXPECT_EQ(graph_error_from([&] { (void)include_first.partition("other"); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(include_first.view().identity(), before_member_conflict.identity());
  EXPECT_EQ(graph_error_from([&] { include_first.auto_partition_remaining(); }).message,
            "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT");
  EXPECT_EQ(include_first.view().identity(), before_member_conflict.identity());

  Graph automatic_first{"automatic-first"};
  automatic_first.auto_partition_remaining();
  auto automatic_partition = automatic_first.partition("anchor");
  const GraphView before_automatic_conflict = automatic_first.view();
  EXPECT_EQ(graph_error_from([&] { automatic_partition.include_all(); }).message,
            "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT");
  EXPECT_EQ(automatic_first.view().identity(), before_automatic_conflict.identity());

  Graph multiple{"multiple"};
  auto first_partition = multiple.partition("first");
  (void)multiple.partition("second");
  const GraphView before_multiple_conflict = multiple.view();
  EXPECT_EQ(graph_error_from([&] { first_partition.include_all(); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(multiple.view().identity(), before_multiple_conflict.identity());

  Graph parent{"parent"};
  const GraphView before_non_root = parent.view();
  EXPECT_EQ(graph_error_from([&] { (void)parent.embed<IncludeAllChildGraph>("child"); }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(parent.view().identity(), before_non_root.identity());
  EXPECT_EQ(graph_error_from([&] {
              (void)parent.embed<AutomaticRemainderChildGraph>("automatic-child");
            }).message,
            "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT");
  EXPECT_EQ(parent.view().identity(), before_non_root.identity());

  Graph all_parent{"all-parent"};
  all_parent.partition("application").include_all();
  const GraphView before_child_manual = all_parent.view();
  EXPECT_EQ(graph_error_from([&] {
              (void)all_parent.embed<ManualPartitionChildGraph>("manual-child");
            }).message,
            "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
  EXPECT_EQ(all_parent.view().identity(), before_child_manual.identity());
}

TEST(GraphComposition, PartitionAuthoringRejectsInvalidNamesAndHandlesTransactionally) {
  Graph graph{"root"};
  auto first = graph.op<PassOp>("first");
  auto partition = graph.partition("serial");
  partition.add(first);
  const GraphView accepted = graph.view();

  EXPECT_EQ(graph_error_from([&] { (void)graph.partition(""); }).message,
            "GRAPH_PARTITION_NAME_INVALID");
  EXPECT_EQ(graph_error_from([&] { (void)graph.partition("serial"); }).message,
            "GRAPH_PARTITION_NAME_DUPLICATE");
  const Error duplicate_member_error = graph_error_from([&] { partition.add(first); });
  EXPECT_EQ(duplicate_member_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(duplicate_member_error.message, "GRAPH_PARTITION_MEMBER_DUPLICATE");

  Graph foreign_graph{"foreign"};
  auto foreign = foreign_graph.op<PassOp>("foreign");
  const Error foreign_error = graph_error_from([&] { partition.add(foreign); });
  EXPECT_EQ(foreign_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(foreign_error.message, "GRAPH_PARTITION_MEMBER_FOREIGN");

  EXPECT_EQ(graph.view().identity(), accepted.identity());
  ASSERT_EQ(graph.view().partitions().size(), 1);
  EXPECT_EQ(graph.view().partitions()[0].members.size(), 1);

  auto other = graph.partition("other");
  const GraphView before_cross_partition_duplicate = graph.view();
  EXPECT_EQ(graph_error_from([&] { other.add(first); }).message,
            "GRAPH_PARTITION_MEMBER_DUPLICATE");
  EXPECT_EQ(graph.view().identity(), before_cross_partition_duplicate.identity());

  auto stale = [] {
    auto owner = std::make_unique<Graph>("temporary");
    auto op = owner->op<PassOp>("op");
    auto authored = owner->partition("partition");
    return std::pair{std::move(authored), std::move(op)};
  }();
  EXPECT_EQ(graph_error_from([&] { stale.first.add(stale.second); }).message,
            "GRAPH_PARTITION_HANDLE_INVALID");
}

// REQ: GRAPH-016
//   a, b, c(PassOp) — no data edges; lifecycle edges only
//   permissions: a(self+external), b→a restart request
//   ordering:    a before b@kStart; b before a@kConfigure (different stage, not a cycle)
TEST(GraphComposition, LifecyclePermissionsAndOrderingAreIndependentAndDenyByDefault) {
  Graph graph;
  auto a = graph.op<PassOp>("a");
  auto b = graph.op<PassOp>("b");
  auto c = graph.op<PassOp>("c");
  EXPECT_TRUE(graph.view().lifecycle_restart_permissions().empty());

  auto lifecycle = graph.lifecycle();
  lifecycle.allow_self_restart(a).allow_external_restart(a).allow_restart_request(b, a);
  EXPECT_EQ(graph.view().lifecycle_restart_permissions().size(), 3);
  EXPECT_EQ(graph_error_from([&] { lifecycle.allow_restart_request(a, a); }).message,
            "GRAPH_LIFECYCLE_PERMISSION_INVALID");
  EXPECT_EQ(graph_error_from([&] { lifecycle.allow_self_restart(a); }).message,
            "GRAPH_LIFECYCLE_PERMISSION_DUPLICATE");

  lifecycle.before(a, b, LifecycleStage::kStart);
  EXPECT_EQ(graph_error_from([&] { lifecycle.before(a, b, LifecycleStage::kStart); }).message,
            "GRAPH_LIFECYCLE_DEPENDENCY_DUPLICATE");
  EXPECT_EQ(graph_error_from([&] { lifecycle.before(b, a, LifecycleStage::kStart); }).message,
            "GRAPH_LIFECYCLE_CYCLE");
  EXPECT_NO_THROW(lifecycle.before(b, a, LifecycleStage::kConfigure));

  Graph other;
  auto foreign = other.op<PassOp>("foreign");
  EXPECT_EQ(graph_error_from([&] { lifecycle.allow_restart_request(c, foreign); }).message,
            "GRAPH_LIFECYCLE_HANDLE_INVALID");
  EXPECT_EQ(graph_error_from([&] { lifecycle.before(a, foreign, LifecycleStage::kStart); }).message,
            "GRAPH_LIFECYCLE_HANDLE_INVALID");
}

// REQ: GRAPH-009
// REQ: GRAPH-016
// REQ: GRAPH-017
//   a, b, c(PassOp); clock_a --[off=1]--> clock_b; clock_a --[off=2]--> clock_c
//   "first" authors lifecycle/clock in forward order; "second" reverses both orderings.
//   The canonical hash must be equal, confirming declaration order is irrelevant.
TEST(GraphComposition, AuxiliaryDeclarationOrderDoesNotChangeCanonicalHash) {
  const auto author = [](Graph& graph, bool reverse) {
    auto a = graph.op<PassOp>("a");
    auto b = graph.op<PassOp>("b");
    auto c = graph.op<PassOp>("c");
    const ClockKey clock_a = graph.add_clock<ManualClock>("clock_a", std::int64_t{0});
    const ClockKey clock_b = graph.add_clock<ManualClock>("clock_b", std::int64_t{0});
    const ClockKey clock_c = graph.add_clock<ManualClock>("clock_c", std::int64_t{0});
    auto lifecycle = graph.lifecycle();
    if (reverse) {
      lifecycle.before(a, c, LifecycleStage::kStart)
          .before(a, b, LifecycleStage::kStart)
          .allow_restart_request(b, c)
          .allow_external_restart(b)
          .allow_self_restart(a);
      graph.add_clock_conversion(clock_a, clock_c, AffineClockConversion{.offset_ns = 2});
      graph.add_clock_conversion(clock_a, clock_b, AffineClockConversion{.offset_ns = 1});
    } else {
      lifecycle.before(a, b, LifecycleStage::kStart)
          .before(a, c, LifecycleStage::kStart)
          .allow_self_restart(a)
          .allow_external_restart(b)
          .allow_restart_request(b, c);
      graph.add_clock_conversion(clock_a, clock_b, AffineClockConversion{.offset_ns = 1});
      graph.add_clock_conversion(clock_a, clock_c, AffineClockConversion{.offset_ns = 2});
    }
  };

  Graph first{"same"};
  author(first, false);
  Graph second{"same"};
  author(second, true);
  EXPECT_EQ(first.view().identity().hash, second.view().identity().hash);
}

// REQ: GRAPH-002
// REQ: GRAPH-009
// REQ: GRAPH-020
//   Each sub-block injects a simulated allocation fault at successive checkpoints and
//   verifies the graph is completely unchanged, then accepts the same mutation on retry.
//   Sub-topologies exercised:
//     operator:              candidate(PassOp)
//     operator+readiness:    candidate(PodReadinessOp)
//     params:                candidate(ParamOp) + set_params rollback
//     interface:             op(PassOp) + export_input/export_output
//     flow:                  source --[output→input]--> target (PassOp×2)
//     tensor-output-layout:  source(TensorOp) --> first, second (TensorPassOp×2)
//     tensor-input-layout:   source(TensorOp) --[output→input]--> target(TensorPassOp)
//     tensor-payload:        source(TensorOp) --[output→input]--> target(TensorPassOp)
//     external:              import_topic<GraphPodPayload>
//     region:                region.contains / constrain / optimize
//     clock:                 add_clock / bind_shared_clock / set_default_clock
TEST(GraphComposition, CompoundMutationFaultsRollbackEveryLogicalCommit) {
  const auto expect_unchanged = [](const GraphView& before, const GraphView& after) {
    EXPECT_EQ(after.generation(), before.generation());
    EXPECT_EQ(after.identity(), before.identity());
  };

  // Key setup: fail_mutation_after(graph, N) injects a std::bad_alloc at the Nth internal
  //            allocation checkpoint, then verifies both generation and identity are stable.
  for (std::size_t checkpoint = 0; checkpoint < 9; ++checkpoint) {
    Graph graph{"operator"};
    const GraphView before = graph.view();
    // Key assertion: expect_unchanged confirms rollback leaves generation and hash intact.
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW((void)graph.op<PassOp>("candidate"), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW((void)graph.op<PassOp>("candidate"));
  }

  for (std::size_t checkpoint = 0; checkpoint < 8; ++checkpoint) {
    Graph graph{"operator-pod-readiness"};
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW((void)graph.op<PodReadinessOp>("candidate"), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW((void)graph.op<PodReadinessOp>("candidate"));
  }

  {
    Graph graph{"params"};
    auto op = graph.op<ParamOp>("candidate");
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(op.set_params(GraphParams{.value = 7}), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_EQ(op->params().value, 0);
    EXPECT_NO_THROW(op.set_params(GraphParams{.value = 7}));
  }

  for (std::size_t checkpoint = 0; checkpoint < 6; ++checkpoint) {
    Graph graph{"interface"};
    auto op = graph.op<PassOp>("op");
    const Input<std::uint32_t> public_input{"public"};
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW(graph.export_input(public_input, op->input), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW(graph.export_input(public_input, op->input));
  }

  for (std::size_t checkpoint = 0; checkpoint < 6; ++checkpoint) {
    Graph graph{"interface-output"};
    auto op = graph.op<PassOp>("op");
    const Output<std::uint32_t> public_output{"public"};
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW(graph.export_output(public_output, op->output), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW(graph.export_output(public_output, op->output));
  }

  {
    Graph graph{"flow"};
    auto source = graph.op<PassOp>("source");
    auto target = graph.op<PassOp>("target");
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW((void)graph.add_flow(source->output, target->input), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_TRUE(detail::GraphCompilerAccess::dense_ids_valid(graph));
    // The opposite edge remains legal only if rollback also removed the provisional adjacency.
    EXPECT_NO_THROW((void)graph.add_flow(target->output, source->input));
    EXPECT_TRUE(detail::GraphCompilerAccess::dense_ids_valid(graph));
  }

  {
    Graph graph{"tensor-output-layout"};
    auto source = graph.op<TensorOp>("source");
    auto first = graph.op<TensorPassOp>("first");
    auto second = graph.op<TensorPassOp>("second");
    (void)graph.add_flow(source->output, first->input);
    (void)graph.add_flow(source->output, second->input);
    (void)graph.configure(source->output);
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    const TensorOutputSpec output_contract{.representation = {.rank = 2U}};
    EXPECT_THROW(graph.configure(source->output).produces_tensor(output_contract), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW(graph.configure(source->output).produces_tensor(output_contract));
  }

  {
    Graph graph{"tensor-input-layout"};
    auto source = graph.op<TensorOp>("source");
    auto target = graph.op<TensorPassOp>("target");
    (void)graph.add_flow(source->output, target->input);
    (void)graph.configure(source->output);
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    // Match the producer maxima so compatibility validation reaches the injected allocation fault.
    const TensorInputSpec input_spec{
        .representation = {.memory_kind = MemoryKind::kCudaDevice},
        .bounds = TensorBounds{.max_rank = 4U, .max_byte_span = 4096U},
    };
    EXPECT_THROW(graph.configure(target->input).expects_tensor(input_spec), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW(graph.configure(target->input).expects_tensor(input_spec));
  }

  {
    Graph graph{"tensor-payload"};
    auto source = graph.op<TensorOp>("source");
    auto target = graph.op<TensorPassOp>("target");
    (void)graph.add_flow(source->output, target->input);
    (void)graph.configure(source->output);
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(graph.configure(source->output).payload_options(PayloadOptions{}), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW(graph.configure(source->output).payload_options(PayloadOptions{}));
  }

  for (std::size_t checkpoint = 0; checkpoint < 8; ++checkpoint) {
    Graph graph{"external"};
    const GraphView before = graph.view();
    TopicOptions options;
    options.endpoint_name = "candidate_endpoint";
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW((void)graph.import_topic<GraphPodPayload>("holomq://candidate", options),
                 std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_TRUE(detail::GraphCompilerAccess::dense_ids_valid(graph));
    auto topic = graph.import_topic<GraphPodPayload>("holomq://candidate", options);
    EXPECT_TRUE(topic.graph_valid());
    EXPECT_TRUE(detail::GraphCompilerAccess::dense_ids_valid(graph));
    ASSERT_EQ(graph.view().external_topics().size(), 1);
    EXPECT_EQ(graph.view().external_topics()[0].declaration_ordinal, 0);
  }

  {
    Graph graph{"region"};
    auto op = graph.op<PassOp>("op");
    const GraphView before_region = graph.view();
    for (std::size_t checkpoint = 0; checkpoint < 2; ++checkpoint) {
      detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
      EXPECT_THROW((void)graph.region("candidate"), std::bad_alloc);
      expect_unchanged(before_region, graph.view());
    }
    auto region = graph.region("candidate");
    const GraphView before_member = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(region.add(op), std::bad_alloc);
    expect_unchanged(before_member, graph.view());
    EXPECT_NO_THROW(region.add(op));

    const GraphView before_constraint = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(region.constrain(RegionConstraint::bounded_memory()), std::bad_alloc);
    expect_unchanged(before_constraint, graph.view());
    EXPECT_NO_THROW(region.constrain(RegionConstraint::bounded_memory()));

    const GraphView before_hint = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(region.optimize(OptimizationHint::kCudaGraphFusion), std::bad_alloc);
    expect_unchanged(before_hint, graph.view());
    EXPECT_NO_THROW(region.optimize(OptimizationHint::kCudaGraphFusion));
  }

  {
    Graph graph{"partition"};
    auto op = graph.op<PassOp>("op");
    const GraphView before_partition = graph.view();
    for (std::size_t checkpoint = 0; checkpoint < 2; ++checkpoint) {
      detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
      EXPECT_THROW((void)graph.partition("candidate"), std::bad_alloc);
      expect_unchanged(before_partition, graph.view());
    }
    auto partition = graph.partition("candidate");
    const GraphView before_member = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(partition.add(op), std::bad_alloc);
    expect_unchanged(before_member, graph.view());
    EXPECT_NO_THROW(partition.add(op));
  }

  for (std::size_t checkpoint = 0; checkpoint < 2; ++checkpoint) {
    Graph graph{"clock"};
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW((void)graph.add_clock<ManualClock>("candidate", std::int64_t{0}), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW((void)graph.add_clock<ManualClock>("candidate", std::int64_t{0}));
  }

  for (std::size_t checkpoint = 0; checkpoint < 2; ++checkpoint) {
    Graph graph{"shared-clock"};
    const GraphView before = graph.view();
    auto clock = std::make_shared<ManualClock>(std::int64_t{0});
    detail::GraphCompilerAccess::fail_mutation_after(graph, checkpoint);
    EXPECT_THROW((void)graph.bind_shared_clock("candidate", clock), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_NO_THROW((void)graph.bind_shared_clock("candidate", clock));
  }

  {
    Graph graph{"default-clock"};
    const ClockKey first = graph.add_clock<ManualClock>("first", std::int64_t{0});
    const ClockKey second = graph.add_clock<ManualClock>("second", std::int64_t{0});
    graph.set_default_clock(first);
    const GraphView before = graph.view();
    detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
    EXPECT_THROW(graph.set_default_clock(second), std::bad_alloc);
    expect_unchanged(before, graph.view());
    EXPECT_EQ(graph.default_clock(), first);
    EXPECT_NO_THROW(graph.set_default_clock(second));
  }
}

// REQ: GRAPH-009
// REQ: GRAPH-019
TEST(GraphComposition, ReadinessSnapshotCarriesStableProvenanceAndWakeRequirement) {
  Graph graph;
  auto op = graph.op<ReadinessOp>("ready");
  (void)op;
  const GraphView view = graph.view();
  ASSERT_EQ(view.readiness_sources().size(), 2);
  EXPECT_EQ(view.readiness_sources()[0].owner, view.readiness_sources()[1].owner);
  EXPECT_EQ(view.readiness_sources()[0].declaration_ordinal, 0);
  EXPECT_EQ(view.readiness_sources()[1].declaration_ordinal, 1);
  EXPECT_EQ(view.readiness_sources()[0].wake_requirement,
            ReadinessWakeRequirement::kDeploymentSelected);
  EXPECT_EQ(view.readiness_sources()[1].wake_requirement,
            ReadinessWakeRequirement::kBusyPollRequired);
}

// REQ: GRAPH-009
// REQ: GRAPH-019
TEST(GraphComposition, CountedReadinessSnapshotRetainsCapacityAndClockProvenance) {
  Graph graph;
  (void)graph.op<CountedReadinessOp>("counted");
  const GraphView view = graph.view();
  ASSERT_EQ(view.readiness_sources().size(), 2);

  const auto& notification = view.readiness_sources()[0];
  EXPECT_EQ(notification.kind, ReadinessSourceKind::kNotification);
  EXPECT_EQ(notification.name, "notification");
  EXPECT_EQ(notification.requested_capacity, 4);
  EXPECT_EQ(notification.requested_sender_reference_capacity, 3);
  EXPECT_EQ(notification.requested_clock, ClockKey::named("notification_clock"));
  EXPECT_EQ(notification.wake_requirement, ReadinessWakeRequirement::kDeploymentSelected);
  EXPECT_FALSE(notification.referenced);

  const auto& doorbell = view.readiness_sources()[1];
  EXPECT_EQ(doorbell.kind, ReadinessSourceKind::kDoorbell);
  EXPECT_EQ(doorbell.name, "doorbell");
  EXPECT_EQ(doorbell.requested_capacity, 5);
  EXPECT_EQ(doorbell.requested_sender_reference_capacity, 0);
  EXPECT_EQ(doorbell.requested_clock, ClockKey::named("doorbell_clock"));
  EXPECT_EQ(doorbell.wake_requirement, ReadinessWakeRequirement::kDeploymentSelected);
  EXPECT_FALSE(doorbell.referenced);
}

// REQ: GRAPH-009
TEST(GraphComposition, AuthoringIdentitySeparatesInstanceVersionAndCanonicalContent) {
  Graph first{"same"};
  (void)first.op<PassOp>("pass");
  const GraphView first_view = first.view();
  const GraphView first_repeat = first.view();
  EXPECT_EQ(first_view.identity(), first_repeat.identity());
  EXPECT_EQ(first_view.identity().version, first_view.generation());

  Graph second{"same"};
  (void)second.op<PassOp>("pass");
  const GraphView second_view = second.view();
  EXPECT_NE(first_view.identity().graph, second_view.identity().graph);
  EXPECT_EQ(first_view.identity().hash, second_view.identity().hash);

  (void)first.op<PassOp>("later");
  const GraphView changed = first.view();
  EXPECT_EQ(changed.identity().graph, first_view.identity().graph);
  EXPECT_GT(changed.identity().version, first_view.identity().version);
  EXPECT_NE(changed.identity().hash, first_view.identity().hash);
}

// REQ: GRAPH-017
//   owned(ManualClock) --[affine: scale=1/1, offset=7ns]--> shared(ManualClock)
//   Idempotent re-registration with equivalent scale accepted; conflicting offset rejected.
//   ThrowingClock factory failure is wrapped and the internal detail is not surfaced.
TEST(GraphComposition, ClockOwnershipConversionsAndFactoryFailuresAreExplicit) {
  Graph graph{"root"};
  const ClockKey owned = graph.add_clock<ManualClock>("owned", std::int64_t{12});
  auto shared_clock = std::make_shared<ManualClock>(std::int64_t{13});
  const ClockKey shared = graph.bind_shared_clock("shared", shared_clock);
  graph.set_default_clock(owned);

  const AffineClockConversion conversion{
      .scale = ClockRateRatio{.numerator = 1, .denominator = 1},
      .offset_ns = 7,
  };
  graph.add_clock_conversion(owned, shared, conversion);
  EXPECT_NO_THROW(graph.add_clock_conversion(
      owned,
      shared,
      AffineClockConversion{.scale = ClockRateRatio{.numerator = 2, .denominator = 2},
                            .offset_ns = 7}));
  const Error conflict = graph_error_from([&] {
    graph.add_clock_conversion(
        owned,
        shared,
        AffineClockConversion{.scale = ClockRateRatio{.numerator = 1, .denominator = 1},
                              .offset_ns = 8});
  });
  EXPECT_EQ(conflict.message, "GRAPH_CLOCK_CONVERSION_CONFLICT");
  EXPECT_THROW(graph.add_clock_conversion(owned, owned, conversion), RuntimeError);
  EXPECT_THROW(graph.add_clock_conversion(ClockKey::named("missing"), shared, conversion),
               RuntimeError);

  try {
    (void)graph.add_clock<ThrowingClock>("throws");
    FAIL() << "throwing user factory was accepted";
  } catch (const RuntimeError& error) {
    // Key assertion: the error token is GRAPH_CLOCK_CONSTRUCTION_FAILED and the internal
    //                constructor detail string is not leaked into the public what() message.
    EXPECT_EQ(error.code(), ErrorCode::kFailure);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_CLOCK_CONSTRUCTION_FAILED"),
              std::string_view::npos);
    EXPECT_EQ(std::string_view{error.what()}.find("private clock construction detail"),
              std::string_view::npos);
  }

  const GraphView view = graph.view();
  ASSERT_EQ(view.clocks().size(), 2);
  EXPECT_FALSE(view.clocks()[0].shared);
  EXPECT_TRUE(view.clocks()[0].is_default);
  EXPECT_TRUE(view.clocks()[1].shared);
  EXPECT_FALSE(view.clocks()[1].is_default);
  ASSERT_EQ(view.clock_conversions().size(), 1);
  EXPECT_EQ(view.clock_conversions()[0].from, owned);
  EXPECT_EQ(view.clock_conversions()[0].to, shared);
  EXPECT_EQ(view.clock_conversions()[0].conversion, conversion);
}

// REQ: GRAPH-017
TEST(GraphComposition, DefaultAndSharedClockRejectionsAreExplicit) {
  Graph graph;
  EXPECT_EQ(graph.default_clock(), ClockKey::default_clock());

  EXPECT_EQ(graph_error_from([&] { graph.set_default_clock(ClockKey::default_clock()); }).message,
            "GRAPH_DEFAULT_CLOCK_INVALID");
  EXPECT_EQ(graph_error_from([&] { graph.set_default_clock(ClockKey::named("")); }).message,
            "GRAPH_DEFAULT_CLOCK_INVALID");
  EXPECT_EQ(graph_error_from([&] { graph.set_default_clock(ClockKey::named("missing")); }).message,
            "GRAPH_CLOCK_NOT_FOUND");
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.bind_shared_clock("null", std::shared_ptr<Clock>{});
            }).message,
            "GRAPH_SHARED_CLOCK_INVALID");
  EXPECT_TRUE(graph.view().clocks().empty());
}

// REQ: GRAPH-015
TEST(GraphComposition, LatencyBudgetRequiresPositiveDuration) {
  EXPECT_EQ(graph_error_from([] {
              (void)RegionConstraint::latency_budget(std::chrono::nanoseconds::zero());
            }).message,
            "GRAPH_REGION_LATENCY_INVALID");
  EXPECT_EQ(graph_error_from([] {
              (void)RegionConstraint::latency_budget(std::chrono::nanoseconds{-1});
            }).message,
            "GRAPH_REGION_LATENCY_INVALID");
  EXPECT_EQ(RegionConstraint::latency_budget(std::chrono::nanoseconds{1}).latency(),
            std::chrono::nanoseconds{1});
}

// REQ: GRAPH-017
TEST(GraphComposition, ClockFactoryValidationConstructsAndDiscardsOneInstance) {
  CountingClock::constructor_calls = 0;
  CountingClock::live_count = 0;

  Graph graph;
  const ClockKey key = graph.add_clock<CountingClock>("counted");

  EXPECT_EQ(key, ClockKey::named("counted"));
  EXPECT_EQ(CountingClock::constructor_calls, 1);
  EXPECT_EQ(CountingClock::live_count, 0);
  ASSERT_EQ(graph.view().clocks().size(), 1);
  EXPECT_EQ(CountingClock::constructor_calls, 1);
  EXPECT_EQ(CountingClock::live_count, 0);
}

// REQ: GRAPH-017
//   a --[conv]--> b --[conv]--> c --X [GRAPH_CLOCK_CONVERSION_CYCLE]--> a
TEST(GraphComposition, ClockConversionGraphRejectsDirectedCycles) {
  Graph graph;
  const ClockKey a = graph.add_clock<ManualClock>("a", std::int64_t{0});
  const ClockKey b = graph.add_clock<ManualClock>("b", std::int64_t{0});
  const ClockKey c = graph.add_clock<ManualClock>("c", std::int64_t{0});
  const AffineClockConversion identity{};
  graph.add_clock_conversion(a, b, identity);
  graph.add_clock_conversion(b, c, identity);
  const Error error = graph_error_from([&] { graph.add_clock_conversion(c, a, identity); });
  EXPECT_EQ(error.code, ErrorCode::kIncompatible);
  EXPECT_EQ(error.message, "GRAPH_CLOCK_CONVERSION_CYCLE");
}

TEST(ClockHealthReport, FactoriesPairAssessmentsWithReasons) {
  constexpr ClockHealthReport healthy = ClockHealthReport::healthy();
  constexpr ClockHealthReport degraded =
      ClockHealthReport::degraded(ClockHealthReason::kCalibrationExpired);
  constexpr ClockHealthReport unhealthy =
      ClockHealthReport::unhealthy(ClockHealthReason::kClockStalled);

  EXPECT_EQ(healthy.assessment, ClockHealthAssessment::kHealthy);
  EXPECT_EQ(healthy.reason, ClockHealthReason::kNone);
  EXPECT_EQ(degraded.assessment, ClockHealthAssessment::kDegraded);
  EXPECT_EQ(degraded.reason, ClockHealthReason::kCalibrationExpired);
  EXPECT_EQ(unhealthy.assessment, ClockHealthAssessment::kUnhealthy);
  EXPECT_EQ(unhealthy.reason, ClockHealthReason::kClockStalled);
}

}  // namespace
}  // namespace holoscan
