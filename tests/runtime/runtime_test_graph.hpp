/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "compiler/process_local_fanout_experiment.hpp"

namespace holoscan {

struct RuntimeInputPredicateTestConfig {
  std::uint32_t minimum_driver_value{};
};

template <>
struct PredicateConfigCodec<RuntimeInputPredicateTestConfig> {
  static constexpr std::string_view schema{"test.runtime.input-predicate"};
  static constexpr std::uint32_t version{1U};
  static constexpr std::size_t max_encoded_bytes{4U};

  [[nodiscard]] static expected<std::vector<std::byte>, Error> encode(
      const RuntimeInputPredicateTestConfig& value);
};

}  // namespace holoscan

namespace holoscan::test {

enum class RuntimeProbeMode : std::uint8_t {
  kComplete,
  kFail,
  kWaitForStop,
  kPollForStop,
  kIgnoreStopUntilReleased,
  kWaitForExternalRelease,
  kWaitForStopThenFail,
  kRequestCompletionThenWaitForStop,
  kThrowSetup,
  kThrowStart,
  kThrowCompute,
  kThrowStop,
  kMismatchedContract,
  kAdvanceClockInStart,
  kThrowClockRead,
  kWaitFromCompute,
  kDestroyFromCompute,
  kWaitFromLifecycle,
  kDestroyFromLifecycle,
  kRequestStop,
  kRequestStopTwice,
};

/// @brief Share bounded process-local observations among runtime conformance operators.
struct RuntimeProbeState {
  static void reset(RuntimeProbeMode requested_mode = RuntimeProbeMode::kComplete);

  static void bind_callback_session(RunSession* session) noexcept;

  static void invoke_session_action_from_callback(RuntimeProbeMode requested_mode);

  [[nodiscard]] static bool callback_action_returned() noexcept;

  static void record(std::int32_t value);

  [[nodiscard]] static std::vector<std::int32_t> snapshot();

  [[nodiscard]] static std::vector<std::thread::id> snapshot_threads();

  [[nodiscard]] static bool wait_until_compute_enters();

  [[nodiscard]] static bool wait_until_active(std::uint32_t expected);

  static void bind_external_ready_fd(int fd) noexcept;
  static void bind_external_stop_gate_fd(int fd) noexcept;

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::vector<std::int32_t> events;
  static std::vector<std::thread::id> event_threads;
  static std::atomic<RuntimeProbeMode> mode;
  static std::atomic<std::uint32_t> constructed;
  static std::atomic<std::uint32_t> active;
  static std::atomic<std::uint32_t> max_active;
  static std::atomic_bool setup_entered;
  static std::atomic_bool compute_entered;
  static std::atomic<std::uint64_t> observed_clock_id;
  static std::atomic<std::uint64_t> observed_activation_clock_id;
  static std::atomic<std::int64_t> observed_activation_time_ns;
  static std::atomic<std::int64_t> observed_clock_time_ns;
  static std::atomic_bool observed_clock_read_succeeded;
  static std::atomic_bool observed_clock_can_read_now;
  static std::atomic_bool observed_clock_can_wait_until;
  static std::atomic_bool observed_clock_steady;
  static std::atomic<std::int64_t> runtime_clock_ns;
  static std::atomic<std::uint32_t> clocks_constructed;
  static std::atomic<RunSession*> callback_session;
  static std::atomic_bool callback_session_action_returned;
  static std::atomic<std::int64_t> stop_observed_time_ns;
  static std::atomic<std::int64_t> callback_return_time_ns;
  static std::atomic<int> external_ready_fd;
  static std::atomic<int> external_stop_gate_fd;
};

/// @brief Retain a fixed prefix of periodic Runtime activation evidence without hot allocation.
struct RuntimeClockProbeState {
  enum class TerminationRequestMode : std::uint8_t {
    kNone,
    kCompletion,
    kStop,
    kCompletionThenStop,
    kStopThenCompletion,
    kCompletionThenFailure,
  };

  static constexpr std::size_t kRecordedActivations = 8U;
  static constexpr std::int64_t kPeriodNs = 5;

  static void reset() noexcept;

  [[nodiscard]] static bool wait_until_compute_count(std::size_t expected);

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<std::size_t> compute_count;
  static std::atomic<TerminationRequestMode> termination_request_mode;
  static std::array<Tag, kRecordedActivations> tags;
  static std::array<TagTime, kRecordedActivations> tag_times;
  static std::array<TimePoint, kRecordedActivations> activation_times;
};

/// @brief Coordinate notification lifecycle and compute evidence with the test thread.
///
/// The compute gate holds one witness active so capacity-through-retirement is observable by
/// \c NotificationSenderIsLifecycleBoundCapacityLimitedAndClosedAtStop.
struct RuntimeNotificationProbeState {
  static void reset();

  static void publish_sender(NotificationSender value, ErrorCode post_error,
                             ErrorCode acquire_error);

  [[nodiscard]] static bool wait_until_sender();

  [[nodiscard]] static NotificationSender sender_copy();

  [[nodiscard]] static bool wait_until_compute_enters();

  [[nodiscard]] static bool wait_until_compute_count(std::size_t expected);

  static void release_blocked_compute();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::optional<NotificationSender> sender;
  static ErrorCode arm_post_error;
  static ErrorCode second_acquire_error;
  static std::size_t compute_count;
  static bool compute_entered;
  static bool release_compute;
};

enum class RuntimeClockPredicatePattern : std::uint8_t {
  kTrueFalseTrue,
  kAlwaysTrue,
};

struct RuntimeClockPredicateEvaluationEvidence {
  TimePoint reference_time{};
  Tag tag{};
  TagTime tag_time{};
  std::uint32_t state_count{};
  std::size_t selection_count{};
  bool result{};
};

struct RuntimeClockPredicateComputeEvidence {
  TimePoint activation_time{};
  Tag tag{};
  TagTime tag_time{};
};

/// @brief Retain a fixed prefix of whole-session no-input predicate evidence.
struct RuntimeClockPredicateProbeState {
  static constexpr std::size_t kRecordedEvaluations = 8U;

  static void reset() noexcept;

  static void record_evaluation(const PredicateEvaluationContext& context,
                                std::uint32_t state_count, bool result) noexcept;

  static void record_compute(ExecutionContext& context, bool request_restart) noexcept;

  [[nodiscard]] static bool wait_until_compute_count(std::size_t expected);

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<std::size_t> evaluation_count;
  static std::atomic<std::size_t> compute_count;
  static std::atomic_bool restart_accepted;
  static std::atomic<ErrorCode> restart_error;
  static std::atomic<std::uint64_t> restart_receipt_id;
  static std::atomic<LifecycleStage> restart_requested_stage;
  static std::array<RuntimeClockPredicateEvaluationEvidence, kRecordedEvaluations> evaluations;
  static std::array<RuntimeClockPredicateComputeEvidence, kRecordedEvaluations> computes;
};

class RuntimeProbeClock final : public Clock {
 public:
  RuntimeProbeClock();

  [[nodiscard]] double time() const override;

  [[nodiscard]] std::int64_t timestamp() const override;

  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t duration_ns) override;

  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t target_time_ns) override;
};

/// @brief Provide a portless Operator whose explicit SpinSource receives one injected test edge.
class RuntimeProbeOperator final : public Operator<> {
 public:
  explicit RuntimeProbeOperator(std::int32_t identity, bool = false);

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  void start() override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  void stop() override;

 private:
  SpinSource spin_;
  std::int32_t identity_{};
};

struct RuntimeProbeParams {
  std::int32_t value{7};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeProbeParams, param_schema("holoscan.test.RuntimeProbeParams", 1),
                      field<&RuntimeProbeParams::value>("value").runtime_mutable());

struct RuntimeDeadlineProbeParams {
  std::chrono::nanoseconds deadline{10};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeDeadlineProbeParams,
                      param_schema("holoscan.test.RuntimeDeadlineProbeParams", 1),
                      field<&RuntimeDeadlineProbeParams::deadline>("deadline")
                          .runtime_mutable()
                          .range(std::chrono::nanoseconds{1}, std::chrono::nanoseconds{100}));

struct EventPolicyProbeParams {
  std::int32_t public_value{7};
  std::int32_t internal_value{8};
  std::int32_t sensitive_value{9};
  std::int32_t external_value{10};
};

HOLOSCAN_PARAM_SCHEMA(
    EventPolicyProbeParams, param_schema("holoscan.test.EventPolicyProbeParams", 1),
    field<&EventPolicyProbeParams::public_value>("public_value").runtime_mutable(),
    field<&EventPolicyProbeParams::internal_value>("internal_value").runtime_mutable().internal(),
    field<&EventPolicyProbeParams::sensitive_value>("sensitive_value")
        .runtime_mutable()
        .sensitive(),
    field<&EventPolicyProbeParams::external_value>("external_value")
        .runtime_mutable()
        .external_tuning());

class EventPolicyParamOperator final : public Operator<EventPolicyProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

struct LifecycleParamProbeParams {
  std::int32_t value{7};
};

HOLOSCAN_PARAM_SCHEMA(
    LifecycleParamProbeParams, param_schema("holoscan.test.LifecycleParamProbeParams", 1),
    field<&LifecycleParamProbeParams::value>("value").lifecycle_mutable(LifecycleStage::kAllocate));

struct LifecycleParamObservation {
  std::int32_t operator_identity{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  std::int32_t value{};
  ParamGeneration generation{};
  const LifecycleParamProbeParams* address{};
};

struct LifecycleParamComputeObservation {
  std::int32_t operator_identity{};
  std::int32_t value{};
  ParamGeneration generation{};
};

struct LifecycleParamProbeState {
  static void reset(bool hold_first_callback = false, bool fail_restart_start = false,
                    std::int32_t failure_identity = 1, bool hold_restart_start_callback = false);

  static void record_lifecycle(LifecycleContext& context, LifecycleStage stage,
                               std::int32_t operator_identity);

  [[nodiscard]] static bool should_fail_start(LifecycleContext& context,
                                              std::int32_t operator_identity);

  static void record_compute(ExecutionContext& context, std::int32_t operator_identity);

  [[nodiscard]] static bool wait_until_compute_count(std::size_t count);

  static void release_first_callback();

  [[nodiscard]] static bool wait_until_restart_start();

  static void release_restart_start_callback();

  [[nodiscard]] static bool restart_start_saw_cancellation();

  [[nodiscard]] static std::vector<LifecycleParamObservation> lifecycle_snapshot();

  [[nodiscard]] static std::vector<LifecycleParamComputeObservation> compute_snapshot();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::vector<LifecycleParamObservation> lifecycle;
  static std::vector<LifecycleParamComputeObservation> compute_observations;
  static bool hold_first;
  static bool release_first;
  static bool fail_start;
  static std::int32_t fail_start_identity;
  static bool failed_start;
  static bool hold_restart_start;
  static bool release_restart_start;
  static bool restart_start_entered;
  static bool restart_start_observed_cancellation;
};

class LifecycleParamProbeOperator final : public Operator<LifecycleParamProbeParams> {
 public:
  explicit LifecycleParamProbeOperator(std::int32_t identity = 1) noexcept;

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  LifecycleStatus allocate(LifecycleContext& context) noexcept;
  LifecycleStatus warm_up(LifecycleContext& context) noexcept;
  LifecycleStatus arm(LifecycleContext& context) noexcept;
  LifecycleStatus start_stage(LifecycleContext& context) noexcept;
  LifecycleStatus stop_stage(LifecycleContext& context) noexcept;
  LifecycleStatus release(LifecycleContext& context) noexcept;

 private:
  LifecycleStatus record(LifecycleContext& context, LifecycleStage stage) const noexcept;

  SpinSource spin_;
  std::int32_t identity_{1};
};

struct RuntimeDurationParamProbeParams {
  std::chrono::milliseconds deadline{3};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeDurationParamProbeParams,
                      param_schema("holoscan.test.RuntimeDurationParamProbeParams", 1),
                      field<&RuntimeDurationParamProbeParams::deadline>("deadline")
                          .runtime_mutable()
                          .range(std::chrono::milliseconds{1}, std::chrono::milliseconds{10}));

class RuntimeDurationParamProbeOperator final : public Operator<RuntimeDurationParamProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

struct RuntimeValidationParamProbeParams {
  std::int32_t bounded_value{7};
  std::int32_t frozen_value{8};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeValidationParamProbeParams,
                      param_schema("holoscan.test.RuntimeValidationParamProbeParams", 1),
                      field<&RuntimeValidationParamProbeParams::bounded_value>("bounded_value")
                          .runtime_mutable()
                          .range(std::int32_t{0}, std::int32_t{100}),
                      field<&RuntimeValidationParamProbeParams::frozen_value>("frozen_value"));

class RuntimeValidationParamProbeOperator final
    : public Operator<RuntimeValidationParamProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

struct RuntimeFloatParamProbeParams {
  float value{std::numeric_limits<float>::quiet_NaN()};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeFloatParamProbeParams,
                      param_schema("holoscan.test.RuntimeFloatParamProbeParams", 1),
                      field<&RuntimeFloatParamProbeParams::value>("value").runtime_mutable());

class RuntimeFloatParamProbeOperator final : public Operator<RuntimeFloatParamProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

struct MixedMutabilityParamProbeParams {
  std::int32_t runtime_value{7};
  std::int32_t lifecycle_value{8};
};

HOLOSCAN_PARAM_SCHEMA(
    MixedMutabilityParamProbeParams,
    param_schema("holoscan.test.MixedMutabilityParamProbeParams", 1),
    field<&MixedMutabilityParamProbeParams::runtime_value>("runtime_value").runtime_mutable(),
    field<&MixedMutabilityParamProbeParams::lifecycle_value>("lifecycle_value")
        .lifecycle_mutable(LifecycleStage::kAllocate));

class MixedMutabilityParamProbeOperator final : public Operator<MixedMutabilityParamProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

/// @brief Preserve one callback's coherent parameter observations.
struct RuntimeParamObservation {
  std::int32_t before_value{};
  std::int32_t after_value{};
  ParamGeneration before_generation{};
  ParamGeneration after_generation{};
  const RuntimeProbeParams* before_address{};
  const RuntimeProbeParams* after_address{};
  Tag before_tag{};
  Tag after_tag{};
  TagTime before_tag_time{};
  TagTime after_tag_time{};
  TimePoint before_activation_time{};
  TimePoint after_activation_time{};
  ActivationStatus before_activation{};
  ActivationStatus after_activation{};
};

/// @brief Coordinate deterministic external-update tests with the parameterized probe.
struct RuntimeParamProbeState {
  static void reset(bool hold_first_callback = false, bool hold_stop_callback = false,
                    bool hold_second_callback = false);

  [[nodiscard]] static bool wait_until_callback_enters(std::size_t count);

  static void release_first_callback();

  static void release_second_callback();

  static void hold_start_callback();

  [[nodiscard]] static bool wait_until_start_enters();

  static void release_start_callback();

  [[nodiscard]] static bool wait_until_stop_enters();

  static void release_stop_callback();

  [[nodiscard]] static std::vector<RuntimeParamObservation> snapshot();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::vector<RuntimeParamObservation> observations;
  static bool hold_first;
  static bool release_first;
  static bool hold_second;
  static bool release_second;
  static bool hold_start;
  static bool start_entered;
  static bool release_start;
  static bool hold_stop;
  static bool stop_entered;
  static bool release_stop;
  static std::atomic<std::int32_t> stop_observed_value;
  static std::atomic_bool stop_observed_params;
};

class RuntimeParamProbeOperator final : public Operator<RuntimeProbeParams> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  void start() override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  void stop() override;

 private:
  SpinSource spin_;
};

/// @brief Record the generation and deadline admitted to one real Runtime callback.
struct RuntimeDeadlineParamProbeState {
  static void reset() noexcept;

  static std::atomic<std::uint32_t> compute_count;
  static std::atomic<std::int64_t> observed_deadline_ns;
  static std::atomic<ParamGeneration> observed_generation;
  static std::atomic<ActivationReason> observed_reason;
  static std::atomic<OnViolation> replay_violation;
};

/// @brief Exercise a runtime-mutable deadline through the production invocation gate.
class RuntimeDeadlineParamProbeOperator final : public Operator<RuntimeDeadlineProbeParams> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  void start() override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

 private:
  SpinSource spin_;
};

/// @brief Exercise a production periodic source without introducing an undocumented count API.
class RuntimeClockProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
};

// Acquires its sole NotificationSender during Arm and blocks the first compute until the test
// observes capacity pressure. This uses ordinary plan/runtime materialization, not test injection.
// Used by: NotificationSenderIsLifecycleBoundCapacityLimitedAndClosedAtStop
class RuntimeNotificationProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  LifecycleStatus arm(LifecycleContext& context) noexcept;

 private:
  NotificationSource notification_;
};

class RuntimeClockPredicateErrorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

class RuntimeClockPredicateProbeOperator final : public Operator<> {
 public:
  RuntimeClockPredicateProbeOperator(RuntimeClockPredicatePattern pattern, PredicateFiring firing,
                                     bool restart_after_first_compute);

  void setup(OperatorSpec&) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

 private:
  RuntimeClockPredicatePattern pattern_{RuntimeClockPredicatePattern::kAlwaysTrue};
  PredicateFiring firing_{PredicateFiring::kOnFalseToTrue};
  bool restart_after_first_compute_{};
};

enum class RuntimeInputContractReplayMode : std::uint8_t {
  kMatch,
  kSelectorMismatch,
  kPredicateFiringMismatch,
  kMaxStartDelayMismatch,
};

/// @brief Supply three process-local POD streams for input-contract reconstruction tests.
class RuntimeInputContractSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Output<std::uint32_t> driver;
  Output<std::uint32_t> additional;
  Output<std::uint32_t> compute_only;
};

enum class RuntimeAlignedChannelScript : std::uint8_t {
  kExactBoundary,
  kMisalignmentProgress,
  kMissingHead,
  kMisalignedMembers,
  kSideInputReference,
};

/// @brief Share bounded observations for end-to-end aligned Channel execution tests.
/// Used by RunSessionTest.ChannelAlignedConnectedSilentSideInputRemainsNeverSeen.
struct RuntimeAlignedChannelProbeState {
  static void reset(RuntimeAlignedChannelScript requested);

  [[nodiscard]] static bool wait_for_source(std::size_t expected);

  [[nodiscard]] static bool wait_for_sink(std::size_t expected);

  static void record_source_compute();

  static void record_sink_compute(const Sample<std::uint32_t>& first,
                                  const Sample<std::uint32_t>& second);

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<RuntimeAlignedChannelScript> script;
  static std::atomic<std::size_t> next_step;
  static std::atomic<std::size_t> source_compute_count;
  static std::atomic<std::size_t> sink_compute_count;
  static std::atomic<std::uint32_t> observed_first;
  static std::atomic<std::uint32_t> observed_second;
  static std::atomic<std::uint32_t> observed_optional;
  static std::atomic<std::int64_t> observed_first_capture;
  static std::atomic<std::int64_t> observed_second_capture;
  static std::atomic<std::int64_t> side_reference_capture;
  static std::atomic<InputState> observed_optional_state;
  static std::atomic<SelectionReason> observed_optional_reason;
  static std::atomic<ErrorCode> observed_optional_error;
  static std::atomic<ErrorCode> observed_optional_batch_error;
};

/// @brief Emit deterministic capture-time scripts into aligned and optional Channel services.
/// Used by RunSessionTest.ChannelAlignedSideInputUsesAlignedReferenceWhenOlderMemberArrivesLast.
class RuntimeAlignedChannelSourceOperator final : public Operator<> {
 public:
  explicit RuntimeAlignedChannelSourceOperator(bool finite = false) : finite_(finite) {}

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> first;
  Output<std::uint32_t> second;
  Output<std::uint32_t> optional;

 private:
  bool finite_{};
};

/// @brief Consume one inclusive aligned tuple and retain optional-side observations.
/// Used by RunSessionTest.ChannelAlignedTriggerAcceptsTheInclusiveExactBoundary.
class RuntimeAlignedChannelSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  Input<std::uint32_t> optional;
  Input<std::uint32_t> additional_optional;
};

/// @brief Record activation of an Operator whose declared ports have no Channel endpoints.
struct RuntimeUnconnectedPortProbeState {
  static void reset() noexcept;

  [[nodiscard]] static bool wait_for_compute();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<std::uint32_t> compute_count;
};

/// @brief Exercise clock activation with replayed but intentionally unconnected ports.
class RuntimeUnconnectedPortClockOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

/// @brief Retain one serial input activation for process-local activation-cut ordering tests.
struct RuntimeActivationCutObservation {
  std::uint32_t value{};
  std::int64_t capture_timestamp_ns{};
  TagTime tag_time{};
};

/// @brief Share ordered sink observations for the post-compute process-local activation cut.
struct RuntimeActivationCutProbeState {
  static void reset();

  [[nodiscard]] static bool wait_for_sinks(std::size_t expected);

  static void record(const Sample<std::uint32_t>& sample, TagTime tag_time);

  [[nodiscard]] static std::vector<RuntimeActivationCutObservation> snapshot();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::vector<RuntimeActivationCutObservation> observations;
};

/// @brief Emit a later-tagged subscriber first and an earlier-tagged subscriber second.
class RuntimeActivationCutSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> later;
  Output<std::uint32_t> earlier;

 private:
  bool emitted_{};
};

/// @brief Record one exact input activation without retaining callback-scoped sample state.
class RuntimeActivationCutSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> input;
};

enum class RuntimeWindowChannelScript : std::uint8_t {
  kSuccessorClose,
  kTimeoutClose,
  kEndOfStreamDrain,
};

/// @brief Retain one end-to-end window activation without borrowing callback-scoped views.
struct RuntimeWindowChannelProbeState {
  static void reset(RuntimeWindowChannelScript requested);

  [[nodiscard]] static bool wait_for_compute();

  [[nodiscard]] static bool wait_for_compute_count(std::size_t expected);

  static void record(InputSelection selection, const Sample<std::uint32_t>& anchor_sample,
                     TimePoint activation_time, Tag tag, TagTime tag_time,
                     std::size_t selected_size, std::int64_t first_capture_ns,
                     std::int64_t last_capture_ns);

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<RuntimeWindowChannelScript> script;
  static std::atomic<std::size_t> compute_count;
  static std::atomic<std::size_t> safe_default_end_of_stream_count;
  static std::atomic<std::size_t> downstream_count;
  static std::array<Tag, 2U> terminal_tags;
  static std::array<TagTime, 2U> terminal_tag_times;
  static std::atomic<InputState> selection_state;
  static std::atomic<SelectionReason> selection_reason;
  static std::atomic<std::int64_t> anchor_capture_ns;
  static std::atomic<std::int64_t> activation_time_ns;
  static std::atomic<std::size_t> batch_size;
  static std::atomic<std::int64_t> batch_first_capture_ns;
  static std::atomic<std::int64_t> batch_last_capture_ns;
};

/// @brief Publish three anchors that bound two pending windows before terminal EOS.
class RuntimeWindowChannelSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> anchor;
  Output<std::uint32_t> windowed;

 private:
  std::uint32_t next_terminal_anchor_{10U};
};

/// @brief Forward every anchor before publishing one logical producer's downstream EOS.
class RuntimeWindowChannelRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> input;
  Output<std::uint32_t> anchor;
  Output<std::uint32_t> windowed;
};

/// @brief Observe the window activation coordinate and optional compute-only side.
/// Used by RunSessionTest.OptionalWindowSidePreservesSuccessfulAndDegradedWindowSelection.
class RuntimeWindowChannelSinkOperator final : public Operator<> {
 public:
  explicit RuntimeWindowChannelSinkOperator(bool optional_side = false)
      : optional_side_(optional_side) {}
  static inline std::atomic_uint32_t optional_checks{};
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> windowed;
  Output<std::uint32_t> output;
  Input<std::uint32_t> side;

 private:
  bool optional_side_{};
};

/// @brief Drain every window result and its downstream EOS before natural completion.
class RuntimeWindowChannelDrainOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> input;
};

enum class RuntimeInputTriggerReplayKind : std::uint8_t {
  kAny,
  kEach,
  kAligned,
  kWindow,
};

/// @brief Replay each input-backed trigger family over two process-local POD streams.
class RuntimeInputTriggerReplayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  static std::atomic<RuntimeInputTriggerReplayKind> kind;
};

/// @brief Replay one phase-separated input predicate with selectable contract corruption.
class RuntimeInputContractPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> driver;
  Input<std::uint32_t> additional;
  Input<std::uint32_t> compute_only;
  static std::atomic<RuntimeInputContractReplayMode> mode;
};

enum class RuntimeClockInputReplayKind : std::uint8_t {
  kClock,
  kPredicate,
};

/// @brief Replay clock-driven scheduling together with retained input-selection policy.
class RuntimeClockInputReplayOperator final : public Operator<> {
 public:
  explicit RuntimeClockInputReplayOperator(RuntimeClockInputReplayKind kind,
                                           std::uint32_t replay_ordinal = 0U);

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> predicate_input;
  Input<std::uint32_t> compute_only;
  static std::atomic<std::uint32_t> mismatch_ordinal;

 private:
  RuntimeClockInputReplayKind kind_;
  std::uint32_t replay_ordinal_{};
};

/// @brief Exercise a clock-driven Exact/SafeDefault input with no graph edge.
/// Used by RunSessionTest.UnwiredExactSafeDefaultRunsWithNotConnectedSelection.
class RuntimeOptionalSafeDefaultOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> optional;
  static std::atomic<std::uint32_t> compute_count;
  static std::atomic<InputState> observed_state;
  static std::atomic<SelectionReason> observed_reason;
  static std::atomic<ErrorCode> scalar_error;
  static std::atomic<ErrorCode> batch_error;
};

enum class RuntimeInputPredicatePattern : std::uint8_t {
  kAlwaysTrue,
  kFalseTrueTrue,
  kTrueFalseTrue,
  kTrueThenError,
  kError,
};

enum class RuntimeInputPredicateMissing : std::uint8_t {
  kFresh,
  kReuseLast,
  kSkip,
  kSafeDefault,
};

struct RuntimeInputPredicateTestState {
  std::uint32_t evaluation_count{};
};

struct RuntimeInputPredicateEvaluationObservation {
  TimePoint reference_time{};
  Tag tag{};
  TagTime tag_time{};
  InputSelection driver_selection{};
  InputSelection additional_selection{};
  SampleMetadata driver_metadata{};
  SampleMetadata additional_metadata{};
  std::uint32_t driver_value{};
  std::uint32_t additional_value{};
  std::uint32_t candidate_state_count{};
  bool value{};
  bool failed{};
};

struct RuntimeInputPredicateComputeObservation {
  TimePoint activation_time{};
  Tag tag{};
  TagTime tag_time{};
  InputSelection driver_selection{};
  InputSelection additional_selection{};
  InputSelection compute_only_selection{};
  SampleMetadata driver_metadata{};
  SampleMetadata additional_metadata{};
  std::uint32_t driver_value{};
  std::uint32_t additional_value{};
  bool has_additional{};
};

[[nodiscard]] expected<bool, Error> runtime_input_predicate_result(
    RuntimeInputPredicatePattern pattern, std::uint32_t evaluation, std::uint32_t driver_value,
    std::uint32_t minimum_driver_value);

/// @brief Retain end-to-end predicate evaluation and compute snapshot evidence.
struct RuntimeInputPredicateChannelState {
  static void reset();

  [[nodiscard]] static bool wait_for_source();

  [[nodiscard]] static bool wait_for_compute(std::size_t expected);

  [[nodiscard]] static bool wait_for_evaluator(std::size_t expected);

  static void record_source();

  static void record_evaluator(const Sample<std::uint32_t>& driver,
                               const Sample<std::uint32_t>& additional,
                               const PredicateEvaluationContext& context,
                               std::uint32_t candidate_state_count,
                               const expected<bool, Error>& result);

  static void record_compute(const Sample<std::uint32_t>& driver,
                             const Sample<std::uint32_t>* additional, ExecutionContext& context,
                             InputSelection driver_selection, InputSelection additional_selection,
                             InputSelection compute_only_selection);

  [[nodiscard]] static std::vector<RuntimeInputPredicateEvaluationObservation>
  evaluation_snapshot();

  [[nodiscard]] static std::vector<RuntimeInputPredicateComputeObservation> compute_snapshot();

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<std::size_t> evaluator_count;
  static std::atomic<std::size_t> compute_count;
  static std::atomic<std::size_t> source_count;
  static std::atomic<std::uint32_t> evaluator_driver_value;
  static std::atomic<std::uint32_t> evaluator_additional_value;
  static std::atomic<std::uint64_t> evaluator_driver_sequence;
  static std::atomic<std::uint64_t> evaluator_additional_sequence;
  static std::atomic<std::uint32_t> compute_driver_value;
  static std::atomic<std::uint32_t> compute_additional_value;
  static std::atomic<std::uint64_t> compute_driver_sequence;
  static std::atomic<std::uint64_t> compute_additional_sequence;
  static std::vector<RuntimeInputPredicateEvaluationObservation> evaluation_observations;
  static std::vector<RuntimeInputPredicateComputeObservation> compute_observations;
};

/// @brief Publish three capture-ordered input snapshots with a newer candidate already queued.
class RuntimeInputPredicateChannelSourceOperator final : public Operator<> {
 public:
  explicit RuntimeInputPredicateChannelSourceOperator(RuntimeInputPredicateMissing missing);

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Output<std::uint32_t> additional;
  Output<std::uint32_t> compute_only;
  Output<std::uint32_t> driver;

 private:
  bool emitted_{};
  RuntimeInputPredicateMissing missing_{RuntimeInputPredicateMissing::kFresh};
};

/// @brief Exercise the complete input-predicate transaction over Channel-backed inputs.
class RuntimeInputPredicateChannelSinkOperator final : public Operator<> {
 public:
  RuntimeInputPredicateChannelSinkOperator(RuntimeInputPredicatePattern pattern,
                                           PredicateFiring firing, bool configured,
                                           RuntimeInputPredicateMissing missing);

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<std::uint32_t> driver;
  Input<std::uint32_t> additional;
  Input<std::uint32_t> compute_only;

 private:
  RuntimeInputPredicatePattern pattern_{RuntimeInputPredicatePattern::kAlwaysTrue};
  PredicateFiring firing_{PredicateFiring::kOnFalseToTrue};
  bool configured_{};
  RuntimeInputPredicateMissing missing_{RuntimeInputPredicateMissing::kFresh};
};

enum class RuntimeComputeCallAction : std::uint8_t {
  kNone,
  kSkipFinalPublication,
  kObserveCallbackProgress,
  kFailOnSecond,
  kFailPublicationOnFirst,
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  kThrowOnFirst,
#endif
  kRequestStopAfterFirst,
  kRequestCompletionAfterFirst,
  kRestartAfterFirst,
  kRestartAfterComputeCalls,
  kRestartRollbackAfterFirst,
};

/// @brief Retain compute-call progress, restart, and final-data evidence for completion tests.
/// Used by RunSessionTest.FailedComputeCallRetainsOnlyPreviouslySuccessfulProgressAndPublication.
struct RuntimeComputeCallChannelProbeState {
  static void reset(bool hold_first_compute = false) noexcept;
  [[nodiscard]] static std::size_t enter_source_compute();
  [[nodiscard]] static bool wait_until_source_compute();
  /// Wait for prior publication witnesses before injecting the source failure in
  /// RunSessionTest.FailedComputeCallRetainsOnlyPreviouslySuccessfulProgressAndPublication.
  [[nodiscard]] static bool wait_until_sink_count(std::size_t expected_count);
  static void release_source_compute();
  static void record_sink(std::uint32_t value) noexcept;

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::atomic<std::size_t> source_count;
  static std::atomic<std::size_t> transform_count;
  static std::atomic<std::size_t> sink_compute_count;
  static std::atomic<std::size_t> sink_count;
  /// Samples received by the sink authored as the plan's first (always blocking) recipient.
  static std::atomic<std::size_t> first_sink_count;
  /// Microseconds each sink compute sleeps after recording its sample, so producers outrun it.
  static std::atomic<std::uint32_t> sink_stall_us;
  /// When set, each sink compute instead holds its sample until the relay has begun a later
  /// publication (at most 20 ms), then 200 us longer, so that publication meets an occupied slot.
  static std::atomic_bool sink_holds_until_relay_progress;
  /// Value the relay is about to publish, for sink_holds_until_relay_progress.
  static std::atomic<std::uint32_t> relay_publishing;
  static std::atomic<std::uint32_t> first_value;
  static std::atomic<std::uint32_t> second_value;
  static std::atomic_bool restart_accepted;
  static std::atomic_bool callback_progress_observed;
  static std::atomic<ErrorCode> restart_error;
  static std::atomic<std::uint64_t> successful_compute_calls_before_restart;
  static std::atomic<std::size_t> restart_forward_start_count;
  static std::atomic<std::size_t> restore_old_start_count;
  static bool hold_source;
  static bool source_entered;
  static bool release_source;
};

/// @brief Emit values from a zero-input source with Operator-local compute-call completion.
/// Used by RunSessionTest.FailedComputeCallRetainsOnlyPreviouslySuccessfulProgressAndPublication.
class RuntimeComputeCallSourceOperator final : public Operator<> {
 public:
  RuntimeComputeCallSourceOperator(
      std::uint64_t limit, RuntimeComputeCallAction action,
      std::uint64_t restart_after_successful_compute_calls = 1U) noexcept;

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  LifecycleStatus start_stage(LifecycleContext& context) noexcept;

  Output<std::uint32_t> output;

 private:
  std::uint64_t limit_{};
  RuntimeComputeCallAction action_{RuntimeComputeCallAction::kNone};
  std::uint64_t restart_after_successful_compute_calls_{1U};
};

/// @brief Keep a valid finite notification source open without publishing a readiness event.
class RuntimeDormantComputeCallSourceOperator final : public Operator<> {
 public:
  explicit RuntimeDormantComputeCallSourceOperator(std::uint64_t limit) noexcept;

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  LifecycleStatus arm(LifecycleContext& context) noexcept;

  Output<std::uint32_t> output;

 private:
  std::uint64_t limit_{};
  NotificationSource notification_;
  NotificationSender sender_;
};

/// @brief Forward one finite input without owning an Operator completion rule.
class RuntimeComputeCallTransformOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

/// @brief Drain one finite source through EOS without carrying a completion rule of its own.
class RuntimeComputeCallSinkOperator final : public Operator<> {
 public:
  explicit RuntimeComputeCallSinkOperator(bool first_sink = false) noexcept
      : first_sink_(first_sink) {}

  void setup(OperatorSpec& spec) override;

  [[nodiscard]] Contract contract() const override;

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Input<std::uint32_t> input;

 private:
  bool first_sink_{};
};

template <typename ClockT, typename... Args>
[[nodiscard]] ExecutionPlan make_runtime_test_plan_with_clock(bool reverse_authoring,
                                                              Args&&... args) {
  Graph graph{"runtime-test"};
  CompileOptions options;
  if (reverse_authoring) {
    const auto last = graph.op<RuntimeProbeOperator>("z-last", 2);
    const auto first = graph.op<RuntimeProbeOperator>("a-first", 1);
    graph.partition("runtime-spin-worker-first").add(first);
    graph.partition("runtime-spin-worker-last").add(last);
    options.deployment
        .bind_thread_placement("runtime-spin-worker-first",
                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
        .bind_thread_placement("runtime-spin-worker-last",
                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  } else {
    const auto probe = graph.op<RuntimeProbeOperator>("operator", 1);
    graph.partition("runtime-spin-worker").add(probe);
    options.deployment.bind_thread_placement("runtime-spin-worker",
                                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  }
  const ClockKey default_clock =
      graph.add_clock<ClockT>("runtime-test-clock", std::forward<Args>(args)...);
  graph.set_default_clock(default_clock);
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_test_plan(bool reverse_authoring = false);

[[nodiscard]] ExecutionPlan make_runtime_stop_dependency_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_deadline_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_two_deadline_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_role_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_clock_test_plan();

/// @brief Compile the production notification-source lifecycle test graph.
/// @return A compiled plan using sleeping-waitset notification placement.
[[nodiscard]] ExecutionPlan make_runtime_notification_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_clock_predicate_error_plan();

[[nodiscard]] ExecutionPlan make_runtime_clock_predicate_plan(
    RuntimeClockPredicatePattern pattern, PredicateFiring firing,
    bool restart_after_first_compute = false);

[[nodiscard]] ExecutionPlan make_runtime_mixed_source_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_input_predicate_replay_plan();

[[nodiscard]] ExecutionPlan make_runtime_clock_input_replay_plan(
    RuntimeClockInputReplayKind kind, Profile profile = Profile::kStandard);

/// @brief Compile a one-operator clock graph with an unwired Exact/SafeDefault input.
/// Used by RunSessionTest.UnwiredExactSafeDefaultRunsWithNotConnectedSelection.
[[nodiscard]] ExecutionPlan make_runtime_optional_safe_default_plan();

/// @brief Compile two clock-input replay owners whose mismatch can be selected by stable ordinal.
[[nodiscard]] ExecutionPlan make_runtime_dual_clock_input_replay_plan();

[[nodiscard]] ExecutionPlan make_runtime_input_predicate_channel_plan(
    RuntimeInputPredicatePattern pattern, PredicateFiring firing, bool configured = false,
    RuntimeInputPredicateMissing missing = RuntimeInputPredicateMissing::kFresh);

/// @brief Compile a compute-call source and two recipients with the selected per-edge capacity.
/// The depth-one default retains pressure coverage; completion tests can provision the full burst.
[[nodiscard]] ExecutionPlan make_runtime_compute_call_channel_plan(
    std::uint64_t limit = 1U, RuntimeComputeCallAction action = RuntimeComputeCallAction::kNone,
    std::uint64_t restart_after_successful_compute_calls = 1U, bool separate_partitions = true,
    bool force_event_wake = false,
    detail::ProcessLocalFanoutExperiment experiment =
        detail::ProcessLocalFanoutExperiment::kDisabled,
    bool busy_poll = false, QueuePolicy second_policy = QueuePolicy::kBlocking,
    bool pump_delivery = false, bool force_holomq_fanout = false, std::uint32_t queue_depth = 1U);

/// @brief Build one compute-call source and one Channel recipient, each in its own partition and
/// placed with `wake_mode`. With `split_processes`, each partition is also its own process group,
/// so the recipient's worker owns one local partition whose input is a cross-process import.
[[nodiscard]] ExecutionPlan make_runtime_idle_recipient_plan(std::uint64_t limit,
                                                             RuntimeComputeCallAction action,
                                                             WakeMode wake_mode,
                                                             bool split_processes);

/// Source -> transform -> sink, each in its own partition unless `colocate_source_and_transform`
/// puts the first two in one partition; every edge has depth one and `queue_policy`.
[[nodiscard]] ExecutionPlan make_runtime_compute_call_chain_plan(
    std::uint64_t limit = 2U, bool colocate_source_and_transform = false,
    QueuePolicy queue_policy = QueuePolicy::kBlocking);

/// Source -> {local, remote}: one depth-one drop-oldest fanout family with one recipient in the
/// source's partition and one in another. compile() delivers it by direct retention;
/// `force_holomq_fanout` keeps it on HoloMQ.
[[nodiscard]] ExecutionPlan make_runtime_drop_oldest_fanout_plan(
    std::uint64_t limit, RuntimeComputeCallAction action = RuntimeComputeCallAction::kNone,
    bool force_holomq_fanout = false);

/// Compile split fanout followed by a direct edge and a complete same-partition fanout family.
/// All edges use the selected capacity; the depth-one default preserves pressure tests.
[[nodiscard]] ExecutionPlan make_runtime_mixed_eventless_fanout_plan(
    std::uint64_t limit, RuntimeComputeCallAction action,
    detail::ProcessLocalFanoutExperiment experiment, std::uint32_t queue_depth = 1U);

[[nodiscard]] ExecutionPlan make_runtime_dormant_compute_call_plan(std::uint64_t limit = 2U);

[[nodiscard]] ExecutionPlan make_runtime_input_trigger_replay_plan(
    RuntimeInputTriggerReplayKind kind);

/// @brief Compile aligned members and optional sides with configurable wiring and source EOS.
/// Used by RunSessionTest.ChannelAlignedUnwiredSideInputDoesNotBlockNaturalCompletion.
[[nodiscard]] ExecutionPlan make_runtime_aligned_channel_plan(bool complete_manual_cover = false,
                                                              bool connect_optional = false,
                                                              bool finite_source = false);

/// @brief Compile one exact named source anchor and one automatic Channel consumer.
/// Currently has no test caller; shares the aligned fixture used by
/// RunSessionTest.ChannelAlignedTriggerAcceptsTheInclusiveExactBoundary.
[[nodiscard]] ExecutionPlan make_runtime_hybrid_aligned_channel_plan(bool connect_optional = false);

[[nodiscard]] ExecutionPlan make_runtime_mixed_channel_unconnected_port_plan();

[[nodiscard]] ExecutionPlan make_runtime_activation_cut_order_plan();

/// @brief Compile window closure or terminal-drain graphs with an optional compute-only side.
/// Used by RunSessionTest.OptionalWindowSidePreservesSuccessfulAndDegradedWindowSelection.
[[nodiscard]] ExecutionPlan make_runtime_window_channel_plan(
    bool separate_execution_partitions = false, bool optional_side = false);

[[nodiscard]] ExecutionPlan make_runtime_multi_lane_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_two_param_test_plan();

[[nodiscard]] ExecutionPlan make_event_policy_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_duration_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_validation_param_test_plan();

[[nodiscard]] ExecutionPlan make_runtime_float_param_test_plan();

[[nodiscard]] ExecutionPlan make_lifecycle_param_test_plan();

[[nodiscard]] ExecutionPlan make_mixed_mutability_param_test_plan();

[[nodiscard]] ExecutionPlan make_two_lifecycle_param_test_plan();

[[nodiscard]] ExecutionPlan make_three_lifecycle_param_test_plan();

[[nodiscard]] ExecutionPlan make_lifecycle_param_dependency_test_plan();

[[nodiscard]] bool has_runtime_diagnostic(const RunSession& session, std::string_view token);

}  // namespace holoscan::test
