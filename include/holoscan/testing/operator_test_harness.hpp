/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/holoscan_testing_export.h>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/testing/schema_snapshot.hpp>
#include <holoscan/testing/tensor_snapshot.hpp>

#include <holoscan/testing/operator_test_adapters.hpp>
#include <holoscan/testing/test_sample.hpp>

namespace holoscan::testing {

namespace detail {
struct RunningFixtureState;
struct RegisteredAdapterAccess;
}  // namespace detail

/// @brief Identify admitted payloads that retain native Tensor backing.
/// @tparam T Exact public Tensor or descriptor-backed generated schema type.
template <typename T>
concept TensorCapturePayload =
    PublishedPayload<T> &&
    (std::same_as<T, Tensor> || holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>);

/// @brief Identify owning value payloads that need no Tensor capture policy.
/// @tparam T Exact admitted scalar, POD, or ordinary generated schema type.
template <typename T>
concept ValueCapturePayload = PublishedPayload<T> && !TensorCapturePayload<T>;

/// @brief Identify Tensor payloads with complete owning snapshot support.
/// @tparam T Exact Tensor or descriptor schema with a metadata-clone operation.
template <typename T>
concept TensorSnapshotPayload =
    TensorCapturePayload<T> && (std::same_as<T, Tensor> || requires(const T& value) {
      {
        DescriptorTensorSchemaNativeTraits<T>::clone_metadata(value)
      } -> std::same_as<expected<T, Error>>;
    });

/// @brief Name the independently owned snapshot of a supported Tensor payload.
/// @tparam T Exact supported Tensor or descriptor-backed schema type.
template <TensorSnapshotPayload T>
using Snapshot = detail::Snapshot<T>;

/// @brief Own an immutable staged input description shared by C++ and native bindings.
/// @note Copying this holoscan.core testing handle shares immutable values. Staging or appending
/// never mutates another handle. Provider modules must remain loaded while their handles live.
class StagedInput final {
 public:
  /// @brief Construct an empty handle, rejected by NativeOperatorTest::stage_input().
  StagedInput() = default;

 private:
  explicit StagedInput(std::shared_ptr<const detail::InputScript> script)
      : script_(std::move(script)) {}
  std::shared_ptr<const detail::InputScript> script_;
  template <PublishedPayload T>
  friend StagedInput staged_samples(std::vector<TestSample<T>> samples);
  friend class NativeOperatorTest;
  friend struct detail::RegisteredAdapterAccess;
};

class CaptureRequest;

template <ValueCapturePayload T>
[[nodiscard]] CaptureRequest value_capture(std::size_t capacity = 1024U);
template <TensorSnapshotPayload T>
[[nodiscard]] CaptureRequest snapshot_capture(std::size_t capacity = 1024U,
                                              std::size_t maximum_bytes = 64U * 1024U * 1024U);
template <TensorCapturePayload T>
[[nodiscard]] CaptureRequest retain_capture(std::size_t capacity = 1024U);

/// @brief Own an immutable bounded capture policy shared by C++ and native bindings.
/// @note Reusing this holoscan.core testing handle creates independent storage for every run.
/// Provider modules must remain loaded while their requests or resulting captures live.
class CaptureRequest final {
 public:
  /// @brief Construct an empty request, rejected by NativeOperatorTest::declare_capture().
  CaptureRequest() = default;

 private:
  explicit CaptureRequest(std::shared_ptr<const detail::CaptureDeclaration> capture)
      : capture_(std::move(capture)) {}
  std::shared_ptr<const detail::CaptureDeclaration> capture_;
  template <ValueCapturePayload T>
  friend CaptureRequest value_capture(std::size_t capacity);
  template <TensorSnapshotPayload T>
  friend CaptureRequest snapshot_capture(std::size_t capacity, std::size_t maximum_bytes);
  template <TensorCapturePayload T>
  friend CaptureRequest retain_capture(std::size_t capacity);
  friend class NativeOperatorTest;
  friend struct detail::RegisteredAdapterAccess;
};

/// @brief Own an ordered batch of timestamped values, including an explicit empty producer.
/// @tparam T Exact input payload type.
/// @param samples Owning envelopes, copied through the native validation and lease boundary.
/// @return Reusable immutable input handle.
/// @throws std::invalid_argument or RuntimeError when bounds, Tensor storage, or leases reject.
template <PublishedPayload T>
[[nodiscard]] StagedInput staged_samples(std::vector<TestSample<T>> samples) {
  return StagedInput{std::make_shared<detail::ValueInputScript<T>>(std::move(samples))};
}

/// @brief Own an ordered batch of values without adding capture timestamps.
/// @tparam T Exact input payload type.
/// @param values Payloads in delivery order, including an explicit empty batch.
/// @return Reusable immutable input handle with the same validation as staged_samples().
/// @throws std::invalid_argument or RuntimeError when input validation rejects.
template <PublishedPayload T>
[[nodiscard]] StagedInput staged_values(std::vector<T> values) {
  std::vector<TestSample<T>> samples;
  samples.reserve(values.size());
  for (auto& value : values) {
    samples.push_back(TestSample<T>{.data = std::move(value)});
  }
  return staged_samples<T>(std::move(samples));
}

/// @brief Request independent owning values from a non-Tensor-bearing output.
/// @tparam T Exact scalar, POD, or ordinary generated schema payload type.
/// @param capacity Maximum stored records; overflow fails the run without dropping records.
/// @return Reusable bounded capture request.
/// @throws std::invalid_argument unless capacity fits a positive uint32 value.
template <ValueCapturePayload T>
[[nodiscard]] CaptureRequest value_capture(std::size_t capacity) {
  return CaptureRequest{std::make_shared<detail::ValueCaptureDeclaration<T>>(capacity)};
}

/// @brief Request independent host snapshots of Tensor-bearing output content.
/// @tparam T Exact payload with complete owning snapshot support.
/// @param capacity Maximum stored records.
/// @param maximum_bytes Maximum bytes across Tensor fields in one snapshot.
/// @return Reusable bounded snapshot request.
/// @throws std::invalid_argument unless capacity fits positive uint32 and byte bound is positive.
template <TensorSnapshotPayload T>
[[nodiscard]] CaptureRequest snapshot_capture(std::size_t capacity, std::size_t maximum_bytes) {
  return CaptureRequest{
      std::make_shared<detail::SnapshotCaptureDeclaration<T>>(capacity, maximum_bytes)};
}

/// @brief Request retained native backing with whole-run capacity and completion proof.
/// @tparam T Exact Tensor or descriptor-backed schema payload type.
/// @param capacity Maximum retained records; unsupported topologies reject before execution.
/// @return Reusable bounded retention request.
/// @throws std::invalid_argument unless capacity fits a positive uint32 value.
template <TensorCapturePayload T>
[[nodiscard]] CaptureRequest retain_capture(std::size_t capacity) {
  return CaptureRequest{std::make_shared<detail::RetainCaptureDeclaration<T>>(capacity)};
}

/// @brief Keep real output Tensors after a finite source test, without copying their bytes.
/// @note Only a finite source with one output is supported. The complete output bound and
/// pool capacity must be established before execution.
/// @note The bounded source profile preserves Runtime kStopped after the collector observes the
/// complete compiler-derived output bound. Input-driven or unproved geometries reject before run.
struct Retain {
  std::size_t max_records{1024U};  ///< Maximum retained samples; never a truncation count.
};

/// @brief Name one owning value in a staged frame.
/// @tparam T Exact input payload type.
template <PublishedPayload T>
struct NamedTestInput {
  std::string name;  ///< Tested operator input name.
  T value;           ///< One owning payload.
};

/// @brief Name a single frame publication without inferring a batch from its value.
/// @tparam T Exact input payload type.
/// @param name Tested operator input name.
/// @param value Owning payload value.
/// @return Named frame entry.
template <PublishedPayload T>
[[nodiscard]] NamedTestInput<T> input(std::string_view name, T value) {
  return NamedTestInput<T>{std::string(name), std::move(value)};
}

/// @brief Own one ordered, heterogeneous source activation's publications.
struct TestFrame {
  /// @brief Hold one named owning payload adapter in emission order.
  struct Field {
    std::string name;   ///< Tested operator input name.
    StagedInput value;  ///< Exactly one staged value through an owning public handle.
  };
  std::vector<Field> fields;  ///< Ordered publications; omitted inputs publish nothing.
};

/// @brief Own an ordered frame without adding capture coordinates.
/// @tparam T Exact payload type of each field.
/// @param values Named publications in caller-selected order; no values create an empty frame.
/// @return Owning frame whose entries preserve the absence of capture time.
/// @throws std::invalid_argument or RuntimeError if a payload cannot be staged.
template <PublishedPayload... T>
[[nodiscard]] TestFrame frame(NamedTestInput<T>... values) {
  TestFrame result;
  result.fields.reserve(sizeof...(T));
  (result.fields.push_back(TestFrame::Field{
       std::move(values.name), staged_values<T>(std::vector<T>{std::move(values.value)})}),
   ...);
  return result;
}

/// @brief Give every ordered frame entry the same explicit capture coordinate.
/// @tparam T Exact payload type of each field.
/// @param coordinate Shared fixture-domain capture coordinate, bound at emission.
/// @param values Named publications in caller-selected order.
/// @return Owning frame, without clock advancement or fixture-side selection.
/// @throws std::invalid_argument or RuntimeError if a payload cannot be staged.
template <PublishedPayload... T>
[[nodiscard]] TestFrame frame_at(std::chrono::nanoseconds coordinate, NamedTestInput<T>... values) {
  TestFrame result;
  result.fields.reserve(sizeof...(T));
  (result.fields.push_back(TestFrame::Field{std::move(values.name),
                                            staged_samples<T>(std::vector<TestSample<T>>{
                                                sample_at(std::move(values.value), coordinate)})}),
   ...);
  return result;
}

/// @brief Select how the immutable grouped input script becomes eligible.
enum class FrameRelease : std::uint8_t {
  kAutomatic = 0,  ///< Use the ordinary finite source's native clock trigger.
  kManual = 1,     ///< Use one bounded native notification per pre-staged frame.
};

/// @brief Configure one blocking or asynchronous attempt without modifying staging.
struct TestRunOptions {
  /// Cooperative steady-clock timeout; must be positive. It cannot interrupt a blocked callback.
  std::chrono::milliseconds timeout{30000};
  /// Accept a clean kStopped outcome. Failures and watchdog expiry always override this
  /// expectation; controlled-clock and Retain completion proofs remain required.
  bool expect_stop{};
  /// Grouped-frame release policy. Manual mode requires run_async() and grouped staging only.
  FrameRelease frames{FrameRelease::kAutomatic};
};

/// @brief Qualify one clock tick after staged inputs become available in Runtime.
/// @note This holoscan.core testing convenience creates a shared SyntheticClock frozen at zero.
/// It currently requires one ordinary partition, scalar inputs with one sample each, an unchanged
/// OnClock/LatestWithin/Skip contract, and a collector for every output. The startup tick at zero
/// is real. Unsupported configurations reject before consuming staging. This is not a step API.
struct ControlledClockRun {
  std::chrono::nanoseconds tick{};  ///< Positive exact period boundary to observe after startup.
  std::size_t observation_capacity{
      1024U};  ///< Bounded Runtime records and total publication attempts.
  std::chrono::milliseconds timeout{30000};  ///< Steady-clock cooperative watchdog, not test time.
};

/// @brief Distinguish the requested tick's actual Runtime outcome.
enum class ClockTestOutcome : std::uint8_t {
  kNotRequested,  ///< The ordinary finite run overload was used.
  kRetired,       ///< Compute ran and its complete retirement succeeded, including no-output calls.
  kSuppressed,    ///< Native selection suppressed compute and finalized the candidate.
  kFailed,        ///< The requested observation or completion proof was not established.
};

/// @brief Keep the outcome and captured outputs of a completed test.
/// @note Copies share immutable result storage. Reading a result may be concurrent after
/// publication; moving or destroying the same wrapper requires external exclusion. Accessors
/// returning values allocate independent copies and may throw. The retained plan also owns its
/// replayable input factories, so retaining a result may keep staged Tensor read leases alive.
/// Reexecuting an automatic finite plan creates fresh collector storage and cannot change this
/// result. A manual-frame plan remains inspectable, but direct replay rejects because no live
/// release driver belongs to the reconstructed session; stage a fresh fixture attempt instead.
class HOLOSCAN_TESTING_EXPORT TestResult {
 public:
  /// @brief Return whether the expected completion and cleanup succeeded without test failures.
  /// @note Clean kCompleted and kRequestedCompleted succeed. kStopped requires expect_stop or
  /// the controlled/Retain completion proof. Capture failures and watchdog expiry always fail.
  [[nodiscard]] bool ok() const noexcept;
  /// @brief Return the actual Runtime terminal disposition.
  [[nodiscard]] RunTermination termination() const noexcept;
  /// @brief Return the exact compiled plan, including rejected admission evidence.
  [[nodiscard]] const ExecutionPlan& plan() const noexcept;
  /// @brief Return copied compile and Runtime diagnostics.
  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept;
  /// @brief Describe terminal failures, rejected admission, and watchdog expiry.
  [[nodiscard]] std::string describe_failure() const;
  /// @brief Return the requested controlled tick outcome, distinct from run termination.
  [[nodiscard]] ClockTestOutcome clock_outcome() const noexcept;
  /// @brief Return the native selection recorded for the controlled tick.
  /// @param name Operator input name; ordinary or unsuccessful observations reject this query.
  /// @throws std::invalid_argument if name is not an input in this controlled observation.
  /// @throws std::logic_error if the controlled tick or its native selection is unavailable.
  [[nodiscard]] InputSelection clock_selection(std::string_view name) const;
  /// @brief Return the failed capture's metadata and error, if collection failed for this output.
  /// @param name Declared capture output; an unknown capture is rejected.
  /// @return Owning failure evidence without the failed payload or a finite read claim.
  /// @throws std::invalid_argument if the requested output or payload type was not captured.
  [[nodiscard]] std::optional<CaptureFailure> capture_failure(std::string_view name) const;

  /// @brief Return an independent copy of the captured samples for one exact payload type.
  /// @tparam T Declared output payload type.
  /// @param name Tested operator output name.
  /// @throws std::invalid_argument if the output was not captured with this type.
  template <PublishedPayload T>
  [[nodiscard]] std::vector<Sample<T>> samples(std::string_view name) const {
    const auto& storage = capture_storage(name);
    const auto* typed = dynamic_cast<const detail::ValueCaptureStorage<T>*>(&storage);
    if (storage.kind() != detail::CaptureKind::kValue || typed == nullptr) {
      throw std::invalid_argument{"operator test capture payload type mismatch; output='" +
                                  std::string(name) + "'"};
    }
    return typed->samples;
  }

  /// @brief Return independently owned Tensor or schema snapshot records.
  /// @tparam T Exact captured payload type.
  /// @param name Captured operator output.
  /// @throws std::invalid_argument if the requested output or payload type was not captured.
  template <TensorSnapshotPayload T>
  [[nodiscard]] std::vector<Snapshot<T>> snapshots(std::string_view name) const {
    const auto& storage = capture_storage(name);
    const auto* typed = dynamic_cast<const detail::SnapshotCaptureStorage<T>*>(&storage);
    if (storage.kind() != detail::CaptureKind::kSnapshot || typed == nullptr) {
      throw std::invalid_argument{
          "operator test snapshot payload or capture policy mismatch; output='" +
          std::string(name) + "'"};
    }
    return typed->snapshots;
  }
  /// @brief Return the only snapshot; reject zero or multiple records.
  /// @tparam T Exact captured payload type.
  /// @param name Captured output.
  /// @throws std::invalid_argument if the requested output or payload type was not captured.
  /// @throws std::logic_error unless exactly one snapshot was captured.
  template <TensorSnapshotPayload T>
  [[nodiscard]] Snapshot<T> snapshot(std::string_view name) const {
    auto result = snapshots<T>(name);
    if (result.size() != 1U)
      throw std::logic_error{
          "operator test snapshot for output '" + std::string(name) +
          "' requires exactly one record; actual=" + std::to_string(result.size())};
    return std::move(result.front());
  }

  /// @brief Return all captured output values in observed publication order.
  /// @tparam T Exact output payload type.
  /// @param name Tested operator output name.
  /// @throws std::invalid_argument if the requested output or payload type was not captured.
  template <PublishedPayload T>
  [[nodiscard]] std::vector<T> values(std::string_view name) const {
    auto captured = samples<T>(name);
    std::vector<T> result;
    result.reserve(captured.size());
    for (auto& sample : captured) {
      result.push_back(std::move(sample.data));
    }
    return result;
  }

  /// @brief Return the only captured value; reject zero or multiple records.
  /// @tparam T Exact output payload type.
  /// @param name Tested operator output name.
  /// @throws std::logic_error unless exactly one value was captured.
  /// @throws std::invalid_argument if the requested output or payload type was not captured.
  template <PublishedPayload T>
  [[nodiscard]] T value(std::string_view name) const {
    auto captured = values<T>(name);
    if (captured.size() != 1U) {
      throw std::logic_error{
          "operator test value for output '" + std::string(name) +
          "' requires exactly one captured record; actual=" + std::to_string(captured.size())};
    }
    return std::move(captured.front());
  }

 private:
  explicit TestResult(std::shared_ptr<const detail::TestResultState> state);
  [[nodiscard]] const detail::CaptureStorage& capture_storage(std::string_view name) const;
  std::shared_ptr<const detail::TestResultState> state_;
  friend class NativeOperatorTest;
  friend class RunningFixture;
  friend struct detail::RegisteredAdapterAccess;
};

/// @brief Own one asynchronous holoscan.core operator-test attempt independently of its builder.
/// @note The caller serializes finish, move and destruction. Destruction requests cooperative
/// stop and joins; nonreturning callbacks or CUDA work can prevent completion indefinitely.
/// Captured payloads become accessible only through the joined result returned by finish().
/// Counts remain cumulative across same-session restart generations. Move assignment is
/// unavailable; destroy/reset an old owner before creating its replacement with optional::emplace.
/// @warning Joining a native session from a Runtime compute, lifecycle, or Dispatcher callback
/// is a fatal contract violation and terminates the process. Call finish() and destroy live
/// owners outside Runtime callbacks.
class HOLOSCAN_TESTING_EXPORT RunningFixture {
 public:
  /// @brief Request stop and join every owned activity before releasing storage.
  /// @warning Destroying a native session from a Runtime callback terminates the process;
  /// destruction never detaches work.
  ~RunningFixture() noexcept;
  /// @brief Transfer the stable attempt owner without moving any live state.
  RunningFixture(RunningFixture&&) noexcept;
  RunningFixture(const RunningFixture&) = delete;
  RunningFixture& operator=(const RunningFixture&) = delete;
  RunningFixture& operator=(RunningFixture&&) = delete;

  /// @brief Join and return the same immutable owning result on every call.
  /// @note A compile-rejected attempt returns its diagnostics without starting a native session.
  /// @throws std::logic_error if this wrapper has been moved from.
  /// @throws std::system_error if the native session or watchdog thread join fails.
  /// @warning Joining from a Runtime callback terminates the process; stop remains cooperative.
  [[nodiscard]] TestResult finish();
  /// @brief Query native join admission without joining; hosted bindings check before waits.
  /// @return The native calling-context error, or success for a cold/completed attempt.
  [[nodiscard]] expected<void, Error> wait_admission() const noexcept;
  /// @brief Request cooperative stop without joining; harmless after move or completion.
  /// @note On return, manual frame admission is closed even while native work is retiring.
  void request_stop() noexcept;
  /// @brief Post one native notification to release at most one pre-staged grouped frame.
  /// @return Native post result; NotReady before Arm/Start, Backpressured until retirement,
  /// StaleEndpoint after invalidation, Closed after stop or compile rejection, or InvalidArgument
  /// outside manual mode or after move. Rejection text directs callers to finish() diagnostics.
  /// @note Accepted notifications are not replayed after cancellation. Submission does not
  /// advance the script; a successful source callback does. No live payload is accepted.
  [[nodiscard]] expected<void, Error> release_frame() const noexcept;
  /// @brief Release at most one frame within a bounded readiness/backpressure retry interval.
  /// @param timeout Nonnegative steady-clock budget; zero performs exactly one immediate attempt.
  /// @return Success admits one notification. InvalidArgument rejects an invalid duration before
  /// posting; Timeout means no notification was admitted within the retry budget. Only NotReady
  /// and Backpressured are retried; stale endpoints, closed attempts and other errors return.
  /// @throws std::system_error if the platform timed wait fails.
  [[nodiscard]] expected<void, Error> release_frame(std::chrono::milliseconds timeout) const;

  /// @brief Return whether native run-owned work remains outstanding; false after move/rejection.
  [[nodiscard]] bool is_running() const noexcept;
  /// @brief Return native run disposition, or failed for a cold rejection or moved-from owner.
  [[nodiscard]] RunTermination termination() const noexcept;
  /// @brief Return the tested operator's identity bound to the current native run.
  /// @return Native identity or an error; rejection before start names finish() for diagnostics.
  [[nodiscard]] expected<OperatorRef, Error> operator_ref() const noexcept;
  /// @brief Bind an explicit identity from this plan to the current native run.
  /// @param plan_ref Plan-only identity; native stale and foreign checks are preserved.
  [[nodiscard]] expected<OperatorRef, Error> operator_ref(OperatorRef plan_ref) const noexcept;
  /// @brief Mint a restriction-only native lifecycle handle.
  /// @param policy Copied target and stage subset; an empty policy grants no authority.
  /// @return Native control handle, or Closed after move or a compile-rejected attempt.
  /// @note Compile-rejection text names finish() for the original admission diagnostics.
  [[nodiscard]] expected<LifecycleControlHandle, Error> lifecycle_control(
      const LifecycleAccessPolicy& policy);
  /// @brief Return the synchronized number of complete records observed by one collector.
  /// @param name Declared captured output; unknown names throw std::invalid_argument.
  /// @note Counts are cumulative for the whole attempt, including every restart generation.
  /// Observation does not establish callback retirement or restart completion.
  /// @throws std::logic_error if this owner was moved from.
  [[nodiscard]] std::size_t output_count(std::string_view name) const;
  /// @brief Wait for a captured prefix without exposing live payload storage.
  /// @param name Declared captured output.
  /// @param minimum Required number of complete records.
  /// @param timeout Nonnegative caller-side timeout; zero queries without waiting.
  /// @return True if the prefix was observed; false on terminal completion or timeout.
  /// @note Counts are cumulative across restart generations. False does not prove suppression
  /// or cancel the run/restart; timeout and termination can race with later status inspection.
  /// @throws std::logic_error if this owner was moved from.
  /// @throws std::invalid_argument if the output is unknown or timeout is negative/unrepresentable.
  /// @throws std::system_error if the platform condition-variable wait fails.
  [[nodiscard]] bool wait_for_output(std::string_view name, std::size_t minimum,
                                     std::chrono::milliseconds timeout) const;

 private:
  explicit RunningFixture(std::unique_ptr<detail::RunningFixtureState> state);
  std::unique_ptr<detail::RunningFixtureState> state_;
  friend class NativeOperatorTest;
};

/// @brief Borrow a single cold fixture build's native Graph and compile settings.
/// @note The view and its handles are valid only during the configuration callback. Placement
/// records borrow names owned by the view and must be copied into native CompileOptions before
/// the callback returns. This view provides authoring access, never a live Runtime operator.
class HOLOSCAN_TESTING_EXPORT NativeTestGraphView {
 public:
  /// @brief Return the fixture Graph for call-scoped native authoring.
  [[nodiscard]] Graph& graph() const noexcept { return graph_; }
  /// @brief Return the configured authoring handle, never a Runtime instance.
  [[nodiscard]] const authoring::OpaqueOperatorRef& tested_operator() const noexcept { return op_; }
  /// @brief Add all unique finite producers to one native partition atomically.
  /// @param partition Partition of this fixture Graph.
  /// @throws RuntimeError if the partition or helper handles are invalid or foreign.
  void add_sources_to(PartitionHandle& partition) const;
  /// @brief Add all output collectors to one native partition atomically.
  /// @param partition Partition of this fixture Graph.
  /// @throws RuntimeError if the partition or helper handles are invalid or foreign.
  void add_collectors_to(PartitionHandle& partition) const;
  /// @brief Resolve a source output into a native placement record.
  /// @param name Operator input fed by this helper.
  /// @param device Explicit CUDA device ordinal.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  [[nodiscard]] TensorOutputDevicePlacement input_source_device_placement(std::string_view name,
                                                                          DeviceId device) const;
  /// @brief Resolve an operator output into a native placement record.
  /// @param name Operator output, including an uncaptured output.
  /// @param device Explicit CUDA device ordinal.
  /// @throws RuntimeError if the operator handle or named output port is invalid.
  [[nodiscard]] TensorOutputDevicePlacement output_device_placement(std::string_view name,
                                                                    DeviceId device) const;
  /// @brief Borrow one exact native helper output for this callback only.
  /// @tparam T Exact declared input payload type.
  /// @param name Operator input fed by the helper.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] const Output<T>& source_output(std::string_view name) const {
    const auto& helper = helper_port(sources_, name);
    const auto* base =
        static_cast<const OutputBase*>(holoscan::detail::OperatorTestingAccess::resolve(
            helper.op,
            helper.port,
            PortDirection::kOutput,
            holoscan::detail::operator_spec_payload_identity<T>(),
            &holoscan::detail::kOperatorSpecLocalTypeToken<T>,
            &typeid(T)));
    return *static_cast<const Output<T>*>(base);
  }
  /// @brief Borrow one exact native collector input for this callback only.
  /// @tparam T Exact declared output payload type.
  /// @param name Captured operator output.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] const Input<T>& capture_input(std::string_view name) const {
    const auto& helper = helper_port(collectors_, name);
    const auto* base =
        static_cast<const InputBase*>(holoscan::detail::OperatorTestingAccess::resolve(
            helper.op,
            helper.port,
            PortDirection::kInput,
            holoscan::detail::operator_spec_payload_identity<T>(),
            &holoscan::detail::kOperatorSpecLocalTypeToken<T>,
            &typeid(T)));
    return *static_cast<const Input<T>*>(base);
  }
  /// @brief Resolve a typed source-to-operator flow for native Graph configuration.
  /// @tparam T Exact payload type at both endpoints.
  /// @param name Staged operator input.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] FlowHandle<T> input_flow(std::string_view name) const {
    const auto& helper = helper_port(sources_, name);
    return holoscan::detail::OperatorTestingAccess::flow<T>(
        graph_, helper.op, helper.port, op_, name);
  }
  /// @brief Resolve a typed operator-to-collector flow for native Graph configuration.
  /// @tparam T Exact payload type at both endpoints.
  /// @param name Captured operator output.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] FlowHandle<T> output_flow(std::string_view name) const {
    const auto& helper = helper_port(collectors_, name);
    return holoscan::detail::OperatorTestingAccess::flow<T>(
        graph_, op_, name, helper.op, helper.port);
  }
  /// @brief Return the per-attempt owning compile-options copy.
  [[nodiscard]] CompileOptions& compile_options() const noexcept { return options_; }

 private:
  NativeTestGraphView(Graph& graph, const authoring::OpaqueOperatorRef& op, CompileOptions& options,
                      const detail::HelperPorts& sources, const detail::HelperPorts& collectors)
      : graph_(graph), op_(op), options_(options), sources_(sources), collectors_(collectors) {}
  [[nodiscard]] static const detail::HelperPort& helper_port(const detail::HelperPorts& ports,
                                                             std::string_view name);
  Graph& graph_;
  const authoring::OpaqueOperatorRef& op_;
  CompileOptions& options_;
  const detail::HelperPorts& sources_;
  const detail::HelperPorts& collectors_;
  mutable std::set<std::string, std::less<>> placement_names_;
  friend class NativeOperatorTest;
};

/// @brief Borrow a native fixture view with typed configured operator access.
/// @tparam Op Exact operator type.
template <std::derived_from<OperatorBase> Op>
class TestGraphView {
 public:
  /// @brief Adapt one active native callback view.
  /// @param view Native call-scoped authoring view.
  explicit TestGraphView(NativeTestGraphView& view) : view_(view) {}
  /// @brief Return the actual fixture Graph.
  [[nodiscard]] Graph& graph() const noexcept { return view_.graph(); }
  /// @brief Re-resolve the typed authoring handle after Params reconstruction.
  [[nodiscard]] OperatorHandle<Op> tested_operator() const {
    return holoscan::detail::OperatorTestingAccess::typed_handle<Op>(view_.tested_operator());
  }
  /// @brief Add finite sources to the supplied fixture partition.
  /// @param partition Current Graph partition.
  /// @throws RuntimeError if the partition or helper handles are invalid or foreign.
  void add_sources_to(PartitionHandle& partition) const { view_.add_sources_to(partition); }
  /// @brief Add collectors to the supplied fixture partition.
  /// @param partition Current Graph partition.
  /// @throws RuntimeError if the partition or helper handles are invalid or foreign.
  void add_collectors_to(PartitionHandle& partition) const { view_.add_collectors_to(partition); }
  /// @brief Borrow the actual source output feeding an operator input.
  /// @tparam T Exact payload type.
  /// @param name Operator input name.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] const Output<T>& source_output(std::string_view name) const {
    return view_.template source_output<T>(name);
  }
  /// @brief Borrow the collector input attached to an operator output.
  /// @tparam T Exact payload type.
  /// @param name Operator output name.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] const Input<T>& capture_input(std::string_view name) const {
    return view_.template capture_input<T>(name);
  }
  /// @brief Return a source-output placement borrowing this callback's helper names.
  /// @param name Operator input name.
  /// @param device Explicit device ordinal.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  [[nodiscard]] TensorOutputDevicePlacement input_source_device_placement(std::string_view name,
                                                                          DeviceId device) const {
    return view_.input_source_device_placement(name, device);
  }
  /// @brief Return a native placement for a declared operator output.
  /// @param name Operator output name.
  /// @param device Explicit device ordinal.
  /// @throws RuntimeError if the operator handle or named output port is invalid.
  [[nodiscard]] TensorOutputDevicePlacement output_device_placement(std::string_view name,
                                                                    DeviceId device) const {
    return view_.output_device_placement(name, device);
  }
  /// @brief Resolve the staged input's flow through the shared native view.
  /// @tparam T Exact payload type at both endpoints.
  /// @param name Staged operator input.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] FlowHandle<T> input_flow(std::string_view name) const {
    return view_.template input_flow<T>(name);
  }
  /// @brief Resolve the captured output's flow through the shared native view.
  /// @tparam T Exact payload type at both endpoints.
  /// @param name Captured operator output.
  /// @throws std::invalid_argument if the named helper is absent.
  /// @throws RuntimeError if native Graph handles or payload types are incompatible.
  template <PublishedPayload T>
  [[nodiscard]] FlowHandle<T> output_flow(std::string_view name) const {
    return view_.template output_flow<T>(name);
  }
  /// @brief Return the current attempt's owning native compile options.
  [[nodiscard]] CompileOptions& compile_options() const noexcept { return view_.compile_options(); }

 private:
  NativeTestGraphView& view_;
};

template <std::derived_from<OperatorBase> Op>
class OperatorTestHarness;

/// @brief Stage named inputs and captures for a C++ operator supplied by a factory.
/// @note This optional testing surface belongs to holoscan.core and links holoscan::testing.
/// Serialize authoring operations. An active staging/build/run operation rejects reentrant
/// mutation instead of replaying a partially changed description. Move and destruction require
/// external exclusion from every operation. Methods on a moved-from fixture throw
/// std::logic_error. Move assignment is unavailable; reconstruct with optional::emplace if needed.
/// Each run constructs fresh operator instances. Cold rejection preserves staged inputs;
/// starting execution consumes them even if startup, compute or cleanup subsequently fails.
/// Captures and configuration persist for the next attempt. The blocking run owns and joins
/// its session. An asynchronous owner retains its own session independently of this builder.
/// The cooperative watchdog cannot terminate nonreturning user code or CUDA work.
/// @note Misuse throws std::invalid_argument or std::logic_error; native Graph authoring errors
/// throw RuntimeError. Compile rejection and execution failure return a failed TestResult.
/// Port setters construct a fresh prototype and validate selectors before changing staging.
/// Defaults are operator name "operator", partition "fixture", RealtimeClock "fixture_clock",
/// OnClock{1ms} finite input helpers, queues derived from staged/captured counts, 1024 captures,
/// 64 MiB per snapshot and a configurable 30-second watchdog.
/// @throws std::logic_error if the fixture is active or moved-from.
/// @throws std::invalid_argument if the value or fixture configuration is unsupported.
/// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
/// @warning Do not call blocking run() from a Runtime callback. Stop is cooperative.
class HOLOSCAN_TESTING_EXPORT NativeOperatorTest {
 public:
  /// @brief Name a replayable ordinary Graph construction operation.
  using OperatorFactory = std::function<authoring::OpaqueOperatorRef(Graph&)>;
  /// @brief Construct a fixture with a fresh ordinary authoring factory per attempt.
  /// @param factory Create the configured operator in the supplied Graph.
  explicit NativeOperatorTest(OperatorFactory factory);
  /// @brief Destroy staged state; an asynchronous running owner keeps its attempt alive.
  ~NativeOperatorTest();
  NativeOperatorTest(const NativeOperatorTest&) = delete;
  NativeOperatorTest& operator=(const NativeOperatorTest&) = delete;
  /// @brief Transfer staging ownership; an active asynchronous attempt retains its shared gate.
  NativeOperatorTest(NativeOperatorTest&&) noexcept;
  NativeOperatorTest& operator=(NativeOperatorTest&&) = delete;

  /// @brief Replace the complete replayable configured operator factory.
  /// @param factory New ordinary Graph construction operation.
  void set_operator_factory(OperatorFactory factory);
  /// @brief Atomically replace a configured factory and its complete capture selection.
  /// @param factory Replayable ordinary Graph construction operation.
  /// @param captures Complete named capture declarations for the new configured ports.
  /// @note Pending inputs are retained and validated against the new exact port identities.
  void set_operator_configuration(OperatorFactory factory,
                                  std::map<std::string, CaptureRequest, std::less<>> captures);
  /// @brief Append one atomic exact-type input batch through an owning public handle.
  /// @param name Tested operator input name.
  /// @param input Immutable pending values; an empty handle is rejected.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void stage_input(std::string_view name, StagedInput input);
  /// @brief Append one heterogeneous frame atomically to the shared finite producer.
  /// @param frame Ordered named publications; absence publishes nothing.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void push_frame(const TestFrame& frame);
  /// @brief Replace capture declaration for one output without altering pending inputs.
  /// @param name Tested operator output name.
  /// @param capture Exact-type bounded public request; an empty handle is rejected.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void declare_capture(std::string_view name, CaptureRequest capture);
  /// @brief Replace or clear one operator input-history override.
  /// @param name Input name on the configured operator.
  /// @param depth Native history depth; nullopt clears this key.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_queue_depth(std::string_view name, std::optional<std::uint32_t> depth);
  /// @brief Copy, replace or clear native options for a staged input flow.
  /// @param name Tested operator input name.
  /// @param options Owned native options; nullopt restores fixture derivation.
  /// @note The matching source or capture must exist when run() builds this configured flow.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if name is empty or an authored bound is zero.
  /// @throws RuntimeError if the named operator port or direction is invalid.
  void input_flow(std::string_view name, std::optional<ConnectionOptions> options);
  /// @brief Copy, replace or clear native options for a requested capture flow.
  /// @param name Tested operator output name.
  /// @param options Owned native options; nullopt restores fixture derivation.
  /// @note The matching source or capture must exist when run() builds this configured flow.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if name is empty or an authored bound is zero.
  /// @throws RuntimeError if the named operator port or direction is invalid.
  void output_flow(std::string_view name, std::optional<ConnectionOptions> options);
  /// @brief Replace or clear operator tensor input contract.
  /// @param name Logical operator port selector.
  /// @param value Native value copied before return; nullopt clears.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_tensor_spec(std::string_view name, std::optional<TensorInputSpec> value);
  /// @brief Replace or clear operator tensor output contract.
  /// @param name Logical operator port selector.
  /// @param value Native value copied before return; nullopt clears.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void output_tensor_spec(std::string_view name, std::optional<TensorOutputSpec> value);
  /// @brief Replace or clear staged source producer promise.
  /// @param name Logical operator port selector.
  /// @param value Native value copied before return; nullopt clears.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_source_tensor_spec(std::string_view name, std::optional<TensorOutputSpec> value);
  /// @brief Replace or clear staged source output cuda placement.
  /// @param name Logical operator port selector.
  /// @param value Native value copied before return; nullopt clears.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_source_device(std::string_view name, std::optional<DeviceId> value);
  /// @brief Replace or clear operator output cuda placement.
  /// @param name Logical operator port selector.
  /// @param value Native value copied before return; nullopt clears.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void output_device(std::string_view name, std::optional<DeviceId> value);
  /// @brief Replace the entire staged owning CompileOptions value.
  /// @param options Value copied again for each fresh attempt.
  /// @throws std::logic_error if the fixture is active or moved-from.
  void set_compile_options(CompileOptions options);
  /// @brief Replace the late native hook; an empty function clears it.
  /// @param hook Callback invoked after wiring and before default partition selection and compile.
  /// @throws std::logic_error if the fixture is active or moved-from.
  void configure_graph(std::function<void(NativeTestGraphView&)> hook);
  /// @brief Compile, execute, join and return an owning finite test result.
  /// @note Cold failure preserves input staging; an attempted run consumes it without replay.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  [[nodiscard]] TestResult run();
  /// @brief Run with an explicit cooperative timeout and termination expectation.
  /// @param options Positive watchdog timeout, expected stop and automatic frame release policy.
  /// @note Use \c run() or \c run({}) for defaults; braces may specify only \c timeout.
  /// @throws std::invalid_argument if the timeout is invalid or manual frame release is requested.
  /// @throws RuntimeError if Graph authoring rejects a staged configuration.
  /// @throws std::logic_error if this fixture is moved-from or already active.
  [[nodiscard]] TestResult run(TestRunOptions options);
  /// @brief Begin one finite attempt and transfer its independent lifetime to the caller.
  /// @note Cold rejection preserves staging; attempted execution consumes it. The same timeout
  /// and expected-stop policy as run(TestRunOptions) applies across the complete attempt.
  /// @param options Cooperative timeout, expected stop and automatic or manual frame release.
  /// @throws std::logic_error if this fixture is moved-from or already active.
  /// @throws std::invalid_argument if the timeout or frame-release configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects a staged configuration.
  [[nodiscard]] RunningFixture run_async(TestRunOptions options = {});
  /// @brief Observe one controlled tick through native availability and retirement barriers.
  /// @param control Tick, bounded observation storage and cooperative steady watchdog settings.
  /// @note Controlled-clock execution is separate from ordinary \c run options.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  [[nodiscard]] TestResult run_controlled_clock(ControlledClockRun control);

 private:
  // Retain the existing exported symbols as private implementation seams, not extension APIs.
  void stage_input(std::string_view name, std::shared_ptr<const detail::InputScript> script);
  void declare_capture(std::string_view name,
                       std::shared_ptr<const detail::CaptureDeclaration> capture);
  [[nodiscard]] RunningFixture run_impl(std::optional<ControlledClockRun> control,
                                        bool observe = false, TestRunOptions options = {});
  [[nodiscard]] detail::OperatorTestState& require_state() const;
  [[nodiscard]] std::string resolve_input_name(
      const std::function<std::string(const authoring::OpaqueOperatorRef&)>& select) const;
  std::unique_ptr<detail::OperatorTestState> state_;
  template <std::derived_from<OperatorBase> Op>
  friend class OperatorTestHarness;
};

/// @brief Provide typed convenience methods over the shared native controller.
/// @tparam Op Concrete operator type; registration is not required.
/// @note The serialization, move and finite-execution contract is the same as
/// \ref NativeOperatorTest. Constructor arguments are copied from a replay tuple for every
/// authoring build and must remain valid when copied; borrowed arguments do not become owning.
/// Member selectors resolve against the configured authoring prototype, not a running instance.
template <std::derived_from<OperatorBase> Op>
class OperatorTestHarness {
 public:
  /// @brief Name the operator parameter type used at the start of each attempt.
  using Params = typename Op::Params;

  /// @brief Construct with replayable ordinary Graph constructor arguments.
  /// @tparam Args Replayable constructor argument types.
  /// @param args Constructor arguments copied into each fresh Graph factory.
  template <typename... Args>
  // Forwarded into the replay tuple in the member initializer (LLVM 18 checks the body only).
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  explicit OperatorTestHarness(Args&&... args)
      : construct_([arguments = std::make_tuple(std::forward<Args>(args)...)](Graph& graph) {
          return std::apply(
              [&graph](const auto&... values) { return graph.op<Op>("operator", values...); },
              arguments);
        }),
        native_(make_factory(construct_, std::nullopt)) {}

  /// @brief Replace the operator parameters for subsequent attempts.
  /// @param params Ordinary Graph parameter value.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void set_params(Params params) { native_.set_operator_factory(make_factory(construct_, params)); }

  /// @brief Append one owning value to a named input.
  /// @tparam T Exact declared payload type, inferred from the value.
  /// @param name Input name on the configured operator.
  /// @param value One payload, never implicitly interpreted as a sequence.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <PublishedPayload T>
  void push(std::string_view name, T value) {
    std::vector<TestSample<T>> values;
    values.push_back(TestSample<T>{.data = std::move(value)});
    native_.stage_input(name, staged_samples<T>(std::move(values)));
  }

  /// @brief Infer an input's payload type and name from its member pointer.
  /// @tparam T Declared input payload type, selected only by the member pointer.
  /// @param port Member pointer on the configured operator prototype.
  /// @param value One payload, using ordinary implicit C++ conversion to the declared type.
  /// @throws std::invalid_argument if the member selector is null or staging rejects the value.
  /// @throws RuntimeError if configured Graph authoring or payload validation fails.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <PublishedPayload T>
  void push(Input<T> Op::* port, std::type_identity_t<T> value) {
    push(resolve_member_name(port, "input"), std::move(value));
  }

  /// @brief Append one value with an explicit fixture-domain capture coordinate.
  /// @tparam T Exact declared payload type, inferred from the value.
  /// @param name Input name on the configured operator.
  /// @param value One owning payload.
  /// @param coordinate Capture coordinate; does not advance the fixture clock.
  /// @throws std::invalid_argument or RuntimeError if authoring or staging rejects the value.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <PublishedPayload T>
  void push_at(std::string_view name, T value, std::chrono::nanoseconds coordinate) {
    push_samples<T>(name, {sample_at(std::move(value), coordinate)});
  }

  /// @brief Select an input member and append one timestamped value.
  /// @tparam T Declared payload type, selected only by the member pointer.
  /// @param port Member pointer on the configured operator prototype.
  /// @param value One payload, using ordinary implicit C++ conversion to the declared type.
  /// @param coordinate Fixture-domain capture coordinate; never advances its clock.
  /// @throws std::invalid_argument if the selector is null or staging rejects the value.
  /// @throws RuntimeError if configured Graph authoring or payload validation fails.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <PublishedPayload T>
  void push_at(Input<T> Op::* port, std::type_identity_t<T> value,
               std::chrono::nanoseconds coordinate) {
    push_at(resolve_member_name(port, "input"), std::move(value), coordinate);
  }

  /// @brief Append an atomic ordered batch, including an explicit empty producer.
  /// @tparam T Exact declared input payload type.
  /// @param name Input name on the configured operator.
  /// @param values Payloads in delivery order.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <PublishedPayload T>
  void push_many(std::string_view name, std::vector<T> values) {
    native_.stage_input(name, staged_values<T>(std::move(values)));
  }

  /// @brief Deduce and append an ordered batch from an initializer list.
  /// @tparam T Common exact input payload type of the list elements.
  /// @param name Input name on the configured operator.
  /// @param values Copied payloads in delivery order; empty lists require an explicit T.
  /// @throws std::invalid_argument or RuntimeError if authoring or staging rejects the batch.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <PublishedPayload T>
  void push_many(std::string_view name, std::initializer_list<T> values) {
    push_many<T>(name, std::vector<T>{values});
  }

  /// @brief Append timestamped values atomically in the supplied delivery order.
  /// @tparam T Exact input payload type.
  /// @param name Input name on the configured operator.
  /// @param samples Owning envelopes; capture coordinates never sort or advance delivery.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <PublishedPayload T>
  void push_samples(std::string_view name, std::vector<TestSample<T>> samples) {
    native_.stage_input(name, staged_samples<T>(std::move(samples)));
  }

  /// @brief Deduce and append an initializer list of owning sample envelopes.
  /// @tparam T Common exact payload type of the TestSample elements.
  /// @param name Input name on the configured operator.
  /// @param samples Copied envelopes in delivery order; empty lists require an explicit T.
  /// @throws std::invalid_argument or RuntimeError if authoring or staging rejects the batch.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <PublishedPayload T>
  void push_samples(std::string_view name, std::initializer_list<TestSample<T>> samples) {
    push_samples<T>(name, std::vector<TestSample<T>>{samples});
  }

  /// @brief Append one ordered frame, rejecting mixed producers and partial staging.
  /// @param frame Named values with absent or explicitly staged capture coordinates.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void push_frame(const TestFrame& frame) { native_.push_frame(frame); }

  /// @brief Capture bounded independent values from one output.
  /// @tparam T Exact declared output payload type.
  /// @param name Output name on the configured operator.
  /// @param capacity Maximum stored records; overflow fails the run instead of dropping output.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <ValueCapturePayload T>
  void capture(std::string_view name, std::size_t capacity = 1024U) {
    native_.declare_capture(name, value_capture<T>(capacity));
  }

  /// @brief Select an output member and capture bounded independent values.
  /// @tparam T Exact declared output payload type, inferred from its member.
  /// @param port Member pointer on the configured operator prototype.
  /// @param capacity Maximum stored records; overflow fails the run without dropping output.
  /// @throws std::invalid_argument if the selector is null or capacity is invalid.
  /// @throws RuntimeError if configured Graph authoring or output validation fails.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <ValueCapturePayload T>
  void capture(Output<T> Op::* port, std::size_t capacity = 1024U) {
    capture<T>(resolve_member_name(port, "output"), capacity);
  }

  /// @brief Retain native backing after proving whole-run capacity and completion.
  /// @tparam T Exact Tensor or descriptor-schema payload type.
  /// @param name Operator output name.
  /// @param policy Finite capture bound; unproved topologies reject before consuming inputs.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <TensorCapturePayload T>
  void capture(std::string_view name, Retain policy) {
    native_.declare_capture(name, retain_capture<T>(policy.max_records));
  }

  /// @brief Select an output member and retain its bounded native backing.
  /// @tparam T Exact declared Tensor or descriptor-backed schema payload type.
  /// @param port Member pointer on the configured operator prototype.
  /// @param policy Finite capture bound; unproved topologies reject before consuming inputs.
  /// @throws std::invalid_argument if the selector is null or capacity is invalid.
  /// @throws RuntimeError if configured Graph authoring or output validation fails.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <TensorCapturePayload T>
  void capture(Output<T> Op::* port, Retain policy) {
    capture<T>(resolve_member_name(port, "output"), policy);
  }

  /// @brief Capture independent bounded host content for Tensor-bearing outputs.
  /// @tparam T Exact native Tensor or schema payload.
  /// @param name Operator output name.
  /// @param capacity Maximum records; overflow fails without dropping earlier captures.
  /// @param maximum_bytes Maximum bytes for one snapshot.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  template <TensorSnapshotPayload T>
  void capture_snapshot(std::string_view name, std::size_t capacity = 1024U,
                        std::size_t maximum_bytes = 64U * 1024U * 1024U) {
    native_.declare_capture(name, snapshot_capture<T>(capacity, maximum_bytes));
  }

  /// @brief Select an output member and capture independent host snapshots.
  /// @tparam T Exact declared Tensor or descriptor-backed schema payload type.
  /// @param port Member pointer on the configured operator prototype.
  /// @param capacity Maximum records; overflow fails without dropping earlier captures.
  /// @param maximum_bytes Maximum bytes for one snapshot.
  /// @throws std::invalid_argument if the selector is null or either bound is invalid.
  /// @throws RuntimeError if configured Graph authoring or output validation fails.
  /// @throws std::logic_error if the fixture is moved-from or already active.
  template <TensorSnapshotPayload T>
  void capture_snapshot(Output<T> Op::* port, std::size_t capacity = 1024U,
                        std::size_t maximum_bytes = 64U * 1024U * 1024U) {
    capture_snapshot<T>(resolve_member_name(port, "output"), capacity, maximum_bytes);
  }

  /// @brief Replace or clear a native input-history override.
  /// @param name Operator input name.
  /// @param depth Native history depth or nullopt to clear.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_queue_depth(std::string_view name, std::optional<std::uint32_t> depth) {
    native_.input_queue_depth(name, depth);
  }
  /// @brief Copy, replace or clear native source-to-operator flow options.
  /// @param name Operator input name.
  /// @param options Native flow value or nullopt to clear.
  /// @note The matching source or capture must exist when run() builds this configured flow.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if name is empty or an authored bound is zero.
  /// @throws RuntimeError if the named operator port or direction is invalid.
  void input_flow(std::string_view name, std::optional<ConnectionOptions> options) {
    native_.input_flow(name, options);
  }
  /// @brief Copy, replace or clear native operator-to-collector flow options.
  /// @param name Operator output name.
  /// @param options Native flow value or nullopt to clear.
  /// @note The matching source or capture must exist when run() builds this configured flow.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if name is empty or an authored bound is zero.
  /// @throws RuntimeError if the named operator port or direction is invalid.
  void output_flow(std::string_view name, std::optional<ConnectionOptions> options) {
    native_.output_flow(name, options);
  }
  /// @brief Replace or clear operator tensor input contract.
  /// @param name Logical operator port selector.
  /// @param value Native setting, deeply owned before return.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_tensor_spec(std::string_view name, std::optional<TensorInputSpec> value) {
    native_.input_tensor_spec(name, value);
  }
  /// @brief Replace or clear operator tensor output contract.
  /// @param name Logical operator port selector.
  /// @param value Native setting, deeply owned before return.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void output_tensor_spec(std::string_view name, std::optional<TensorOutputSpec> value) {
    native_.output_tensor_spec(name, value);
  }
  /// @brief Replace or clear staged source producer promise.
  /// @param name Logical operator port selector.
  /// @param value Native setting, deeply owned before return.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_source_tensor_spec(std::string_view name, std::optional<TensorOutputSpec> value) {
    native_.input_source_tensor_spec(name, value);
  }
  /// @brief Replace or clear staged source output cuda placement.
  /// @param name Logical operator port selector.
  /// @param value Native setting, deeply owned before return.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void input_source_device(std::string_view name, std::optional<DeviceId> value) {
    native_.input_source_device(name, value);
  }
  /// @brief Replace or clear operator output cuda placement.
  /// @param name Logical operator port selector.
  /// @param value Native setting, deeply owned before return.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  void output_device(std::string_view name, std::optional<DeviceId> value) {
    native_.output_device(name, value);
  }
  /// @brief Replace the complete owning native compile settings.
  /// @param options Settings copied for each fresh attempt.
  /// @throws std::logic_error if the fixture is active or moved-from.
  void set_compile_options(CompileOptions options) {
    native_.set_compile_options(std::move(options));
  }
  /// @brief Replace the late typed native Graph hook; nullptr clears it.
  /// @param hook Cold callback over fresh configured Graph handles and compile settings.
  /// @throws std::logic_error if the fixture is active or moved-from.
  void configure_graph(std::function<void(TestGraphView<Op>&)> hook) {
    if (!hook) {
      native_.configure_graph(nullptr);
      return;
    }
    native_.configure_graph([hook = std::move(hook)](NativeTestGraphView& native_view) {
      TestGraphView<Op> view{native_view};
      hook(view);
    });
  }

  /// @brief Run the finite fixture through the shared native controller.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  [[nodiscard]] TestResult run() { return native_.run(); }
  /// @brief Run with an explicit cooperative timeout and expected terminal state.
  /// @param options Watchdog settings, expected stop and automatic frame release policy.
  /// @note Use \c run() or \c run({}) for defaults; braces may specify only \c timeout.
  /// @throws std::invalid_argument if options are invalid or manual frame release is requested.
  /// @throws RuntimeError if Graph authoring fails before execution.
  /// @throws std::logic_error if the fixture is active or moved-from.
  [[nodiscard]] TestResult run(TestRunOptions options) { return native_.run(options); }
  /// @brief Begin a finite attempt whose owner survives this builder.
  /// @param options Cooperative timeout, expected stop and automatic or manual frame release.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the timeout or frame-release configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects a staged configuration.
  [[nodiscard]] RunningFixture run_async(TestRunOptions options = {}) {
    return native_.run_async(options);
  }
  /// @brief Observe one qualified controlled tick through the shared native controller.
  /// @param control Tick, observation capacity and steady-clock watchdog settings.
  /// @note Controlled-clock execution is separate from ordinary \c run options.
  /// @throws std::logic_error if the fixture is active or moved-from.
  /// @throws std::invalid_argument if the value or fixture configuration is unsupported.
  /// @throws RuntimeError if native Graph authoring rejects the selector or configuration.
  [[nodiscard]] TestResult run_controlled_clock(ControlledClockRun control) {
    return native_.run_controlled_clock(control);
  }

 private:
  template <typename Port>
  [[nodiscard]] std::string resolve_member_name(Port Op::* port, std::string_view direction) const {
    // Native resolution checks moved-from/reentrant access before invoking this callback.
    return native_.resolve_input_name([port, direction](const auto& op) {
      if (port == nullptr) {
        throw std::invalid_argument{"operator test " + std::string(direction) +
                                    " member selector must not be null"};
      }
      const auto handle = holoscan::detail::OperatorTestingAccess::typed_handle<Op>(op);
      const auto access = handle.operator->();
      return std::string{(access.operator->()->*port).name()};
    });
  }

  using Construct = std::function<OperatorHandle<Op>(Graph&)>;
  static NativeOperatorTest::OperatorFactory make_factory(Construct construct,
                                                          std::optional<Params> params) {
    return [construct = std::move(construct), params = std::move(params)](Graph& graph) {
      auto handle = construct(graph);
      if (params) {
        handle.set_params(*params);
      }
      return holoscan::detail::OperatorTestingAccess::erase(handle);
    };
  }
  Construct construct_;
  NativeOperatorTest native_;
};

/// @brief Create a typed finite operator test without global operator registration.
/// @tparam Op Concrete operator type.
/// @tparam Args Replayable constructor argument types.
/// @param args Arguments forwarded into the fixture's owning factory.
/// @return Move-only fixture using ordinary Graph and Runtime semantics.
template <std::derived_from<OperatorBase> Op, typename... Args>
[[nodiscard]] OperatorTestHarness<Op> create_operator_test(Args&&... args) {
  return OperatorTestHarness<Op>(std::forward<Args>(args)...);
}

}  // namespace holoscan::testing
