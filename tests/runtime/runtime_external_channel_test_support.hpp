/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/cuda_test_utils.hpp"
#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_payload_frame.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime_test_graph.hpp"

// External Channel fixtures shared by the import, rebind, export and topology test files.
namespace holoscan {

namespace test::external_channel {

// Collects up to kCapacity samples received by ExternalImportSink for post-run assertion.
// Captures data values and sequence numbers to verify ordering and metadata preservation.
// Representative test: UnmonitoredScalarImportDrainsEverySampleInPublisherOrder.
struct ExternalImportProbe {
  static constexpr std::size_t kCapacity = 4U;

  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    values.fill(0U);
    sequences.fill(0U);
    count.store(0U, std::memory_order_release);
  }

  static void record(const Sample<std::uint32_t>& sample) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < kCapacity) {
      values[index] = sample.data;
      sequences[index] = sample.metadata.sequence;
      count.store(index + 1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(std::size_t expected,
                                           std::chrono::milliseconds timeout = std::chrono::seconds{
                                               5}) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<std::uint32_t, kCapacity> values{};
  static inline std::array<std::uint64_t, kCapacity> sequences{};
};

// Receives uint32_t samples from an imported external topic and records each one via
// ExternalImportProbe for value and ordering assertions after the session ends.
// Representative test: UnmonitoredScalarImportDrainsEverySampleInPublisherOrder.
class ExternalImportSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = std::chrono::seconds{1}})
        .on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    ExternalImportProbe::record(*sample);
    return {};
  }

  Input<std::uint32_t> input;
};

// Source that gates emission on a test-controlled rendezvous: blocks in compute until
// allow_emit() is called, then sends a fixed scalar payload with optional capture-time
// metadata. Representative test: CanonicalScalarGraphExportReachesExternalSubscriber.
class ExternalScalarExportSource final : public Operator<> {
 public:
  static void reset() noexcept {
    const std::lock_guard lock{coordination_mutex};
    compute_started = false;
    emit_allowed = false;
    emit_completed = false;
    compute_count.store(0U, std::memory_order_relaxed);
    emit_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    capture_time.reset();
  }

  [[nodiscard]] static bool wait_for_compute_started(
      std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return compute_started; });
  }

  [[nodiscard]] static bool wait_for_emit_completed(
      std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    std::unique_lock lock{coordination_mutex};
    return coordination_ready.wait_for(lock, timeout, [] { return emit_completed; });
  }

  static void allow_emit() noexcept {
    {
      const std::lock_guard lock{coordination_mutex};
      emit_allowed = true;
    }
    coordination_ready.notify_all();
  }

  static void set_capture_time(TimePoint capture) noexcept {
    const std::lock_guard lock{coordination_mutex};
    capture_time = capture;
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{50}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    std::optional<TimePoint> selected_capture;
    {
      std::unique_lock lock{coordination_mutex};
      compute_started = true;
      coordination_ready.notify_all();
      if (!coordination_ready.wait_for(
              lock, std::chrono::seconds{5}, [] { return emit_allowed; })) {
        emit_error.store(ErrorCode::kTimeout, std::memory_order_relaxed);
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      selected_capture = capture_time;
    }
    SampleMetadata metadata{.trace_id = 17U, .frame_id = 23U};
    if (selected_capture) {
      metadata.capture_timestamp_ns = selected_capture->timestamp_ns;
      metadata.source_clock_id = selected_capture->clock_id;
    }
    auto emitted = output.emit(Sample<std::uint32_t>{
        0xdecafbadU,
        metadata,
    });
    if (!emitted) {
      emit_error.store(emitted.error().code, std::memory_order_relaxed);
    }
    {
      const std::lock_guard lock{coordination_mutex};
      emit_completed = true;
    }
    coordination_ready.notify_all();
    return emitted;
  }

  Output<std::uint32_t> output;
  static inline std::atomic<std::uint32_t> compute_count{};
  static inline std::atomic<ErrorCode> emit_error{ErrorCode::kSuccess};
  static inline std::mutex coordination_mutex;
  static inline std::condition_variable coordination_ready;
  static inline bool compute_started{};
  static inline bool emit_allowed{};
  static inline bool emit_completed{};
  static inline std::optional<TimePoint> capture_time;
};

inline constexpr DLDataType kExternalTensorDtype{kDLUInt, 8U, 1U};

inline constexpr std::array<std::int64_t, 1U> kExternalTensorShape{8};

inline constexpr std::array<TensorAxisBounds, 1U> kExternalTensorAxes{
    TensorAxisBounds{.max_extent = 8U, .max_abs_stride_bytes = 1U}};

using ExternalImagePayload = schema::DescriptorFlatBufferPayloadFor<schema::Image>;

inline constexpr TensorFieldSelector<schema::ImageT> kExternalImageData{4U, 0U};

inline constexpr std::array<std::int64_t, 2U> kExternalImageShape{1, 8};

[[nodiscard]] expected<Tensor, Error> make_external_image_tensor(const std::uint8_t value);

[[nodiscard]] DescriptorTensorTopicContract<schema::ImageT> external_image_contract();

[[nodiscard]] TopicOptions external_image_import_options(std::string_view endpoint_name);

struct ExternalImageImportProbe {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    height.store(0U, std::memory_order_relaxed);
    width.store(0U, std::memory_order_relaxed);
    first_byte.store(0U, std::memory_order_relaxed);
    capture_timestamp_ns.store(0, std::memory_order_relaxed);
    source_clock_id.store(0U, std::memory_order_relaxed);
    trace_id.store(0U, std::memory_order_relaxed);
    frame_id.store(0U, std::memory_order_relaxed);
    flags.store(0U, std::memory_order_relaxed);
    sequence.store(0U, std::memory_order_relaxed);
  }

  static void record(const Sample<schema::ImageT>& sample) noexcept {
    height.store(sample.data.height, std::memory_order_relaxed);
    width.store(sample.data.width, std::memory_order_relaxed);
    const auto* bytes = sample.data.data ? sample.data.data->data_as<std::uint8_t>() : nullptr;
    first_byte.store(bytes == nullptr ? 0U : bytes[0], std::memory_order_relaxed);
    capture_timestamp_ns.store(sample.metadata.capture_timestamp_ns, std::memory_order_relaxed);
    source_clock_id.store(sample.metadata.source_clock_id.value, std::memory_order_relaxed);
    trace_id.store(sample.metadata.trace_id, std::memory_order_relaxed);
    frame_id.store(sample.metadata.frame_id, std::memory_order_relaxed);
    flags.store(static_cast<std::uint32_t>(sample.metadata.flags), std::memory_order_relaxed);
    sequence.store(sample.metadata.sequence, std::memory_order_relaxed);
    count.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(std::uint32_t expected,
                                           std::chrono::milliseconds timeout = std::chrono::seconds{
                                               5}) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_sample(
      std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    return wait_for_count(1U, timeout);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> count{};
  static inline std::atomic<std::uint32_t> height{};
  static inline std::atomic<std::uint32_t> width{};
  static inline std::atomic<std::uint8_t> first_byte{};
  static inline std::atomic<std::int64_t> capture_timestamp_ns{};
  static inline std::atomic<std::uint64_t> source_clock_id{};
  static inline std::atomic<std::uint64_t> trace_id{};
  static inline std::atomic<std::uint64_t> frame_id{};
  static inline std::atomic<std::uint32_t> flags{};
  static inline std::atomic<std::uint64_t> sequence{};
};

class ExternalImageImportSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(kExternalImageData)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                               .dtype = kExternalTensorDtype,
                                               .rank = 2U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    ExternalImageImportProbe::record(*sample);
    return {};
  }

  Input<schema::ImageT> input;
};

// Keeps an output edge live while an externally imported descriptor-schema input is awaiting a
// lifecycle restart.  This is intentionally a forwarding operator rather than a terminal sink:
// closing its output during the first generation's terminal drain would reject the restart.
class ExternalImageForwarder final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(kExternalImageData)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                               .dtype = kExternalTensorDtype,
                                               .rank = 2U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .tensor_field(kExternalImageData)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = kExternalTensorDtype,
                               .rank = 2U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(kExternalImageShape),
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return output.emit(std::move(*sample));
  }

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;
};

// Internal sink that receives Tensor samples from a fanout edge. Verifies that a dropped
// optional external export does not suppress delivery on the internal path.
// Representative test: OptionalTensorDropDoesNotPoisonAcceptedInternalFanout.
class ExternalTensorExportSink final : public Operator<> {
 public:
  static void reset() noexcept {
    received.store(0U, std::memory_order_relaxed);
    last_sequence.store(0U, std::memory_order_relaxed);
  }

  [[nodiscard]] static bool wait_for_sample(
      std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return received.load(std::memory_order_acquire) != 0U; });
  }

  [[nodiscard]] static bool wait_for_count(const std::uint32_t expected,
                                           std::chrono::milliseconds timeout = std::chrono::seconds{
                                               5}) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [expected] { return received.load(std::memory_order_acquire) >= expected; });
  }

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    last_sequence.store(sample->metadata.sequence, std::memory_order_relaxed);
    received.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
    return {};
  }

  Input<Tensor> input;
  static inline std::atomic<std::uint32_t> received{};
  static inline std::atomic<std::uint64_t> last_sequence{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
};

// Builds TopicOptions for an optional host Tensor export with fixed 8-byte single-axis
// bounds. Shared across optional-tensor tests to avoid repeating boilerplate.
[[nodiscard]] TopicOptions optional_tensor_export_options(std::string_view endpoint_name);

// Returns a process-, time-, and counter-stamped endpoint name to prevent resource collisions
// between independently launched test processes sharing the same HoloMQ daemon.
[[nodiscard]] std::string unique_endpoint_name();

// Polls update_connections() until the publisher observes at least one subscriber or timeout.
// Called after run_async() to synchronize before the external test publisher emits.
template <PublishedPayload T>
[[nodiscard]] expected<void, Error> wait_for_subscriber(
    ChannelPublisher<T>& publisher, std::chrono::milliseconds timeout = std::chrono::seconds{5},
    std::chrono::milliseconds stable_for = std::chrono::milliseconds{0}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::optional<std::chrono::steady_clock::time_point> connected_since;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      const auto now = std::chrono::steady_clock::now();
      connected_since = connected_since.value_or(now);
      if (now - *connected_since >= stable_for) {
        return {};
      }
    } else {
      connected_since.reset();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<void, Error> publish_external_host_tensor(
    ChannelPublisher<Tensor>& publisher, std::optional<TimePoint> capture_time,
    const std::byte fill = std::byte{0x2a});

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<void, Error> wait_for_schema_subscriber(
    ChannelTensorSchemaPublisher<Payload>& publisher,
    std::chrono::milliseconds timeout = std::chrono::seconds{5},
    std::chrono::milliseconds stable_for = std::chrono::milliseconds{0}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::optional<std::chrono::steady_clock::time_point> connected_since;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      const auto now = std::chrono::steady_clock::now();
      connected_since = connected_since.value_or(now);
      if (now - *connected_since >= stable_for) {
        return {};
      }
    } else {
      connected_since.reset();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_schema_operation(
    channel_internal::TensorSchemaPayloadSubscriber& subscriber,
    std::chrono::milliseconds timeout = std::chrono::seconds{5});

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_schema_operation(ChannelTensorSchemaSubscriber<Payload>& subscriber,
                                         std::chrono::milliseconds timeout = std::chrono::seconds{
                                             5}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] bool wait_for_session_terminal(
    const RunSession& session, std::chrono::milliseconds timeout = std::chrono::seconds{5});

// Polls has_samples() until a publisher is seen on two consecutive checks, guarding against
// a transient first observation before the Runtime-owned publisher is fully live.
template <typename T>
[[nodiscard]] expected<void, Error> wait_for_publisher(
    ChannelSubscriber<T>& subscriber, std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    // Refresh receiver-side mappings without consuming data, then require two observations while
    // the subscriber remains alive. The advisory Boolean coordinates this bounded test handoff;
    // it is not treated as proof of the later publication result.
    auto refreshed = subscriber.has_samples();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed.error()));
    }
    if (subscriber.has_publishers()) {
      ++consecutive_observations;
      if (consecutive_observations == 2U) {
        return {};
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// Polls has_samples() and drains one sample when ready, retrying until the deadline.
// Returns kTimeout if no sample arrives within the window.
[[nodiscard]] expected<Sample<std::uint32_t>, Error> wait_for_sample(
    ChannelSubscriber<std::uint32_t>& subscriber,
    std::chrono::milliseconds timeout = std::chrono::seconds{5});

}  // namespace test::external_channel
}  // namespace holoscan
