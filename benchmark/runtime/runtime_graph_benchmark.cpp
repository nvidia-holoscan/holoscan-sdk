/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#if defined(_WIN32)
#include <process.h>
#elif defined(__linux__)
#include <unistd.h>
#else
#include <unistd.h>
#endif

#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime_multi_tensor_benchmark_traits.hpp"

#include "../common/benchmark_pod.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_graph_benchmark_common.hpp"

namespace holoscan {

namespace {

using BenchmarkClock = std::chrono::steady_clock;
using benchmarking::benchmark_thread_identity;
using benchmarking::BenchmarkSupportThreadControl;
using benchmarking::emit_optional_cpu;
using benchmarking::emit_support_thread_evidence;
using benchmarking::json_escape;
using benchmarking::kDescriptorBackedImagePayloadBytes;
using benchmarking::kMaximumSamples;
using benchmarking::kMaximumTensorBytes;
using benchmarking::kTensorByteDtype;
using benchmarking::kTensorTransportPayloadBytes;
using benchmarking::make_placement;
using benchmarking::payload_validation_name;
using benchmarking::PayloadValidation;
using benchmarking::SupportThreadBinding;
using benchmarking::SupportThreadRole;
using benchmarking::tensor_descriptor_matches;
using benchmarking::tensor_memory_kind;
using benchmarking::tensor_storage_name;
using benchmarking::TensorStorage;
using benchmarking::validation_timing_name;
using MultiTensorMessage = runtime_benchmark::MultiTensorMessageT;
using MultiTensorFields = runtime_benchmark::MultiTensorMessageTensorFields;

constexpr std::uint32_t kMaximumRelayCount = 64U;
constexpr std::uint32_t kMaximumFanout = 8U;
constexpr std::size_t kMaximumRuntimePhaseTraceRecords = 2'000'000U;
constexpr std::size_t kBenchmarkCacheLineBytes = 64U;
constexpr std::size_t kMaximumImageSinkComputeAllocations = 6U;
constexpr std::size_t kMaximumMultiTensorSinkComputeAllocations = 4U;
// Seven activation/callback checkpoints plus bounded readiness, Channel, and controller wake
// evidence. The capacity is deliberately conservative because parked execution partitions can
// append controller park/return records between otherwise complete activations.
constexpr std::uint64_t kTraceRecordsPerExpectedActivation = 14U;
constexpr std::uint64_t kTensorTraceRecordsPerPublication = 5U;
constexpr std::uint64_t kInlinePayloadTraceRecordsPerPublication = 5U;
constexpr int kSkipped = 77;
constexpr std::uint8_t kSignalPayload = 0xA5U;
constexpr std::uint8_t kImagePayloadByte = 0x3CU;
constexpr std::uint8_t kMultiTensorPayloadByte = 0xA5U;

enum class Workload { kLatency, kNotificationLatency, kSaturated };
enum class BulkWorkProfile { kFullWrite, kDescriptorOnly };
enum class Scenario {
  kSourceOnly,
  kPod,
  kSignal,
  kSampleForward,
  kSchema,
  kTensor,
  kImage,
  kMultiTensor
};
enum class OperatorLaneMode { kSeparate, kShared, kPerSink };

struct Options {
  PayloadValidation validation{PayloadValidation::kFinal};
  Workload workload{Workload::kLatency};
  Scenario scenario{Scenario::kPod};
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  TensorStorage tensor_storage{TensorStorage::kPageableHost};
  BulkWorkProfile bulk_work_profile{BulkWorkProfile::kFullWrite};
  OperatorLaneMode operator_lanes{OperatorLaneMode::kSeparate};
  std::size_t tensor_bytes{1U << 20U};
  int tensor_device{0};
  std::uint64_t samples{100'000U};
  std::uint64_t warmup{1'000U};
  std::uint32_t queue_depth{1U};
  std::uint32_t relay_count{1U};
  std::uint32_t fanout{1U};
  std::uint32_t notification_capacity{1U};
  std::uint32_t publisher_period_us{1'000U};
  std::uint32_t timeout_ms{30'000U};
  int source_cpu{-1};
  int relay_cpu{-1};
  int sink_cpu{-1};
  std::vector<int> sink_cpus;
  int controller_cpu{-1};
  int channel_pump_cpu{-1};
  int notification_driver_cpu{-1};
  bool busy_poll{};
  bool runtime_phase_counters{};
  bool disable_direct_continuation_elision{};
  bool disable_closed_island_composite{};
  bool force_holomq_process_local_inline{};
  bool test_fail_sink_after_relay_pressure{};
  std::string output_path;
};

[[nodiscard]] constexpr bool direct_retained_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSignal ||
         scenario == Scenario::kSampleForward || scenario == Scenario::kSchema ||
         scenario == Scenario::kTensor || scenario == Scenario::kImage ||
         scenario == Scenario::kMultiTensor;
}

[[nodiscard]] constexpr bool fanout_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSchema;
}

[[nodiscard]] constexpr bool lossy_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] constexpr bool lossy_scenario_supported(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSchema ||
         scenario == Scenario::kTensor || scenario == Scenario::kImage;
}

[[nodiscard]] constexpr const char* queue_policy_name(const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return "blocking";
    case QueuePolicy::kDropOldest:
      return "drop_oldest";
    default:
      return "unknown";
  }
}

[[nodiscard]] constexpr bool uses_channel_pump(const Options& options) noexcept {
  return direct_retained_scenario(options.scenario) &&
         (options.force_holomq_process_local_inline || options.fanout > 1U);
}

[[nodiscard]] constexpr const char* process_local_delivery_name(const Options& options) noexcept {
  if (!direct_retained_scenario(options.scenario)) {
    return "not_applicable";
  }
  return uses_channel_pump(options) ? "holomq" : "direct_retained";
}

struct Percentiles {
  std::uint64_t minimum{};
  std::uint64_t p50{};
  std::uint64_t p90{};
  std::uint64_t p99{};
  std::optional<std::uint64_t> p999;
  std::uint64_t maximum{};
};

struct RuntimePhaseReceiverSummary {
  std::size_t receiver_index{};
  bool source{};
  std::optional<std::uint32_t> downstream_index;
  std::uint64_t activations{};
  std::vector<std::uint64_t> previous_cut_to_readiness_poll_return;
  std::vector<std::uint64_t> readiness_poll_return_to_candidate;
  std::vector<std::uint64_t> candidate_to_dispatch_start;
  std::vector<std::uint64_t> compute_dispatch;
  std::vector<std::uint64_t> dispatch_return_to_retirement;
  std::vector<std::uint64_t> retirement_to_cut;
  std::vector<std::uint64_t> dispatch_return_to_cut;
  std::vector<std::uint64_t> callback_pre_return_to_dispatch_return;
  std::vector<std::uint64_t> dispatch_start_to_callback_entry;
};

struct RuntimePhaseTraceSummary {
  bool enabled{};
  bool activation_trace_applicable{};
  bool valid{};
  bool capacity_exhausted{};
  bool storage_invariants_valid{};
  std::size_t capacity{};
  std::size_t attempted_records{};
  std::size_t records{};
  std::uint64_t dropped{};
  std::uint64_t readiness_polls{};
  std::uint64_t unmatched_readiness_polls{};
  std::uint64_t complete_activations{};
  std::uint64_t source_activation_runs{};
  std::uint64_t maximum_consecutive_source_activations{};
  std::uint64_t source_to_source_transitions{};
  std::uint64_t source_to_downstream_transitions{};
  std::uint64_t downstream_to_source_transitions{};
  std::uint64_t downstream_to_downstream_transitions{};
  std::uint64_t complete_tensor_emits{};
  std::vector<std::uint64_t> tensor_emit_total;
  std::vector<std::uint64_t> tensor_emit_validation;
  std::vector<std::uint64_t> tensor_pre_transport;
  std::vector<std::uint64_t> tensor_transport_publication;
  std::vector<std::uint64_t> tensor_post_transport;
  std::uint64_t complete_inline_payload_emits{};
  std::vector<std::uint64_t> inline_payload_emit_total;
  std::vector<std::uint64_t> inline_payload_encoding;
  std::vector<std::uint64_t> inline_payload_pre_transport;
  std::vector<std::uint64_t> inline_payload_transport_publication;
  std::vector<std::uint64_t> inline_payload_post_transport;
  std::vector<RuntimePhaseReceiverSummary> receivers;
};

[[nodiscard]] const char* workload_name(const Workload workload) noexcept {
  switch (workload) {
    case Workload::kLatency:
      return "latency";
    case Workload::kNotificationLatency:
      return "notification_latency";
    case Workload::kSaturated:
      return "saturated";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool is_latency_workload(const Workload workload) noexcept {
  return workload != Workload::kSaturated;
}

[[nodiscard]] constexpr bool is_notification_workload(const Workload workload) noexcept {
  return workload != Workload::kLatency;
}

[[nodiscard]] const char* scenario_name(const Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::kSourceOnly:
      return "source_only";
    case Scenario::kPod:
      return "pod";
    case Scenario::kSignal:
      return "signal";
    case Scenario::kSampleForward:
      return "sample_forward";
    case Scenario::kSchema:
      return "schema";
    case Scenario::kTensor:
      return "tensor";
    case Scenario::kImage:
      return "image";
    case Scenario::kMultiTensor:
      return "multi_tensor";
  }
  return "unknown";
}

[[nodiscard]] std::string benchmark_name(const Options& options) {
  std::string result;
  switch (options.scenario) {
    case Scenario::kSourceOnly:
      result = "runtime_graph_source_activation";
      break;
    case Scenario::kPod:
      result = "runtime_graph_pod_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kSignal:
      result = "runtime_graph_signal_1to1";
      break;
    case Scenario::kSampleForward:
      result = "runtime_graph_pod_chain_1to1_relays_" + std::to_string(options.relay_count);
      break;
    case Scenario::kSchema:
      result = "runtime_graph_schema_time_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kTensor:
      result = "runtime_graph_tensor_1to1";
      break;
    case Scenario::kImage:
      result = "runtime_graph_image_1to1";
      break;
    case Scenario::kMultiTensor:
      result = "runtime_graph_multi_tensor_1to1";
      break;
  }
  if (result.empty()) {
    result = "runtime_graph_unknown";
  }
  if (options.force_holomq_process_local_inline) {
    result += "_holomq_fallback";
  }
  if (options.queue_policy != QueuePolicy::kBlocking) {
    result += "_";
    result += queue_policy_name(options.queue_policy);
  }
  if (options.workload == Workload::kSaturated) {
    result += "_saturated";
  } else if (options.workload == Workload::kNotificationLatency) {
    result += "_notification_latency";
  }
  return result;
}

[[nodiscard]] const char* benchmark_answer(const Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::kSourceOnly:
      return "notification-to-source callback cost without Channel or downstream input polling";
    case Scenario::kPod:
      return "minimal typed process-local publish/receive activation cost end to end";
    case Scenario::kSignal:
      return "source-to-sink Runtime floor with only one accounted causal trigger byte";
    case Scenario::kSampleForward:
      return "metadata-preserving Sample<T> forwarding cost across a linear Operator chain";
    case Scenario::kSchema:
      return "production generated-schema encode, verify, materialize, and activation cost";
    case Scenario::kTensor:
      return "production direct Tensor loan, publish, readiness, receive, and retirement cost";
    case Scenario::kImage:
      return "production Image metadata and embedded Tensor retention cost";
    case Scenario::kMultiTensor:
      return "generated-schema metadata and two preallocated Tensor field retention cost";
  }
  return "unknown benchmark claim";
}

[[nodiscard]] std::size_t payload_size(const Options& options) noexcept {
  if (options.scenario == Scenario::kSourceOnly) {
    return 0U;
  }
  if (options.scenario == Scenario::kSignal) {
    return sizeof(std::uint8_t);
  }
  if (options.scenario == Scenario::kTensor || options.scenario == Scenario::kImage) {
    return options.tensor_bytes;
  }
  if (options.scenario == Scenario::kMultiTensor) {
    return 2U * options.tensor_bytes;
  }
  if (options.scenario == Scenario::kSchema) {
    return sizeof(schema::TimeT);
  }
  return sizeof(BenchmarkPOD64);
}

[[nodiscard]] std::optional<std::size_t> inline_transport_bytes(const Options& options) noexcept {
  if (!uses_channel_pump(options)) {
    return 0U;
  }
  if (options.scenario == Scenario::kTensor) {
    return kTensorTransportPayloadBytes;
  }
  if (options.scenario == Scenario::kImage) {
    return kDescriptorBackedImagePayloadBytes;
  }
  if (options.scenario == Scenario::kMultiTensor) {
    return std::nullopt;
  }
  if (options.scenario == Scenario::kSchema) {
    // Tensor and Image use the private channel_internal::TensorTransportPayload and
    // schema::DescriptorFlatBufferPayloadFor<schema::Image> types; the Channel benchmark
    // statically verifies the mirrored fixed sizes above. Generated schemas instead submit a
    // variable encoded prefix of RuntimePortTypeOps::wire_capacity, and Runtime does not expose
    // that exact count in this result, so preserve it as unknown rather than reporting the bound.
    return std::nullopt;
  }
  return payload_size(options);
}

[[nodiscard]] std::size_t external_buffer_bytes(const Options& options) noexcept {
  if (options.scenario == Scenario::kMultiTensor) {
    return 2U * options.tensor_bytes;
  }
  return options.scenario == Scenario::kTensor || options.scenario == Scenario::kImage
             ? options.tensor_bytes
             : 0U;
}

[[nodiscard]] const char* payload_write_profile(const Options& options) noexcept {
  if (options.scenario == Scenario::kSourceOnly) {
    return "none";
  }
  if (options.scenario == Scenario::kImage) {
    return "metadata_per_sample_reused_external_buffer";
  }
  if (options.scenario == Scenario::kMultiTensor) {
    return options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly
               ? "metadata_per_sample_preinitialized_field_pools"
               : "full_payload_per_sample";
  }
  if (options.scenario == Scenario::kTensor &&
      options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly) {
    return "preinitialized_pool_backing";
  }
  return "full_payload_per_sample";
}

[[nodiscard]] const char* bulk_work_profile(const Options& options) noexcept {
  if (options.scenario == Scenario::kTensor) {
    return options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly ? "descriptor_only"
                                                                         : "full_write";
  }
  if (options.scenario == Scenario::kImage) {
    return "descriptor_only";
  }
  if (options.scenario == Scenario::kMultiTensor) {
    return options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly ? "descriptor_only"
                                                                         : "full_write";
  }
  return "not_applicable";
}

[[nodiscard]] std::uint32_t edge_count(const Options& options) noexcept {
  if (options.scenario == Scenario::kSourceOnly) {
    return 0U;
  }
  if (options.scenario == Scenario::kSampleForward) {
    return options.relay_count + 1U;
  }
  return fanout_scenario(options.scenario) ? options.fanout : 1U;
}

[[nodiscard]] std::int64_t now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             BenchmarkClock::now().time_since_epoch())
      .count();
}

[[nodiscard]] BenchmarkPOD64 make_payload(const std::uint64_t sequence) noexcept {
  const std::uint32_t flags = static_cast<std::uint32_t>(sequence) ^ 0xA5A5'5A5AU;
  BenchmarkPOD64 payload{
      .sequence = sequence,
      .auxiliary0 = flags,
      .auxiliary1 = sequence * 3U,
  };
  payload.checksum = benchmark_pod_checksum(payload);
  return payload;
}

template <typename Payload>
[[nodiscard]] Payload make_benchmark_payload(const std::uint64_t sequence) noexcept {
  if constexpr (std::is_same_v<Payload, BenchmarkPOD64>) {
    return make_payload(sequence);
  } else if constexpr (std::is_same_v<Payload, schema::TimeT>) {
    return schema::TimeT{
        .time_ns = static_cast<std::int64_t>(sequence),
        .acqtime_ns = static_cast<std::int64_t>(sequence ^ 0x5A5A'5A5AU),
    };
  } else {
    static_assert(std::is_same_v<Payload, std::uint8_t>);
    return kSignalPayload;
  }
}

[[nodiscard]] std::uint64_t process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(_getpid());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] bool payload_valid(const BenchmarkPOD64& payload) noexcept {
  return benchmark_pod_valid(payload);
}

[[nodiscard]] bool payload_valid(const std::uint8_t payload) noexcept {
  return payload == kSignalPayload;
}

[[nodiscard]] bool payload_valid(const schema::TimeT& payload) noexcept {
  const std::uint64_t sequence = static_cast<std::uint64_t>(payload.time_ns);
  return payload.time_ns >= 0 &&
         payload.acqtime_ns == static_cast<std::int64_t>(sequence ^ 0x5A5A'5A5AU);
}

// Field order groups state by benchmark phase and cache-line ownership. Reordering for minimum
// object padding would obscure those synchronization boundaries.
class BenchmarkRunState {  // NOLINT(clang-analyzer-optin.performance.Padding)
 public:
  explicit BenchmarkRunState(Options options)
      : options_(std::move(options)),
        source_timestamps_(is_latency_workload(options_.workload) ? options_.samples : 0U),
        source_pre_emit_timestamps_(is_latency_workload(options_.workload) ? options_.samples : 0U),
        source_post_emit_timestamps_(is_latency_workload(options_.workload) ? options_.samples
                                                                            : 0U),
        source_pre_return_timestamps_(is_latency_workload(options_.workload) ? options_.samples
                                                                             : 0U),
        tensor_loan_acquired_timestamps_(options_.scenario == Scenario::kTensor &&
                                                 is_latency_workload(options_.workload)
                                             ? options_.samples
                                             : 0U),
        tensor_write_committed_timestamps_(options_.scenario == Scenario::kTensor &&
                                                   is_latency_workload(options_.workload)
                                               ? options_.samples
                                               : 0U),
        tensor_receive_start_timestamps_(options_.scenario == Scenario::kTensor &&
                                                 is_latency_workload(options_.workload)
                                             ? options_.samples
                                             : 0U),
        sink_timestamps_(is_latency_workload(options_.workload) ? options_.samples : 0U),
        sink_timestamps_by_index_(
            is_latency_workload(options_.workload) ? options_.fanout : 0U,
            std::vector<std::int64_t>(is_latency_workload(options_.workload) ? options_.samples
                                                                             : 0U)),
        sink_callback_timestamps_by_index_(
            is_latency_workload(options_.workload) ? options_.fanout : 0U,
            std::vector<std::int64_t>(is_latency_workload(options_.workload) ? options_.samples
                                                                             : 0U)),
        notification_post_timestamps_(
            options_.workload == Workload::kNotificationLatency ? options_.samples : 0U),
        source_callback_timestamps_(
            options_.workload == Workload::kNotificationLatency ? options_.samples : 0U),
        sink_deliveries_(options_.fanout, 0U),
        sink_final_sequences_(options_.fanout, 0U),
        sequence_sink_delivery_count_(options_.fanout > 1U ? final_sequence() + 1U : 0U),
        sequence_sink_deliveries_(
            sequence_sink_delivery_count_ == 0U
                ? nullptr
                : std::make_unique<
                      std::atomic<std::uint32_t>[]>(  // NOLINT(cppcoreguidelines-avoid-c-arrays)
                      sequence_sink_delivery_count_)),
        tensor_source_pointers_(
            options_.scenario == Scenario::kTensor
                ? std::make_unique<
                      std::atomic<std::uintptr_t>[]>(  // NOLINT(cppcoreguidelines-avoid-c-arrays)
                      final_sequence() + 1U)
                : nullptr),
        tensor_validation_buffer_(options_.scenario == Scenario::kTensor ||
                                          options_.scenario == Scenario::kImage ||
                                          options_.scenario == Scenario::kMultiTensor
                                      ? options_.tensor_bytes
                                      : 0U),
        image_backing_(options_.scenario == Scenario::kImage ? options_.tensor_bytes : 0U,
                       kImagePayloadByte) {
    for (std::size_t index = 0U; index < sequence_sink_delivery_count_; ++index) {
      sequence_sink_deliveries_[index].store(0U, std::memory_order_relaxed);
    }
    if (options_.scenario == Scenario::kMultiTensor) {
      for (auto& pointers : multi_tensor_source_pointers_) {
        pointers = std::make_unique<
            std::atomic<std::uintptr_t>[]>(  // NOLINT(cppcoreguidelines-avoid-c-arrays)
            final_sequence() + 1U);
      }
    }
  }

  BenchmarkRunState(const BenchmarkRunState&) = delete;
  BenchmarkRunState& operator=(const BenchmarkRunState&) = delete;

  [[nodiscard]] expected<void, Error> initialize_image_tensor() {
    if (options_.scenario != Scenario::kImage) {
      return {};
    }
    const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(options_.tensor_bytes)};
    auto tensor = make_tensor(shape,
                              kTensorByteDtype,
                              TensorByteStridesView{std::nullopt},
                              MemoryStorageType::kSystem,
                              image_backing_.data(),
                              static_cast<MemoryBuffer::release_function_t>(nullptr));
    if (!tensor) {
      return make_unexpected(std::move(tensor.error()));
    }
    try {
      image_tensor_ = std::make_shared<Tensor>(std::move(*tensor));
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return {};
  }

  [[nodiscard]] const std::shared_ptr<Tensor>& image_tensor() const noexcept {
    return image_tensor_;
  }

  void set_descriptor_warmup_slots(const std::size_t slots) noexcept {
    descriptor_warmup_slots_ = slots;
  }

  [[nodiscard]] std::size_t descriptor_warmup_slots() const noexcept {
    return descriptor_warmup_slots_;
  }

  [[nodiscard]] const Options& options() const noexcept { return options_; }

  [[nodiscard]] bool saturated() const noexcept {
    return options_.workload == Workload::kSaturated;
  }

  [[nodiscard]] bool notification_latency() const noexcept {
    return options_.workload == Workload::kNotificationLatency;
  }

  [[nodiscard]] bool notification_driven() const noexcept {
    return is_notification_workload(options_.workload);
  }

  [[nodiscard]] std::uint64_t allocation_probe_sequence() const noexcept {
    return options_.warmup + 1U;
  }

  [[nodiscard]] std::uint64_t first_measured_sequence() const noexcept {
    return allocation_probe_sequence() + 1U;
  }

  [[nodiscard]] std::uint64_t final_sequence() const noexcept {
    return first_measured_sequence() + options_.samples - 1U;
  }

  [[nodiscard]] std::chrono::nanoseconds source_period() const noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::microseconds{options_.publisher_period_us});
  }

  [[nodiscard]] bool measured(const std::uint64_t sequence) const noexcept {
    return sequence >= first_measured_sequence() && sequence <= final_sequence();
  }

  [[nodiscard]] bool source_timestamp_required(const std::uint64_t sequence) const noexcept {
    return measured(sequence) &&
           (is_latency_workload(options_.workload) || sequence == first_measured_sequence());
  }

  [[nodiscard]] bool source_only_timestamp_required(const std::uint64_t sequence) const noexcept {
    return source_timestamp_required(sequence) ||
           (options_.workload == Workload::kSaturated && sequence == final_sequence());
  }

  [[nodiscard]] bool source_post_emit_timestamp_required(
      const std::uint64_t sequence) const noexcept {
    return is_latency_workload(options_.workload) && measured(sequence) &&
           options_.scenario != Scenario::kSourceOnly;
  }

  [[nodiscard]] bool sink_timestamp_required(const std::uint64_t sequence) const noexcept {
    return measured(sequence) &&
           (is_latency_workload(options_.workload) || sequence == final_sequence());
  }

  [[nodiscard]] std::size_t measured_index(const std::uint64_t sequence) const noexcept {
    return static_cast<std::size_t>(sequence - first_measured_sequence());
  }

  void record_source_callback(const std::uint64_t sequence,
                              const std::int64_t timestamp = 0) noexcept {
    ++source_callbacks_;
    if (measured(sequence)) {
      ++measured_source_callbacks_;
      if (notification_latency()) {
        source_callback_timestamps_[measured_index(sequence)] = timestamp;
      }
    }
  }

  void record_source_compute_entry() noexcept {
    record_callback_thread(source_callback_tid_,
                           source_callback_identity_is_native_,
                           source_callback_native_tid_syscall_unavailable_,
                           source_callback_multiple_threads_);
    source_compute_entries_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_relay_callback() noexcept {
    record_callback_thread(relay_callback_tid_,
                           relay_callback_identity_is_native_,
                           relay_callback_native_tid_syscall_unavailable_,
                           relay_callback_multiple_threads_);
    ++relay_callbacks_;
  }

  void record_sink_callback() noexcept {
    record_callback_thread(sink_callback_tid_,
                           sink_callback_identity_is_native_,
                           sink_callback_native_tid_syscall_unavailable_,
                           sink_callback_multiple_threads_);
    sink_callbacks_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_backpressure() noexcept {
    backpressure_count_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_source_backpressure() noexcept {
    source_backpressure_count_.fetch_add(1U, std::memory_order_relaxed);
    record_backpressure();
  }

  void record_source_allocation_probe(const std::size_t count) noexcept {
    source_probe_allocations_ = count;
  }

  void record_sink_allocation_probe(const std::size_t count) noexcept {
    sink_probe_allocations_.fetch_add(count, std::memory_order_relaxed);
  }

  void record_relay_allocation_probe(const std::size_t count) noexcept {
    relay_probe_allocations_ += count;
  }

  void record_relay_retry() noexcept {
    relay_retry_observed_.store(true, std::memory_order_release);
  }

  [[nodiscard]] bool relay_retry_observed() const noexcept {
    return relay_retry_observed_.load(std::memory_order_acquire);
  }

  void prepare_source_timestamp(const std::uint64_t sequence,
                                const std::int64_t timestamp) noexcept {
    if (!measured(sequence)) {
      return;
    }
    if (sequence == first_measured_sequence() && !notification_latency()) {
      measured_start_ns_ = timestamp;
    }
    if (is_latency_workload(options_.workload)) {
      source_timestamps_[measured_index(sequence)] = timestamp;
    }
  }

  void prepare_source_pre_emit_timestamp(const std::uint64_t sequence,
                                         const std::int64_t timestamp) noexcept {
    if (is_latency_workload(options_.workload) && measured(sequence)) {
      source_pre_emit_timestamps_[measured_index(sequence)] = timestamp;
    }
  }

  void prepare_tensor_source_phase_timestamps(
      const std::uint64_t sequence, const std::int64_t loan_acquired_timestamp,
      const std::int64_t write_committed_timestamp) noexcept {
    if (options_.scenario == Scenario::kTensor && is_latency_workload(options_.workload) &&
        measured(sequence)) {
      const std::size_t index = measured_index(sequence);
      tensor_loan_acquired_timestamps_[index] = loan_acquired_timestamp;
      tensor_write_committed_timestamps_[index] = write_committed_timestamp;
    }
  }

  void prepare_tensor_receive_start_timestamp(const std::uint64_t sequence,
                                              const std::int64_t timestamp) noexcept {
    if (options_.scenario == Scenario::kTensor && is_latency_workload(options_.workload) &&
        measured(sequence)) {
      tensor_receive_start_timestamps_[measured_index(sequence)] = timestamp;
    }
  }

  void prepare_notification_post_timestamp(const std::uint64_t sequence,
                                           const std::int64_t timestamp) noexcept {
    if (!notification_latency() || !measured(sequence)) {
      return;
    }
    notification_post_timestamps_[measured_index(sequence)] = timestamp;
    if (sequence == first_measured_sequence()) {
      measured_start_ns_ = timestamp;
    }
  }

  void record_publication(const std::uint64_t sequence,
                          const std::int64_t post_emit_timestamp) noexcept {
    if (source_post_emit_timestamp_required(sequence)) {
      source_post_emit_timestamps_[measured_index(sequence)] = post_emit_timestamp;
    }
    publications_.fetch_add(1U, std::memory_order_relaxed);
    if (sequence == final_sequence()) {
      source_finished_.store(true, std::memory_order_release);
    }
  }

  void record_source_pre_return(const std::uint64_t sequence,
                                const std::int64_t timestamp) noexcept {
    if (source_post_emit_timestamp_required(sequence)) {
      source_pre_return_timestamps_[measured_index(sequence)] = timestamp;
    }
  }

  void record_sink_callback_entry(const std::uint32_t sink_index, const std::uint64_t sequence,
                                  const std::int64_t timestamp) noexcept {
    if (is_latency_workload(options_.workload) && measured(sequence) &&
        sink_index < sink_callback_timestamps_by_index_.size()) {
      sink_callback_timestamps_by_index_[sink_index][measured_index(sequence)] = timestamp;
    }
  }

  void record_relay_publication() noexcept { ++relay_publications_; }

  void record_delivery(const std::uint32_t sink_index, const std::uint64_t sequence,
                       const std::int64_t timestamp) {
    if (sink_index >= sink_deliveries_.size() ||
        (options_.fanout > 1U && sequence >= sequence_sink_delivery_count_)) {
      fail(ErrorCode::kProtocolError);
      return;
    }
    ++sink_deliveries_[sink_index];
    sink_final_sequences_[sink_index] = sequence;
    deliveries_.fetch_add(1U, std::memory_order_release);
    if (measured(sequence)) {
      measured_deliveries_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (is_latency_workload(options_.workload) && measured(sequence)) {
      sink_timestamps_by_index_[sink_index][measured_index(sequence)] = timestamp;
    }
    const std::uint32_t completed_sinks =
        options_.fanout == 1U
            ? 1U
            : sequence_sink_deliveries_[sequence].fetch_add(1U, std::memory_order_acq_rel) + 1U;
    if (completed_sinks > options_.fanout) {
      fail(ErrorCode::kProtocolError);
      return;
    }
    if (completed_sinks != options_.fanout) {
      return;
    }
    if (is_latency_workload(options_.workload) && measured(sequence)) {
      // The final sink arrival defines the fanout completion and latency boundary.
      sink_timestamps_[measured_index(sequence)] = timestamp;
    }
    record_activation_completion(sequence, timestamp);
  }

  void record_source_only_completion(const std::uint64_t sequence, const std::int64_t timestamp) {
    if (sequence == final_sequence()) {
      source_finished_.store(true, std::memory_order_release);
    }
    record_activation_completion(sequence, timestamp);
  }

  void record_activation_completion(const std::uint64_t sequence, const std::int64_t timestamp) {
    activation_completions_.fetch_add(1U, std::memory_order_release);
    if (notification_latency()) {
      completion_cv_.notify_all();
    }
    if (sequence != final_sequence()) {
      return;
    }
    measured_end_ns_ = timestamp;
    {
      const std::lock_guard lock(completion_mutex_);
      completed_ = true;
    }
    completion_cv_.notify_all();
  }

  void record_validation_error() noexcept {
    validation_errors_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_tensor_source_pointer(const std::uint64_t sequence, const void* pointer) noexcept {
    tensor_source_pointers_[sequence].store(reinterpret_cast<std::uintptr_t>(pointer),
                                            std::memory_order_release);
  }

  [[nodiscard]] bool tensor_pointer_matches(const std::uint64_t sequence,
                                            const void* pointer) const noexcept {
    return tensor_source_pointers_[sequence].load(std::memory_order_acquire) ==
           reinterpret_cast<std::uintptr_t>(pointer);
  }

  void record_multi_tensor_source_pointer(const std::size_t field, const std::uint64_t sequence,
                                          const void* pointer) noexcept {
    multi_tensor_source_pointers_[field][sequence].store(reinterpret_cast<std::uintptr_t>(pointer),
                                                         std::memory_order_release);
  }

  [[nodiscard]] bool multi_tensor_pointer_matches(const std::size_t field,
                                                  const std::uint64_t sequence,
                                                  const void* pointer) const noexcept {
    return multi_tensor_source_pointers_[field][sequence].load(std::memory_order_acquire) ==
           reinterpret_cast<std::uintptr_t>(pointer);
  }

  void record_descriptor_backing_initialization(const std::uint64_t sequence) noexcept {
    descriptor_backing_initializations_.fetch_add(1U, std::memory_order_relaxed);
    if (measured(sequence)) {
      descriptor_backing_initializations_during_measurement_.fetch_add(1U,
                                                                       std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::uint64_t descriptor_backing_initializations() const noexcept {
    return descriptor_backing_initializations_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t descriptor_backing_initializations_during_measurement()
      const noexcept {
    return descriptor_backing_initializations_during_measurement_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool validate_tensor_payload(const Tensor& tensor, const std::uint64_t sequence,
                                             cudaStream_t stream = nullptr) noexcept {
    return validate_tensor_bytes(tensor, static_cast<std::uint8_t>(sequence), stream);
  }

  [[nodiscard]] bool validate_tensor_bytes(const Tensor& tensor, const std::uint8_t expected,
                                           cudaStream_t stream = nullptr) noexcept {
    const auto copied = tensor.copy_to_host(
        tensor_validation_buffer_.data(), tensor_validation_buffer_.size(), stream);
    if (!copied) {
      fail(copied.error().code);
      return false;
    }
    return std::all_of(tensor_validation_buffer_.begin(),
                       tensor_validation_buffer_.end(),
                       [expected](const std::uint8_t value) { return value == expected; });
  }

  void retain_final_tensor(Tensor tensor, const std::uint8_t expected) {
    const std::lock_guard lock(completion_mutex_);
    final_tensor_.emplace(std::move(tensor));
    final_tensor_expected_byte_ = expected;
  }

  void retain_final_multi_tensor(Tensor first, Tensor second, const std::uint8_t expected) {
    const std::lock_guard lock(completion_mutex_);
    final_tensor_.emplace(std::move(first));
    final_second_tensor_.emplace(std::move(second));
    final_tensor_expected_byte_ = expected;
  }

  void record_tensor_consumer_stream(cudaStream_t stream) noexcept {
    tensor_consumer_stream_.store(reinterpret_cast<std::uintptr_t>(stream),
                                  std::memory_order_release);
  }

  [[nodiscard]] bool validate_retained_tensor() noexcept {
    std::optional<Tensor> retained;
    std::uint8_t expected{};
    {
      const std::lock_guard lock(completion_mutex_);
      if (!final_tensor_.has_value()) {
        return false;
      }
      retained.emplace(std::move(*final_tensor_));
      final_tensor_.reset();
      expected = final_tensor_expected_byte_;
    }
    const auto stream =
        reinterpret_cast<cudaStream_t>(tensor_consumer_stream_.load(std::memory_order_acquire));
    return validate_tensor_bytes(*retained, expected, stream);
  }

  [[nodiscard]] bool validate_retained_multi_tensor() noexcept {
    std::optional<Tensor> first;
    std::optional<Tensor> second;
    std::uint8_t expected{};
    {
      const std::lock_guard lock(completion_mutex_);
      if (!final_tensor_.has_value() || !final_second_tensor_.has_value()) {
        return false;
      }
      first.emplace(std::move(*final_tensor_));
      second.emplace(std::move(*final_second_tensor_));
      final_tensor_.reset();
      final_second_tensor_.reset();
      expected = final_tensor_expected_byte_;
    }
    return validate_tensor_bytes(*first, expected) && validate_tensor_bytes(*second, expected);
  }

  void fail(const ErrorCode code) noexcept {
    ErrorCode expected = ErrorCode::kSuccess;
    static_cast<void>(first_error_.compare_exchange_strong(
        expected, code, std::memory_order_relaxed, std::memory_order_relaxed));
    {
      const std::lock_guard lock(completion_mutex_);
      failed_ = true;
    }
    completion_cv_.notify_all();
  }

  void publish_notification_sender(NotificationSender sender) {
    {
      const std::lock_guard lock(completion_mutex_);
      notification_sender_.emplace(std::move(sender));
    }
    completion_cv_.notify_all();
  }

  [[nodiscard]] std::optional<NotificationSender> take_notification_sender() {
    std::unique_lock lock(completion_mutex_);
    const bool ready =
        completion_cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
          return notification_sender_.has_value() || failed_;
        });
    if (!ready || !notification_sender_.has_value()) {
      return std::nullopt;
    }
    return std::exchange(notification_sender_, std::nullopt);
  }

  void drive_notifications(const NotificationSender& sender) noexcept {
    while (!source_finished_.load(std::memory_order_acquire) &&
           first_error_.load(std::memory_order_relaxed) == ErrorCode::kSuccess) {
      const std::uint64_t accepted = notifications_accepted_.load(std::memory_order_acquire);
      const std::uint64_t completed = lossy_queue_policy(options_.queue_policy)
                                          ? source_compute_entries_.load(std::memory_order_acquire)
                                          : activation_completions_.load(std::memory_order_acquire);
      if (accepted >= completed && accepted - completed >= options_.notification_capacity) {
        std::this_thread::yield();
        continue;
      }
      notification_attempts_.fetch_add(1U, std::memory_order_relaxed);
      auto posted = sender.post();
      if (posted) {
        notifications_accepted_.fetch_add(1U, std::memory_order_relaxed);
        continue;
      }
      if (posted.error().code == ErrorCode::kBackpressured) {
        notification_backpressure_.fetch_add(1U, std::memory_order_relaxed);
        std::this_thread::yield();
        continue;
      }
      if (posted.error().code == ErrorCode::kNotReady) {
        notification_not_ready_.fetch_add(1U, std::memory_order_relaxed);
        std::this_thread::yield();
        continue;
      }
      if (posted.error().code == ErrorCode::kClosed &&
          source_finished_.load(std::memory_order_acquire)) {
        return;
      }
      fail(posted.error().code);
      return;
    }
  }

  void drive_notification_latency(const NotificationSender& sender) noexcept {
    for (std::uint64_t sequence = 1U;
         sequence <= final_sequence() && first_error() == ErrorCode::kSuccess;
         ++sequence) {
      if (sequence > 1U && !wait_for_activation_completion(sequence - 1U)) {
        fail(ErrorCode::kTimeout);
        return;
      }
      std::this_thread::sleep_for(source_period());
      while (first_error() == ErrorCode::kSuccess) {
        notification_attempts_.fetch_add(1U, std::memory_order_relaxed);
        prepare_notification_post_timestamp(sequence, now_ns());
        auto posted = sender.post();
        if (posted) {
          notifications_accepted_.fetch_add(1U, std::memory_order_relaxed);
          break;
        }
        if (posted.error().code == ErrorCode::kBackpressured) {
          notification_backpressure_.fetch_add(1U, std::memory_order_relaxed);
          std::this_thread::yield();
          continue;
        }
        if (posted.error().code == ErrorCode::kNotReady) {
          notification_not_ready_.fetch_add(1U, std::memory_order_relaxed);
          std::this_thread::yield();
          continue;
        }
        fail(posted.error().code);
        return;
      }
    }
  }

  [[nodiscard]] bool wait() {
    std::unique_lock lock(completion_mutex_);
    return completion_cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
      return completed_ || failed_;
    });
  }

  [[nodiscard]] bool completed() const noexcept {
    const std::lock_guard lock(completion_mutex_);
    return completed_;
  }

  [[nodiscard]] ErrorCode first_error() const noexcept {
    return first_error_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t source_callbacks() const noexcept { return source_callbacks_; }

  [[nodiscard]] std::uint64_t source_compute_entries() const noexcept {
    return source_compute_entries_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t relay_callbacks() const noexcept { return relay_callbacks_; }

  [[nodiscard]] std::uint64_t sink_callbacks() const noexcept {
    return sink_callbacks_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t source_callback_tid() const noexcept {
    return source_callback_tid_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t relay_callback_tid() const noexcept {
    return relay_callback_tid_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t sink_callback_tid() const noexcept {
    return sink_callback_tid_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool source_callback_identity_is_native() const noexcept {
    return source_callback_identity_is_native_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool relay_callback_identity_is_native() const noexcept {
    return relay_callback_identity_is_native_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool sink_callback_identity_is_native() const noexcept {
    return sink_callback_identity_is_native_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool source_callback_native_tid_syscall_unavailable() const noexcept {
    return source_callback_native_tid_syscall_unavailable_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool relay_callback_native_tid_syscall_unavailable() const noexcept {
    return relay_callback_native_tid_syscall_unavailable_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool sink_callback_native_tid_syscall_unavailable() const noexcept {
    return sink_callback_native_tid_syscall_unavailable_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool source_callback_multiple_threads() const noexcept {
    return source_callback_multiple_threads_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool relay_callback_multiple_threads() const noexcept {
    return relay_callback_multiple_threads_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool sink_callback_multiple_threads() const noexcept {
    return sink_callback_multiple_threads_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t publications() const noexcept {
    return publications_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t measured_source_callbacks() const noexcept {
    return measured_source_callbacks_;
  }

  [[nodiscard]] std::uint64_t relay_publications() const noexcept { return relay_publications_; }

  [[nodiscard]] std::uint64_t deliveries() const noexcept {
    return deliveries_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t measured_deliveries() const noexcept {
    return measured_deliveries_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t consumer_visible_replacements() const noexcept {
    if (!lossy_queue_policy(options_.queue_policy) || options_.fanout != 1U) {
      return 0U;
    }
    const std::uint64_t accepted = publications();
    const std::uint64_t observed = deliveries();
    return accepted >= observed ? accepted - observed : 0U;
  }

  [[nodiscard]] std::uint64_t sink_deliveries(const std::uint32_t sink_index) const noexcept {
    return sink_deliveries_[sink_index];
  }

  [[nodiscard]] std::uint64_t sink_final_sequence(const std::uint32_t sink_index) const noexcept {
    return sink_final_sequences_[sink_index];
  }

  [[nodiscard]] bool sink_evidence_valid(const std::uint64_t expected_sequence,
                                         const std::uint64_t expected_deliveries) const noexcept {
    return std::all_of(
               sink_deliveries_.begin(),
               sink_deliveries_.end(),
               [expected_deliveries](const auto count) { return count == expected_deliveries; }) &&
           std::all_of(
               sink_final_sequences_.begin(),
               sink_final_sequences_.end(),
               [expected_sequence](const auto sequence) { return sequence == expected_sequence; });
  }

  [[nodiscard]] std::uint64_t activation_completions() const noexcept {
    return activation_completions_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t backpressure_count() const noexcept {
    return backpressure_count_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t source_backpressure_count() const noexcept {
    return source_backpressure_count_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t validation_errors() const noexcept {
    return validation_errors_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t notification_attempts() const noexcept {
    return notification_attempts_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t notifications_accepted() const noexcept {
    return notifications_accepted_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t notification_backpressure() const noexcept {
    return notification_backpressure_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t notification_not_ready() const noexcept {
    return notification_not_ready_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t source_probe_allocations() const noexcept {
    return source_probe_allocations_;
  }

  [[nodiscard]] std::size_t sink_probe_allocations() const noexcept {
    return sink_probe_allocations_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t relay_probe_allocations() const noexcept {
    return relay_probe_allocations_;
  }

  [[nodiscard]] bool allocation_probe_valid() const noexcept {
    return source_probe_allocations() <= source_allocation_limit() &&
           relay_probe_allocations() <= relay_allocation_limit() &&
           sink_probe_allocations() <= sink_allocation_limit();
  }

  [[nodiscard]] constexpr std::size_t source_allocation_limit() const noexcept { return 0U; }

  [[nodiscard]] constexpr std::size_t relay_allocation_limit() const noexcept { return 0U; }

  [[nodiscard]] std::size_t sink_allocation_limit() const noexcept {
    if (options_.scenario == Scenario::kImage) {
      return kMaximumImageSinkComputeAllocations;
    }
    return options_.scenario == Scenario::kMultiTensor ? kMaximumMultiTensorSinkComputeAllocations
                                                       : 0U;
  }

  [[nodiscard]] std::int64_t measured_start_ns() const noexcept { return measured_start_ns_; }

  [[nodiscard]] std::int64_t measured_end_ns() const noexcept { return measured_end_ns_; }

  [[nodiscard]] std::vector<std::uint64_t> latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return timestamp_deltas(notification_post_timestamps_, source_callback_timestamps_);
    }
    return timestamp_deltas(
        notification_latency() ? notification_post_timestamps_ : source_timestamps_,
        sink_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> notification_to_source_latencies() const {
    return timestamp_deltas(notification_post_timestamps_, source_callback_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_to_sink_latencies() const {
    if (!notification_latency() || options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_timestamps_, sink_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> publication_to_receive_latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_pre_emit_timestamps_, sink_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_emit_latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_timestamps_, source_post_emit_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_publication_call_latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_pre_emit_timestamps_, source_post_emit_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_post_emit_to_sink_latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_post_emit_timestamps_, sink_timestamps_);
  }

  [[nodiscard]] std::uint64_t source_post_emit_to_sink_overlaps() const noexcept {
    if (options_.scenario == Scenario::kSourceOnly) {
      return 0U;
    }
    return timestamp_overlaps(source_post_emit_timestamps_, sink_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_post_emit_to_sink_latencies(
      const std::uint32_t sink_index) const {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_timestamps_by_index_.size()) {
      return {};
    }
    return timestamp_deltas(source_post_emit_timestamps_, sink_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::uint64_t source_post_emit_to_sink_overlaps(
      const std::uint32_t sink_index) const noexcept {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_timestamps_by_index_.size()) {
      return 0U;
    }
    return timestamp_overlaps(source_post_emit_timestamps_, sink_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_post_emit_to_pre_return_latencies() const {
    if (options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_post_emit_timestamps_, source_pre_return_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_pre_return_to_sink_callback_latencies(
      const std::uint32_t sink_index) const {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_callback_timestamps_by_index_.size()) {
      return {};
    }
    return timestamp_deltas(source_pre_return_timestamps_,
                            sink_callback_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::uint64_t source_pre_return_to_sink_callback_overlaps(
      const std::uint32_t sink_index) const noexcept {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_callback_timestamps_by_index_.size()) {
      return 0U;
    }
    return timestamp_overlaps(source_pre_return_timestamps_,
                              sink_callback_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::vector<std::uint64_t> sink_callback_to_post_receive_latencies(
      const std::uint32_t sink_index) const {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_callback_timestamps_by_index_.size()) {
      return {};
    }
    return timestamp_deltas(sink_callback_timestamps_by_index_[sink_index],
                            sink_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_to_sink_latencies(
      const std::uint32_t sink_index) const {
    if (options_.scenario == Scenario::kSourceOnly ||
        sink_index >= sink_timestamps_by_index_.size()) {
      return {};
    }
    return timestamp_deltas(source_timestamps_, sink_timestamps_by_index_[sink_index]);
  }

  [[nodiscard]] std::vector<std::uint64_t> source_callback_entry_to_pre_emit_latencies() const {
    if (!notification_latency() || options_.scenario == Scenario::kSourceOnly) {
      return {};
    }
    return timestamp_deltas(source_callback_timestamps_, source_pre_emit_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> tensor_loan_acquire_latencies() const {
    return timestamp_deltas(source_timestamps_, tensor_loan_acquired_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> tensor_write_commit_latencies() const {
    return timestamp_deltas(tensor_loan_acquired_timestamps_, tensor_write_committed_timestamps_);
  }

  [[nodiscard]] std::vector<std::uint64_t> tensor_receive_call_latencies() const {
    return timestamp_deltas(tensor_receive_start_timestamps_, sink_timestamps_);
  }

  [[nodiscard]] std::int64_t source_pre_return_timestamp(
      const std::uint64_t sequence) const noexcept {
    if (!measured(sequence)) {
      return 0;
    }
    const std::size_t index = measured_index(sequence);
    return index < source_pre_return_timestamps_.size() ? source_pre_return_timestamps_[index] : 0;
  }

  [[nodiscard]] std::int64_t sink_callback_timestamp(const std::uint32_t sink_index,
                                                     const std::uint64_t sequence) const noexcept {
    if (!measured(sequence) || sink_index >= sink_callback_timestamps_by_index_.size()) {
      return 0;
    }
    const std::size_t index = measured_index(sequence);
    return index < sink_callback_timestamps_by_index_[sink_index].size()
               ? sink_callback_timestamps_by_index_[sink_index][index]
               : 0;
  }

 private:
  void record_callback_thread(std::atomic<std::uint64_t>& first_tid,
                              std::atomic_bool& identity_is_native,
                              std::atomic_bool& native_tid_syscall_unavailable,
                              std::atomic_bool& multiple_threads) noexcept {
    if (!options_.runtime_phase_counters) {
      return;
    }
    const benchmarking::BenchmarkThreadIdentity current = benchmark_thread_identity();
    std::uint64_t expected = 0U;
    if (!first_tid.compare_exchange_strong(
            expected, current.value, std::memory_order_relaxed, std::memory_order_relaxed)) {
      if (expected != current.value ||
          identity_is_native.load(std::memory_order_relaxed) != current.is_native ||
          native_tid_syscall_unavailable.load(std::memory_order_relaxed) !=
              current.native_tid_syscall_unavailable) {
        multiple_threads.store(true, std::memory_order_relaxed);
      }
    } else {
      identity_is_native.store(current.is_native, std::memory_order_relaxed);
      native_tid_syscall_unavailable.store(current.native_tid_syscall_unavailable,
                                           std::memory_order_relaxed);
    }
  }

  [[nodiscard]] bool wait_for_activation_completion(const std::uint64_t expected) {
    std::unique_lock lock(completion_mutex_);
    const bool ready =
        completion_cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [&] {
          return activation_completions_.load(std::memory_order_acquire) >= expected || failed_;
        });
    return ready && activation_completions_.load(std::memory_order_acquire) >= expected;
  }

  [[nodiscard]] static std::vector<std::uint64_t> timestamp_deltas(
      const std::vector<std::int64_t>& starts, const std::vector<std::int64_t>& ends) {
    std::vector<std::uint64_t> result;
    if (starts.size() != ends.size()) {
      return result;
    }
    result.reserve(starts.size());
    for (std::size_t index = 0U; index < starts.size(); ++index) {
      const std::int64_t start = starts[index];
      const std::int64_t end = ends[index];
      if (start <= 0 || end < start) {
        continue;
      }
      result.push_back(static_cast<std::uint64_t>(end - start));
    }
    return result;
  }

  [[nodiscard]] static std::uint64_t timestamp_overlaps(
      const std::vector<std::int64_t>& starts, const std::vector<std::int64_t>& ends) noexcept {
    if (starts.size() != ends.size()) {
      return 0U;
    }
    std::uint64_t overlaps = 0U;
    for (std::size_t index = 0U; index < starts.size(); ++index) {
      const std::int64_t start = starts[index];
      const std::int64_t end = ends[index];
      if (start > 0 && end > 0 && end < start) {
        ++overlaps;
      }
    }
    return overlaps;
  }

  Options options_;
  std::vector<std::int64_t> source_timestamps_;
  std::vector<std::int64_t> source_pre_emit_timestamps_;
  std::vector<std::int64_t> source_post_emit_timestamps_;
  std::vector<std::int64_t> source_pre_return_timestamps_;
  std::vector<std::int64_t> tensor_loan_acquired_timestamps_;
  std::vector<std::int64_t> tensor_write_committed_timestamps_;
  std::vector<std::int64_t> tensor_receive_start_timestamps_;
  std::vector<std::int64_t> sink_timestamps_;
  std::vector<std::vector<std::int64_t>> sink_timestamps_by_index_;
  std::vector<std::vector<std::int64_t>> sink_callback_timestamps_by_index_;
  std::vector<std::int64_t> notification_post_timestamps_;
  std::vector<std::int64_t> source_callback_timestamps_;
  std::vector<std::uint64_t> sink_deliveries_;
  std::vector<std::uint64_t> sink_final_sequences_;
  std::size_t sequence_sink_delivery_count_{};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<std::atomic<std::uint32_t>[]> sequence_sink_deliveries_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<std::atomic<std::uintptr_t>[]> tensor_source_pointers_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::array<std::unique_ptr<std::atomic<std::uintptr_t>[]>, 2U> multi_tensor_source_pointers_;
  std::vector<std::uint8_t> tensor_validation_buffer_;
  std::vector<std::uint8_t> image_backing_;
  std::shared_ptr<Tensor> image_tensor_;
  std::atomic<std::uintptr_t> tensor_consumer_stream_{};
  std::uint64_t source_callbacks_{};
  std::atomic<std::uint64_t> source_compute_entries_{};
  std::uint64_t measured_source_callbacks_{};
  std::uint64_t relay_callbacks_{};
  alignas(kBenchmarkCacheLineBytes) std::atomic<std::uint64_t> sink_callbacks_{};
  std::atomic<std::uint64_t> source_callback_tid_{};
  std::atomic<std::uint64_t> relay_callback_tid_{};
  std::atomic<std::uint64_t> sink_callback_tid_{};
  std::atomic_bool source_callback_identity_is_native_{};
  std::atomic_bool relay_callback_identity_is_native_{};
  std::atomic_bool sink_callback_identity_is_native_{};
  std::atomic_bool source_callback_native_tid_syscall_unavailable_{};
  std::atomic_bool relay_callback_native_tid_syscall_unavailable_{};
  std::atomic_bool sink_callback_native_tid_syscall_unavailable_{};
  std::atomic_bool source_callback_multiple_threads_{};
  std::atomic_bool relay_callback_multiple_threads_{};
  std::atomic_bool sink_callback_multiple_threads_{};
  alignas(kBenchmarkCacheLineBytes) std::atomic<std::uint64_t> publications_{};
  std::uint64_t relay_publications_{};
  alignas(kBenchmarkCacheLineBytes) std::atomic<std::uint64_t> deliveries_{};
  std::atomic<std::uint64_t> measured_deliveries_{};
  std::atomic<std::uint64_t> activation_completions_{};
  std::atomic<std::uint64_t> backpressure_count_{};
  std::atomic<std::uint64_t> source_backpressure_count_{};
  std::atomic<std::uint64_t> descriptor_backing_initializations_{};
  std::atomic<std::uint64_t> descriptor_backing_initializations_during_measurement_{};
  std::atomic_bool relay_retry_observed_{};
  std::atomic<std::uint64_t> validation_errors_{};
  std::size_t source_probe_allocations_{};
  std::size_t relay_probe_allocations_{};
  std::atomic<std::size_t> sink_probe_allocations_{};
  std::int64_t measured_start_ns_{};
  std::int64_t measured_end_ns_{};
  std::atomic<ErrorCode> first_error_{ErrorCode::kSuccess};
  std::atomic_bool source_finished_{};
  std::atomic<std::uint64_t> notification_attempts_{};
  std::atomic<std::uint64_t> notifications_accepted_{};
  std::atomic<std::uint64_t> notification_backpressure_{};
  std::atomic<std::uint64_t> notification_not_ready_{};
  mutable std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  bool completed_{};
  bool failed_{};
  std::size_t descriptor_warmup_slots_{};
  std::optional<Tensor> final_tensor_;
  std::optional<Tensor> final_second_tensor_;
  std::uint8_t final_tensor_expected_byte_{};
  std::optional<NotificationSender> notification_sender_;
};

[[nodiscard]] std::size_t runtime_phase_trace_capacity(const BenchmarkRunState& state) noexcept {
  const Options& options = state.options();
  const std::uint64_t downstream_activations =
      options.scenario == Scenario::kSourceOnly
          ? 0U
          : (options.scenario == Scenario::kSampleForward ? options.relay_count + 1U
                                                          : options.fanout);
  const std::uint64_t activations_per_sequence = 1U + downstream_activations;
  const std::uint64_t extra_source_activations =
      is_notification_workload(options.workload) ? options.notification_capacity : 1U;
  if (state.final_sequence() >
      (std::numeric_limits<std::uint64_t>::max() - extra_source_activations) /
          activations_per_sequence) {
    return kMaximumRuntimePhaseTraceRecords;
  }
  const std::uint64_t expected_activations =
      state.final_sequence() * activations_per_sequence + extra_source_activations;
  if (expected_activations >
      kMaximumRuntimePhaseTraceRecords / kTraceRecordsPerExpectedActivation) {
    return kMaximumRuntimePhaseTraceRecords;
  }
  std::uint64_t expected_records = expected_activations * kTraceRecordsPerExpectedActivation;
  if (options.scenario == Scenario::kTensor) {
    if (state.final_sequence() >
        (kMaximumRuntimePhaseTraceRecords - expected_records) / kTensorTraceRecordsPerPublication) {
      return kMaximumRuntimePhaseTraceRecords;
    }
    expected_records += state.final_sequence() * kTensorTraceRecordsPerPublication;
  }
  const std::uint64_t inline_emits_per_sequence =
      options.scenario == Scenario::kSampleForward
          ? options.relay_count + 1U
          : (options.scenario == Scenario::kPod || options.scenario == Scenario::kSignal ||
                     options.scenario == Scenario::kSchema
                 ? 1U
                 : 0U);
  const std::uint64_t inline_records_per_sequence =
      inline_emits_per_sequence * kInlinePayloadTraceRecordsPerPublication;
  if (inline_records_per_sequence != 0U) {
    if (state.final_sequence() >
        (kMaximumRuntimePhaseTraceRecords - expected_records) / inline_records_per_sequence) {
      return kMaximumRuntimePhaseTraceRecords;
    }
    expected_records += state.final_sequence() * inline_records_per_sequence;
  }
  return static_cast<std::size_t>(expected_records);
}

void append_timestamp_delta(std::vector<std::uint64_t>& values, const std::int64_t start,
                            const std::int64_t end, bool& valid) {
  if (start <= 0 || end < start) {
    valid = false;
    return;
  }
  values.push_back(static_cast<std::uint64_t>(end - start));
}

[[nodiscard]] RuntimePhaseReceiverSummary& phase_receiver(RuntimePhaseTraceSummary& summary,
                                                          const std::size_t receiver_index) {
  const auto existing =
      std::find_if(summary.receivers.begin(), summary.receivers.end(), [&](const auto& receiver) {
        return receiver.receiver_index == receiver_index;
      });
  if (existing != summary.receivers.end()) {
    return *existing;
  }
  const bool source = summary.receivers.empty();
  summary.receivers.push_back(RuntimePhaseReceiverSummary{
      .receiver_index = receiver_index,
      .source = source,
      .downstream_index = source ? std::nullopt
                                 : std::optional<std::uint32_t>{static_cast<std::uint32_t>(
                                       summary.receivers.size() - 1U)},
  });
  return summary.receivers.back();
}

// The summarizer validates a stateful event grammar and is clearer as one chronological pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RuntimePhaseTraceSummary summarize_phase_trace(
    const BenchmarkRunState& state, const detail::RuntimePhaseCounters* counters) {
  RuntimePhaseTraceSummary summary;
  if (counters == nullptr) {
    return summary;
  }
  summary.enabled = true;
  summary.activation_trace_applicable =
      state.options().operator_lanes == OperatorLaneMode::kShared ||
      state.options().scenario == Scenario::kSourceOnly;
  summary.capacity_exhausted = counters->trace_capacity_exhausted();
  summary.capacity = counters->trace_capacity();
  summary.attempted_records = counters->trace_attempted();
  summary.records = counters->trace_records().size();
  summary.dropped = counters->trace_dropped();
  summary.storage_invariants_valid = summary.records <= summary.capacity &&
                                     summary.attempted_records >= summary.records &&
                                     summary.attempted_records - summary.records == summary.dropped;
  // Detailed activation attribution is a serial trace. Independent partition owners can append
  // valid records concurrently, so their total order is not an activation order. Keep the bounded
  // storage and phase-counter evidence, but do not publish a misleading serial attribution.
  if (!summary.activation_trace_applicable) {
    summary.valid =
        summary.storage_invariants_valid && !summary.capacity_exhausted && summary.dropped == 0U;
    return summary;
  }
  summary.valid = true;

  struct ActivationTrace {
    std::size_t receiver_index{detail::kNoRuntimePhaseReceiver};
    SourceId source{};
    std::uint64_t source_sequence{};
    std::optional<std::int64_t> readiness_poll_returned;
    std::optional<std::int64_t> readiness_preceding_cut;
    std::int64_t candidate_frozen{};
    std::int64_t dispatch_started{};
    std::int64_t callback_entered{};
    std::int64_t callback_returned{};
    std::int64_t dispatch_returned{};
    std::int64_t retirement_completed{};
  } activation;
  struct PendingReadinessPoll {
    SourceId source{};
    std::int64_t timestamp_ns{};
    std::int64_t preceding_cut_ns{};
  };
  struct PendingTensorEmit {
    std::uint64_t output_port_id{};
    std::uint64_t source_sequence{};
    std::int64_t started{};
    std::int64_t validated{};
    std::int64_t transport_started{};
    std::int64_t transport_returned{};
  };
  struct PendingInlinePayloadEmit {
    std::uint64_t output_port_id{};
    std::uint64_t source_sequence{};
    std::int64_t started{};
    std::int64_t encoded{};
    std::int64_t transport_started{};
    std::int64_t transport_returned{};
  };
  std::vector<PendingReadinessPoll> pending_readiness_polls;
  std::vector<PendingTensorEmit> pending_tensor_emits;
  std::vector<PendingInlinePayloadEmit> pending_inline_payload_emits;
  const auto pending_tensor_emit = [&](const detail::RuntimePhaseTraceRecord& record) {
    return std::find_if(pending_tensor_emits.begin(),
                        pending_tensor_emits.end(),
                        [&](const PendingTensorEmit& pending) {
                          return pending.output_port_id == record.output_port_id &&
                                 pending.source_sequence == record.source_sequence;
                        });
  };
  const auto pending_inline_payload_emit = [&](const detail::RuntimePhaseTraceRecord& record) {
    return std::find_if(pending_inline_payload_emits.begin(),
                        pending_inline_payload_emits.end(),
                        [&](const PendingInlinePayloadEmit& pending) {
                          return pending.output_port_id == record.output_port_id &&
                                 pending.source_sequence == record.source_sequence;
                        });
  };
  std::optional<bool> previous_activation_source;
  std::uint64_t consecutive_source_activations{};
  std::int64_t previous_cut{};

  for (const detail::RuntimePhaseTraceRecord& record : counters->trace_records()) {
    switch (record.point) {
      case detail::RuntimePhaseTracePoint::kChannelTransportWaitReturned:
      case detail::RuntimePhaseTracePoint::kChannelIngressRetained:
      case detail::RuntimePhaseTracePoint::kChannelEndOfStreamRetained:
      case detail::RuntimePhaseTracePoint::kChannelReadinessWakeSignaled:
      case detail::RuntimePhaseTracePoint::kControllerWakeSignaled:
      case detail::RuntimePhaseTracePoint::kControllerParkStarted:
      case detail::RuntimePhaseTracePoint::kControllerTimeoutReturned:
      case detail::RuntimePhaseTracePoint::kDirectCompositeStarted:
      case detail::RuntimePhaseTracePoint::kDirectCompositeSuccessorEntered:
      case detail::RuntimePhaseTracePoint::kControllerWakeReturned:
      case detail::RuntimePhaseTracePoint::kDirectCompositeReturned:
        break;
      case detail::RuntimePhaseTracePoint::kReadinessPollReturned:
        ++summary.readiness_polls;
        // Multiple fanout bindings can report readiness for one logical source before any
        // corresponding candidate freezes. Retain each poll and consume them in source-local FIFO
        // order so trace validation remains one-to-one.
        pending_readiness_polls.push_back(PendingReadinessPoll{
            .source = record.source,
            .timestamp_ns = record.timestamp_ns,
            .preceding_cut_ns = previous_cut,
        });
        break;
      case detail::RuntimePhaseTracePoint::kCandidateFrozen:
        if (activation.candidate_frozen != 0) {
          summary.valid = false;
          return summary;
        }
        activation.receiver_index = record.receiver_index;
        activation.source = record.source;
        activation.source_sequence = record.source_sequence;
        if (const auto pending =
                std::find_if(pending_readiness_polls.begin(),
                             pending_readiness_polls.end(),
                             [&](const auto& poll) { return poll.source == record.source; });
            pending != pending_readiness_polls.end()) {
          activation.readiness_poll_returned = pending->timestamp_ns;
          if (pending->preceding_cut_ns != 0) {
            activation.readiness_preceding_cut = pending->preceding_cut_ns;
          }
          pending_readiness_polls.erase(pending);
        }
        activation.candidate_frozen = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchStarted:
        if (activation.candidate_frozen == 0 || activation.dispatch_started != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.dispatch_started = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeCallbackEntered:
        if (activation.dispatch_started == 0 || activation.callback_entered != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.callback_entered = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeCallbackReturned:
        if (activation.callback_entered == 0 || activation.callback_returned != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.callback_returned = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchReturned:
        if (activation.callback_returned == 0 || activation.dispatch_returned != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.dispatch_returned = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationRetirementCompleted:
        if (activation.dispatch_returned == 0 || activation.retirement_completed != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.retirement_completed = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationCutCompleted: {
        if (activation.retirement_completed == 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        RuntimePhaseReceiverSummary& receiver = phase_receiver(summary, activation.receiver_index);
        ++receiver.activations;
        ++summary.complete_activations;
        if (previous_activation_source.has_value()) {
          if (*previous_activation_source && receiver.source) {
            ++summary.source_to_source_transitions;
          } else if (*previous_activation_source) {
            ++summary.source_to_downstream_transitions;
          } else if (receiver.source) {
            ++summary.downstream_to_source_transitions;
          } else {
            ++summary.downstream_to_downstream_transitions;
          }
        }
        if (receiver.source) {
          if (!previous_activation_source.value_or(false)) {
            ++summary.source_activation_runs;
            consecutive_source_activations = 0U;
          }
          ++consecutive_source_activations;
          summary.maximum_consecutive_source_activations = std::max(
              summary.maximum_consecutive_source_activations, consecutive_source_activations);
        } else {
          consecutive_source_activations = 0U;
        }
        previous_activation_source = receiver.source;
        append_timestamp_delta(receiver.candidate_to_dispatch_start,
                               activation.candidate_frozen,
                               activation.dispatch_started,
                               summary.valid);
        append_timestamp_delta(receiver.compute_dispatch,
                               activation.dispatch_started,
                               activation.dispatch_returned,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_return_to_retirement,
                               activation.dispatch_returned,
                               activation.retirement_completed,
                               summary.valid);
        append_timestamp_delta(receiver.retirement_to_cut,
                               activation.retirement_completed,
                               record.timestamp_ns,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_return_to_cut,
                               activation.dispatch_returned,
                               record.timestamp_ns,
                               summary.valid);
        if (activation.readiness_poll_returned.has_value()) {
          append_timestamp_delta(receiver.readiness_poll_return_to_candidate,
                                 *activation.readiness_poll_returned,
                                 activation.candidate_frozen,
                                 summary.valid);
          if (activation.readiness_preceding_cut.has_value()) {
            append_timestamp_delta(receiver.previous_cut_to_readiness_poll_return,
                                   *activation.readiness_preceding_cut,
                                   *activation.readiness_poll_returned,
                                   summary.valid);
          }
        }
        if (receiver.source) {
          const std::int64_t pre_return =
              state.source_pre_return_timestamp(activation.source_sequence);
          if (pre_return != 0) {
            append_timestamp_delta(receiver.callback_pre_return_to_dispatch_return,
                                   pre_return,
                                   activation.dispatch_returned,
                                   summary.valid);
          }
        } else if (receiver.downstream_index.has_value() &&
                   fanout_scenario(state.options().scenario)) {
          const std::int64_t callback_entry =
              state.sink_callback_timestamp(*receiver.downstream_index, activation.source_sequence);
          if (callback_entry != 0) {
            append_timestamp_delta(receiver.dispatch_start_to_callback_entry,
                                   activation.dispatch_started,
                                   callback_entry,
                                   summary.valid);
          }
        }
        previous_cut = record.timestamp_ns;
        activation = {};
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorEmitStarted:
        if (record.output_port_id == 0U || record.source_sequence == 0U ||
            pending_tensor_emit(record) != pending_tensor_emits.end()) {
          summary.valid = false;
          return summary;
        }
        pending_tensor_emits.push_back(PendingTensorEmit{
            .output_port_id = record.output_port_id,
            .source_sequence = record.source_sequence,
            .started = record.timestamp_ns,
        });
        break;
      case detail::RuntimePhaseTracePoint::kTensorEmitValidated: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated != 0) {
          summary.valid = false;
          return summary;
        }
        pending->validated = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorTransportPublicationStarted: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated == 0 ||
            pending->transport_started != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_started = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorTransportPublicationReturned: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->transport_started == 0 ||
            pending->transport_returned != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_returned = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorEmitCompleted: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated == 0 ||
            (pending->transport_started == 0) != (pending->transport_returned == 0)) {
          summary.valid = false;
          return summary;
        }
        append_timestamp_delta(
            summary.tensor_emit_total, pending->started, record.timestamp_ns, summary.valid);
        append_timestamp_delta(
            summary.tensor_emit_validation, pending->started, pending->validated, summary.valid);
        if (pending->transport_started != 0) {
          append_timestamp_delta(summary.tensor_pre_transport,
                                 pending->validated,
                                 pending->transport_started,
                                 summary.valid);
          append_timestamp_delta(summary.tensor_transport_publication,
                                 pending->transport_started,
                                 pending->transport_returned,
                                 summary.valid);
          append_timestamp_delta(summary.tensor_post_transport,
                                 pending->transport_returned,
                                 record.timestamp_ns,
                                 summary.valid);
        }
        ++summary.complete_tensor_emits;
        pending_tensor_emits.erase(pending);
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadEmitStarted:
        if (record.output_port_id == 0U || record.source_sequence == 0U ||
            pending_inline_payload_emit(record) != pending_inline_payload_emits.end()) {
          summary.valid = false;
          return summary;
        }
        pending_inline_payload_emits.push_back(PendingInlinePayloadEmit{
            .output_port_id = record.output_port_id,
            .source_sequence = record.source_sequence,
            .started = record.timestamp_ns,
        });
        break;
      case detail::RuntimePhaseTracePoint::kInlinePayloadEncoded: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded != 0) {
          summary.valid = false;
          return summary;
        }
        pending->encoded = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadTransportPublicationStarted: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded == 0 ||
            pending->transport_started != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_started = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadTransportPublicationReturned: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->transport_started == 0 ||
            pending->transport_returned != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_returned = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadEmitCompleted: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded == 0 ||
            (pending->transport_started == 0) != (pending->transport_returned == 0)) {
          summary.valid = false;
          return summary;
        }
        append_timestamp_delta(summary.inline_payload_emit_total,
                               pending->started,
                               record.timestamp_ns,
                               summary.valid);
        append_timestamp_delta(
            summary.inline_payload_encoding, pending->started, pending->encoded, summary.valid);
        if (pending->transport_started != 0) {
          append_timestamp_delta(summary.inline_payload_pre_transport,
                                 pending->encoded,
                                 pending->transport_started,
                                 summary.valid);
          append_timestamp_delta(summary.inline_payload_transport_publication,
                                 pending->transport_started,
                                 pending->transport_returned,
                                 summary.valid);
          append_timestamp_delta(summary.inline_payload_post_transport,
                                 pending->transport_returned,
                                 record.timestamp_ns,
                                 summary.valid);
        }
        ++summary.complete_inline_payload_emits;
        pending_inline_payload_emits.erase(pending);
        break;
      }
      default:
        summary.valid = false;
        return summary;
    }
    if (!summary.valid) {
      return summary;
    }
  }
  summary.unmatched_readiness_polls += pending_readiness_polls.size();
  summary.valid = activation.candidate_frozen == 0 && summary.dropped == 0U &&
                  summary.unmatched_readiness_polls == 0U && pending_tensor_emits.empty() &&
                  pending_inline_payload_emits.empty();
  return summary;
}

BenchmarkRunState* g_run_state{};

[[nodiscard]] BenchmarkRunState& run_state() noexcept {
  return *g_run_state;
}

[[nodiscard]] std::string_view operator_partition_name(
    const Options& options, const std::string_view separate_partition) noexcept {
  return options.operator_lanes == OperatorLaneMode::kShared ? "runtime-graph-shared"
                                                             : separate_partition;
}

[[nodiscard]] std::string sink_partition_name(const Options& options,
                                              const std::uint32_t sink_index) {
  if (options.operator_lanes == OperatorLaneMode::kShared) {
    return "runtime-graph-shared";
  }
  if (options.operator_lanes == OperatorLaneMode::kPerSink || options.fanout > 1U) {
    return "runtime-graph-sink-" + std::to_string(sink_index + 1U);
  }
  return "runtime-graph-sink";
}

[[nodiscard]] LifecycleStatus acquire_benchmark_notification_sender(
    LifecycleContext& context, NotificationSource& source) noexcept {
  auto sender = context.notification_sender(source);
  if (!sender) {
    run_state().fail(sender.error().code);
    return LifecycleStatus::kFatalFailure;
  }
  run_state().publish_notification_sender(std::move(*sender));
  return LifecycleStatus::kOk;
}

// Isolates the production notification, controller, Dispatcher, and one-lane callback path from
// Channel endpoints, the Channel pump, and downstream input polling.
class RuntimeBenchmarkSourceOnly final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(run_state().options().notification_capacity)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeBenchmarkSourceOnly::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.source_only_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    state.prepare_source_timestamp(next_sequence_, callback_timestamp);
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      state.record_source_allocation_probe(allocations.count());
    }
    state.record_source_only_completion(next_sequence_, callback_timestamp);
    ++next_sequence_;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
};

template <typename Payload>
class RuntimeBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output");
    if (run_state().notification_driven()) {
      spec.notification_source(notification_, "benchmark-notification")
          .capacity(run_state().options().notification_capacity)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeBenchmarkSource::arm);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (run_state().notification_driven()) {
      result.trigger(OnNotified{.event = notification_});
    } else {
      result.trigger(OnClock{.period = run_state().source_period()});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);

    const Payload payload = make_benchmark_payload<Payload>(next_sequence_);
    const EmitOptions emit_options{
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
        .flags = EmitFlags::kDegraded,
    };
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;

    expected<void, Error> emitted;
    state.prepare_source_pre_emit_timestamp(
        next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      emitted = output_.emit(payload, emit_options);
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = output_.emit(payload, emit_options);
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        state.record_source_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted.error()));
    }

    const std::int64_t post_emit_timestamp =
        state.source_post_emit_timestamp_required(next_sequence_) ? now_ns() : 0;
    if (source_timestamp != 0) {
      state.prepare_source_timestamp(next_sequence_, source_timestamp);
    }
    const std::uint64_t published_sequence = next_sequence_;
    state.record_publication(published_sequence, post_emit_timestamp);
    ++next_sequence_;
    state.record_source_pre_return(
        published_sequence,
        state.source_post_emit_timestamp_required(published_sequence) ? now_ns() : 0);
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  Output<Payload> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

template <typename Payload>
class RuntimeBenchmarkSink final : public Operator<> {
 public:
  explicit RuntimeBenchmarkSink(const std::uint32_t sink_index = 0U) : sink_index_(sink_index) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        is_latency_workload(state.options().workload) ? now_ns() : 0;
    state.record_sink_callback();

    if (state.options().test_fail_sink_after_relay_pressure) {
      const auto deadline =
          BenchmarkClock::now() + std::chrono::milliseconds{state.options().timeout_ms};
      while (!state.relay_retry_observed()) {
        const ErrorCode failure = state.first_error();
        if (failure != ErrorCode::kSuccess) {
          return make_unexpected(Error{failure});
        }
        if (context.cancellation().stop_requested()) {
          state.fail(ErrorCode::kClosed);
          return make_unexpected(Error{ErrorCode::kClosed});
        }
        if (BenchmarkClock::now() >= deadline) {
          state.fail(ErrorCode::kTimeout);
          return make_unexpected(Error{ErrorCode::kTimeout});
        }
        std::this_thread::yield();
      }
      state.fail(ErrorCode::kFailure);
      return make_unexpected(
          Error{ErrorCode::kFailure, "intentional sink failure after relay pressure"});
    }

    expected<Sample<Payload>, Error> sample = make_unexpected(Error{});
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }

    const std::uint64_t sequence = sample->metadata.frame_id;
    state.record_sink_callback_entry(sink_index_, sequence, callback_timestamp);
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const bool identity_valid =
        sequence != 0U && sequence >= next_sequence_ && sample->metadata.trace_id == sequence &&
        sample->metadata.degraded() &&
        (lossy_queue_policy(state.options().queue_policy) || sequence == next_sequence_);
    const bool sequence_valid = [&sample, sequence] {
      if constexpr (std::is_same_v<Payload, BenchmarkPOD64>) {
        return sample->data.sequence == sequence;
      } else if constexpr (std::is_same_v<Payload, schema::TimeT>) {
        return sample->data.time_ns == static_cast<std::int64_t>(sequence);
      } else {
        return true;
      }
    }();
    const bool validate_payload = state.options().validation == PayloadValidation::kAll ||
                                  (state.options().validation == PayloadValidation::kFinal &&
                                   sequence == state.final_sequence());
    if (!identity_valid || !sequence_valid || (validate_payload && !payload_valid(sample->data))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    state.record_delivery(sink_index_, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<Payload> input_;
  std::uint32_t sink_index_{};
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeTensorBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.output(output_, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = tensor_memory_kind(options.tensor_storage),
            .dtype = kTensorByteDtype,
            .rank = 1U,
        })
        .tensor_allocation(TensorAllocationBounds{
            .max_rank = 1U,
            .max_byte_span = options.tensor_bytes,
        });
    if (run_state().notification_driven()) {
      spec.notification_source(notification_, "benchmark-notification")
          .capacity(run_state().options().notification_capacity)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeTensorBenchmarkSource::arm);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (run_state().notification_driven()) {
      result.trigger(OnNotified{.event = notification_});
    } else {
      result.trigger(OnClock{.period = run_state().source_period()});
    }
    return result;
  }

  void start() override { initialized_backings_.reserve(run_state().descriptor_warmup_slots()); }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;
    const std::array<std::int64_t, 1U> shape{
        static_cast<std::int64_t>(state.options().tensor_bytes)};

    const auto publish = [&]() -> expected<void, Error> {
      auto loan =
          output_.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kTensorByteDtype});
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }
      const std::int64_t loan_acquired_timestamp = source_timestamp != 0 ? now_ns() : 0;
      void* pointer{};
      bool initialized_backing{};
      const bool descriptor_only =
          state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
      const auto initialize = [&](const TensorOutputWriteGuard& writer) {
        pointer = writer.data();
        initialized_backing = descriptor_only && std::find(initialized_backings_.begin(),
                                                           initialized_backings_.end(),
                                                           pointer) == initialized_backings_.end();
        return !descriptor_only || initialized_backing;
      };
      const std::uint8_t fill =
          descriptor_only ? kMultiTensorPayloadByte : static_cast<std::uint8_t>(next_sequence_);
      expected<void, Error> committed;
      if (state.options().tensor_storage == TensorStorage::kDevice) {
        committed = loan->write_and_commit(
            context.cuda_stream(),
            [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
              if (initialize(writer) &&
                  cudaMemsetAsync(writer.data(),
                                  static_cast<int>(fill),
                                  writer.byte_size(),
                                  context.cuda_stream().get()) != cudaSuccess) {
                return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
              }
              return {};
            });
      } else {
        committed = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
          if (initialize(writer)) {
            std::memset(writer.data(), static_cast<int>(fill), writer.byte_size());
          }
        });
      }
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
      if (initialized_backing) {
        initialized_backings_.push_back(pointer);
        state.record_descriptor_backing_initialization(next_sequence_);
        if (state.measured(next_sequence_)) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
      const std::int64_t write_committed_timestamp = source_timestamp != 0 ? now_ns() : 0;
      state.prepare_tensor_source_phase_timestamps(
          next_sequence_, loan_acquired_timestamp, write_committed_timestamp);
      state.record_tensor_source_pointer(next_sequence_, pointer);
      state.prepare_source_pre_emit_timestamp(
          next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
      return output_.emit(std::move(*loan),
                          EmitOptions{
                              .frame_id = next_sequence_,
                              .trace = TraceContext::root(next_sequence_),
                              .flags = EmitFlags::kDegraded,
                          });
    };

    expected<void, Error> emitted;
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      emitted = publish();
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = publish();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        state.record_source_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted.error()));
    }
    const std::int64_t post_emit_timestamp =
        state.source_post_emit_timestamp_required(next_sequence_) ? now_ns() : 0;
    if (source_timestamp != 0) {
      state.prepare_source_timestamp(next_sequence_, source_timestamp);
    }
    const std::uint64_t published_sequence = next_sequence_;
    state.record_publication(published_sequence, post_emit_timestamp);
    ++next_sequence_;
    state.record_source_pre_return(
        published_sequence,
        state.source_post_emit_timestamp_required(published_sequence) ? now_ns() : 0);
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  Output<Tensor> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  std::vector<void*> initialized_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeTensorBenchmarkSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input")
        .queue_depth(options.queue_depth)
        .expects_tensor(TensorPortLayout{
            .memory_kind = tensor_memory_kind(options.tensor_storage),
            .dtype = kTensorByteDtype,
            .rank = 1U,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override {
    const std::size_t warmup_slots = run_state().descriptor_warmup_slots();
    retained_warmup_tensors_.reserve(run_state().options().queue_depth);
    observed_backings_.reserve(warmup_slots);
    pool_warmup_complete_ = warmup_slots == 0U;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_sink_callback();
    state.record_sink_callback_entry(0U, next_sequence_, callback_timestamp);
    expected<Sample<Tensor>, Error> sample = make_unexpected(Error{});
    const std::int64_t receive_start_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.prepare_tensor_receive_start_timestamp(next_sequence_, receive_start_timestamp);
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }

    const std::uint64_t sequence = sample->metadata.frame_id;
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const bool metadata_valid =
        sequence != 0U && sequence <= state.final_sequence() && sequence >= next_sequence_ &&
        sample->metadata.trace_id == sequence &&
        (lossy_queue_policy(state.options().queue_policy) || sequence == next_sequence_) &&
        sample->metadata.degraded();
    const bool descriptor_valid = tensor_descriptor_matches(sample->data,
                                                            state.options().tensor_bytes,
                                                            state.options().tensor_storage,
                                                            state.options().tensor_device,
                                                            false);
    const bool pointer_valid = state.tensor_pointer_matches(sequence, sample->data.data());
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kMultiTensorPayloadByte
            : static_cast<std::uint8_t>(sequence);
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      state.record_tensor_consumer_stream(context.cuda_stream().get());
    }
    if (!metadata_valid || !descriptor_valid || !pointer_valid ||
        (validate_payload &&
         !state.validate_tensor_bytes(sample->data, expected, context.cuda_stream().get()))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    if (state.options().validation == PayloadValidation::kFinal &&
        sequence == state.final_sequence()) {
      state.retain_final_tensor(std::move(sample->data), expected);
    }
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (!pool_warmup_complete_ && sequence <= state.options().warmup && warmup_slots != 0U) {
      const void* const backing = sample->data.data();
      if (std::find(observed_backings_.begin(), observed_backings_.end(), backing) ==
          observed_backings_.end()) {
        observed_backings_.push_back(backing);
      }
      if (observed_backings_.size() == warmup_slots) {
        retained_warmup_tensors_.clear();
        pool_warmup_complete_ = true;
      } else {
        if (retained_warmup_tensors_.size() == state.options().queue_depth) {
          retained_warmup_tensors_.erase(retained_warmup_tensors_.begin());
        }
        retained_warmup_tensors_.push_back(sample->data);
      }
    }
    state.record_delivery(0U, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<Tensor> input_;
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::vector<Tensor> retained_warmup_tensors_;
  std::vector<const void*> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeMultiTensorBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    const TensorFieldOutputContract field_contract{
        .layout =
            TensorPortLayout{
                .memory_kind = MemoryKind::kHost,
                .dtype = kTensorByteDtype,
                .rank = 1U,
            },
        .max_byte_span = options.tensor_bytes,
        .payload_options =
            TensorFieldPayloadOptions{
                .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
            },
    };
    auto& builder = spec.output(output_, "output").max_emits_per_compute(1U);
    builder.tensor_fields(std::array{MultiTensorFields::first, MultiTensorFields::second},
                          field_contract);
    if (run_state().notification_driven()) {
      spec.notification_source(notification_, "benchmark-notification")
          .capacity(options.notification_capacity)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeMultiTensorBenchmarkSource::arm);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (run_state().notification_driven()) {
      result.trigger(OnNotified{.event = notification_});
    } else {
      result.trigger(OnClock{.period = run_state().source_period()});
    }
    return result;
  }

  void start() override {
    for (auto& inventory : initialized_backings_) {
      inventory.reserve(run_state().descriptor_warmup_slots());
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;

    const auto publish = [&]() -> expected<void, Error> {
      const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
      const std::array requests{
          tensor_shape(MultiTensorFields::first, shape),
          tensor_shape(MultiTensorFields::second, shape),
      };
      auto bundle = output_.allocate_tensors(requests);
      if (!bundle) {
        return make_unexpected(std::move(bundle.error()));
      }
      const bool descriptor_only =
          state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
      const std::uint8_t fill =
          descriptor_only ? kMultiTensorPayloadByte : static_cast<std::uint8_t>(next_sequence_);
      std::array<void*, 2U> field_pointers{};
      std::array<bool, 2U> initialized_backing{};
      const auto write_field = [&](const std::size_t index) {
        return [&, index](const TensorOutputWriteGuard& writer) {
          field_pointers[index] = writer.data();
          initialized_backing[index] =
              descriptor_only && std::find(initialized_backings_[index].begin(),
                                           initialized_backings_[index].end(),
                                           writer.data()) == initialized_backings_[index].end();
          if (!descriptor_only || initialized_backing[index]) {
            std::memset(writer.data(), static_cast<int>(fill), writer.byte_size());
          }
        };
      };
      auto committed = bundle->write_all(host_field(MultiTensorFields::first, write_field(0U)),
                                         host_field(MultiTensorFields::second, write_field(1U)));
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
      for (std::size_t index = 0U; index < initialized_backing.size(); ++index) {
        if (initialized_backing[index]) {
          initialized_backings_[index].push_back(field_pointers[index]);
          state.record_descriptor_backing_initialization(next_sequence_);
          if (state.measured(next_sequence_)) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
        }
        state.record_multi_tensor_source_pointer(index, next_sequence_, field_pointers[index]);
      }
      MultiTensorMessage message;
      message.sequence = next_sequence_;
      state.prepare_source_pre_emit_timestamp(
          next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
      return output_.emit_tensors(std::move(*bundle),
                                  message,
                                  EmitOptions{
                                      .frame_id = next_sequence_,
                                      .trace = TraceContext::root(next_sequence_),
                                      .flags = EmitFlags::kDegraded,
                                  });
    };

    expected<void, Error> emitted;
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      emitted = publish();
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = publish();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        state.record_source_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted.error()));
    }
    const std::int64_t post_emit_timestamp =
        state.source_post_emit_timestamp_required(next_sequence_) ? now_ns() : 0;
    if (source_timestamp != 0) {
      state.prepare_source_timestamp(next_sequence_, source_timestamp);
    }
    const std::uint64_t published_sequence = next_sequence_;
    state.record_publication(published_sequence, post_emit_timestamp);
    ++next_sequence_;
    state.record_source_pre_return(
        published_sequence,
        state.source_post_emit_timestamp_required(published_sequence) ? now_ns() : 0);
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  Output<MultiTensorMessage> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  std::array<std::vector<void*>, 2U> initialized_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeMultiTensorBenchmarkSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& builder = spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
    auto first = builder.tensor_field(MultiTensorFields::first);
    auto second = builder.tensor_field(MultiTensorFields::second);
    configure_field(first);
    configure_field(second);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override {
    const std::size_t warmup_slots = run_state().descriptor_warmup_slots();
    retained_warmup_tensors_.reserve(run_state().options().queue_depth);
    for (auto& inventory : observed_backings_) {
      inventory.reserve(warmup_slots);
    }
    pool_warmup_complete_ = warmup_slots == 0U;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_sink_callback();
    state.record_sink_callback_entry(0U, next_sequence_, callback_timestamp);
    expected<Sample<MultiTensorMessage>, Error> sample = make_unexpected(Error{});
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }

    const std::uint64_t sequence = sample->metadata.frame_id;
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const bool metadata_valid = sequence != 0U && sequence <= state.final_sequence() &&
                                sequence == next_sequence_ && sample->data.sequence == sequence &&
                                sample->metadata.trace_id == sequence &&
                                sample->metadata.degraded() && sample->data.first != nullptr &&
                                sample->data.second != nullptr;
    const bool tensors_valid =
        metadata_valid &&
        tensor_descriptor_matches(*sample->data.first,
                                  state.options().tensor_bytes,
                                  TensorStorage::kPageableHost,
                                  0,
                                  false) &&
        tensor_descriptor_matches(*sample->data.second,
                                  state.options().tensor_bytes,
                                  TensorStorage::kPageableHost,
                                  0,
                                  false) &&
        state.multi_tensor_pointer_matches(0U, sequence, sample->data.first->data()) &&
        state.multi_tensor_pointer_matches(1U, sequence, sample->data.second->data());
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kMultiTensorPayloadByte
            : static_cast<std::uint8_t>(sequence);
    if (!tensors_valid ||
        (validate_payload && (!state.validate_tensor_bytes(*sample->data.first, expected) ||
                              !state.validate_tensor_bytes(*sample->data.second, expected)))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    if (state.options().validation == PayloadValidation::kFinal &&
        sequence == state.final_sequence()) {
      state.retain_final_multi_tensor(*sample->data.first, *sample->data.second, expected);
    }
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (!pool_warmup_complete_ && sequence <= state.options().warmup && warmup_slots != 0U) {
      const std::array backings{sample->data.first->data(), sample->data.second->data()};
      for (std::size_t index = 0U; index < backings.size(); ++index) {
        auto& inventory = observed_backings_[index];
        if (std::find(inventory.begin(), inventory.end(), backings[index]) == inventory.end()) {
          inventory.push_back(backings[index]);
        }
      }
      const bool inventory_complete = std::ranges::all_of(
          observed_backings_,
          [&](const auto& inventory) { return inventory.size() == warmup_slots; });
      if (inventory_complete) {
        retained_warmup_tensors_.clear();
        pool_warmup_complete_ = true;
      } else {
        if (retained_warmup_tensors_.size() == state.options().queue_depth) {
          retained_warmup_tensors_.erase(retained_warmup_tensors_.begin());
        }
        retained_warmup_tensors_.push_back(
            std::array<Tensor, 2U>{*sample->data.first, *sample->data.second});
      }
    }
    state.record_delivery(0U, sequence, receive_timestamp);
    ++next_sequence_;
    return {};
  }

 private:
  static void configure_field(TensorFieldInputSpecBuilder& builder) {
    builder.expects_tensor(TensorPortLayout{
        .memory_kind = MemoryKind::kHost,
        .dtype = kTensorByteDtype,
        .rank = 1U,
    });
  }

  Input<MultiTensorMessage> input_;
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::vector<std::array<Tensor, 2U>> retained_warmup_tensors_;
  std::array<std::vector<const void*>, 2U> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeImageBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output");
    if (run_state().notification_driven()) {
      spec.notification_source(notification_, "benchmark-notification")
          .capacity(run_state().options().notification_capacity)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeImageBenchmarkSource::arm);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (run_state().notification_driven()) {
      result.trigger(OnNotified{.event = notification_});
    } else {
      result.trigger(OnClock{.period = run_state().source_period()});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;

    const auto publish = [&]() -> expected<void, Error> {
      schema::ImageT image;
      image.height = 1;
      image.width = static_cast<std::int32_t>(state.options().tensor_bytes);
      image.encoding = schema::ImageEncoding_CUSTOM;
      image.data = state.image_tensor();
      state.prepare_source_pre_emit_timestamp(
          next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
      return output_.emit(std::move(image),
                          EmitOptions{
                              .frame_id = next_sequence_,
                              .trace = TraceContext::root(next_sequence_),
                              .flags = EmitFlags::kDegraded,
                          });
    };

    expected<void, Error> emitted;
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      emitted = publish();
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = publish();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        state.record_source_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted.error()));
    }
    const std::int64_t post_emit_timestamp =
        state.source_post_emit_timestamp_required(next_sequence_) ? now_ns() : 0;
    if (source_timestamp != 0) {
      state.prepare_source_timestamp(next_sequence_, source_timestamp);
    }
    const std::uint64_t published_sequence = next_sequence_;
    state.record_publication(published_sequence, post_emit_timestamp);
    ++next_sequence_;
    state.record_source_pre_return(
        published_sequence,
        state.source_post_emit_timestamp_required(published_sequence) ? now_ns() : 0);
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  Output<schema::ImageT> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeImageBenchmarkSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_sink_callback();
    state.record_sink_callback_entry(0U, next_sequence_, callback_timestamp);
    expected<Sample<schema::ImageT>, Error> sample = make_unexpected(Error{});
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }

    const std::uint64_t sequence = sample->metadata.frame_id;
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const Tensor* tensor = sample->data.data.get();
    const bool metadata_valid =
        sequence != 0U && sequence <= state.final_sequence() && sequence >= next_sequence_ &&
        sample->metadata.trace_id == sequence &&
        (lossy_queue_policy(state.options().queue_policy) || sequence == next_sequence_) &&
        sample->metadata.degraded();
    const bool image_valid =
        sample->data.height == 1 &&
        sample->data.width == static_cast<std::int32_t>(state.options().tensor_bytes) &&
        sample->data.encoding == schema::ImageEncoding_CUSTOM && tensor != nullptr;
    bool tensor_valid = image_valid;
    if (tensor_valid) {
      tensor_valid =
          tensor_descriptor_matches(
              *tensor, state.options().tensor_bytes, TensorStorage::kPageableHost, 0, false) &&
          tensor->data() == state.image_tensor()->data();
    }
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    if (!metadata_valid || !tensor_valid ||
        (validate_payload && !state.validate_tensor_bytes(*tensor, kImagePayloadByte))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    if (state.options().validation == PayloadValidation::kFinal &&
        sequence == state.final_sequence()) {
      state.retain_final_tensor(*tensor, kImagePayloadByte);
    }
    state.record_delivery(0U, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<schema::ImageT> input_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeBenchmarkRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
    spec.output(output_, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    state.record_relay_callback();

    expected<Sample<BenchmarkPOD64>, Error> sample = make_unexpected(Error{});
    expected<void, Error> emitted;
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      if (sample) {
        emitted = output_.emit(*sample);
      }
      state.record_relay_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
      if (sample) {
        emitted = output_.emit(*sample);
      }
    }
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto retry_deadline =
        BenchmarkClock::now() + std::chrono::milliseconds{state.options().timeout_ms};
    while (!emitted) {
      if (emitted.error().code != ErrorCode::kBackpressured &&
          emitted.error().code != ErrorCode::kNotReady) {
        state.fail(emitted.error().code);
        return make_unexpected(std::move(emitted.error()));
      }
      const ErrorCode failure = state.first_error();
      if (failure != ErrorCode::kSuccess) {
        return make_unexpected(Error{failure});
      }
      if (context.cancellation().stop_requested()) {
        state.fail(ErrorCode::kClosed);
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (BenchmarkClock::now() >= retry_deadline) {
        state.fail(ErrorCode::kTimeout);
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      state.record_backpressure();
      state.record_relay_retry();
      std::this_thread::yield();
      emitted = output_.emit(*sample);
    }
    state.record_relay_publication();
    ++next_sequence_;
    return {};
  }

 private:
  Input<BenchmarkPOD64> input_;
  Output<BenchmarkPOD64> output_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// Graph construction intentionally keeps all benchmark scenarios in one comparable path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ExecutionPlan make_plan(const Options& options) {
  // A benchmark runner starts independent repetitions in quick succession. Give each process a
  // distinct service namespace so backend discovery cleanup from the prior repetition cannot
  // affect the next measurement.
  std::string graph_name{"runtime-graph-benchmark-"};
  graph_name += scenario_name(options.scenario);
  graph_name += "-";
  graph_name += workload_name(options.workload);
  graph_name += "-" + std::to_string(process_id());
  Graph graph{std::move(graph_name)};
  std::optional<PartitionHandle> shared_partition;
  if (options.operator_lanes == OperatorLaneMode::kShared) {
    shared_partition.emplace(graph.partition("runtime-graph-shared"));
  }
  auto source_partition =
      shared_partition ? *shared_partition : graph.partition("runtime-graph-source");
  std::optional<PartitionHandle> common_sink_partition;
  if (!shared_partition && options.operator_lanes != OperatorLaneMode::kPerSink &&
      options.scenario != Scenario::kSourceOnly && options.fanout == 1U) {
    common_sink_partition.emplace(graph.partition("runtime-graph-sink"));
  }
  std::optional<PartitionHandle> relay_partition;
  if (!shared_partition && options.scenario == Scenario::kSampleForward) {
    relay_partition.emplace(graph.partition("runtime-graph-relay"));
  }
  const auto add_sink = [&](const auto& sink, const std::uint32_t index = 0U) {
    if (shared_partition) {
      shared_partition->add(sink);
    } else if (options.operator_lanes == OperatorLaneMode::kPerSink || options.fanout > 1U) {
      graph.partition(sink_partition_name(options, index)).add(sink);
    } else {
      common_sink_partition->add(sink);
    }
  };
  const ConnectionOptions connection{
      .queue_policy = options.queue_policy,
      .queue_depth = options.queue_depth,
  };
  if (options.scenario == Scenario::kSourceOnly) {
    source_partition.add(graph.op<RuntimeBenchmarkSourceOnly>("source"));
  } else if (options.scenario == Scenario::kSignal) {
    auto source = graph.op<RuntimeBenchmarkSource<std::uint8_t>>("source");
    auto sink = graph.op<RuntimeBenchmarkSink<std::uint8_t>>("sink");
    source_partition.add(source);
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kSampleForward) {
    auto source = graph.op<RuntimeBenchmarkSource<BenchmarkPOD64>>("source");
    auto sink = graph.op<RuntimeBenchmarkSink<BenchmarkPOD64>>("sink");
    source_partition.add(source);
    add_sink(sink);
    std::vector<OperatorHandle<RuntimeBenchmarkRelay>> relays;
    relays.reserve(options.relay_count);
    for (std::uint32_t index = 0U; index < options.relay_count; ++index) {
      auto relay = graph.op<RuntimeBenchmarkRelay>("relay-" + std::to_string(index + 1U));
      if (relays.empty()) {
        graph.add_flow(source->output_, relay->input_, connection);
      } else {
        graph.add_flow(relays.back()->output_, relay->input_, connection);
      }
      if (shared_partition) {
        shared_partition->add(relay);
      } else {
        relay_partition->add(relay);
      }
      relays.push_back(relay);
    }
    graph.add_flow(relays.back()->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kTensor) {
    auto source = graph.op<RuntimeTensorBenchmarkSource>("source");
    auto sink = graph.op<RuntimeTensorBenchmarkSink>("sink");
    source_partition.add(source);
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kSchema) {
    auto source = graph.op<RuntimeBenchmarkSource<schema::TimeT>>("source");
    source_partition.add(source);
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeBenchmarkSink<schema::TimeT>>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, connection);
    }
  } else if (options.scenario == Scenario::kImage) {
    auto source = graph.op<RuntimeImageBenchmarkSource>("source");
    auto sink = graph.op<RuntimeImageBenchmarkSink>("sink");
    source_partition.add(source);
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kMultiTensor) {
    auto source = graph.op<RuntimeMultiTensorBenchmarkSource>("source");
    auto sink = graph.op<RuntimeMultiTensorBenchmarkSink>("sink");
    source_partition.add(source);
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else {
    auto source = graph.op<RuntimeBenchmarkSource<BenchmarkPOD64>>("source");
    source_partition.add(source);
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeBenchmarkSink<BenchmarkPOD64>>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, connection);
    }
  }
  const ClockKey clock = graph.add_clock<RealtimeClock>("benchmark-clock");
  graph.set_default_clock(clock);

  CompileOptions compile_options;
  if (options.operator_lanes == OperatorLaneMode::kShared) {
    compile_options.deployment.bind_thread_placement(
        "runtime-graph-shared", make_placement(options.source_cpu, options.busy_poll));
  } else {
    compile_options.deployment.bind_thread_placement(
        "runtime-graph-source", make_placement(options.source_cpu, options.busy_poll));
    if (options.scenario == Scenario::kSampleForward) {
      compile_options.deployment.bind_thread_placement(
          "runtime-graph-relay", make_placement(options.relay_cpu, options.busy_poll));
    }
    if (options.operator_lanes == OperatorLaneMode::kPerSink || options.fanout > 1U) {
      for (std::uint32_t index = 0U; index < options.fanout; ++index) {
        const int cpu = options.operator_lanes == OperatorLaneMode::kPerSink
                            ? (options.sink_cpus.empty() ? -1 : options.sink_cpus[index])
                            : options.sink_cpu;
        compile_options.deployment.bind_thread_placement(
            "runtime-graph-sink-" + std::to_string(index + 1U),
            make_placement(cpu, options.busy_poll));
      }
    } else if (options.scenario != Scenario::kSourceOnly) {
      compile_options.deployment.bind_thread_placement(
          "runtime-graph-sink", make_placement(options.sink_cpu, options.busy_poll));
    }
  }
  if (options.scenario == Scenario::kTensor && options.tensor_storage == TensorStorage::kDevice) {
    compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = DeviceId{options.tensor_device},
    });
  }
  if (!options.force_holomq_process_local_inline) {
    return compile(graph, std::move(compile_options));
  }
  return detail::compile_with_process_local_holomq(graph, std::move(compile_options));
}

[[nodiscard]] std::optional<Percentiles> summarize_latencies(std::vector<std::uint64_t> values,
                                                             const bool include_p999) {
  if (values.empty()) {
    return std::nullopt;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&](const long double quantile) {
    std::size_t index =
        static_cast<std::size_t>(std::ceil(quantile * static_cast<long double>(values.size())));
    index = std::clamp(index, std::size_t{1U}, values.size()) - 1U;
    return values[index];
  };
  return Percentiles{
      .minimum = values.front(),
      .p50 = percentile(0.50L),
      .p90 = percentile(0.90L),
      .p99 = percentile(0.99L),
      .p999 = include_p999 ? std::optional<std::uint64_t>{percentile(0.999L)} : std::nullopt,
      .maximum = values.back(),
  };
}

void emit_optional_metric(std::ostream& output, const bool available, const double value) {
  if (available) {
    output << value;
  } else {
    output << "null";
  }
}

void emit_optional_cpu_array(std::ostream& output, const Options& options) {
  output << '[';
  const std::uint32_t count = options.scenario == Scenario::kSourceOnly ? 0U : options.fanout;
  for (std::uint32_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      output << ',';
    }
    if (options.operator_lanes == OperatorLaneMode::kShared) {
      emit_optional_cpu(output, options.source_cpu);
    } else if (options.operator_lanes == OperatorLaneMode::kPerSink) {
      emit_optional_cpu(output, options.sink_cpus.empty() ? -1 : options.sink_cpus[index]);
    } else {
      emit_optional_cpu(output, options.sink_cpu);
    }
  }
  output << ']';
}

[[nodiscard]] const char* operator_lane_mode_name(const OperatorLaneMode mode) noexcept {
  switch (mode) {
    case OperatorLaneMode::kSeparate:
      return "separate";
    case OperatorLaneMode::kShared:
      return "shared";
    case OperatorLaneMode::kPerSink:
      return "per_sink";
  }
  return "unknown";
}

void emit_operator_partition_uses(std::ostream& output, const ExecutionPlan& plan) {
  output << '[';
  bool first = true;
  for (const auto& partition : plan.execution_partitions()) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"partition\":\""
           << json_escape(partition.authored_name.value_or("implicit-" +
                                                           std::to_string(partition.id.value)))
           << "\",\"lane\":" << partition.execution_lane
           << ",\"operator_count\":" << partition.members.size() << '}';
  }
  output << ']';
}

void emit_percentiles(std::ostream& output, const std::vector<std::uint64_t>& samples,
                      const bool include_p999) {
  const std::optional<Percentiles> percentiles = summarize_latencies(samples, include_p999);
  if (!percentiles.has_value()) {
    output << "null";
    return;
  }
  output << "{\"samples\":" << samples.size() << ",\"min\":" << percentiles->minimum
         << ",\"p50\":" << percentiles->p50 << ",\"p90\":" << percentiles->p90
         << ",\"p99\":" << percentiles->p99 << ",\"p999\":";
  if (percentiles->p999.has_value()) {
    output << *percentiles->p999;
  } else {
    output << "null";
  }
  output << ",\"max\":" << percentiles->maximum << '}';
}

[[nodiscard]] std::vector<std::uint64_t> calibrate_timestamp_read_pairs() {
  constexpr std::size_t kCalibrationSamples = 4'096U;
  std::vector<std::uint64_t> samples;
  samples.reserve(kCalibrationSamples);
  for (std::size_t index = 0U; index < kCalibrationSamples; ++index) {
    const std::int64_t start_ns = now_ns();
    const std::int64_t end_ns = now_ns();
    samples.push_back(static_cast<std::uint64_t>(end_ns - start_ns));
  }
  return samples;
}

[[nodiscard]] const char* termination_name(const RunTermination termination) noexcept {
  switch (termination) {
    case RunTermination::kRunning:
      return "running";
    case RunTermination::kCompleted:
      return "completed";
    case RunTermination::kStopped:
      return "stopped";
    case RunTermination::kFailed:
      return "failed";
  }
  return "unknown";
}

[[nodiscard]] const char* partition_selection_source_name(
    const PlanSelectionSource source) noexcept {
  switch (source) {
    case PlanSelectionSource::kCoreDefault:
      return "core_default";
    case PlanSelectionSource::kCompilerPolicy:
      return "compiler_policy";
    case PlanSelectionSource::kCompilerService:
      return "compiler_service";
    case PlanSelectionSource::kTestOverride:
      return "test_override";
    case PlanSelectionSource::kEnvironmentOverride:
      return "environment_override";
    case PlanSelectionSource::kExplicitFallback:
      return "explicit_fallback";
  }
  return "unknown";
}

void emit_algorithm_key(std::ostream& output, const std::optional<AlgorithmKey>& algorithm) {
  if (!algorithm.has_value()) {
    output << "null";
    return;
  }
  output << "{\"id\":\"" << json_escape(algorithm->id) << "\",\"version\":" << algorithm->version
         << '}';
}

void emit_phase_counters(std::ostream& output, const detail::RuntimePhaseCounters* counters) {
  output << "  \"runtime_phase_counters\":{";
  if (counters == nullptr) {
    output << "\"enabled\":false},\n";
    return;
  }
  const auto snapshot = counters->snapshot();
  const auto value = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  output
      << "\"enabled\":true"
      << ",\"readiness_scans\":" << value(detail::RuntimePhaseCounter::kReadinessScans)
      << ",\"no_progress_scans\":" << value(detail::RuntimePhaseCounter::kNoProgressScans)
      << ",\"wait_calls\":" << value(detail::RuntimePhaseCounter::kWaitCalls)
      << ",\"wake_signals\":" << value(detail::RuntimePhaseCounter::kWakeSignals)
      << ",\"wake_returns\":" << value(detail::RuntimePhaseCounter::kWakeReturns)
      << ",\"timeout_returns\":" << value(detail::RuntimePhaseCounter::kTimeoutReturns)
      << ",\"stop_signals\":" << value(detail::RuntimePhaseCounter::kStopSignals)
      << ",\"candidates_frozen\":" << value(detail::RuntimePhaseCounter::kCandidatesFrozen)
      << ",\"candidates_selected\":" << value(detail::RuntimePhaseCounter::kCandidatesSelected)
      << ",\"readiness_poll_dispatches\":"
      << value(detail::RuntimePhaseCounter::kReadinessPollDispatches)
      << ",\"direct_continuation_receipts_accepted\":"
      << value(detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted)
      << ",\"direct_continuation_poll_elisions\":"
      << value(detail::RuntimePhaseCounter::kDirectContinuationPollElisions)
      << ",\"direct_composite_attempts\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeAttempts)
      << ",\"direct_composite_completions\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeCompletions)
      << ",\"direct_composite_fallbacks\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeFallbacks)
      << ",\"direct_composite_failures\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeFailures)
      << ",\"direct_composite_callbacks_attempted\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted)
      << ",\"direct_composite_callbacks_completed\":"
      << value(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted)
      << ",\"compute_dispatches\":" << value(detail::RuntimePhaseCounter::kComputeDispatches)
      << ",\"activations_completed\":" << value(detail::RuntimePhaseCounter::kActivationsCompleted)
      << ",\"channel_prepare_calls\":" << value(detail::RuntimePhaseCounter::kChannelPrepareCalls)
      << ",\"channel_emit_attempts\":" << value(detail::RuntimePhaseCounter::kChannelEmitAttempts)
      << ",\"channel_emit_accepted\":" << value(detail::RuntimePhaseCounter::kChannelEmitAccepted)
      << ",\"channel_emit_backpressured\":"
      << value(detail::RuntimePhaseCounter::kChannelEmitBackpressured)
      << ",\"channel_samples_received\":"
      << value(detail::RuntimePhaseCounter::kChannelSamplesReceived)
      << ",\"channel_readiness_published\":"
      << value(detail::RuntimePhaseCounter::kChannelReadinessPublished)
      << ",\"channel_input_reads\":" << value(detail::RuntimePhaseCounter::kChannelInputReads)
      << ",\"channel_witness_retired\":"
      << value(detail::RuntimePhaseCounter::kChannelWitnessRetired)
      << ",\"channel_retire_calls\":" << value(detail::RuntimePhaseCounter::kChannelRetireCalls)
      << ",\"channel_activation_cuts\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCuts)
      << ",\"channel_activation_cut_drains\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutDrains)
      << ",\"channel_activation_cut_skips\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutSkips)
      << ",\"channel_receive_backpressured\":"
      << value(detail::RuntimePhaseCounter::kChannelReceiveBackpressured)
      << ",\"lane_dispatches\":" << value(detail::RuntimePhaseCounter::kLaneDispatches)
      << ",\"lane_compute_dispatches\":"
      << value(detail::RuntimePhaseCounter::kLaneComputeDispatches)
      << ",\"lane_dispatcher_dispatches\":"
      << value(detail::RuntimePhaseCounter::kLaneDispatcherDispatches)
      << ",\"lane_lifecycle_dispatches\":"
      << value(detail::RuntimePhaseCounter::kLaneLifecycleDispatches)
      << ",\"lane_unknown_dispatches\":"
      << value(detail::RuntimePhaseCounter::kLaneUnknownDispatches)
      << ",\"lane_busy_task_publications\":"
      << value(detail::RuntimePhaseCounter::kLaneBusyTaskPublications)
      << ",\"lane_sleeping_wake_signals\":"
      << value(detail::RuntimePhaseCounter::kLaneSleepingWakeSignals)
      << ",\"lane_callbacks_entered\":" << value(detail::RuntimePhaseCounter::kLaneCallbacksEntered)
      << ",\"lane_callbacks_exited\":" << value(detail::RuntimePhaseCounter::kLaneCallbacksExited)
      << ",\"lane_completions_observed\":"
      << value(detail::RuntimePhaseCounter::kLaneCompletionsObserved)
      << ",\"lane_serialization_contentions\":"
      << value(detail::RuntimePhaseCounter::kLaneSerializationContentions)
      << ",\"lane_inline_owner_invocations\":"
      << value(detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations) << "},\n";
}

[[nodiscard]] constexpr std::string_view thread_usage_role_name(
    const detail::RuntimeThreadUsageRole role) noexcept {
  switch (role) {
    case detail::RuntimeThreadUsageRole::kController:
      return "controller";
    case detail::RuntimeThreadUsageRole::kExecutionLane:
      return "execution_lanes";
    case detail::RuntimeThreadUsageRole::kChannelPump:
      return "channel_pump";
    case detail::RuntimeThreadUsageRole::kNotificationDriver:
      return "notification_driver";
    case detail::RuntimeThreadUsageRole::kCount:
      break;
  }
  return "unknown";
}

void emit_thread_usage(std::ostream& output, const detail::RuntimePhaseCounters* counters) {
  output << "  \"runtime_thread_usage\":{";
  if (counters == nullptr) {
    output << "\"enabled\":false},\n";
    return;
  }
  const auto usage = counters->thread_usage_snapshot();
  detail::RuntimeThreadUsageRecord aggregate;
  const auto add = [](std::uint64_t& target, const std::uint64_t value) {
    target = value > std::numeric_limits<std::uint64_t>::max() - target
                 ? std::numeric_limits<std::uint64_t>::max()
                 : target + value;
  };
  for (const auto& record : usage) {
    add(aggregate.samples, record.samples);
    add(aggregate.user_cpu_ns, record.user_cpu_ns);
    add(aggregate.system_cpu_ns, record.system_cpu_ns);
    add(aggregate.voluntary_context_switches, record.voluntary_context_switches);
    add(aggregate.involuntary_context_switches, record.involuntary_context_switches);
    add(aggregate.minor_page_faults, record.minor_page_faults);
    add(aggregate.major_page_faults, record.major_page_faults);
  }
  output << "\"enabled\":true,\"supported\":"
         << (detail::runtime_thread_usage_detail::kSupported ? "true" : "false")
         << ",\"scope\":\"whole_runtime_thread_lifetime_including_startup_and_teardown\""
         << ",\"roles\":[";
  for (std::size_t index = 0U; index < usage.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const auto& record = usage[index];
    output << "{\"role\":\"" << thread_usage_role_name(record.role)
           << "\",\"supported\":" << (record.supported ? "true" : "false")
           << ",\"samples\":" << record.samples << ",\"user_cpu_ns\":" << record.user_cpu_ns
           << ",\"system_cpu_ns\":" << record.system_cpu_ns
           << ",\"voluntary_context_switches\":" << record.voluntary_context_switches
           << ",\"involuntary_context_switches\":" << record.involuntary_context_switches
           << ",\"minor_page_faults\":" << record.minor_page_faults
           << ",\"major_page_faults\":" << record.major_page_faults << '}';
  }
  output << "],\"aggregate\":{\"scope\":\"sum_of_runtime_roles\""
         << ",\"samples\":" << aggregate.samples << ",\"user_cpu_ns\":" << aggregate.user_cpu_ns
         << ",\"system_cpu_ns\":" << aggregate.system_cpu_ns
         << ",\"voluntary_context_switches\":" << aggregate.voluntary_context_switches
         << ",\"involuntary_context_switches\":" << aggregate.involuntary_context_switches
         << ",\"minor_page_faults\":" << aggregate.minor_page_faults
         << ",\"major_page_faults\":" << aggregate.major_page_faults << "}},\n";
}

void emit_phase_trace(std::ostream& output, const RuntimePhaseTraceSummary& trace,
                      const bool include_p999) {
  output << "  \"runtime_phase_trace\":{";
  if (!trace.enabled) {
    output << "\"enabled\":false},\n";
    return;
  }
  output << "\"enabled\":true,\"activation_trace_applicable\":"
         << (trace.activation_trace_applicable ? "true" : "false")
         << ",\"valid\":" << (trace.valid ? "true" : "false")
         << ",\"capacity_exhausted\":" << (trace.capacity_exhausted ? "true" : "false")
         << ",\"storage_invariants_valid\":" << (trace.storage_invariants_valid ? "true" : "false")
         << ",\"capacity\":" << trace.capacity
         << ",\"attempted_records\":" << trace.attempted_records << ",\"records\":" << trace.records
         << ",\"dropped\":" << trace.dropped << ",\"readiness_polls\":" << trace.readiness_polls
         << ",\"unmatched_readiness_polls\":" << trace.unmatched_readiness_polls
         << ",\"complete_activations\":" << trace.complete_activations
         << ",\"source_activation_runs\":" << trace.source_activation_runs
         << ",\"maximum_consecutive_source_activations\":"
         << trace.maximum_consecutive_source_activations
         << ",\"source_to_source_transitions\":" << trace.source_to_source_transitions
         << ",\"source_to_downstream_transitions\":" << trace.source_to_downstream_transitions
         << ",\"downstream_to_source_transitions\":" << trace.downstream_to_source_transitions
         << ",\"downstream_to_downstream_transitions\":"
         << trace.downstream_to_downstream_transitions
         << ",\"complete_tensor_emits\":" << trace.complete_tensor_emits
         << ",\"tensor_emit_total_ns\":";
  emit_percentiles(output, trace.tensor_emit_total, include_p999);
  output << ",\"tensor_emit_validation_ns\":";
  emit_percentiles(output, trace.tensor_emit_validation, include_p999);
  output << ",\"tensor_pre_transport_ns\":";
  emit_percentiles(output, trace.tensor_pre_transport, include_p999);
  output << ",\"tensor_transport_publication_ns\":";
  emit_percentiles(output, trace.tensor_transport_publication, include_p999);
  output << ",\"tensor_post_transport_ns\":";
  emit_percentiles(output, trace.tensor_post_transport, include_p999);
  output << ",\"complete_inline_payload_emits\":" << trace.complete_inline_payload_emits
         << ",\"inline_payload_emit_total_ns\":";
  emit_percentiles(output, trace.inline_payload_emit_total, include_p999);
  output << ",\"inline_payload_encoding_ns\":";
  emit_percentiles(output, trace.inline_payload_encoding, include_p999);
  output << ",\"inline_payload_pre_transport_ns\":";
  emit_percentiles(output, trace.inline_payload_pre_transport, include_p999);
  output << ",\"inline_payload_transport_publication_ns\":";
  emit_percentiles(output, trace.inline_payload_transport_publication, include_p999);
  output << ",\"inline_payload_post_transport_ns\":";
  emit_percentiles(output, trace.inline_payload_post_transport, include_p999);
  output << ",\"receivers\":[";
  for (std::size_t index = 0U; index < trace.receivers.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const RuntimePhaseReceiverSummary& receiver = trace.receivers[index];
    output << "{\"receiver_index\":" << receiver.receiver_index << ",\"role\":\""
           << (receiver.source ? "source" : "downstream") << "\",\"downstream_index\":";
    if (receiver.downstream_index.has_value()) {
      output << *receiver.downstream_index;
    } else {
      output << "null";
    }
    output << ",\"activations\":" << receiver.activations
           << ",\"previous_cut_to_readiness_poll_return_ns\":";
    emit_percentiles(output, receiver.previous_cut_to_readiness_poll_return, include_p999);
    output << ",\"readiness_poll_return_to_candidate_ns\":";
    emit_percentiles(output, receiver.readiness_poll_return_to_candidate, include_p999);
    output << ",\"candidate_to_dispatch_start_ns\":";
    emit_percentiles(output, receiver.candidate_to_dispatch_start, include_p999);
    output << ",\"compute_dispatch_ns\":";
    emit_percentiles(output, receiver.compute_dispatch, include_p999);
    output << ",\"dispatch_return_to_retirement_ns\":";
    emit_percentiles(output, receiver.dispatch_return_to_retirement, include_p999);
    output << ",\"retirement_to_cut_ns\":";
    emit_percentiles(output, receiver.retirement_to_cut, include_p999);
    output << ",\"dispatch_return_to_cut_ns\":";
    emit_percentiles(output, receiver.dispatch_return_to_cut, include_p999);
    output << ",\"callback_pre_return_to_dispatch_return_ns\":";
    emit_percentiles(output, receiver.callback_pre_return_to_dispatch_return, include_p999);
    output << ",\"dispatch_start_to_callback_entry_ns\":";
    emit_percentiles(output, receiver.dispatch_start_to_callback_entry, include_p999);
    output << '}';
  }
  output << "]},\n";
}

// Keep the stable evidence JSON fields together for review and downstream parsing.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_callback_thread_evidence(std::ostream& output, const BenchmarkRunState& state,
                                   const BenchmarkSupportThreadControl& thread_control) {
  const bool enabled = state.options().runtime_phase_counters;
  if (!enabled) {
    output << "  \"callback_thread_evidence\":{\"enabled\":false},\n";
    return;
  }
  std::optional<std::uint64_t> controller_native_tid;
  std::optional<bool> controller_identity_is_native;
  std::optional<bool> controller_native_tid_syscall_unavailable;
  for (const benchmarking::SupportThreadEvidence& evidence : thread_control.evidence()) {
    if (evidence.role == SupportThreadRole::kController) {
      controller_native_tid = evidence.native_tid;
      controller_identity_is_native = evidence.thread_identity_is_native;
      controller_native_tid_syscall_unavailable = evidence.native_tid_syscall_unavailable;
      break;
    }
  }
  const std::array callback_tids{
      state.source_callback_tid(), state.relay_callback_tid(), state.sink_callback_tid()};
  const std::array callback_identity_is_native{state.source_callback_identity_is_native(),
                                               state.relay_callback_identity_is_native(),
                                               state.sink_callback_identity_is_native()};
  const std::array callback_native_tid_syscall_unavailable{
      state.source_callback_native_tid_syscall_unavailable(),
      state.relay_callback_native_tid_syscall_unavailable(),
      state.sink_callback_native_tid_syscall_unavailable()};
  const std::array callback_counts{
      state.source_compute_entries(), state.relay_callbacks(), state.sink_callbacks()};
  const std::array callback_multiple_threads{state.source_callback_multiple_threads(),
                                             state.relay_callback_multiple_threads(),
                                             state.sink_callback_multiple_threads()};
  const std::array<std::string_view, 3U> role_names{"source", "relay", "sink"};
  bool valid = enabled && controller_native_tid.has_value() &&
               controller_identity_is_native.has_value() &&
               controller_native_tid_syscall_unavailable.has_value();
  std::optional<std::uint64_t> shared_tid;
  std::optional<bool> shared_identity_is_native;
  std::optional<bool> shared_native_tid_syscall_unavailable;
  bool shared_lane_single_thread =
      enabled && state.options().operator_lanes == OperatorLaneMode::kShared;
  for (std::size_t index = 0U; index < callback_tids.size(); ++index) {
    if (callback_counts[index] == 0U) {
      continue;
    }
    valid = valid && callback_tids[index] != 0U;
    if (callback_multiple_threads[index]) {
      shared_lane_single_thread = false;
    }
    if (!shared_tid.has_value()) {
      shared_tid = callback_tids[index];
      shared_identity_is_native = callback_identity_is_native[index];
      shared_native_tid_syscall_unavailable = callback_native_tid_syscall_unavailable[index];
    } else if (*shared_tid != callback_tids[index]) {
      shared_lane_single_thread = false;
    }
    if (shared_identity_is_native != callback_identity_is_native[index]) {
      shared_lane_single_thread = false;
    }
    if (shared_native_tid_syscall_unavailable != callback_native_tid_syscall_unavailable[index]) {
      shared_lane_single_thread = false;
    }
  }
  const bool identity_kind_matches =
      shared_identity_is_native.has_value() && controller_identity_is_native.has_value() &&
      shared_native_tid_syscall_unavailable.has_value() &&
      controller_native_tid_syscall_unavailable.has_value() &&
      shared_identity_is_native == controller_identity_is_native &&
      shared_native_tid_syscall_unavailable == controller_native_tid_syscall_unavailable;
  valid = valid && shared_tid.has_value() && identity_kind_matches;
  const bool callbacks_on_partition_owner =
      shared_lane_single_thread && controller_native_tid.has_value() && identity_kind_matches &&
      shared_tid == controller_native_tid;
  const std::string_view identity_kind = identity_kind_matches && *shared_identity_is_native
                                             ? "native_tid"
                                         : identity_kind_matches ? "std_thread_id_hash"
                                                                 : "mixed";

  output << "  \"callback_thread_evidence\":{\"enabled\":" << (enabled ? "true" : "false")
         << ",\"valid\":" << (valid ? "true" : "false") << ",\"thread_identity_kind\":\""
         << identity_kind << "\",\"native_tid_syscall_unavailable\":"
         << (controller_native_tid_syscall_unavailable.value_or(false) ? "true" : "false")
         << ",\"controller_native_tid\":";
  if (controller_native_tid.has_value()) {
    output << *controller_native_tid;
  } else {
    output << "null";
  }
  output << ",\"shared_lane_single_thread\":" << (shared_lane_single_thread ? "true" : "false")
         << ",\"callbacks_on_partition_owner\":"
         << (callbacks_on_partition_owner ? "true" : "false") << ",\"roles\":[";
  for (std::size_t index = 0U; index < callback_tids.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"role\":\"" << role_names[index] << "\",\"callbacks\":" << callback_counts[index]
           << ",\"native_tid\":";
    if (callback_tids[index] != 0U) {
      output << callback_tids[index];
    } else {
      output << "null";
    }
    output << ",\"multiple_threads\":" << (callback_multiple_threads[index] ? "true" : "false")
           << '}';
  }
  output << "]},\n";
}

[[nodiscard]] bool phase_counters_valid(const BenchmarkRunState& state,
                                        const detail::RuntimePhaseCounters* counters) noexcept {
  if (counters == nullptr) {
    return true;
  }
  const auto snapshot = counters->snapshot();
  const auto value = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  const std::uint64_t expected_compute_dispatches =
      state.source_compute_entries() + state.relay_callbacks() + state.sink_callbacks();
  const std::uint64_t expected_emit_attempts = state.source_callbacks() + state.relay_callbacks();
  const std::uint64_t expected_accepted_emits = state.publications() + state.relay_publications();
  const std::uint64_t expected_input_operations = state.relay_callbacks() + state.sink_callbacks();
  const std::uint64_t frozen = value(detail::RuntimePhaseCounter::kCandidatesFrozen);
  const bool runtime_valid = frozen == expected_compute_dispatches &&
                             value(detail::RuntimePhaseCounter::kCandidatesSelected) == frozen &&
                             value(detail::RuntimePhaseCounter::kComputeDispatches) == frozen &&
                             value(detail::RuntimePhaseCounter::kActivationsCompleted) == frozen &&
                             value(detail::RuntimePhaseCounter::kWakeReturns) +
                                     value(detail::RuntimePhaseCounter::kTimeoutReturns) ==
                                 value(detail::RuntimePhaseCounter::kWaitCalls) &&
                             value(detail::RuntimePhaseCounter::kNoProgressScans) <=
                                 value(detail::RuntimePhaseCounter::kReadinessScans);
  if (!runtime_valid) {
    return false;
  }
  const std::uint64_t composite_attempts =
      value(detail::RuntimePhaseCounter::kDirectCompositeAttempts);
  const std::uint64_t composite_completions =
      value(detail::RuntimePhaseCounter::kDirectCompositeCompletions);
  const std::uint64_t composite_fallbacks =
      value(detail::RuntimePhaseCounter::kDirectCompositeFallbacks);
  const std::uint64_t composite_failures =
      value(detail::RuntimePhaseCounter::kDirectCompositeFailures);
  const std::uint64_t composite_callbacks_attempted =
      value(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted);
  const std::uint64_t composite_callbacks_completed =
      value(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted);
  const bool composite_valid =
      composite_callbacks_attempted <= expected_compute_dispatches &&
      composite_callbacks_completed <= composite_callbacks_attempted &&
      composite_callbacks_attempted >= composite_attempts &&
      composite_callbacks_attempted - composite_attempts <= composite_attempts &&
      composite_completions + composite_fallbacks + composite_failures <= composite_attempts &&
      (!state.options().disable_closed_island_composite ||
       (composite_attempts == 0U && composite_completions == 0U && composite_fallbacks == 0U &&
        composite_failures == 0U && composite_callbacks_attempted == 0U &&
        composite_callbacks_completed == 0U));
  if (!composite_valid) {
    return false;
  }
  const std::uint64_t lane_dispatches = value(detail::RuntimePhaseCounter::kLaneDispatches);
  const std::uint64_t lane_compute_dispatches =
      value(detail::RuntimePhaseCounter::kLaneComputeDispatches);
  const std::uint64_t lane_dispatcher_dispatches =
      value(detail::RuntimePhaseCounter::kLaneDispatcherDispatches);
  const std::uint64_t inline_owner_invocations =
      value(detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations);
  const std::uint64_t expected_lane_compute_dispatches =
      expected_compute_dispatches - composite_callbacks_attempted;
  const std::uint64_t expected_lane_dispatcher_dispatches =
      value(detail::RuntimePhaseCounter::kReadinessPollDispatches) + composite_attempts;
  const std::uint64_t lane_lifecycle_dispatches =
      value(detail::RuntimePhaseCounter::kLaneLifecycleDispatches);
  const std::uint64_t lane_unknown_dispatches =
      value(detail::RuntimePhaseCounter::kLaneUnknownDispatches);
  const std::uint64_t lane_busy_publications =
      value(detail::RuntimePhaseCounter::kLaneBusyTaskPublications);
  const std::uint64_t lane_sleeping_wakes =
      value(detail::RuntimePhaseCounter::kLaneSleepingWakeSignals);
  const std::uint64_t lane_callbacks_entered =
      value(detail::RuntimePhaseCounter::kLaneCallbacksEntered);
  const std::uint64_t lane_callbacks_exited =
      value(detail::RuntimePhaseCounter::kLaneCallbacksExited);
  const std::uint64_t lane_completions =
      value(detail::RuntimePhaseCounter::kLaneCompletionsObserved);
  const bool ordinary_lane_valid =
      inline_owner_invocations == 0U &&
      lane_compute_dispatches == expected_lane_compute_dispatches &&
      lane_dispatcher_dispatches == expected_lane_dispatcher_dispatches &&
      lane_dispatches ==
          lane_compute_dispatches + lane_dispatcher_dispatches + lane_lifecycle_dispatches &&
      lane_unknown_dispatches == 0U && lane_callbacks_entered == lane_dispatches &&
      lane_callbacks_exited == lane_dispatches && lane_completions == lane_dispatches &&
      (state.options().busy_poll
           ? lane_busy_publications == lane_dispatches && lane_sleeping_wakes == 0U
           : lane_sleeping_wakes == lane_dispatches && lane_busy_publications == 0U);
  const bool inline_lane_valid =
      inline_owner_invocations != 0U && lane_compute_dispatches == 0U &&
      lane_dispatcher_dispatches == 0U && lane_dispatches == lane_lifecycle_dispatches &&
      lane_unknown_dispatches == 0U &&
      (state.options().busy_poll
           ? lane_busy_publications == lane_dispatches && lane_sleeping_wakes == 0U
           : lane_sleeping_wakes == lane_dispatches && lane_busy_publications == 0U) &&
      lane_callbacks_entered == inline_owner_invocations + lane_dispatches &&
      lane_callbacks_exited == inline_owner_invocations + lane_dispatches &&
      lane_completions == lane_dispatches &&
      inline_owner_invocations >=
          expected_lane_compute_dispatches + expected_lane_dispatcher_dispatches;
  const bool lane_valid = ordinary_lane_valid || inline_lane_valid;
  if (!lane_valid) {
    return false;
  }
  if (state.options().scenario == Scenario::kSourceOnly) {
    return value(detail::RuntimePhaseCounter::kChannelPrepareCalls) == 0U &&
           value(detail::RuntimePhaseCounter::kChannelEmitAttempts) == 0U &&
           value(detail::RuntimePhaseCounter::kChannelSamplesReceived) == 0U;
  }
  const std::uint64_t activation_cut_drains =
      value(detail::RuntimePhaseCounter::kChannelActivationCutDrains);
  const std::uint64_t activation_cut_skips =
      value(detail::RuntimePhaseCounter::kChannelActivationCutSkips);
  const bool activation_cut_valid =
      uses_channel_pump(state.options())
          ? activation_cut_drains + activation_cut_skips == expected_compute_dispatches
          : activation_cut_drains == 0U && activation_cut_skips == 0U;
  const std::uint64_t expected_activation_cuts =
      uses_channel_pump(state.options()) ? expected_compute_dispatches : 0U;
  return value(detail::RuntimePhaseCounter::kChannelPrepareCalls) == expected_compute_dispatches &&
         value(detail::RuntimePhaseCounter::kChannelEmitAttempts) == expected_emit_attempts &&
         value(detail::RuntimePhaseCounter::kChannelEmitAccepted) == expected_accepted_emits &&
         value(detail::RuntimePhaseCounter::kChannelEmitBackpressured) ==
             state.backpressure_count() &&
         value(detail::RuntimePhaseCounter::kChannelSamplesReceived) == expected_input_operations &&
         value(detail::RuntimePhaseCounter::kChannelReadinessPublished) ==
             expected_input_operations &&
         value(detail::RuntimePhaseCounter::kChannelInputReads) == expected_input_operations &&
         value(detail::RuntimePhaseCounter::kChannelWitnessRetired) == expected_input_operations &&
         value(detail::RuntimePhaseCounter::kChannelRetireCalls) == expected_compute_dispatches &&
         value(detail::RuntimePhaseCounter::kChannelActivationCuts) == expected_activation_cuts &&
         activation_cut_valid;
}

// Keep the benchmark's stable JSON schema visible in one emitter.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_result(std::ostream& output, const BenchmarkRunState& state, const ExecutionPlan& plan,
                 const RunSession& session, const BenchmarkSupportThreadControl& thread_control,
                 const detail::RuntimePhaseCounters* phase_counters,
                 const RuntimePhaseTraceSummary& phase_trace,
                 const std::optional<std::uint64_t> final_validation_ns) {
  const Options& options = state.options();
  const std::int64_t start_ns = state.measured_start_ns();
  const std::int64_t end_ns = state.measured_end_ns();
  const std::uint64_t duration_ns =
      start_ns > 0 && end_ns > start_ns ? static_cast<std::uint64_t>(end_ns - start_ns) : 0U;
  const double publications_per_second =
      duration_ns == 0U || options.scenario == Scenario::kSourceOnly
          ? 0.0
          : static_cast<double>(options.samples) * 1.0e9 / static_cast<double>(duration_ns);
  const double aggregate_deliveries_per_second =
      duration_ns == 0U ? 0.0
                        : static_cast<double>(state.measured_deliveries()) * 1.0e9 /
                              static_cast<double>(duration_ns);
  const double logical_payload_bytes_per_second =
      publications_per_second * static_cast<double>(payload_size(options));
  const std::optional<std::size_t> transport_bytes = inline_transport_bytes(options);
  const double aggregate_payload_bytes_per_second =
      aggregate_deliveries_per_second * static_cast<double>(payload_size(options));
  const std::uint64_t consumer_visible_replacements = state.consumer_visible_replacements();
  const double replacement_rate = state.publications() == 0U
                                      ? 0.0
                                      : static_cast<double>(consumer_visible_replacements) /
                                            static_cast<double>(state.publications());
  const bool throughput_metrics_available = options.workload == Workload::kSaturated;
  const double configured_publication_rate_per_second =
      is_latency_workload(options.workload)
          ? 1.0e6 / static_cast<double>(options.publisher_period_us)
          : 0.0;
  const double source_activations_per_second =
      duration_ns == 0U ? 0.0
                        : static_cast<double>(state.measured_source_callbacks()) * 1.0e9 /
                              static_cast<double>(duration_ns);
  const std::vector<std::uint64_t> latencies = state.latencies();
  const std::vector<std::uint64_t> notification_to_source_latencies =
      state.notification_to_source_latencies();
  const std::vector<std::uint64_t> source_callback_entry_to_pre_emit_latencies =
      state.source_callback_entry_to_pre_emit_latencies();
  const std::vector<std::uint64_t> tensor_loan_acquire_latencies =
      state.tensor_loan_acquire_latencies();
  const std::vector<std::uint64_t> tensor_write_commit_latencies =
      state.tensor_write_commit_latencies();
  const std::vector<std::uint64_t> tensor_receive_call_latencies =
      state.tensor_receive_call_latencies();
  const std::vector<std::uint64_t> source_emit_latencies = state.source_emit_latencies();
  const std::vector<std::uint64_t> source_publication_call_latencies =
      state.source_publication_call_latencies();
  const std::vector<std::uint64_t> source_post_emit_to_pre_return_latencies =
      state.source_post_emit_to_pre_return_latencies();
  const std::vector<std::uint64_t> source_post_emit_to_sink_latencies =
      state.source_post_emit_to_sink_latencies();
  const std::vector<std::uint64_t> source_to_sink_latencies = state.source_to_sink_latencies();
  const std::vector<std::uint64_t> publication_to_receive_latencies =
      state.publication_to_receive_latencies();
  const bool include_p999 = options.samples >= 10'000U;
  const std::vector<std::uint64_t> timestamp_read_pair_overhead = calibrate_timestamp_read_pairs();

  output
      << "{\n"
      << "  \"schema\":\"holoscan.ea1.runtime-benchmark/v1\",\n"
      << "  \"benchmark\":\"" << benchmark_name(options) << "\",\n"
      << "  \"answers\":\"" << benchmark_answer(options.scenario) << "\",\n"
      << "  \"scenario\":\"" << scenario_name(options.scenario) << "\",\n"
      << "  \"process_local_inline_delivery\":\"" << process_local_delivery_name(options) << "\",\n"
      << "  \"runtime_controls\":{\"direct_continuation_elision_disabled\":"
      << (options.disable_direct_continuation_elision ? "true" : "false")
      << ",\"closed_island_composite_disabled\":"
      << (options.disable_closed_island_composite ? "true" : "false") << "},\n"
      << "  \"queue_policy\":\"" << queue_policy_name(options.queue_policy) << "\",\n"
      << "  \"payload\":\""
      << (options.scenario == Scenario::kSourceOnly
              ? "none"
              : (options.scenario == Scenario::kSignal
                     ? "trigger_token"
                     : (options.scenario == Scenario::kTensor
                            ? "tensor"
                            : (options.scenario == Scenario::kImage
                                   ? "image"
                                   : (options.scenario == Scenario::kMultiTensor
                                          ? "multi_tensor"
                                          : (options.scenario == Scenario::kSchema ? "schema"
                                                                                   : "pod"))))))
      << "\",\n"
      << "  \"workload\":\"" << workload_name(options.workload) << "\",\n"
      << "  \"source_trigger\":\""
      << (is_notification_workload(options.workload) ? "notification" : "clock") << "\",\n"
      << "  \"process_id\":" << process_id() << ",\n"
      << "  \"clock\":{\"source\":\"std::chrono::steady_clock\",\"steady\":"
      << (BenchmarkClock::is_steady ? "true" : "false") << ",\"scope\":\"single_process\"},\n"
      << "  \"timestamp_instrumentation\":{\"calibration\":\"post_measurement_back_to_back_pair\""
      << ",\"clock_reads_per_sample\":2,\"pair_overhead_ns\":";
  emit_percentiles(output, timestamp_read_pair_overhead, false);
  output << "},\n"
         << "  \"partitioning\":{\"requested_algorithm\":";
  const PartitioningPlanRecord& partitioning = plan.partitioning_plan();
  const TensorAllocationPlanRecord* tensor_allocation =
      plan.tensor_allocations().empty() ? nullptr : &plan.tensor_allocations().front();
  const TensorFieldAllocationPlanRecord* tensor_field_allocation =
      plan.tensor_field_allocations().empty() ? nullptr : &plan.tensor_field_allocations().front();
  const bool bulk_payload = options.scenario == Scenario::kTensor ||
                            options.scenario == Scenario::kImage ||
                            options.scenario == Scenario::kMultiTensor;
  const std::uint32_t bulk_pool_slots =
      tensor_allocation != nullptr
          ? tensor_allocation->slot_count
          : (tensor_field_allocation != nullptr ? tensor_field_allocation->slot_count
                                                : (bulk_payload ? 1U : 0U));
  const std::uint32_t bulk_release_capacity =
      tensor_allocation != nullptr
          ? tensor_allocation->release_capacity
          : (tensor_field_allocation != nullptr ? tensor_field_allocation->release_capacity
                                                : (bulk_payload ? 1U : 0U));
  emit_algorithm_key(output, partitioning.requested_algorithm);
  output << ",\"accepted_algorithm\":";
  emit_algorithm_key(output, partitioning.accepted_algorithm);
  output << ",\"selection_source\":\""
         << partition_selection_source_name(partitioning.selection_source) << "\"},\n"
         << "  \"geometry\":{\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
         << ",\"allocation_probe_samples\":1" << ",\"queue_depth\":" << options.queue_depth
         << ",\"notification_capacity\":"
         << (is_notification_workload(options.workload) ? options.notification_capacity : 0U)
         << ",\"publication_backpressure_capacity\":" << options.queue_depth
         << ",\"fanout\":" << (options.scenario == Scenario::kSourceOnly ? 0U : options.fanout)
         << ",\"relay_count\":"
         << (options.scenario == Scenario::kSampleForward ? options.relay_count : 0U)
         << ",\"pipeline_edges\":" << edge_count(options)
         << ",\"payload_bytes\":" << payload_size(options)
         << ",\"application_payload_bytes\":" << payload_size(options)
         << ",\"inline_transport_bytes\":";
  if (transport_bytes) {
    output << *transport_bytes;
  } else {
    output << "null";
  }
  output << ",\"external_buffer_bytes\":" << external_buffer_bytes(options) << ",\"pool_slots\":";
  if (bulk_payload) {
    output << bulk_pool_slots;
  } else {
    output << "null";
  }
  output << ",\"release_capacity\":";
  if (bulk_payload) {
    output << bulk_release_capacity;
  } else {
    output << "null";
  }
  output << ",\"max_fanout_release_streams\":";
  if (bulk_payload) {
    output << options.fanout;
  } else {
    output << "null";
  }
  output << ",\"payload_write_profile\":\"" << payload_write_profile(options)
         << "\",\"bulk_work_profile\":\"" << bulk_work_profile(options)
         << "\",\"publisher_period_us\":";
  if (is_latency_workload(options.workload)) {
    output << options.publisher_period_us;
  } else {
    output << "null";
  }
  output << ",\"source_trigger_period_ns\":";
  if (options.workload == Workload::kLatency) {
    output << state.source_period().count();
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"tensor\":{";
  if (bulk_payload) {
    const bool fused_linear_stream =
        std::any_of(plan.tensor_device_resources().begin(),
                    plan.tensor_device_resources().end(),
                    [](const TensorDeviceResourcePlanRecord& resource) {
                      return !resource.fused_linear_edges.empty();
                    });
    output << "\"storage\":\""
           << tensor_storage_name(options.scenario == Scenario::kImage ||
                                          options.scenario == Scenario::kMultiTensor
                                      ? TensorStorage::kPageableHost
                                      : options.tensor_storage)
           << "\",\"device\":"
           << (options.scenario == Scenario::kImage || options.scenario == Scenario::kMultiTensor
                   ? 0
                   : options.tensor_device)
           << ",\"direct_allocation_records\":"
           << (options.scenario == Scenario::kMultiTensor ? plan.tensor_field_allocations().size()
                                                          : plan.tensor_allocations().size())
           << ",\"device_resource_records\":" << plan.tensor_device_resources().size()
           << ",\"fused_linear_stream\":" << (fused_linear_stream ? "true" : "false")
           << ",\"pointer_identity_checked\":true,\"descriptor_schema\":"
           << (options.scenario == Scenario::kImage || options.scenario == Scenario::kMultiTensor
                   ? "true"
                   : "false");
    if (options.scenario == Scenario::kMultiTensor) {
      output << ",\"native_type\":\"runtime_benchmark::MultiTensorMessageT\""
             << ",\"field_count\":2,\"fields\":["
             << "{\"name\":\"first\",\"dtype\":\"uint8\",\"rank\":1,\"bytes\":"
             << options.tensor_bytes << "},"
             << "{\"name\":\"second\",\"dtype\":\"uint8\",\"rank\":1,\"bytes\":"
             << options.tensor_bytes << "}]";
    }
  }
  output << "},\n"
         << "  \"wake\":{\"event_wake\":" << (uses_channel_pump(options) ? "true" : "false")
         << ",\"busy_poll\":" << (options.busy_poll ? "true" : "false")
         << ",\"receive_polling\":false},\n"
         << "  \"placement\":{\"wake_mode\":\""
         << (options.busy_poll ? "busy_poll" : "sleeping_waitset") << "\",\"operator_lanes\":\""
         << operator_lane_mode_name(options.operator_lanes)
         << "\",\"operator_lane_count\":" << plan.execution_lanes().size()
         << ",\"source_compile_accepted_cpu\":";
  emit_optional_cpu(output, options.source_cpu);
  output << ",\"relay_compile_accepted_cpu\":";
  if (options.scenario == Scenario::kSampleForward) {
    emit_optional_cpu(output,
                      options.operator_lanes == OperatorLaneMode::kShared ? options.source_cpu
                                                                          : options.relay_cpu);
  } else {
    output << "null";
  }
  output << ",\"sink_compile_accepted_cpu\":";
  if (options.scenario == Scenario::kSourceOnly ||
      options.operator_lanes == OperatorLaneMode::kPerSink) {
    output << "null";
  } else {
    emit_optional_cpu(output,
                      options.operator_lanes == OperatorLaneMode::kShared ? options.source_cpu
                                                                          : options.sink_cpu);
  }
  output << ",\"sink_compile_accepted_cpus\":";
  emit_optional_cpu_array(output, options);
  output << ",\"operator_partition_uses\":";
  emit_operator_partition_uses(output, plan);
  output << ",\"support_threads\":";
  emit_support_thread_evidence(output, thread_control);
  output << "},\n"
         << "  \"timing\":{\"measured_start_ns\":" << start_ns << ",\"measured_end_ns\":" << end_ns
         << ",\"completion_window_ns\":" << duration_ns
         << ",\"source_activations_per_second\":" << source_activations_per_second
         << ",\"configured_publication_rate_per_second\":";
  emit_optional_metric(
      output, is_latency_workload(options.workload), configured_publication_rate_per_second);
  output << ",\"logical_publications_per_second\":";
  emit_optional_metric(output, throughput_metrics_available, publications_per_second);
  output << ",\"aggregate_deliveries_per_second\":";
  emit_optional_metric(output, throughput_metrics_available, aggregate_deliveries_per_second);
  output << ",\"logical_payload_bytes_per_second\":";
  emit_optional_metric(output, throughput_metrics_available, logical_payload_bytes_per_second);
  output << ",\"aggregate_payload_bytes_per_second\":";
  emit_optional_metric(output, throughput_metrics_available, aggregate_payload_bytes_per_second);
  output << ",\"latency_ns\":";
  emit_percentiles(output, latencies, include_p999);
  output << ",\"notification_to_source_ns\":";
  emit_percentiles(output, notification_to_source_latencies, include_p999);
  output << ",\"source_callback_entry_to_pre_emit_ns\":";
  emit_percentiles(output, source_callback_entry_to_pre_emit_latencies, include_p999);
  output << ",\"tensor_loan_acquire_ns\":";
  emit_percentiles(output, tensor_loan_acquire_latencies, include_p999);
  output << ",\"tensor_write_commit_ns\":";
  emit_percentiles(output, tensor_write_commit_latencies, include_p999);
  output << ",\"tensor_receive_call_ns\":";
  emit_percentiles(output, tensor_receive_call_latencies, include_p999);
  output << ",\"source_emit_ns\":";
  emit_percentiles(output, source_emit_latencies, include_p999);
  output << ",\"source_publication_call_ns\":";
  emit_percentiles(output, source_publication_call_latencies, include_p999);
  output << ",\"source_post_emit_to_pre_return_ns\":";
  emit_percentiles(output, source_post_emit_to_pre_return_latencies, include_p999);
  output << ",\"source_post_emit_to_sink_ns\":";
  emit_percentiles(output, source_post_emit_to_sink_latencies, include_p999);
  output << ",\"source_post_emit_to_sink_overlaps\":" << state.source_post_emit_to_sink_overlaps();
  output << ",\"source_to_sink_ns\":";
  emit_percentiles(output, source_to_sink_latencies, include_p999);
  output << ",\"publication_to_receive_ns\":";
  emit_percentiles(output, publication_to_receive_latencies, include_p999);
  output << ",\"source_to_sink_by_sink_ns\":[";
  const std::uint32_t timing_fanout =
      options.scenario == Scenario::kSourceOnly ? 0U : options.fanout;
  for (std::uint32_t index = 0U; index < timing_fanout; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index << ",\"latency_ns\":";
    emit_percentiles(output, state.source_to_sink_latencies(index), include_p999);
    output << '}';
  }
  output << ']';
  output << ",\"source_post_emit_to_sink_by_sink_ns\":[";
  for (std::uint32_t index = 0U; index < timing_fanout; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index
           << ",\"overlaps\":" << state.source_post_emit_to_sink_overlaps(index)
           << ",\"latency_ns\":";
    emit_percentiles(output, state.source_post_emit_to_sink_latencies(index), include_p999);
    output << '}';
  }
  output << ']';
  output << ",\"source_pre_return_to_sink_callback_by_sink_ns\":[";
  for (std::uint32_t index = 0U; index < timing_fanout; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index
           << ",\"overlaps\":" << state.source_pre_return_to_sink_callback_overlaps(index)
           << ",\"latency_ns\":";
    emit_percentiles(
        output, state.source_pre_return_to_sink_callback_latencies(index), include_p999);
    output << '}';
  }
  output << ']';
  output << ",\"sink_callback_to_post_receive_by_sink_ns\":[";
  for (std::uint32_t index = 0U; index < timing_fanout; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index << ",\"latency_ns\":";
    emit_percentiles(output, state.sink_callback_to_post_receive_latencies(index), include_p999);
    output << '}';
  }
  output << ']';
  output << "},\n"
         << "  \"counts\":{\"source_callbacks\":" << state.source_callbacks()
         << ",\"source_compute_entries\":" << state.source_compute_entries()
         << ",\"measured_source_callbacks\":" << state.measured_source_callbacks()
         << ",\"relay_callbacks\":" << state.relay_callbacks()
         << ",\"sink_callbacks\":" << state.sink_callbacks()
         << ",\"publications\":" << state.publications()
         << ",\"relay_publications\":" << state.relay_publications()
         << ",\"deliveries\":" << state.deliveries()
         << ",\"consumer_visible_replacements\":" << consumer_visible_replacements
         << ",\"replacement_rate\":" << replacement_rate
         << ",\"activation_completions\":" << state.activation_completions()
         << ",\"backpressured_emits\":" << state.backpressure_count()
         << ",\"source_backpressured_emits\":" << state.source_backpressure_count()
         << ",\"descriptor_backing_initializations\":" << state.descriptor_backing_initializations()
         << ",\"descriptor_backing_initializations_during_measurement\":"
         << state.descriptor_backing_initializations_during_measurement()
         << ",\"notification_post_attempts\":" << state.notification_attempts()
         << ",\"notifications_accepted\":" << state.notifications_accepted()
         << ",\"notification_backpressure\":" << state.notification_backpressure()
         << ",\"notification_not_ready\":" << state.notification_not_ready()
         << ",\"validation_errors\":" << state.validation_errors() << ",\"per_sink\":[";
  const std::uint32_t reported_fanout =
      options.scenario == Scenario::kSourceOnly ? 0U : options.fanout;
  for (std::uint32_t index = 0U; index < reported_fanout; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index << ",\"deliveries\":" << state.sink_deliveries(index)
           << ",\"final_sequence\":" << state.sink_final_sequence(index) << '}';
  }
  output << "]},\n"
         << "  \"allocation_probe\":{\"excluded_from_measurement\":true"
         << ",\"source_compute_allocations\":" << state.source_probe_allocations()
         << ",\"source_compute_allocation_limit\":" << state.source_allocation_limit()
         << ",\"relay_compute_allocations\":" << state.relay_probe_allocations()
         << ",\"relay_compute_allocation_limit\":" << state.relay_allocation_limit()
         << ",\"sink_compute_allocations\":" << state.sink_probe_allocations()
         << ",\"sink_compute_allocation_limit\":" << state.sink_allocation_limit() << "},\n";
  emit_phase_counters(output, phase_counters);
  emit_thread_usage(output, phase_counters);
  emit_phase_trace(output, phase_trace, include_p999);
  emit_callback_thread_evidence(output, state, thread_control);
  output << "  \"validation\":{\"mode\":\"" << payload_validation_name(options.validation)
         << "\",\"timing\":\"" << validation_timing_name(options.validation)
         << "\",\"passed\":" << (state.validation_errors() == 0U ? "true" : "false")
         << ",\"final_validation_ns\":";
  if (final_validation_ns) {
    output << *final_validation_ns;
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"session\":{\"termination\":\"" << termination_name(session.termination())
         << "\",\"diagnostics\":[";
  bool first = true;
  for (const Diagnostic& diagnostic : session.diagnostics()) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << "{\"token\":\"" << json_escape(diagnostic.token) << "\",\"service\":\""
           << json_escape(diagnostic.service) << "\",\"endpoint\":\""
           << json_escape(diagnostic.endpoint) << "\",\"required\":\""
           << json_escape(diagnostic.required) << "\",\"provided\":\""
           << json_escape(diagnostic.provided) << "\"}";
  }
  output << "]}\n}\n";
}

template <typename Integer>
[[nodiscard]] bool parse_integer(const std::string_view text, Integer& output) noexcept {
  const char* const begin = text.data();
  const char* const end = text.data() + text.size();
  const auto [position, error] = std::from_chars(begin, end, output);
  return error == std::errc() && position == end;
}

[[nodiscard]] bool parse_cpu_list(std::string_view text, std::vector<int>& output) {
  output.clear();
  while (!text.empty()) {
    const std::size_t separator = text.find(',');
    const std::string_view token = text.substr(0U, separator);
    int cpu{};
    if (token.empty() || !parse_integer(token, cpu)) {
      return false;
    }
    output.push_back(cpu);
    if (separator == std::string_view::npos) {
      return true;
    }
    text.remove_prefix(separator + 1U);
  }
  return false;
}

[[nodiscard]] bool consume_value(const int argc, char** argv, int& index, std::string_view& value) {
  if (index + 1 >= argc) {
    return false;
  }
  value = argv[++index];
  return true;
}

void print_usage(std::ostream& output) {
  output
      << "runtime_graph_benchmark [options]\n\n"
      << "  --workload <latency|notification_latency|saturated>\n"
      << "  --scenario <source_only|pod|signal|sample_forward|schema|tensor|image|multi_tensor>\n"
      << "  --operator-lanes <separate|shared|per_sink>\n"
      << "  --tensor-storage <pageable_host|pinned_host|device>\n"
      << "  --bulk-work-profile <full_write|descriptor_only>\n"
      << "  --tensor-bytes <n>\n"
      << "  --tensor-device <n>\n"
      << "  --samples <n>\n"
      << "  --warmup <n>\n"
      << "  --queue-policy <blocking|drop_oldest>\n"
      << "  --queue-depth <n>\n"
      << "  --relay-count <n>\n"
      << "  --fanout <n>\n"
      << "  --notification-capacity <n>\n"
      << "  --publisher-period-us <n>\n"
      << "  --payload-validation <none|final|all>\n"
      << "  --source-cpu <n>\n"
      << "  --relay-cpu <n>\n"
      << "  --sink-cpu <n>\n"
      << "  --sink-cpus <n,n,...>\n"
      << "  --controller-cpu <n>\n"
      << "  --channel-pump-cpu <n>\n"
      << "  --notification-driver-cpu <n>\n"
      << "  --busy-poll\n"
      << "  --runtime-phase-counters\n"
      << "  --disable-direct-continuation-elision\n"
      << "  --disable-closed-island-composite\n"
      << "  --force-holomq-process-local-inline\n"
      << "  --timeout-ms <n>\n"
      << "  --output <path>\n";
}

// Keep command-line validation adjacent to each option's parsing branch.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<Options> parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      print_usage(std::cout);
      return std::nullopt;
    }
    if (argument == "--busy-poll") {
      options.busy_poll = true;
      continue;
    }
    if (argument == "--runtime-phase-counters") {
      options.runtime_phase_counters = true;
      continue;
    }
    if (argument == "--disable-direct-continuation-elision") {
      options.disable_direct_continuation_elision = true;
      continue;
    }
    if (argument == "--disable-closed-island-composite") {
      options.disable_closed_island_composite = true;
      continue;
    }
    if (argument == "--force-holomq-process-local-inline") {
      options.force_holomq_process_local_inline = true;
      continue;
    }
    if (argument == "--test-fail-sink-after-relay-pressure") {
      options.test_fail_sink_after_relay_pressure = true;
      continue;
    }

    std::string_view value;
    if (!consume_value(argc, argv, index, value)) {
      std::cerr << argument << " requires a value\n";
      return std::nullopt;
    }
    if (argument == "--workload") {
      if (value == "latency") {
        options.workload = Workload::kLatency;
      } else if (value == "notification_latency") {
        options.workload = Workload::kNotificationLatency;
      } else if (value == "saturated") {
        options.workload = Workload::kSaturated;
      } else {
        std::cerr << "--workload must be latency, notification_latency, or saturated\n";
        return std::nullopt;
      }
    } else if (argument == "--scenario") {
      if (value == "source_only") {
        options.scenario = Scenario::kSourceOnly;
      } else if (value == "pod") {
        options.scenario = Scenario::kPod;
      } else if (value == "signal") {
        options.scenario = Scenario::kSignal;
      } else if (value == "sample_forward") {
        options.scenario = Scenario::kSampleForward;
      } else if (value == "schema") {
        options.scenario = Scenario::kSchema;
      } else if (value == "tensor") {
        options.scenario = Scenario::kTensor;
      } else if (value == "image") {
        options.scenario = Scenario::kImage;
      } else if (value == "multi_tensor") {
        options.scenario = Scenario::kMultiTensor;
      } else {
        std::cerr << "--scenario must be source_only, pod, signal, sample_forward, schema, tensor, "
                     "image, or multi_tensor\n";
        return std::nullopt;
      }
    } else if (argument == "--queue-policy") {
      if (value == "blocking") {
        options.queue_policy = QueuePolicy::kBlocking;
      } else if (value == "drop_oldest") {
        options.queue_policy = QueuePolicy::kDropOldest;
      } else {
        std::cerr << "--queue-policy must be blocking or drop_oldest\n";
        return std::nullopt;
      }
    } else if (argument == "--operator-lanes") {
      if (value == "separate") {
        options.operator_lanes = OperatorLaneMode::kSeparate;
      } else if (value == "shared") {
        options.operator_lanes = OperatorLaneMode::kShared;
      } else if (value == "per_sink") {
        options.operator_lanes = OperatorLaneMode::kPerSink;
      } else {
        std::cerr << "--operator-lanes must be separate, shared, or per_sink\n";
        return std::nullopt;
      }
    } else if (argument == "--tensor-storage") {
      if (value == "pageable_host") {
        options.tensor_storage = TensorStorage::kPageableHost;
      } else if (value == "pinned_host") {
        options.tensor_storage = TensorStorage::kPinnedHost;
      } else if (value == "device") {
        options.tensor_storage = TensorStorage::kDevice;
      } else {
        std::cerr << "--tensor-storage must be pageable_host, pinned_host, or device\n";
        return std::nullopt;
      }
    } else if (argument == "--bulk-work-profile") {
      if (value == "full_write") {
        options.bulk_work_profile = BulkWorkProfile::kFullWrite;
      } else if (value == "descriptor_only") {
        options.bulk_work_profile = BulkWorkProfile::kDescriptorOnly;
      } else {
        std::cerr << "--bulk-work-profile must be full_write or descriptor_only\n";
        return std::nullopt;
      }
    } else if (argument == "--tensor-bytes") {
      if (!parse_integer(value, options.tensor_bytes)) {
        std::cerr << "--tensor-bytes must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--tensor-device") {
      if (!parse_integer(value, options.tensor_device)) {
        std::cerr << "--tensor-device must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--samples") {
      if (!parse_integer(value, options.samples)) {
        std::cerr << "--samples must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--warmup") {
      if (!parse_integer(value, options.warmup)) {
        std::cerr << "--warmup must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--queue-depth") {
      if (!parse_integer(value, options.queue_depth)) {
        std::cerr << "--queue-depth must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--relay-count") {
      if (!parse_integer(value, options.relay_count)) {
        std::cerr << "--relay-count must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--fanout") {
      if (!parse_integer(value, options.fanout)) {
        std::cerr << "--fanout must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--notification-capacity") {
      if (!parse_integer(value, options.notification_capacity)) {
        std::cerr << "--notification-capacity must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--publisher-period-us") {
      if (!parse_integer(value, options.publisher_period_us)) {
        std::cerr << "--publisher-period-us must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--timeout-ms") {
      if (!parse_integer(value, options.timeout_ms)) {
        std::cerr << "--timeout-ms must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--source-cpu") {
      if (!parse_integer(value, options.source_cpu)) {
        std::cerr << "--source-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--relay-cpu") {
      if (!parse_integer(value, options.relay_cpu)) {
        std::cerr << "--relay-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--sink-cpu") {
      if (!parse_integer(value, options.sink_cpu)) {
        std::cerr << "--sink-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--sink-cpus") {
      if (!parse_cpu_list(value, options.sink_cpus)) {
        std::cerr << "--sink-cpus must be a comma-separated CPU list\n";
        return std::nullopt;
      }
    } else if (argument == "--controller-cpu") {
      if (!parse_integer(value, options.controller_cpu)) {
        std::cerr << "--controller-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--channel-pump-cpu") {
      if (!parse_integer(value, options.channel_pump_cpu)) {
        std::cerr << "--channel-pump-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--notification-driver-cpu") {
      if (!parse_integer(value, options.notification_driver_cpu)) {
        std::cerr << "--notification-driver-cpu must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--payload-validation") {
      if (value == "none") {
        options.validation = PayloadValidation::kNone;
      } else if (value == "final") {
        options.validation = PayloadValidation::kFinal;
      } else if (value == "all") {
        options.validation = PayloadValidation::kAll;
      } else {
        std::cerr << "--payload-validation must be none, final, or all\n";
        return std::nullopt;
      }
    } else if (argument == "--output") {
      options.output_path = value;
    } else {
      std::cerr << "unknown option: " << argument << '\n';
      return std::nullopt;
    }
  }

  if (options.samples == 0U || options.samples > kMaximumSamples) {
    std::cerr << "--samples must be between 1 and " << kMaximumSamples << '\n';
    return std::nullopt;
  }
  if (options.warmup > kMaximumSamples ||
      options.warmup > std::numeric_limits<std::uint64_t>::max() - options.samples - 1U) {
    std::cerr << "--warmup exceeds the supported bounded run size\n";
    return std::nullopt;
  }
  if (options.queue_depth == 0U) {
    std::cerr << "--queue-depth must be positive\n";
    return std::nullopt;
  }
  if (lossy_queue_policy(options.queue_policy) && !lossy_scenario_supported(options.scenario)) {
    std::cerr << "--queue-policy drop_oldest supports --scenario pod, schema, tensor, or image\n";
    return std::nullopt;
  }
  if (lossy_queue_policy(options.queue_policy) && options.fanout != 1U) {
    std::cerr << "--queue-policy drop_oldest currently requires --fanout 1\n";
    return std::nullopt;
  }
  if (options.relay_count == 0U || options.relay_count > kMaximumRelayCount) {
    std::cerr << "--relay-count must be between 1 and " << kMaximumRelayCount << '\n';
    return std::nullopt;
  }
  if (options.scenario != Scenario::kSampleForward && options.relay_count != 1U) {
    std::cerr << "--relay-count requires --scenario sample_forward\n";
    return std::nullopt;
  }
  if (options.test_fail_sink_after_relay_pressure && options.scenario != Scenario::kSampleForward) {
    std::cerr << "--test-fail-sink-after-relay-pressure requires --scenario sample_forward\n";
    return std::nullopt;
  }
  if (options.fanout == 0U || options.fanout > kMaximumFanout) {
    std::cerr << "--fanout must be between 1 and " << kMaximumFanout << '\n';
    return std::nullopt;
  }
  if (!fanout_scenario(options.scenario) && options.fanout != 1U) {
    std::cerr << "--fanout greater than one requires --scenario pod or schema\n";
    return std::nullopt;
  }
  if (options.notification_capacity == 0U) {
    std::cerr << "--notification-capacity must be positive\n";
    return std::nullopt;
  }
  if (options.timeout_ms == 0U) {
    std::cerr << "--timeout-ms must be positive\n";
    return std::nullopt;
  }
  if (options.source_cpu < -1 || options.relay_cpu < -1 || options.sink_cpu < -1 ||
      options.controller_cpu < -1 || options.channel_pump_cpu < -1 ||
      options.notification_driver_cpu < -1 ||
      std::any_of(
          options.sink_cpus.begin(),
          options.sink_cpus.end(),
          [](const int cpu) { return cpu < -1; })) {
    std::cerr << "CPU values must be -1 or a nonnegative ordinal\n";
    return std::nullopt;
  }
  if (options.scenario != Scenario::kSampleForward && options.relay_cpu >= 0) {
    std::cerr << "--relay-cpu requires --scenario sample_forward\n";
    return std::nullopt;
  }
  if (options.scenario == Scenario::kSourceOnly && options.sink_cpu >= 0) {
    std::cerr << "--sink-cpu is not used by --scenario source_only\n";
    return std::nullopt;
  }
  if (!uses_channel_pump(options) && options.channel_pump_cpu >= 0) {
    std::cerr << "--channel-pump-cpu requires a transport-backed scenario\n";
    return std::nullopt;
  }
  if (options.force_holomq_process_local_inline && !direct_retained_scenario(options.scenario)) {
    std::cerr << "--force-holomq-process-local-inline requires pod, signal, sample_forward, "
                 "schema, tensor, image, or multi_tensor\n";
    return std::nullopt;
  }
  if (options.force_holomq_process_local_inline && options.fanout != 1U) {
    std::cerr << "--force-holomq-process-local-inline requires --fanout 1\n";
    return std::nullopt;
  }
  if (options.disable_direct_continuation_elision && options.force_holomq_process_local_inline) {
    std::cerr << "--disable-direct-continuation-elision and "
                 "--force-holomq-process-local-inline are mutually exclusive\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kShared && options.sink_cpu >= 0) {
    std::cerr << "--sink-cpu is not used with --operator-lanes shared\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kShared && options.relay_cpu >= 0) {
    std::cerr << "--relay-cpu is not used with --operator-lanes shared\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kPerSink &&
      (!fanout_scenario(options.scenario) || options.fanout <= 1U)) {
    std::cerr << "--operator-lanes per_sink requires pod or schema fanout greater than one\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kPerSink && options.sink_cpu >= 0) {
    std::cerr << "--sink-cpu is not used with --operator-lanes per_sink\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kSeparate && options.fanout > 1U &&
      options.sink_cpu >= 0) {
    std::cerr << "--sink-cpu with fanout greater than one requires "
                 "--operator-lanes per_sink and --sink-cpus\n";
    return std::nullopt;
  }
  if (options.operator_lanes != OperatorLaneMode::kPerSink && !options.sink_cpus.empty()) {
    std::cerr << "--sink-cpus requires --operator-lanes per_sink\n";
    return std::nullopt;
  }
  if (!options.sink_cpus.empty() && options.sink_cpus.size() != options.fanout) {
    std::cerr << "--sink-cpus must provide exactly one CPU per fanout sink\n";
    return std::nullopt;
  }
  if (options.scenario == Scenario::kSourceOnly && !is_notification_workload(options.workload)) {
    std::cerr << "--scenario source_only requires a notification workload\n";
    return std::nullopt;
  }
  if (!is_notification_workload(options.workload) && options.notification_driver_cpu >= 0) {
    std::cerr << "--notification-driver-cpu requires a notification workload\n";
    return std::nullopt;
  }
  if (!is_notification_workload(options.workload) && options.notification_capacity != 1U) {
    std::cerr << "--notification-capacity requires a notification workload\n";
    return std::nullopt;
  }
  if (options.workload == Workload::kSaturated &&
      options.notification_capacity > options.queue_depth) {
    std::cerr << "--notification-capacity must not exceed --queue-depth\n";
    return std::nullopt;
  }
  if (options.workload == Workload::kNotificationLatency && options.notification_capacity != 1U) {
    std::cerr << "notification latency requires --notification-capacity 1\n";
    return std::nullopt;
  }
  std::vector<int> assigned_cpus{
      options.source_cpu,
      options.scenario == Scenario::kSourceOnly ||
              options.operator_lanes != OperatorLaneMode::kSeparate
          ? -1
          : options.sink_cpu,
      options.controller_cpu,
      uses_channel_pump(options) ? options.channel_pump_cpu : -1,
      options.scenario == Scenario::kSampleForward &&
              options.operator_lanes == OperatorLaneMode::kSeparate
          ? options.relay_cpu
          : -1,
      is_notification_workload(options.workload) ? options.notification_driver_cpu : -1};
  assigned_cpus.insert(assigned_cpus.end(), options.sink_cpus.begin(), options.sink_cpus.end());
  std::sort(assigned_cpus.begin(), assigned_cpus.end());
  const auto first_pinned = std::find_if(
      assigned_cpus.begin(), assigned_cpus.end(), [](const int cpu) { return cpu >= 0; });
  if (std::adjacent_find(first_pinned, assigned_cpus.end()) != assigned_cpus.end()) {
    std::cerr << "Runtime benchmark roles must not share one logical CPU\n";
    return std::nullopt;
  }
  if (is_latency_workload(options.workload) && options.publisher_period_us == 0U) {
    std::cerr << "--publisher-period-us must be positive\n";
    return std::nullopt;
  }
  if (options.tensor_bytes == 0U || options.tensor_bytes > kMaximumTensorBytes ||
      options.tensor_bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    std::cerr << "--tensor-bytes must be between 1 and " << kMaximumTensorBytes << '\n';
    return std::nullopt;
  }
  if (options.tensor_device < 0) {
    std::cerr << "--tensor-device must be nonnegative\n";
    return std::nullopt;
  }
  if ((options.scenario == Scenario::kImage || options.scenario == Scenario::kMultiTensor) &&
      options.tensor_storage != TensorStorage::kPageableHost) {
    std::cerr << "--scenario image and multi_tensor currently require --tensor-storage "
                 "pageable_host\n";
    return std::nullopt;
  }
  if ((options.scenario == Scenario::kImage || options.scenario == Scenario::kMultiTensor) &&
      options.tensor_device != 0) {
    std::cerr << "--scenario image and multi_tensor currently require --tensor-device 0\n";
    return std::nullopt;
  }
  if (options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly &&
      options.scenario != Scenario::kTensor && options.scenario != Scenario::kMultiTensor) {
    std::cerr << "--bulk-work-profile descriptor_only requires --scenario tensor or multi_tensor\n";
    return std::nullopt;
  }
  return options;
}

// Benchmark setup is intentionally linear so every failure cleans up the same shared state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_benchmark(Options options) {
  if (options.scenario == Scenario::kTensor &&
      options.tensor_storage != TensorStorage::kPageableHost) {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || options.tensor_device >= device_count ||
        cudaSetDevice(options.tensor_device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      std::cerr << "SKIP: requested CUDA Tensor storage is unavailable\n";
      return kSkipped;
    }
  }
  BenchmarkRunState state{std::move(options)};
  auto image_tensor = state.initialize_image_tensor();
  if (!image_tensor) {
    std::cerr << "failed to initialize Image Tensor backing\n";
    return 2;
  }
  g_run_state = &state;
  auto thread_control =
      std::make_shared<BenchmarkSupportThreadControl>(std::vector<SupportThreadBinding>{
          {.role = SupportThreadRole::kController,
           .requested_cpu = state.options().controller_cpu,
           .required = true},
          {.role = SupportThreadRole::kChannelPump,
           .requested_cpu = state.options().channel_pump_cpu,
           .required = uses_channel_pump(state.options())},
          {.role = SupportThreadRole::kNotificationDriver,
           .requested_cpu = state.options().notification_driver_cpu,
           .required = state.options().workload != Workload::kLatency},
      });
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
      make_plan(state.options()), thread_control);
  if (state.options().disable_direct_continuation_elision) {
    plan = detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(plan);
  }
  if (state.options().disable_closed_island_composite) {
    plan = detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(plan);
  }
  std::vector<detail::RuntimePhaseTraceRecord> phase_trace_storage;
  std::shared_ptr<detail::RuntimePhaseCounters> phase_counters;
  if (state.options().runtime_phase_counters) {
    phase_trace_storage.resize(runtime_phase_trace_capacity(state));
    phase_counters = std::make_shared<detail::RuntimePhaseCounters>(std::span{phase_trace_storage});
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, phase_counters);
  }
  if (!plan.ok()) {
    std::cerr << "failed to compile runtime Graph benchmark plan\n" << plan.json() << '\n';
    g_run_state = nullptr;
    return 2;
  }
  if (state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly &&
      (state.options().scenario == Scenario::kTensor ||
       state.options().scenario == Scenario::kMultiTensor)) {
    std::size_t warmup_slots{};
    if (state.options().scenario == Scenario::kTensor) {
      const auto allocations = plan.tensor_allocations();
      if (allocations.size() != 1U || allocations.front().slot_count == 0U) {
        std::cerr << "Tensor compiled pool inventory is invalid\n";
        g_run_state = nullptr;
        return 2;
      }
      warmup_slots = std::min(static_cast<std::size_t>(allocations.front().slot_count),
                              static_cast<std::size_t>(allocations.front().max_outstanding_loans) +
                                  2U * static_cast<std::size_t>(state.options().queue_depth));
    } else {
      const auto fields = plan.tensor_field_allocations();
      if (fields.size() != 2U || fields.front().slot_count == 0U ||
          fields.front().slot_count != fields.back().slot_count ||
          fields.front().max_outstanding_loans != fields.back().max_outstanding_loans) {
        std::cerr << "multi-Tensor compiled field-pool inventory is invalid\n";
        g_run_state = nullptr;
        return 2;
      }
      warmup_slots = std::min(static_cast<std::size_t>(fields.front().slot_count),
                              static_cast<std::size_t>(fields.front().max_outstanding_loans) +
                                  2U * static_cast<std::size_t>(state.options().queue_depth));
    }
    if (warmup_slots == 0U || state.options().warmup < warmup_slots) {
      std::cerr << "descriptor-only warmup cannot cover the hot-path Tensor slots\n";
      g_run_state = nullptr;
      return 2;
    }
    state.set_descriptor_warmup_slots(warmup_slots);
  }

  RunSession session = run_async(plan);
  std::thread notification_driver;
  if (state.notification_driven()) {
    auto sender = state.take_notification_sender();
    if (sender.has_value()) {
      notification_driver = std::thread([&state, thread_control, sender = std::move(*sender)]() {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        try {
#endif
          auto entered = thread_control->enter_notification_driver();
          if (!entered) {
            state.fail(entered.error().code);
            return;
          }
          if (state.notification_latency()) {
            state.drive_notification_latency(sender);
          } else {
            state.drive_notifications(sender);
          }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        } catch (...) {
          state.fail(ErrorCode::kResourceExhausted);
        }
#endif
      });
    } else {
      state.fail(ErrorCode::kTimeout);
    }
  }
  const bool signaled = state.wait();
  if (!signaled) {
    std::cerr << "runtime Graph benchmark timeout: notifications=" << state.notifications_accepted()
              << " publications=" << state.publications() << " deliveries=" << state.deliveries()
              << " completions=" << state.activation_completions() << '\n';
    state.fail(ErrorCode::kTimeout);
  }
  if (notification_driver.joinable()) {
    notification_driver.join();
  }
  bool retained_tensor_valid = true;
  std::optional<std::uint64_t> final_validation_ns;
  if ((state.options().scenario == Scenario::kTensor ||
       state.options().scenario == Scenario::kImage ||
       state.options().scenario == Scenario::kMultiTensor) &&
      state.options().validation == PayloadValidation::kFinal && signaled && state.completed()) {
    const std::int64_t validation_start_ns = now_ns();
    retained_tensor_valid = state.options().scenario == Scenario::kMultiTensor
                                ? state.validate_retained_multi_tensor()
                                : state.validate_retained_tensor();
    final_validation_ns = static_cast<std::uint64_t>(now_ns() - validation_start_ns);
    if (!retained_tensor_valid) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
    }
  }
  session.request_stop();
  session.wait();
  const RuntimePhaseTraceSummary phase_trace = summarize_phase_trace(state, phase_counters.get());

  std::ofstream output_file;
  std::ostream* output = &std::cout;
  if (!state.options().output_path.empty()) {
    output_file.open(state.options().output_path, std::ios::out | std::ios::trunc);
    if (!output_file) {
      std::cerr << "failed to open benchmark output: " << state.options().output_path << '\n';
      g_run_state = nullptr;
      return 2;
    }
    output = &output_file;
  }
  emit_result(*output,
              state,
              plan,
              session,
              *thread_control,
              phase_counters.get(),
              phase_trace,
              final_validation_ns);

  const std::uint64_t expected_total = state.final_sequence();
  const bool source_only = state.options().scenario == Scenario::kSourceOnly;
  const bool lossy = lossy_queue_policy(state.options().queue_policy);
  const std::uint64_t expected_publications = source_only ? 0U : expected_total;
  const std::uint64_t expected_deliveries =
      source_only ? 0U : (lossy ? state.deliveries() : expected_total * state.options().fanout);
  const std::uint64_t expected_deliveries_per_sink =
      source_only || !lossy ? expected_total : expected_deliveries;
  const std::uint64_t expected_sink_callbacks = expected_deliveries;
  const std::uint64_t expected_activation_completions =
      source_only || !lossy ? expected_total : expected_deliveries;
  const std::uint64_t expected_relay = state.options().scenario == Scenario::kSampleForward
                                           ? expected_total * state.options().relay_count
                                           : 0U;
  const std::uint64_t expected_tensor_emits =
      state.options().scenario == Scenario::kTensor ? expected_publications : 0U;
  const std::uint64_t expected_inline_payload_emits =
      state.options().scenario == Scenario::kSourceOnly ||
              state.options().scenario == Scenario::kTensor ||
              state.options().scenario == Scenario::kImage ||
              state.options().scenario == Scenario::kMultiTensor
          ? 0U
          : expected_publications + expected_relay;
  const bool descriptor_initialization_valid =
      state.options().bulk_work_profile != BulkWorkProfile::kDescriptorOnly ||
      (state.descriptor_backing_initializations() >=
           (state.options().scenario == Scenario::kMultiTensor ? 2U : 1U) &&
       state.descriptor_backing_initializations_during_measurement() == 0U);
  const bool notification_counts_valid =
      !state.notification_driven() ||
      ((state.notification_latency() ? state.notifications_accepted() == expected_total
                                     : state.notifications_accepted() >= expected_total) &&
       state.notification_attempts() >= state.notifications_accepted());
  const bool timing_valid =
      state.measured_start_ns() > 0 && state.measured_end_ns() > state.measured_start_ns();
  const bool replacement_counts_valid =
      !lossy ||
      (state.deliveries() > 0U && state.deliveries() <= state.publications() &&
       state.consumer_visible_replacements() + state.deliveries() == state.publications() &&
       (state.options().workload == Workload::kSaturated ||
        state.consumer_visible_replacements() == 0U));
  // Aggregate backpressure also includes relay retries, which do not re-run the source.
  const std::uint64_t expected_source_callbacks =
      expected_total + state.source_backpressure_count();
  const bool valid =
      signaled && state.completed() && state.first_error() == ErrorCode::kSuccess && timing_valid &&
      state.publications() == expected_publications && state.deliveries() == expected_deliveries &&
      state.activation_completions() == expected_activation_completions &&
      state.source_callbacks() == expected_source_callbacks &&
      state.relay_publications() == expected_relay && state.relay_callbacks() == expected_relay &&
      state.sink_callbacks() == expected_sink_callbacks && retained_tensor_valid &&
      (source_only || state.sink_evidence_valid(expected_total, expected_deliveries_per_sink)) &&
      state.validation_errors() == 0U && state.allocation_probe_valid() &&
      descriptor_initialization_valid && replacement_counts_valid && notification_counts_valid &&
      thread_control->required_roles_entered() &&
      phase_counters_valid(state, phase_counters.get()) &&
      (!phase_trace.enabled ||
       (phase_trace.valid && phase_trace.dropped == 0U &&
        phase_trace.unmatched_readiness_polls == 0U &&
        (!phase_trace.activation_trace_applicable ||
         (phase_trace.complete_activations ==
              state.source_compute_entries() + state.relay_callbacks() + state.sink_callbacks() &&
          phase_trace.complete_tensor_emits == expected_tensor_emits &&
          phase_trace.complete_inline_payload_emits == expected_inline_payload_emits)))) &&
      session.termination() == RunTermination::kStopped && session.diagnostics().empty();
  if (!valid) {
    std::cerr << "runtime Graph benchmark failed: signaled=" << signaled
              << " completed=" << state.completed()
              << " error=" << static_cast<std::uint32_t>(state.first_error())
              << " publications=" << state.publications()
              << " relay_publications=" << state.relay_publications()
              << " deliveries=" << state.deliveries()
              << " source_callbacks=" << state.source_callbacks()
              << " backpressured_emits=" << state.backpressure_count()
              << " source_backpressured_emits=" << state.source_backpressure_count()
              << " expected_source_callbacks=" << expected_source_callbacks
              << " completions=" << state.activation_completions() << " expected=" << expected_total
              << " start_ns=" << state.measured_start_ns() << " end_ns=" << state.measured_end_ns()
              << " termination=" << termination_name(session.termination()) << '\n';
  }
  g_run_state = nullptr;
  return valid ? 0 : 1;
}

}  // namespace
}  // namespace holoscan

int main(const int argc, char** argv) {
  try {
    std::optional<holoscan::Options> options = holoscan::parse_options(argc, argv);
    if (!options.has_value()) {
      return argc == 2 && std::string_view(argv[1]) == "--help" ? 0 : 2;
    }
    return holoscan::run_benchmark(std::move(*options));
  } catch (const std::exception& error) {
    std::cerr << "runtime Graph benchmark setup failed: " << error.what() << '\n';
    return 2;
  }
}
