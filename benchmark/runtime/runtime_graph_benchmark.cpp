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
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <fmt/format.h>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

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
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime_multi_tensor_benchmark_traits.hpp"
#include "runtime_third_party_benchmark_traits.hpp"

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
using benchmarking::emit_runtime_diagnostics;
using benchmarking::emit_runtime_thread_usage;
using benchmarking::emit_support_thread_evidence;
using benchmarking::json_escape;
using benchmarking::kDescriptorBackedImagePayloadBytes;
using benchmarking::kMaximumSamples;
using benchmarking::kMaximumTensorBytes;
using benchmarking::kTensorByteDtype;
using benchmarking::kTensorTransportPayloadBytes;
using benchmarking::make_placement;
using benchmarking::make_preinitialized_byte_tensor;
using benchmarking::payload_validation_name;
using benchmarking::PayloadValidation;
using benchmarking::PreinitializedTensorStorage;
using benchmarking::SupportThreadBinding;
using benchmarking::SupportThreadRole;
using benchmarking::tensor_descriptor_matches;
using benchmarking::tensor_memory_kind;
using benchmarking::tensor_storage_name;
using benchmarking::TensorStorage;
using benchmarking::validation_timing_name;
using MultiTensorMessage = runtime_benchmark::MultiTensorMessageT;
using MultiTensorFields = runtime_benchmark::MultiTensorMessageTensorFields;
using ThirdPartyTelemetry = task30_benchmark::TelemetryT;

constexpr std::uint32_t kMaximumRelayCount = 64U;
constexpr std::uint32_t kMaximumFanout = 32U;
constexpr std::size_t kMaximumRuntimePhaseTraceRecords = 2'000'000U;
constexpr std::size_t kBenchmarkCacheLineBytes = 64U;
constexpr std::size_t kMaximumImageSinkComputeAllocations = 6U;
constexpr std::size_t kMaximumMultiTensorSinkComputeAllocations = 4U;
// Ten activation/callback checkpoints plus bounded readiness, Channel, and controller wake
// evidence. The capacity is deliberately conservative because parked execution partitions can
// append controller park/return records between otherwise complete activations.
// Independently owned fan-out lanes add scheduler-dependent controller park/return and Channel
// wait records. Reserve one additional record per expected activation as proportional
// control-plane headroom so phase evidence cannot truncate a valid run.
constexpr std::uint64_t kTraceRecordsPerExpectedActivation = 19U;
constexpr std::uint64_t kTensorTraceRecordsPerPublication = 5U;
constexpr std::uint64_t kInlinePayloadTraceRecordsPerPublication = 5U;
constexpr std::uint32_t kMaximumOfferedRateHz = 1'000'000U;
constexpr std::uint32_t kMaximumOfferedBurstSize = 64U;
constexpr std::uint32_t kMaximumTestTensorPoolRetries = 16U;
constexpr double kMaximumOfferedRateRelativeError = 0.02;
constexpr int kSkipped = 77;
constexpr std::uint8_t kSignalPayload = 0xA5U;
constexpr std::uint8_t kImagePayloadByte = 0x3CU;
constexpr std::uint8_t kMultiTensorPayloadByte = 0xA5U;
constexpr std::uint8_t kMultiTensorModePayloadByte = 0x6DU;
constexpr std::array<std::uint32_t, 5U> kSupportedCallbackWorkUs{0U, 1U, 5U, 20U, 100U};

enum class Workload { kLatency, kNotificationLatency, kSaturated };
enum class BulkWorkProfile { kFullWrite, kDescriptorOnly };
enum class Scenario {
  kSourceOnly,
  kPod,
  kSignal,
  kSampleForward,
  kSchema,
  kThirdPartySchema,
  kTensor,
  kImage,
  kMultiTensor,
  kMultiTensorLoan,
  kMultiTensorStaging,
};
enum class OperatorLaneMode { kSeparate, kShared, kPerSink };
enum class PartitionCover { kLegacy, kAutomatic, kIncludeAll, kHybrid };
enum class GreenContextLayout {
  kNone,
  kSamePartition,
  kDedicatedToDedicated,
  kDedicatedToResidual
};

[[nodiscard]] constexpr bool multi_tensor_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kMultiTensorLoan || scenario == Scenario::kMultiTensorStaging;
}

[[nodiscard]] constexpr bool tensor_backed_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kTensor || scenario == Scenario::kImage ||
         scenario == Scenario::kMultiTensor || multi_tensor_scenario(scenario);
}

struct Options {
  PayloadValidation validation{PayloadValidation::kFinal};
  Workload workload{Workload::kLatency};
  Scenario scenario{Scenario::kPod};
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  TensorStorage tensor_storage{TensorStorage::kPageableHost};
  BulkWorkProfile bulk_work_profile{BulkWorkProfile::kFullWrite};
  OperatorLaneMode operator_lanes{OperatorLaneMode::kSeparate};
  PartitionCover partition_cover{PartitionCover::kLegacy};
  GreenContextLayout green_context_layout{GreenContextLayout::kNone};
  std::size_t tensor_bytes{1U << 20U};
  int tensor_device{0};
  std::uint32_t green_context_sm_count{};
  std::uint64_t samples{100'000U};
  std::uint64_t warmup{1'000U};
  std::uint32_t queue_depth{1U};
  std::vector<std::uint32_t> sink_queue_depths;
  std::uint32_t relay_count{1U};
  std::uint32_t fanout{1U};
  // Authors the POD two-output/two-sink applicability rows.
  bool multi_output{};
  // Authors the POD two-output/one-two-input-sink applicability rows.
  bool multi_input{};
  // Authors one POD output split across blocking and drop-oldest HoloMQ services.
  bool mixed_policy_fanout{};
  std::uint32_t notification_capacity{1U};
  // Enables controlled POD drop-oldest fanout rows; zero keeps completion-driven pacing.
  std::uint32_t offered_rate_hz{};
  // Bounds publications emitted by each fixed-rate clock activation.
  std::uint32_t offered_burst_size{1U};
  std::uint32_t publisher_period_us{1'000U};
  std::uint32_t callback_work_us{};
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
  bool cuda_resource_counters{};
  bool disable_direct_continuation_elision{};
  bool disable_closed_island_composite{};
  bool force_holomq_process_local_inline{};
  // Selects matched forced-wake rows without changing the active Graph topology.
  bool force_holomq_event_wake{};
  // Retains only the Runtime pump on an otherwise pump-free all-eventless HoloMQ process.
  bool force_process_local_channel_pump{};
  bool test_fail_sink_after_relay_pressure{};
  std::uint32_t test_source_tensor_pool_in_activation_retries{};
  bool test_gate_pod_sink_for_pressure{};
  std::string output_path;
  std::string plan_output_path;
};

struct BenchmarkLifecycleTiming {
  std::uint64_t compile_ns{};
  std::uint64_t run_async_return_ns{};
  std::uint64_t run_async_to_first_callback_ns{};
  std::uint64_t terminal_wait_ns{};
};

[[nodiscard]] constexpr bool direct_retained_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSignal ||
         scenario == Scenario::kSampleForward || scenario == Scenario::kSchema ||
         scenario == Scenario::kThirdPartySchema || tensor_backed_scenario(scenario);
}

[[nodiscard]] constexpr bool fanout_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSchema ||
         scenario == Scenario::kThirdPartySchema || scenario == Scenario::kTensor ||
         scenario == Scenario::kImage || multi_tensor_scenario(scenario);
}

[[nodiscard]] std::uint32_t subscriber_queue_depth(const Options& options,
                                                   const std::uint32_t sink_index) noexcept {
  return options.sink_queue_depths.empty() ? options.queue_depth
                                           : options.sink_queue_depths[sink_index];
}

[[nodiscard]] constexpr QueuePolicy subscriber_queue_policy(
    const Options& options, const std::uint32_t sink_index) noexcept {
  return options.mixed_policy_fanout && sink_index == 1U ? QueuePolicy::kDropOldest
                                                         : options.queue_policy;
}

[[nodiscard]] constexpr std::uint32_t source_output_count(const Options& options) noexcept {
  return options.scenario == Scenario::kSourceOnly
             ? 0U
             : (options.multi_output || options.multi_input ? 2U : 1U);
}

[[nodiscard]] constexpr std::uint32_t subscribers_per_output(const Options& options) noexcept {
  return options.scenario == Scenario::kSourceOnly
             ? 0U
             : (options.multi_output || options.multi_input ? 1U : options.fanout);
}

[[nodiscard]] constexpr std::uint32_t source_emit_count(const Options& options) noexcept {
  return options.multi_output || options.multi_input ? 2U : 1U;
}

[[nodiscard]] constexpr std::uint32_t sink_operator_count(const Options& options) noexcept {
  return options.scenario == Scenario::kSourceOnly ? 0U
                                                   : (options.multi_input ? 1U : options.fanout);
}

[[nodiscard]] constexpr std::uint32_t inputs_per_sink(const Options& options) noexcept {
  return options.scenario == Scenario::kSourceOnly ? 0U : (options.multi_input ? 2U : 1U);
}

[[nodiscard]] constexpr bool holomq_event_wake_control_scenario(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSchema ||
         scenario == Scenario::kThirdPartySchema || tensor_backed_scenario(scenario);
}

[[nodiscard]] constexpr bool lossy_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] constexpr bool lossy_subscriber(const Options& options,
                                              const std::uint32_t sink_index) noexcept {
  return lossy_queue_policy(subscriber_queue_policy(options, sink_index));
}

[[nodiscard]] constexpr bool has_lossy_subscriber(const Options& options) noexcept {
  for (std::uint32_t index = 0U; index < sink_operator_count(options); ++index) {
    if (lossy_subscriber(options, index)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] constexpr bool offered_rate_mode(const Options& options) noexcept {
  return options.offered_rate_hz != 0U;
}

[[nodiscard]] std::chrono::nanoseconds offered_period(const Options& options) noexcept {
  return std::chrono::nanoseconds{1'000'000'000ULL / options.offered_rate_hz};
}

[[nodiscard]] constexpr bool lossy_scenario_supported(const Scenario scenario) noexcept {
  return scenario == Scenario::kPod || scenario == Scenario::kSchema ||
         scenario == Scenario::kThirdPartySchema || scenario == Scenario::kTensor ||
         scenario == Scenario::kImage || multi_tensor_scenario(scenario);
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

[[nodiscard]] constexpr bool uses_holomq(const Options& options) noexcept {
  return direct_retained_scenario(options.scenario) &&
         (options.force_holomq_process_local_inline || options.fanout > 1U);
}

[[nodiscard]] constexpr bool uses_channel_pump(const Options& options) noexcept {
  return uses_holomq(options) &&
         (options.force_process_local_channel_pump || options.force_holomq_event_wake ||
          options.operator_lanes != OperatorLaneMode::kShared);
}

[[nodiscard]] constexpr const char* process_local_delivery_name(const Options& options) noexcept {
  if (!direct_retained_scenario(options.scenario)) {
    return "not_applicable";
  }
  return uses_holomq(options) ? "holomq" : "direct_retained";
}

struct Percentiles {
  std::uint64_t minimum{};
  std::uint64_t p50{};
  std::uint64_t p90{};
  std::uint64_t p95{};
  std::uint64_t p99{};
  std::optional<std::uint64_t> p999;
  std::uint64_t maximum{};
};

struct RuntimePhaseReceiverSummary {
  std::size_t receiver_index{};
  bool source{};
  std::optional<std::uint32_t> downstream_index;
  std::uint64_t activations{};
  std::vector<std::uint64_t> previous_cut_to_candidate;
  std::vector<std::uint64_t> previous_cut_to_readiness_poll_return;
  std::vector<std::uint64_t> readiness_poll_return_to_candidate;
  std::vector<std::uint64_t> candidate_to_selection;
  std::vector<std::uint64_t> selection_to_input_preparation;
  std::vector<std::uint64_t> input_preparation_to_activation;
  std::vector<std::uint64_t> activation_to_dispatch_start;
  std::vector<std::uint64_t> candidate_to_dispatch_start;
  std::vector<std::uint64_t> dispatch_start_to_callback_entry;
  std::vector<std::uint64_t> callback_entry_to_return;
  std::vector<std::uint64_t> callback_return_to_dispatch_return;
  std::vector<std::uint64_t> compute_dispatch;
  std::vector<std::uint64_t> dispatch_return_to_retirement;
  std::vector<std::uint64_t> retirement_to_cut;
  std::vector<std::uint64_t> dispatch_return_to_cut;
  std::vector<std::uint64_t> callback_pre_return_to_dispatch_return;
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
  std::uint64_t terminal_frozen_candidates{};
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

[[nodiscard]] constexpr bool notification_driver_used(const Options& options) noexcept {
  return is_notification_workload(options.workload) && !offered_rate_mode(options);
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
    case Scenario::kThirdPartySchema:
      return "third_party_schema";
    case Scenario::kTensor:
      return "tensor";
    case Scenario::kImage:
      return "image";
    case Scenario::kMultiTensor:
      return "multi_tensor";
    case Scenario::kMultiTensorLoan:
      return "multi_tensor_loan";
    case Scenario::kMultiTensorStaging:
      return "multi_tensor_staging";
  }
  return "unknown";
}

[[nodiscard]] const char* green_context_layout_name(const GreenContextLayout layout) noexcept {
  switch (layout) {
    case GreenContextLayout::kNone:
      return "none";
    case GreenContextLayout::kSamePartition:
      return "same_partition";
    case GreenContextLayout::kDedicatedToDedicated:
      return "dedicated_to_dedicated";
    case GreenContextLayout::kDedicatedToResidual:
      return "dedicated_to_residual";
  }
  return "unknown";
}

[[nodiscard]] const char* partition_cover_name(const PartitionCover cover) noexcept {
  switch (cover) {
    case PartitionCover::kLegacy:
      return "legacy";
    case PartitionCover::kAutomatic:
      return "automatic";
    case PartitionCover::kIncludeAll:
      return "include_all";
    case PartitionCover::kHybrid:
      return "hybrid";
  }
  return "unknown";
}

[[nodiscard]] const char* partitioning_mode_name(const PartitioningMode mode) noexcept {
  switch (mode) {
    case PartitioningMode::kAutomatic:
      return "automatic";
    case PartitioningMode::kConstrainedAutomatic:
      return "constrained_automatic";
    case PartitioningMode::kManual:
      return "manual";
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
      result =
          options.multi_input
              ? "runtime_graph_pod_2to1"
              : (options.multi_output ? "runtime_graph_pod_2x1"
                                      : "runtime_graph_pod_1to" + std::to_string(options.fanout));
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
    case Scenario::kThirdPartySchema:
      result = "runtime_graph_third_party_telemetry_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kTensor:
      result = "runtime_graph_tensor_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kImage:
      result = "runtime_graph_image_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kMultiTensor:
      result = "runtime_graph_multi_tensor_1to1";
      break;
    case Scenario::kMultiTensorLoan:
      result = "runtime_graph_multi_tensor_loan_1to" + std::to_string(options.fanout);
      break;
    case Scenario::kMultiTensorStaging:
      result = "runtime_graph_multi_tensor_staging_1to" + std::to_string(options.fanout);
      break;
  }
  if (result.empty()) {
    result = "runtime_graph_unknown";
  }
  if (options.force_holomq_process_local_inline) {
    result += "_holomq_fallback";
  }
  if (options.force_holomq_event_wake) {
    result += "_event_wake_control";
  }
  if (options.force_process_local_channel_pump) {
    result += "_channel_pump_control";
  }
  if (options.mixed_policy_fanout) {
    result += "_mixed_policy";
  }
  if (options.queue_policy != QueuePolicy::kBlocking) {
    result += "_";
    result += queue_policy_name(options.queue_policy);
  }
  if (offered_rate_mode(options)) {
    result += "_offered_" + std::to_string(options.offered_rate_hz) + "hz_burst" +
              std::to_string(options.offered_burst_size);
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
    case Scenario::kThirdPartySchema:
      return "third-party generated-schema encode, verify, materialize, and activation cost";
    case Scenario::kTensor:
      return "production direct Tensor loan, publish, readiness, receive, and retirement cost";
    case Scenario::kImage:
      return "production Image metadata and embedded Tensor retention cost";
    case Scenario::kMultiTensor:
      return "generated-schema metadata and two preallocated Tensor field retention cost";
    case Scenario::kMultiTensorLoan:
      return "production multi-Tensor schema bundle loan, publish, receive, and retirement cost";
    case Scenario::kMultiTensorStaging:
      return "production multi-Tensor schema staging, publish, receive, and retirement cost";
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
  if (multi_tensor_scenario(options.scenario)) {
    return options.tensor_bytes * 2U;
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
  if (options.scenario == Scenario::kThirdPartySchema) {
    return sizeof(ThirdPartyTelemetry);
  }
  return sizeof(BenchmarkPOD64);
}

[[nodiscard]] std::optional<std::size_t> inline_transport_bytes(const Options& options) noexcept {
  if (!uses_holomq(options)) {
    return 0U;
  }
  if (options.scenario == Scenario::kTensor) {
    return kTensorTransportPayloadBytes;
  }
  if (options.scenario == Scenario::kImage) {
    return kDescriptorBackedImagePayloadBytes;
  }
  if (options.scenario == Scenario::kMultiTensor || multi_tensor_scenario(options.scenario)) {
    // Descriptor metadata uses a variable encoded prefix of the generated schema's bounded wire
    // capacity. The Runtime does not expose the exact per-publication prefix length here.
    return std::nullopt;
  }
  if (options.scenario == Scenario::kSchema || options.scenario == Scenario::kThirdPartySchema) {
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
  if (multi_tensor_scenario(options.scenario)) {
    return options.tensor_bytes * 2U;
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
  if (multi_tensor_scenario(options.scenario)) {
    return "preinitialized_immutable_backing";
  }
  return "full_payload_per_sample";
}

[[nodiscard]] const char* bulk_work_profile(const Options& options) noexcept {
  if (options.scenario == Scenario::kTensor) {
    return options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly ? "descriptor_only"
                                                                         : "full_write";
  }
  if (options.scenario == Scenario::kImage || multi_tensor_scenario(options.scenario)) {
    return "descriptor_only";
  }
  if (options.scenario == Scenario::kMultiTensor) {
    return options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly ? "descriptor_only"
                                                                         : "full_write";
  }
  return "not_applicable";
}

[[nodiscard]] const char* payload_name(const Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::kSourceOnly:
      return "none";
    case Scenario::kSignal:
      return "trigger_token";
    case Scenario::kTensor:
      return "tensor";
    case Scenario::kImage:
      return "image";
    case Scenario::kMultiTensor:
      return "multi_tensor";
    case Scenario::kMultiTensorLoan:
      return "multi_tensor_loan";
    case Scenario::kMultiTensorStaging:
      return "multi_tensor_staging";
    case Scenario::kSchema:
      return "schema";
    case Scenario::kThirdPartySchema:
      return "third_party_schema";
    case Scenario::kPod:
    case Scenario::kSampleForward:
      return "pod";
  }
  return "unknown";
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

inline void callback_work_pause() noexcept {
#if defined(__i386__) || defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

void run_controlled_callback_work(const std::uint32_t duration_us) noexcept {
  if (duration_us == 0U) {
    return;
  }
  const auto deadline = BenchmarkClock::now() + std::chrono::microseconds{duration_us};
  while (BenchmarkClock::now() < deadline) {
    callback_work_pause();
  }
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
  } else if constexpr (std::is_same_v<Payload, ThirdPartyTelemetry>) {
    return ThirdPartyTelemetry{
        .sequence = sequence,
        .source_id = static_cast<std::uint32_t>(sequence) ^ 0xC001'C0DEU,
        .status = static_cast<std::uint16_t>(sequence ^ 0x5A5AU),
        .valid = true,
        .reading = static_cast<double>(sequence) * 0.25,
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

[[nodiscard]] bool payload_valid(const ThirdPartyTelemetry& payload) noexcept {
  return payload.sequence != 0U &&
         payload.source_id == (static_cast<std::uint32_t>(payload.sequence) ^ 0xC001'C0DEU) &&
         payload.status == static_cast<std::uint16_t>(payload.sequence ^ 0x5A5AU) &&
         payload.valid && payload.reading == static_cast<double>(payload.sequence) * 0.25;
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
        sink_callback_tids_by_index_(
            options_.runtime_phase_counters
                ? std::make_unique<std::atomic<std::uint64_t>[]>(sink_operator_count(options_))
                : nullptr),
        sink_callback_identity_is_native_by_index_(
            options_.runtime_phase_counters
                ? std::make_unique<std::atomic_bool[]>(sink_operator_count(options_))
                : nullptr),
        sink_callback_native_tid_syscall_unavailable_by_index_(
            options_.runtime_phase_counters
                ? std::make_unique<std::atomic_bool[]>(sink_operator_count(options_))
                : nullptr),
        sink_callback_multiple_threads_by_index_(
            options_.runtime_phase_counters
                ? std::make_unique<std::atomic_bool[]>(sink_operator_count(options_))
                : nullptr),
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
        multi_tensor_source_pointers_{
            multi_tensor_scenario(options_.scenario)
                ? std::make_unique<std::atomic<std::uintptr_t>[]>(  // NOLINT
                      final_sequence() + 1U)
                : nullptr,
            multi_tensor_scenario(options_.scenario)
                ? std::make_unique<std::atomic<std::uintptr_t>[]>(  // NOLINT
                      final_sequence() + 1U)
                : nullptr},
        tensor_validation_buffer_(tensor_backed_scenario(options_.scenario) ? options_.tensor_bytes
                                                                            : 0U) {
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
    auto tensor = make_preinitialized_byte_tensor(options_.tensor_bytes,
                                                  options_.tensor_storage,
                                                  options_.tensor_device,
                                                  kImagePayloadByte,
                                                  image_storage_,
                                                  options_.fanout);
    if (!tensor) {
      return make_unexpected(std::move(tensor.error()));
    }
    image_tensor_ = std::move(*tensor);
    record_descriptor_backing_initialization(0U);
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
    return notification_driver_used(options_);
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

  // Fixed-rate drop-oldest rows clock bursts while preserving the per-publication offer rate.
  [[nodiscard]] std::chrono::nanoseconds source_period() const noexcept {
    if (offered_rate_mode(options_)) {
      return offered_period(options_) * options_.offered_burst_size;
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::microseconds{options_.publisher_period_us});
  }

  [[nodiscard]] bool measured(const std::uint64_t sequence) const noexcept {
    return sequence >= first_measured_sequence() && sequence <= final_sequence();
  }

  // Bounded-burst rows count clock activations; other rows emit at most once per callback.
  [[nodiscard]] std::uint64_t expected_source_callbacks() const noexcept {
    if (!offered_rate_mode(options_)) {
      return final_sequence() + source_backpressure_count();
    }
    const auto callbacks_for = [&](const std::uint64_t publications) {
      return (publications + options_.offered_burst_size - 1U) / options_.offered_burst_size;
    };
    return callbacks_for(allocation_probe_sequence()) + callbacks_for(options_.samples);
  }

  [[nodiscard]] std::uint64_t expected_measured_source_callbacks() const noexcept {
    if (!offered_rate_mode(options_)) {
      return options_.samples;
    }
    return (options_.samples + options_.offered_burst_size - 1U) / options_.offered_burst_size;
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

  void record_source_compute_entry(const bool terminal) noexcept {
    if (first_callback_entry_ns_.load(std::memory_order_relaxed) == 0) {
      std::int64_t expected{};
      static_cast<void>(first_callback_entry_ns_.compare_exchange_strong(
          expected, now_ns(), std::memory_order_relaxed, std::memory_order_relaxed));
    }
    record_callback_thread(source_callback_tid_,
                           source_callback_identity_is_native_,
                           source_callback_native_tid_syscall_unavailable_,
                           source_callback_multiple_threads_);
    source_compute_entries_.fetch_add(1U, std::memory_order_relaxed);
    // A periodic candidate may already be frozen while the final fanout sample retires. Report
    // that no-op drain activation explicitly instead of assuming a platform-independent count.
    if (terminal) {
      source_terminal_compute_entries_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] std::int64_t first_callback_entry_ns() const noexcept {
    return first_callback_entry_ns_.load(std::memory_order_relaxed);
  }

  void record_relay_callback() noexcept {
    record_callback_thread(relay_callback_tid_,
                           relay_callback_identity_is_native_,
                           relay_callback_native_tid_syscall_unavailable_,
                           relay_callback_multiple_threads_);
    ++relay_callbacks_;
  }

  void record_sink_callback() noexcept { record_sink_callback(0U); }

  void record_sink_callback(const std::uint32_t sink_index) noexcept {
    record_callback_thread(sink_callback_tid_,
                           sink_callback_identity_is_native_,
                           sink_callback_native_tid_syscall_unavailable_,
                           sink_callback_multiple_threads_);
    if (sink_callback_tids_by_index_ && sink_index < sink_operator_count(options_)) {
      record_callback_thread(sink_callback_tids_by_index_[sink_index],
                             sink_callback_identity_is_native_by_index_[sink_index],
                             sink_callback_native_tid_syscall_unavailable_by_index_[sink_index],
                             sink_callback_multiple_threads_by_index_[sink_index]);
    }
    sink_callbacks_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_backpressure() noexcept {
    backpressure_count_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_source_backpressure() noexcept {
    source_backpressure_count_.fetch_add(1U, std::memory_order_relaxed);
    record_backpressure();
  }

  // Raw-Tensor rows distinguish asynchronous pool-retirement retries from emit backpressure.
  void record_source_tensor_pool_retry() noexcept {
    source_tensor_pool_retries_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_source_tensor_pool_in_activation_retry() noexcept {
    source_tensor_pool_in_activation_retries_.fetch_add(1U, std::memory_order_relaxed);
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

  // Fixed-rate drop-oldest rows measure accepted offers independently of sink progress.
  void record_publication_offer(const std::uint64_t sequence,
                                const std::int64_t timestamp) noexcept {
    if (!measured(sequence)) {
      return;
    }
    if (sequence == first_measured_sequence()) {
      publication_offer_start_ns_ = timestamp;
    } else {
      const std::int64_t expected_timestamp =
          publication_offer_start_ns_ +
          static_cast<std::int64_t>((sequence - first_measured_sequence()) /
                                    options_.offered_burst_size) *
              source_period().count();
      if (timestamp > expected_timestamp) {
        maximum_publication_offer_lateness_ns_ =
            std::max(maximum_publication_offer_lateness_ns_,
                     static_cast<std::uint64_t>(timestamp - expected_timestamp));
      }
    }
    if (sequence == final_sequence()) {
      publication_offer_end_ns_ = timestamp;
    }
  }

  void record_publication(const std::uint64_t sequence,
                          const std::int64_t post_emit_timestamp) noexcept {
    if (source_post_emit_timestamp_required(sequence)) {
      source_post_emit_timestamps_[measured_index(sequence)] = post_emit_timestamp;
    }
    if (offered_rate_mode(options_)) {
      record_publication_offer(sequence, now_ns());
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
    return tensor_source_pointer(sequence) == reinterpret_cast<std::uintptr_t>(pointer);
  }

  [[nodiscard]] std::uintptr_t tensor_source_pointer(const std::uint64_t sequence) const noexcept {
    if (tensor_source_pointers_ == nullptr || sequence > final_sequence()) {
      return 0U;
    }
    return tensor_source_pointers_[sequence].load(std::memory_order_acquire);
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

  void retain_final_multi_tensors(Tensor first, Tensor second, const std::uint8_t expected) {
    const std::lock_guard lock(completion_mutex_);
    final_multi_tensors_[0].emplace(std::move(first));
    final_multi_tensors_[1].emplace(std::move(second));
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
    const auto stream =
        reinterpret_cast<cudaStream_t>(tensor_consumer_stream_.load(std::memory_order_acquire));
    return validate_tensor_bytes(*first, expected, stream) &&
           validate_tensor_bytes(*second, expected, stream);
  }

  [[nodiscard]] bool validate_retained_multi_tensors() noexcept {
    std::array<std::optional<Tensor>, 2U> retained;
    std::uint8_t expected{};
    {
      const std::lock_guard lock(completion_mutex_);
      if (!final_multi_tensors_[0].has_value() || !final_multi_tensors_[1].has_value()) {
        return false;
      }
      retained[0].emplace(std::move(*final_multi_tensors_[0]));
      retained[1].emplace(std::move(*final_multi_tensors_[1]));
      final_multi_tensors_[0].reset();
      final_multi_tensors_[1].reset();
      expected = final_tensor_expected_byte_;
    }
    const auto stream =
        reinterpret_cast<cudaStream_t>(tensor_consumer_stream_.load(std::memory_order_acquire));
    const bool first_valid = validate_tensor_bytes(*retained[0], expected, stream);
    const bool second_valid = validate_tensor_bytes(*retained[1], expected, stream);
    if (!first_valid || !second_valid) {
      HOLOSCAN_LOG_ERROR(
          "retained multi-Tensor validation failed: expected={} first_valid={} second_valid={}",
          static_cast<unsigned int>(expected),
          first_valid,
          second_valid);
    }
    return first_valid && second_valid;
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

  void publish_pressure_gate_notification_sender(NotificationSender sender) {
    {
      const std::lock_guard lock(completion_mutex_);
      pressure_gate_notification_sender_.emplace(std::move(sender));
    }
    completion_cv_.notify_all();
  }

  [[nodiscard]] std::optional<NotificationSender> take_pressure_gate_notification_sender() {
    std::unique_lock lock(completion_mutex_);
    const bool ready =
        completion_cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
          return pressure_gate_notification_sender_.has_value() || failed_;
        });
    if (!ready || !pressure_gate_notification_sender_.has_value()) {
      return std::nullopt;
    }
    return std::exchange(pressure_gate_notification_sender_, std::nullopt);
  }

  void drive_notifications(const NotificationSender& sender) noexcept {
    while (!source_finished_.load(std::memory_order_acquire) &&
           first_error_.load(std::memory_order_relaxed) == ErrorCode::kSuccess) {
      const std::uint64_t accepted = notifications_accepted_.load(std::memory_order_acquire);
      // Lossy measurement can begin only after every steady-state descriptor backing has
      // circulated through a completed activation.
      const bool descriptor_warmup_pending =
          descriptor_warmup_slots_ != 0U &&
          publications_.load(std::memory_order_acquire) < first_measured_sequence();
      const std::uint64_t completed =
          has_lossy_subscriber(options_) && !descriptor_warmup_pending
              ? source_compute_entries_.load(std::memory_order_acquire)
              : activation_completions_.load(std::memory_order_acquire) +
                    source_tensor_pool_retries_.load(std::memory_order_acquire);
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

  // The exact depth-one forced-HoloMQ diagnostic creates staged ingress without blocking an
  // activation callback. It first publishes one captured sample, waits for the Channel readiness
  // witness, and admits one independent sink gate. Only after that callback retires and establishes
  // ReuseLast ownership does it drive the remaining source notifications with the gate withheld.
  // A final one-at-a-time gate releases the newest survivor. The ordinary notification driver is a
  // separate function so counters-off benchmark rows execute no recurring seam branch.
  void drive_pressured_pod_diagnostic_notifications(
      const NotificationSender& data_sender, const NotificationSender& gate_sender,
      const detail::RuntimePhaseCounters& phase_counters) noexcept {
    const auto deadline = BenchmarkClock::now() + std::chrono::milliseconds{options_.timeout_ms};
    const auto phase_value = [&](const detail::RuntimePhaseCounter counter) {
      const auto snapshot = phase_counters.snapshot();
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    const auto wait_until = [&](const auto& ready) {
      while (first_error() == ErrorCode::kSuccess && !ready()) {
        if (BenchmarkClock::now() >= deadline) {
          fail(ErrorCode::kTimeout);
          return false;
        }
        std::this_thread::yield();
      }
      return first_error() == ErrorCode::kSuccess;
    };
    const auto post_until_accepted = [&](const NotificationSender& sender, const bool data) {
      while (first_error() == ErrorCode::kSuccess) {
        if (BenchmarkClock::now() >= deadline) {
          fail(ErrorCode::kTimeout);
          return false;
        }
        (data ? notification_attempts_ : pressure_gate_notification_attempts_)
            .fetch_add(1U, std::memory_order_relaxed);
        auto posted = sender.post();
        if (posted) {
          (data ? notifications_accepted_ : pressure_gate_notifications_accepted_)
              .fetch_add(1U, std::memory_order_relaxed);
          return true;
        }
        if (posted.error().code == ErrorCode::kBackpressured) {
          (data ? notification_backpressure_ : pressure_gate_notification_backpressure_)
              .fetch_add(1U, std::memory_order_relaxed);
          std::this_thread::yield();
          continue;
        }
        if (posted.error().code == ErrorCode::kNotReady) {
          (data ? notification_not_ready_ : pressure_gate_notification_not_ready_)
              .fetch_add(1U, std::memory_order_relaxed);
          std::this_thread::yield();
          continue;
        }
        fail(posted.error().code);
        return false;
      }
      return false;
    };

    const std::uint64_t initial_readiness =
        phase_value(detail::RuntimePhaseCounter::kChannelReadinessPublished);
    const std::uint64_t initial_witness_retired =
        phase_value(detail::RuntimePhaseCounter::kChannelWitnessRetired);
    if (!post_until_accepted(data_sender, true) ||
        !wait_until([&] { return publications() >= 1U; }) || !wait_until([&] {
          return phase_value(detail::RuntimePhaseCounter::kChannelReadinessPublished) >
                 initial_readiness;
        }) ||
        !wait_until([&] {
          return phase_value(detail::RuntimePhaseCounter::kChannelWitnessRetired) >
                 initial_witness_retired;
        })) {
      return;
    }
    pressure_gate_first_readiness_published_.store(
        phase_value(detail::RuntimePhaseCounter::kChannelReadinessPublished),
        std::memory_order_relaxed);
    pressure_gate_first_witness_retired_.store(
        phase_value(detail::RuntimePhaseCounter::kChannelWitnessRetired),
        std::memory_order_relaxed);

    const std::uint64_t initial_activations_completed =
        phase_value(detail::RuntimePhaseCounter::kActivationsCompleted);
    if (!post_until_accepted(gate_sender, false) ||
        !wait_until([&] { return deliveries() >= 1U && sink_final_sequence(0U) == 1U; }) ||
        !wait_until([&] {
          return phase_value(detail::RuntimePhaseCounter::kActivationsCompleted) >
                 initial_activations_completed;
        })) {
      return;
    }
    pressure_gate_first_delivery_publications_.store(publications(), std::memory_order_relaxed);

    const std::uint64_t initial_reuse_queries =
        phase_value(detail::RuntimePhaseCounter::kTemporalReuseLastRetainedRecordQueries);
    for (std::uint64_t sequence = 2U; sequence <= final_sequence(); ++sequence) {
      const std::uint64_t prior_promotion_passes =
          phase_value(detail::RuntimePhaseCounter::kChannelDirectIngressPromotionPasses);
      const std::uint64_t prior_activations_completed =
          phase_value(detail::RuntimePhaseCounter::kActivationsCompleted);
      if (!post_until_accepted(data_sender, true) ||
          !wait_until([&] { return publications() >= sequence; }) || !wait_until([&] {
            return phase_value(detail::RuntimePhaseCounter::kChannelDirectIngressPromotionPasses) >
                   prior_promotion_passes;
          }) ||
          !wait_until([&] {
            return phase_value(detail::RuntimePhaseCounter::kActivationsCompleted) >
                   prior_activations_completed;
          })) {
        return;
      }
      if (sequence == 2U && !wait_until([&] {
            return phase_value(
                       detail::RuntimePhaseCounter::kTemporalReuseLastRetainedRecordQueries) >
                   initial_reuse_queries;
          })) {
        return;
      }
    }
    if (!wait_until([&] { return source_finished(); })) {
      return;
    }
    pressure_gate_source_completion_publications_.store(publications(), std::memory_order_relaxed);
    pressure_gate_source_retirement_activations_completed_.store(
        phase_value(detail::RuntimePhaseCounter::kActivationsCompleted), std::memory_order_relaxed);

    while (sink_final_sequence(0U) != final_sequence() && first_error() == ErrorCode::kSuccess) {
      const std::uint64_t prior_deliveries = deliveries();
      if (!post_until_accepted(gate_sender, false) ||
          !wait_until([&] { return deliveries() > prior_deliveries; })) {
        return;
      }
      if (sink_final_sequence(0U) > final_sequence() || deliveries() > final_sequence()) {
        fail(ErrorCode::kProtocolError);
        return;
      }
    }
  }

  [[nodiscard]] bool source_finished() const noexcept {
    return source_finished_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t pressure_gate_first_readiness_published() const noexcept {
    return pressure_gate_first_readiness_published_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_first_witness_retired() const noexcept {
    return pressure_gate_first_witness_retired_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_first_delivery_publications() const noexcept {
    return pressure_gate_first_delivery_publications_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_source_completion_publications() const noexcept {
    return pressure_gate_source_completion_publications_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_source_retirement_activations_completed()
      const noexcept {
    return pressure_gate_source_retirement_activations_completed_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_notification_attempts() const noexcept {
    return pressure_gate_notification_attempts_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_notifications_accepted() const noexcept {
    return pressure_gate_notifications_accepted_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_notification_backpressure() const noexcept {
    return pressure_gate_notification_backpressure_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t pressure_gate_notification_not_ready() const noexcept {
    return pressure_gate_notification_not_ready_.load(std::memory_order_relaxed);
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

  [[nodiscard]] bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock lock(completion_mutex_);
    return completion_cv_.wait_for(lock, timeout, [this] { return completed_ || failed_; });
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

  [[nodiscard]] std::uint64_t source_terminal_compute_entries() const noexcept {
    return source_terminal_compute_entries_.load(std::memory_order_relaxed);
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

  [[nodiscard]] std::uint64_t sink_callback_tid(const std::uint32_t sink_index) const noexcept {
    return sink_callback_tids_by_index_ && sink_index < sink_operator_count(options_)
               ? sink_callback_tids_by_index_[sink_index].load(std::memory_order_relaxed)
               : 0U;
  }

  [[nodiscard]] bool sink_callback_multiple_threads(const std::uint32_t sink_index) const noexcept {
    return sink_callback_multiple_threads_by_index_ && sink_index < sink_operator_count(options_) &&
           sink_callback_multiple_threads_by_index_[sink_index].load(std::memory_order_relaxed);
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

  // Lossy fanout rows contribute one accepted delivery opportunity per sink.
  [[nodiscard]] std::uint64_t consumer_visible_replacements() const noexcept {
    std::uint64_t replacements{};
    for (std::uint32_t index = 0U; index < options_.fanout; ++index) {
      replacements += sink_replacements(index);
    }
    return replacements;
  }

  [[nodiscard]] std::uint64_t sink_deliveries(const std::uint32_t sink_index) const noexcept {
    return sink_deliveries_[sink_index];
  }

  [[nodiscard]] std::uint64_t sink_final_sequence(const std::uint32_t sink_index) const noexcept {
    return sink_final_sequences_[sink_index];
  }

  [[nodiscard]] std::uint64_t sink_replacements(const std::uint32_t sink_index) const noexcept {
    if (!lossy_subscriber(options_, sink_index)) {
      return 0U;
    }
    const std::uint64_t observed = sink_deliveries(sink_index);
    return publications() >= observed ? publications() - observed : 0U;
  }

  // Mixed-depth and lossy fanout rows require independent evidence from every leg.
  [[nodiscard]] bool sink_evidence_valid(const std::uint64_t expected_sequence,
                                         const std::uint64_t expected_deliveries) const noexcept {
    std::uint64_t aggregate{};
    for (std::uint32_t index = 0U; index < options_.fanout; ++index) {
      const std::uint64_t count = sink_deliveries(index);
      const bool count_valid = lossy_subscriber(options_, index)
                                   ? count > 0U && count <= publications()
                                   : count == expected_deliveries;
      if (!count_valid || sink_final_sequence(index) != expected_sequence) {
        return false;
      }
      aggregate += count;
    }
    return aggregate == deliveries();
  }

  [[nodiscard]] bool lossy_activation_completions_valid() const noexcept {
    if (!has_lossy_subscriber(options_)) {
      return true;
    }
    if (options_.fanout == 1U) {
      return activation_completions() == deliveries();
    }
    const std::uint64_t minimum_deliveries = std::ranges::min(sink_deliveries_);
    return activation_completions() > 0U && activation_completions() <= minimum_deliveries;
  }

  [[nodiscard]] bool every_sink_has_replacements() const noexcept {
    return std::ranges::all_of(sink_deliveries_,
                               [&](const std::uint64_t count) { return count < publications(); });
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

  [[nodiscard]] std::uint64_t source_tensor_pool_retries() const noexcept {
    return source_tensor_pool_retries_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t source_tensor_pool_in_activation_retries() const noexcept {
    return source_tensor_pool_in_activation_retries_.load(std::memory_order_relaxed);
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

  // These fields supply the runner's fixed-rate drop-oldest schedule quality gate.
  [[nodiscard]] std::int64_t publication_offer_start_ns() const noexcept {
    return publication_offer_start_ns_;
  }

  [[nodiscard]] std::int64_t publication_offer_end_ns() const noexcept {
    return publication_offer_end_ns_;
  }

  [[nodiscard]] std::uint64_t maximum_publication_offer_lateness_ns() const noexcept {
    return maximum_publication_offer_lateness_ns_;
  }

  [[nodiscard]] double accepted_publication_offer_rate_per_second() const noexcept {
    if (!offered_rate_mode(options_) || options_.samples <= options_.offered_burst_size ||
        publication_offer_end_ns_ <= publication_offer_start_ns_) {
      return 0.0;
    }
    return static_cast<double>(options_.samples - options_.offered_burst_size) * 1.0e9 /
           static_cast<double>(publication_offer_end_ns_ - publication_offer_start_ns_);
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
      return kMaximumImageSinkComputeAllocations * options_.fanout;
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
  // Per-leg counts enable fanout, mixed-depth, and drop-oldest accounting rows.
  std::vector<std::uint64_t> sink_deliveries_;
  std::vector<std::uint64_t> sink_final_sequences_;
  // Per-sink identities prove that fan-out partitions retain stable, distinct callback owners.
  std::unique_ptr<std::atomic<std::uint64_t>[]> sink_callback_tids_by_index_;
  std::unique_ptr<std::atomic_bool[]> sink_callback_identity_is_native_by_index_;
  std::unique_ptr<std::atomic_bool[]> sink_callback_native_tid_syscall_unavailable_by_index_;
  std::unique_ptr<std::atomic_bool[]> sink_callback_multiple_threads_by_index_;
  std::size_t sequence_sink_delivery_count_{};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<std::atomic<std::uint32_t>[]> sequence_sink_deliveries_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<std::atomic<std::uintptr_t>[]> tensor_source_pointers_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::array<std::unique_ptr<std::atomic<std::uintptr_t>[]>, 2U> multi_tensor_source_pointers_;
  std::vector<std::uint8_t> tensor_validation_buffer_;
  std::shared_ptr<PreinitializedTensorStorage> image_storage_;
  std::shared_ptr<Tensor> image_tensor_;
  std::atomic<std::uintptr_t> tensor_consumer_stream_{};
  std::uint64_t source_callbacks_{};
  std::atomic<std::uint64_t> source_compute_entries_{};
  std::atomic<std::uint64_t> source_terminal_compute_entries_{};
  std::atomic<std::int64_t> first_callback_entry_ns_{};
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
  // Device-backed raw-Tensor rows can temporarily lap the bounded backing pool.
  std::atomic<std::uint64_t> source_tensor_pool_retries_{};
  std::atomic<std::uint64_t> source_tensor_pool_in_activation_retries_{};
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
  // Accepted-offer timing is populated only by fixed-rate drop-oldest fanout rows.
  std::int64_t publication_offer_start_ns_{};
  std::int64_t publication_offer_end_ns_{};
  std::uint64_t maximum_publication_offer_lateness_ns_{};
  mutable std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  bool completed_{};
  bool failed_{};
  std::size_t descriptor_warmup_slots_{};
  std::optional<Tensor> final_tensor_;
  std::optional<Tensor> final_second_tensor_;
  std::array<std::optional<Tensor>, 2U> final_multi_tensors_;
  std::uint8_t final_tensor_expected_byte_{};
  std::optional<NotificationSender> notification_sender_;
  std::optional<NotificationSender> pressure_gate_notification_sender_;
  std::atomic<std::uint64_t> pressure_gate_first_readiness_published_{};
  std::atomic<std::uint64_t> pressure_gate_first_witness_retired_{};
  std::atomic<std::uint64_t> pressure_gate_first_delivery_publications_{};
  std::atomic<std::uint64_t> pressure_gate_source_completion_publications_{};
  std::atomic<std::uint64_t> pressure_gate_source_retirement_activations_completed_{};
  std::atomic<std::uint64_t> pressure_gate_notification_attempts_{};
  std::atomic<std::uint64_t> pressure_gate_notifications_accepted_{};
  std::atomic<std::uint64_t> pressure_gate_notification_backpressure_{};
  std::atomic<std::uint64_t> pressure_gate_notification_not_ready_{};
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
      notification_driver_used(options) ? options.notification_capacity : 1U;
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
                     options.scenario == Scenario::kSchema ||
                     options.scenario == Scenario::kThirdPartySchema
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
    std::optional<std::int64_t> preceding_cut;
    std::int64_t candidate_frozen{};
    std::int64_t candidate_selected{};
    std::int64_t input_selection_prepared{};
    std::int64_t activation_prepared{};
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
        if (previous_cut != 0) {
          activation.preceding_cut = previous_cut;
        }
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
      case detail::RuntimePhaseTracePoint::kCandidateSelected:
        if (activation.candidate_frozen == 0 || activation.candidate_selected != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.candidate_selected = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kInputSelectionPrepared:
        if (activation.candidate_selected == 0 || activation.input_selection_prepared != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.input_selection_prepared = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationPrepared:
        if (activation.input_selection_prepared == 0 || activation.activation_prepared != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.activation_prepared = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchStarted:
        if (activation.candidate_frozen == 0 || activation.dispatch_started != 0 ||
            ((activation.candidate_selected != 0 || activation.input_selection_prepared != 0 ||
              activation.activation_prepared != 0) &&
             (activation.candidate_selected == 0 || activation.input_selection_prepared == 0 ||
              activation.activation_prepared == 0)) ||
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
        if (activation.preceding_cut.has_value()) {
          append_timestamp_delta(receiver.previous_cut_to_candidate,
                                 *activation.preceding_cut,
                                 activation.candidate_frozen,
                                 summary.valid);
        }
        if (activation.candidate_selected != 0) {
          append_timestamp_delta(receiver.candidate_to_selection,
                                 activation.candidate_frozen,
                                 activation.candidate_selected,
                                 summary.valid);
          append_timestamp_delta(receiver.selection_to_input_preparation,
                                 activation.candidate_selected,
                                 activation.input_selection_prepared,
                                 summary.valid);
          append_timestamp_delta(receiver.input_preparation_to_activation,
                                 activation.input_selection_prepared,
                                 activation.activation_prepared,
                                 summary.valid);
          append_timestamp_delta(receiver.activation_to_dispatch_start,
                                 activation.activation_prepared,
                                 activation.dispatch_started,
                                 summary.valid);
        }
        append_timestamp_delta(receiver.candidate_to_dispatch_start,
                               activation.candidate_frozen,
                               activation.dispatch_started,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_start_to_callback_entry,
                               activation.dispatch_started,
                               activation.callback_entered,
                               summary.valid);
        append_timestamp_delta(receiver.callback_entry_to_return,
                               activation.callback_entered,
                               activation.callback_returned,
                               summary.valid);
        append_timestamp_delta(receiver.callback_return_to_dispatch_return,
                               activation.callback_returned,
                               activation.dispatch_returned,
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
          // Backpressure retries advance the Runtime activation sequence without advancing the
          // benchmark publication sequence. Keep this optional attribution only while the two
          // sequence domains still identify a timestamp inside the current callback.
          if (pre_return >= activation.callback_entered &&
              pre_return <= activation.callback_returned) {
            append_timestamp_delta(receiver.callback_pre_return_to_dispatch_return,
                                   pre_return,
                                   activation.dispatch_returned,
                                   summary.valid);
          }
        }
        previous_cut = record.timestamp_ns;
        activation = {};
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorEmitStarted:
        if (record.output_port_id == 0U || record.source_sequence == 0U) {
          summary.valid = false;
          return summary;
        }
        if (const auto pending = pending_tensor_emit(record);
            pending != pending_tensor_emits.end()) {
          // A recoverable backpressure return has no terminal emit trace point. Starting the same
          // sequence again therefore closes that incomplete attempt before tracking the retry.
          pending_tensor_emits.erase(pending);
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
        if (record.output_port_id == 0U || record.source_sequence == 0U) {
          summary.valid = false;
          return summary;
        }
        if (const auto pending = pending_inline_payload_emit(record);
            pending != pending_inline_payload_emits.end()) {
          // Inline payload emission has the same recoverable-backpressure trace shape as Tensor.
          pending_inline_payload_emits.erase(pending);
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
  const bool terminal_frozen_candidate =
      activation.candidate_frozen != 0 && activation.candidate_selected == 0 &&
      activation.input_selection_prepared == 0 && activation.activation_prepared == 0 &&
      activation.dispatch_started == 0 && activation.callback_entered == 0 &&
      activation.callback_returned == 0 && activation.dispatch_returned == 0 &&
      activation.retirement_completed == 0;
  summary.terminal_frozen_candidates = terminal_frozen_candidate ? 1U : 0U;
  summary.valid = (activation.candidate_frozen == 0 || terminal_frozen_candidate) &&
                  summary.dropped == 0U && summary.unmatched_readiness_polls == 0U &&
                  pending_tensor_emits.empty() && pending_inline_payload_emits.empty();
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

[[nodiscard]] LifecycleStatus acquire_benchmark_pressure_gate_notification_sender(
    LifecycleContext& context, NotificationSource& source) noexcept {
  auto sender = context.notification_sender(source);
  if (!sender) {
    run_state().fail(sender.error().code);
    return LifecycleStatus::kFatalFailure;
  }
  run_state().publish_pressure_gate_notification_sender(std::move(*sender));
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
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.source_only_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    run_controlled_callback_work(state.options().callback_work_us);
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

// Drives POD, signal, generated-schema, third-party-schema, and chain-source rows. Only the
// fixed-rate POD drop-oldest fanout row enables its bounded clock burst.
template <typename Payload>
class RuntimeBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.output(output_, "output")
        .max_emits_per_compute(offered_rate_mode(options) ? options.offered_burst_size : 1U);
    if (options.multi_output || options.multi_input) {
      spec.output(second_output_, "second-output").max_emits_per_compute(1U);
    }
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

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    run_controlled_callback_work(state.options().callback_work_us);

    std::uint64_t emit_count =
        offered_rate_mode(state.options()) ? state.options().offered_burst_size : 1U;
    emit_count = std::min(emit_count, state.final_sequence() - next_sequence_ + 1U);
    if (next_sequence_ <= state.allocation_probe_sequence() &&
        next_sequence_ + emit_count - 1U > state.allocation_probe_sequence()) {
      emit_count = state.allocation_probe_sequence() - next_sequence_ + 1U;
    }
    const std::optional<TimePoint> paired_capture_time =
        state.options().multi_input ? std::optional{context.activation_time()} : std::nullopt;
    for (std::uint64_t index = 0U; index < emit_count; ++index) {
      auto emitted = publish_one(state, paired_capture_time);
      if (!emitted) {
        return make_unexpected(std::move(emitted.error()));
      }
      if (!*emitted) {
        return {};
      }
    }
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_notification_sender(context, notification_);
  }

 private:
  [[nodiscard]] expected<bool, Error> publish_one(
      BenchmarkRunState& state, const std::optional<TimePoint>& paired_capture_time) {
    const Payload payload = make_benchmark_payload<Payload>(next_sequence_);
    const EmitOptions emit_options{
        .capture_time = paired_capture_time,
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
        .flags = EmitFlags::kDegraded,
    };
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;

    const auto emit_payload = [&]() -> expected<void, Error> {
      auto first = output_.emit(payload, emit_options);
      if (!first || (!state.options().multi_output && !state.options().multi_input)) {
        return first;
      }
      auto second = second_output_.emit(payload, emit_options);
      if (!second && (second.error().code == ErrorCode::kBackpressured ||
                      second.error().code == ErrorCode::kNotReady)) {
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate,
                                     "second output publication was not accepted"});
      }
      return second;
    };
    expected<void, Error> emitted;
    state.prepare_source_pre_emit_timestamp(
        next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      emitted = emit_payload();
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = emit_payload();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured) {
        state.record_source_backpressure();
        return false;
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
    return true;
  }

  Output<Payload> output_;
  Output<Payload> second_output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// The counter-enabled pressure diagnostic needs valid capture coordinates without adding that
// behavior to the ordinary benchmark source. Its bounded driver posts exactly the requested data
// notifications while the independently gated sink drains a production-shaped staged tail. This
// stopped, nonterminal row proves exact final-data delivery; focused terminal tests own EOS.
class RuntimeBenchmarkPressuredPODDiagnosticSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output").max_emits_per_compute(1U);
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(run_state().options().notification_capacity)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm,
                           &RuntimeBenchmarkPressuredPODDiagnosticSource::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    state.record_source_compute_entry(false);
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    run_controlled_callback_work(state.options().callback_work_us);

    const BenchmarkPOD64 payload = make_benchmark_payload<BenchmarkPOD64>(next_sequence_);
    const EmitOptions emit_options{
        .capture_time = context.activation_time(),
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
        .flags = EmitFlags::kDegraded,
    };
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.prepare_source_pre_emit_timestamp(
        next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);

    expected<void, Error> emitted;
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
  Output<BenchmarkPOD64> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// Receives the scalar, POD, and generated-schema rows, including their per-leg lossy evidence.
template <typename Payload>
class RuntimeBenchmarkSink final : public Operator<> {
 public:
  explicit RuntimeBenchmarkSink(const std::uint32_t sink_index = 0U) : sink_index_(sink_index) {}

  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input").queue_depth(subscriber_queue_depth(options, sink_index_));
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
    state.record_sink_callback(sink_index_);
    run_controlled_callback_work(state.options().callback_work_us);

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
        (lossy_subscriber(state.options(), sink_index_) || sequence == next_sequence_);
    const bool sequence_valid = [&sample, sequence] {
      if constexpr (std::is_same_v<Payload, BenchmarkPOD64>) {
        return sample->data.sequence == sequence;
      } else if constexpr (std::is_same_v<Payload, schema::TimeT>) {
        return sample->data.time_ns == static_cast<std::int64_t>(sequence);
      } else if constexpr (std::is_same_v<Payload, ThirdPartyTelemetry>) {
        return sample->data.sequence == sequence;
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

// Creates deterministic depth-one staged ingress without adding a seam branch to the ordinary
// RuntimeBenchmarkSink callback measured by counters-off A/C rows. An independent notification
// gate owns activation authority while arrivals populate LatestWithin retention. Every gate is
// posted only after the driver has observed the preceding Channel/callback acknowledgement, so no
// callback waits for another activation while the Runtime serial dispatch token is held.
class RuntimeBenchmarkPressuredPODDiagnosticSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(1U);
    spec.notification_source(pressure_gate_, "pressure-gate")
        .capacity(1U)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeBenchmarkPressuredPODDiagnosticSink::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = pressure_gate_})
        .input(input_, LatestWithin{.max_age = std::chrono::seconds{30}})
        .on_missing(input_, ReuseLast{.max_age = std::chrono::seconds{30}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    state.record_sink_callback();
    run_controlled_callback_work(state.options().callback_work_us);

    expected<Sample<BenchmarkPOD64>, Error> sample = make_unexpected(Error{});
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
    state.record_sink_callback_entry(0U, sequence, 0);
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const bool validate_payload = state.options().validation == PayloadValidation::kAll ||
                                  (state.options().validation == PayloadValidation::kFinal &&
                                   sequence == state.final_sequence());
    const bool valid = sequence != 0U && sequence >= next_sequence_ &&
                       sample->metadata.trace_id == sequence && sample->metadata.degraded() &&
                       sample->data.sequence == sequence &&
                       (!validate_payload || payload_valid(sample->data));
    if (!valid) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    state.record_delivery(0U, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    return acquire_benchmark_pressure_gate_notification_sender(context, pressure_gate_);
  }

 private:
  Input<BenchmarkPOD64> input_;
  NotificationSource pressure_gate_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// Covers the two-output/one-target applicability row. One OnEach activation consumes both paired
// POD samples, so the row distinguishes two input reads from one sink callback.
class RuntimeBenchmarkMultiInputPODSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(first_, "first").queue_depth(subscriber_queue_depth(options, 0U));
    spec.input(second_, "second").queue_depth(subscriber_queue_depth(options, 1U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    // Canonical source-port order ingests `first_` before `second_`; triggering on the latter
    // proves partial multi-input readiness is retained until the exact peer can be selected.
    result.trigger(OnEach{second_}).input(first_, Exact{}).on_missing(first_, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        is_latency_workload(state.options().workload) ? now_ns() : 0;
    state.record_sink_callback();
    run_controlled_callback_work(state.options().callback_work_us);

    expected<Sample<BenchmarkPOD64>, Error> first = make_unexpected(Error{});
    expected<Sample<BenchmarkPOD64>, Error> second = make_unexpected(Error{});
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      first = first_.receive();
      second = second_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      first = first_.receive();
      second = second_.receive();
    }
    if (!first || !second) {
      Error error = !first ? std::move(first.error()) : std::move(second.error());
      state.fail(error.code);
      return make_unexpected(std::move(error));
    }

    const std::uint64_t sequence = first->metadata.frame_id;
    const auto valid_sample = [&](const Sample<BenchmarkPOD64>& sample) {
      return sample.metadata.frame_id == sequence && sample.metadata.trace_id == sequence &&
             sample.metadata.degraded() && sample.data.sequence == sequence;
    };
    const bool validate_payload = state.options().validation == PayloadValidation::kAll ||
                                  (state.options().validation == PayloadValidation::kFinal &&
                                   sequence == state.final_sequence());
    if (sequence == 0U || sequence != next_sequence_ || !valid_sample(*first) ||
        !valid_sample(*second) ||
        (validate_payload && (!payload_valid(first->data) || !payload_valid(second->data)))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    state.record_sink_callback_entry(0U, sequence, callback_timestamp);
    state.record_sink_callback_entry(1U, sequence, callback_timestamp);
    const std::int64_t first_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    state.record_delivery(0U, sequence, first_timestamp);
    const std::int64_t second_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    state.record_delivery(1U, sequence, second_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<BenchmarkPOD64> first_;
  Input<BenchmarkPOD64> second_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// Raw-Tensor rows add production loan/write timing and backing/stream lifetime evidence.
class RuntimeTensorBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.output(output_, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = tensor_memory_kind(options.tensor_storage),
                    .dtype = kTensorByteDtype,
                    .rank = 1U,
                },
            .bounds = tensor_bounds(options.tensor_bytes),
            .storage = TensorOutputStorage::kRuntimePool,
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

  void start() override {
    initialized_backings_.reserve(run_state().descriptor_warmup_slots());
    injected_pool_retries_remaining_ =
        run_state().options().test_source_tensor_pool_in_activation_retries;
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    std::int64_t source_timestamp{};
    std::int64_t callback_timestamp{};
    bool callback_started{};
    bool callback_recorded{};
    bool tensor_pool_exhausted{};
    const std::array<std::int64_t, 1U> shape{
        static_cast<std::int64_t>(state.options().tensor_bytes)};

    const auto publish = [&]() -> expected<void, Error> {
      tensor_pool_exhausted = false;
      if (!callback_started) {
        callback_timestamp =
            state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
        run_controlled_callback_work(state.options().callback_work_us);
        callback_started = true;
      }
      source_timestamp = state.source_timestamp_required(next_sequence_) ? now_ns() : 0;
      if (injected_pool_retries_remaining_ != 0U) {
        --injected_pool_retries_remaining_;
        tensor_pool_exhausted = true;
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      auto loan =
          output_.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kTensorByteDtype});
      const std::int64_t loan_acquired_timestamp = source_timestamp != 0 ? now_ns() : 0;
      if (!loan) {
        tensor_pool_exhausted = loan.error().code == ErrorCode::kResourceExhausted;
        return make_unexpected(std::move(loan.error()));
      }
      // A notification-driven activation can retry publication while a bounded Tensor backing
      // retires. The retry remains part of this compute callback and must not be counted as a
      // second callback or repeat the controlled callback workload.
      if (!callback_recorded) {
        state.record_source_callback(next_sequence_, callback_timestamp);
        callback_recorded = true;
      }
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
    while (!emitted && state.notification_driven() &&
           (emitted.error().code == ErrorCode::kBackpressured ||
            emitted.error().code == ErrorCode::kNotReady ||
            emitted.error().code == ErrorCode::kResourceExhausted)) {
      if (emitted.error().code == ErrorCode::kResourceExhausted && tensor_pool_exhausted) {
        state.record_source_tensor_pool_in_activation_retry();
      } else {
        state.record_backpressure();
      }
      if (context.cancellation().stop_requested()) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      std::this_thread::yield();
      emitted = publish();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kResourceExhausted && tensor_pool_exhausted) {
        state.record_source_tensor_pool_retry();
        return {};
      }
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady ||
          emitted.error().code == ErrorCode::kResourceExhausted) {
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
  std::uint32_t injected_pool_retries_remaining_{};
  std::vector<void*> initialized_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeTensorBenchmarkSink final : public Operator<> {
 public:
  explicit RuntimeTensorBenchmarkSink(const std::uint32_t sink_index = 0U)
      : sink_index_(sink_index) {}

  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input")
        .queue_depth(subscriber_queue_depth(options, sink_index_))
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = tensor_memory_kind(options.tensor_storage),
                    .dtype = kTensorByteDtype,
                    .rank = 1U,
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override {
    const std::size_t warmup_slots = run_state().descriptor_warmup_slots();
    retained_warmup_tensors_.reserve(warmup_slots == 0U ? 0U : warmup_slots - 1U);
    observed_backings_.reserve(warmup_slots);
    pool_warmup_complete_ = warmup_slots == 0U;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        is_latency_workload(state.options().workload) ? now_ns() : 0;
    state.record_sink_callback(sink_index_);
    run_controlled_callback_work(state.options().callback_work_us);
    expected<Sample<Tensor>, Error> sample = make_unexpected(Error{});
    const std::int64_t receive_start_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
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
    state.prepare_tensor_receive_start_timestamp(sequence, receive_start_timestamp);
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
      const DLDevice actual_device = sample->data.device();
      const std::string diagnostic = fmt::format(
          "BENCHMARK_TENSOR_VALIDATION_FAILED "
          "sequence={} "
          "expected_sequence={} "
          "metadata_valid={} "
          "descriptor_valid={} "
          "pointer_valid={} "
          "source_pointer={} "
          "received_pointer={} "
          "rank={} "
          "bytes={} "
          "device_type={} "
          "device_id={}",
          sequence,                                               // sequence
          next_sequence_,                                         // expected_sequence
          static_cast<int>(metadata_valid),                       // metadata_valid
          static_cast<int>(descriptor_valid),                     // descriptor_valid
          static_cast<int>(pointer_valid),                        // pointer_valid
          state.tensor_source_pointer(sequence),                  // source_pointer
          reinterpret_cast<std::uintptr_t>(sample->data.data()),  // received_pointer
          sample->data.ndim(),                                    // rank
          sample->data.nbytes(),                                  // bytes
          static_cast<int>(actual_device.device_type),            // device_type
          actual_device.device_id);                               // device_id
      return make_unexpected(Error{ErrorCode::kProtocolError, diagnostic});
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
        // Keep backing ownership without retaining the finite Sample read claim.
        const std::size_t retention_limit =
            std::min(warmup_slots - 1U, static_cast<std::size_t>(state.options().queue_depth) + 1U);
        if (retained_warmup_tensors_.size() == retention_limit) {
          retained_warmup_tensors_.erase(retained_warmup_tensors_.begin());
        }
        retained_warmup_tensors_.push_back(sample->data);
      }
    }
    state.record_delivery(sink_index_, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<Tensor> input_;
  std::uint32_t sink_index_{};
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
    const TensorOutputSpec field_contract{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = tensor_memory_kind(options.tensor_storage),
                .dtype = kTensorByteDtype,
                .rank = 1U,
            },
        .bounds = tensor_bounds(options.tensor_bytes),
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
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
    injected_pool_retries_remaining_ =
        run_state().options().test_source_tensor_pool_in_activation_retries;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    std::int64_t source_timestamp{};
    std::int64_t callback_timestamp{};
    bool callback_started{};
    bool callback_recorded{};
    bool tensor_pool_exhausted{};

    const auto publish = [&]() -> expected<void, Error> {
      tensor_pool_exhausted = false;
      if (!callback_started) {
        callback_timestamp =
            state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
        run_controlled_callback_work(state.options().callback_work_us);
        callback_started = true;
      }
      source_timestamp = state.source_timestamp_required(next_sequence_) ? now_ns() : 0;
      if (injected_pool_retries_remaining_ != 0U) {
        --injected_pool_retries_remaining_;
        tensor_pool_exhausted = true;
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
      const std::array requests{
          tensor_shape(MultiTensorFields::first, shape),
          tensor_shape(MultiTensorFields::second, shape),
      };
      auto bundle = output_.allocate_tensors(requests);
      if (!bundle) {
        tensor_pool_exhausted = bundle.error().code == ErrorCode::kResourceExhausted;
        return make_unexpected(std::move(bundle.error()));
      }
      // Acquiring a pooled bundle starts the publication attempt. Allocation exhaustion before
      // this point is a retry, not a completed source callback.
      if (!callback_recorded) {
        state.record_source_callback(next_sequence_, callback_timestamp);
        callback_recorded = true;
      }
      const bool descriptor_only =
          state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
      const std::uint8_t fill =
          descriptor_only ? kMultiTensorPayloadByte : static_cast<std::uint8_t>(next_sequence_);
      std::array<void*, 2U> field_pointers{};
      std::array<bool, 2U> initialized_backing{};
      const auto write_field = [&](const std::size_t index) {
        return [&, index](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          field_pointers[index] = writer.data();
          initialized_backing[index] =
              descriptor_only && std::find(initialized_backings_[index].begin(),
                                           initialized_backings_[index].end(),
                                           writer.data()) == initialized_backings_[index].end();
          if (!descriptor_only || initialized_backing[index]) {
            if (state.options().tensor_storage == TensorStorage::kDevice) {
              if (cudaMemsetAsync(writer.data(),
                                  static_cast<int>(fill),
                                  writer.byte_size(),
                                  context.cuda_stream().get()) != cudaSuccess) {
                return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
              }
            } else {
              std::memset(writer.data(), static_cast<int>(fill), writer.byte_size());
            }
          }
          return {};
        };
      };
      expected<void, Error> committed;
      if (state.options().tensor_storage == TensorStorage::kDevice) {
        const ::cuda::stream_ref stream = context.cuda_stream();
        if (stream.get() == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        committed =
            bundle->write_all(device_field(MultiTensorFields::first, stream, write_field(0U)),
                              device_field(MultiTensorFields::second, stream, write_field(1U)));
      } else {
        committed = bundle->write_all(host_field(MultiTensorFields::first, write_field(0U)),
                                      host_field(MultiTensorFields::second, write_field(1U)));
      }
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
    while (!emitted && state.notification_driven() &&
           (emitted.error().code == ErrorCode::kBackpressured ||
            emitted.error().code == ErrorCode::kNotReady ||
            emitted.error().code == ErrorCode::kResourceExhausted)) {
      if (emitted.error().code == ErrorCode::kResourceExhausted && tensor_pool_exhausted) {
        state.record_source_tensor_pool_in_activation_retry();
      } else {
        state.record_backpressure();
      }
      if (context.cancellation().stop_requested()) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      std::this_thread::yield();
      emitted = publish();
    }
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kResourceExhausted && tensor_pool_exhausted) {
        state.record_source_tensor_pool_retry();
        return {};
      }
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady ||
          emitted.error().code == ErrorCode::kResourceExhausted) {
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
  std::uint32_t injected_pool_retries_remaining_{};
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
    retained_warmup_tensors_.reserve(warmup_slots == 0U ? 0U : warmup_slots - 1U);
    for (auto& inventory : observed_backings_) {
      inventory.reserve(warmup_slots);
    }
    pool_warmup_complete_ = warmup_slots == 0U;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        state.sink_timestamp_required(next_sequence_) ? now_ns() : 0;
    state.record_sink_callback();
    run_controlled_callback_work(state.options().callback_work_us);
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
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      state.record_tensor_consumer_stream(context.cuda_stream().get());
    }
    state.record_sink_callback_entry(0U, sequence, callback_timestamp);
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
                                  state.options().tensor_storage,
                                  state.options().tensor_device,
                                  false) &&
        tensor_descriptor_matches(*sample->data.second,
                                  state.options().tensor_bytes,
                                  state.options().tensor_storage,
                                  state.options().tensor_device,
                                  false) &&
        state.multi_tensor_pointer_matches(0U, sequence, sample->data.first->data()) &&
        state.multi_tensor_pointer_matches(1U, sequence, sample->data.second->data());
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kMultiTensorPayloadByte
            : static_cast<std::uint8_t>(sequence);
    if (!tensors_valid ||
        (validate_payload && (!state.validate_tensor_bytes(
                                  *sample->data.first, expected, context.cuda_stream().get()) ||
                              !state.validate_tensor_bytes(
                                  *sample->data.second, expected, context.cuda_stream().get())))) {
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
        // Keep backing ownership without retaining the finite Sample read claim.
        const std::size_t retention_limit =
            std::min(warmup_slots - 1U, static_cast<std::size_t>(state.options().queue_depth) + 1U);
        if (retained_warmup_tensors_.size() == retention_limit) {
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
  void configure_field(TensorFieldInputSpecBuilder& builder) const {
    builder.expects_tensor(TensorInputSpec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = tensor_memory_kind(run_state().options().tensor_storage),
                .dtype = kTensorByteDtype,
                .rank = 1U,
            },
    });
  }

  Input<MultiTensorMessage> input_;
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::vector<std::array<Tensor, 2U>> retained_warmup_tensors_;
  std::array<std::vector<const void*>, 2U> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

// Image rows isolate descriptor publication while reusing one preinitialized Tensor backing.
class RuntimeImageBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.output(output_, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = tensor_memory_kind(options.tensor_storage),
                    .dtype = kTensorByteDtype,
                    .rank = 1U,
                },
            .bounds = tensor_bounds(options.tensor_bytes),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
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
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    run_controlled_callback_work(state.options().callback_work_us);
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
  explicit RuntimeImageBenchmarkSink(const std::uint32_t sink_index = 0U)
      : sink_index_(sink_index) {}

  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input")
        .queue_depth(subscriber_queue_depth(options, sink_index_))
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = tensor_memory_kind(options.tensor_storage),
                    .dtype = kTensorByteDtype,
                    .rank = 1U,
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        is_latency_workload(state.options().workload) ? now_ns() : 0;
    state.record_sink_callback(sink_index_);
    run_controlled_callback_work(state.options().callback_work_us);
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
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      state.record_tensor_consumer_stream(context.cuda_stream().get());
    }
    state.record_sink_callback_entry(sink_index_, sequence, callback_timestamp);
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
      tensor_valid = tensor_descriptor_matches(*tensor,
                                               state.options().tensor_bytes,
                                               state.options().tensor_storage,
                                               state.options().tensor_device,
                                               false) &&
                     tensor->data() == state.image_tensor()->data();
    }
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    if (!metadata_valid || !tensor_valid ||
        (validate_payload &&
         !state.validate_tensor_bytes(*tensor, kImagePayloadByte, context.cuda_stream().get()))) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    if (state.options().validation == PayloadValidation::kFinal &&
        sequence == state.final_sequence()) {
      state.retain_final_tensor(*tensor, kImagePayloadByte);
    }
    state.record_delivery(sink_index_, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  Input<schema::ImageT> input_;
  std::uint32_t sink_index_{};
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

// Multi-Tensor loan and staging rows share the same preallocated two-field backing geometry.
class RuntimeMultiTensorModeBenchmarkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    const TensorOutputSpec field_contract{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = tensor_memory_kind(options.tensor_storage),
                .dtype = kTensorByteDtype,
                .rank = 1U,
            },
        .bounds = tensor_bounds(options.tensor_bytes),
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    auto& builder = spec.output(output_, "output").max_emits_per_compute(1U);
    builder.tensor_fields(std::array{MultiTensorFields::first, MultiTensorFields::second},
                          field_contract);
    if (run_state().notification_driven()) {
      spec.notification_source(notification_, "benchmark-notification")
          .capacity(run_state().options().notification_capacity)
          .sender_reference_capacity(1U);
      spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeMultiTensorModeBenchmarkSource::arm);
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
    for (auto& inventory : initialized_loan_backings_) {
      inventory.reserve(kMaximumDescriptorTensorFieldPoolSlots);
    }
    if (run_state().options().scenario == Scenario::kMultiTensorStaging &&
        !initialize_foreign_tensors()) {
      run_state().fail(ErrorCode::kResourceExhausted);
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const bool terminal = next_sequence_ > state.final_sequence();
    state.record_source_compute_entry(terminal);
    if (terminal) {
      return {};
    }
    const std::int64_t callback_timestamp =
        state.notification_latency() && state.measured(next_sequence_) ? now_ns() : 0;
    state.record_source_callback(next_sequence_, callback_timestamp);
    run_controlled_callback_work(state.options().callback_work_us);
    const std::int64_t source_timestamp =
        state.source_timestamp_required(next_sequence_) ? now_ns() : 0;

    const auto publish = [&]() {
      return state.options().scenario == Scenario::kMultiTensorLoan ? publish_loan(context)
                                                                    : publish_staged();
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
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady ||
          emitted.error().code == ErrorCode::kResourceExhausted) {
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
  struct ForeignTensorSlot {
    std::shared_ptr<PreinitializedTensorStorage> first_storage;
    std::shared_ptr<PreinitializedTensorStorage> second_storage;
    std::shared_ptr<Tensor> first;
    std::shared_ptr<Tensor> second;
  };

  [[nodiscard]] bool initialize_foreign_tensors() {
    try {
      const std::size_t bytes = run_state().options().tensor_bytes;
      const std::size_t slots =
          static_cast<std::size_t>(run_state().options().queue_depth) * 2U + 3U;
      foreign_slots_.resize(slots);
      for (auto& slot : foreign_slots_) {
        auto first = make_preinitialized_byte_tensor(bytes,
                                                     run_state().options().tensor_storage,
                                                     run_state().options().tensor_device,
                                                     kMultiTensorModePayloadByte,
                                                     slot.first_storage,
                                                     run_state().options().fanout);
        auto second = make_preinitialized_byte_tensor(bytes,
                                                      run_state().options().tensor_storage,
                                                      run_state().options().tensor_device,
                                                      kMultiTensorModePayloadByte,
                                                      slot.second_storage,
                                                      run_state().options().fanout);
        if (!first || !second) {
          return false;
        }
        slot.first = std::move(*first);
        slot.second = std::move(*second);
        run_state().record_descriptor_backing_initialization(0U);
        run_state().record_descriptor_backing_initialization(0U);
      }
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }

  [[nodiscard]] EmitOptions emit_options() const noexcept {
    return EmitOptions{
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
        .flags = EmitFlags::kDegraded,
    };
  }

  [[nodiscard]] bool loan_backing_initialized(const std::size_t field, void* const backing) const {
    const auto& inventory = initialized_loan_backings_[field];
    return std::find(inventory.begin(), inventory.end(), backing) != inventory.end();
  }

  [[nodiscard]] expected<void, Error> publish_loan(ExecutionContext& context) {
    BenchmarkRunState& state = run_state();
    const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
    const std::array requests{
        tensor_shape(MultiTensorFields::first, shape),
        tensor_shape(MultiTensorFields::second, shape),
    };
    auto bundle = output_.allocate_tensors(requests);
    if (!bundle) {
      return make_unexpected(std::move(bundle.error()));
    }
    std::array<void*, 2U> new_backings{};
    bool initialized_during_measurement = false;
    const auto write_field = [&](const std::size_t field) {
      return [&, field](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
        if (!loan_backing_initialized(field, writer.data())) {
          if (state.options().tensor_storage == TensorStorage::kDevice) {
            if (cudaMemsetAsync(writer.data(),
                                static_cast<int>(kMultiTensorModePayloadByte),
                                writer.byte_size(),
                                context.cuda_stream().get()) != cudaSuccess) {
              return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
            }
          } else {
            std::memset(
                writer.data(), static_cast<int>(kMultiTensorModePayloadByte), writer.byte_size());
          }
          new_backings[field] = writer.data();
          initialized_during_measurement =
              initialized_during_measurement || state.measured(next_sequence_);
        }
        state.record_multi_tensor_source_pointer(field, next_sequence_, writer.data());
        return {};
      };
    };
    expected<void, Error> committed;
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      const ::cuda::stream_ref stream = context.cuda_stream();
      if (stream.get() == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      committed =
          bundle->write_all(device_field(MultiTensorFields::first, stream, write_field(0U)),
                            device_field(MultiTensorFields::second, stream, write_field(1U)));
    } else {
      committed = bundle->write_all(host_field(MultiTensorFields::first, write_field(0U)),
                                    host_field(MultiTensorFields::second, write_field(1U)));
    }
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    for (std::size_t field = 0U; field < new_backings.size(); ++field) {
      if (new_backings[field] != nullptr) {
        initialized_loan_backings_[field].push_back(new_backings[field]);
        state.record_descriptor_backing_initialization(next_sequence_);
      }
    }
    if (initialized_during_measurement) {
      state.record_validation_error();
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "multi-Tensor backing initialized during measurement"});
    }
    MultiTensorMessage message;
    message.sequence = next_sequence_;
    state.prepare_source_pre_emit_timestamp(
        next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
    return output_.emit_tensors(std::move(*bundle), std::move(message), emit_options());
  }

  [[nodiscard]] expected<void, Error> publish_staged() {
    if (foreign_slots_.empty()) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    ForeignTensorSlot& slot = foreign_slots_[(next_sequence_ - 1U) % foreign_slots_.size()];
    BenchmarkRunState& state = run_state();
    state.record_multi_tensor_source_pointer(0U, next_sequence_, slot.first->data());
    state.record_multi_tensor_source_pointer(1U, next_sequence_, slot.second->data());
    MultiTensorMessage message;
    message.sequence = next_sequence_;
    message.first = slot.first;
    message.second = slot.second;
    state.prepare_source_pre_emit_timestamp(
        next_sequence_, state.source_timestamp_required(next_sequence_) ? now_ns() : 0);
    return output_.emit(std::move(message), emit_options());
  }

  Output<MultiTensorMessage> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  std::array<std::vector<void*>, 2U> initialized_loan_backings_;
  std::vector<ForeignTensorSlot> foreign_slots_;

  friend ExecutionPlan make_plan(const Options& options);
};

class RuntimeMultiTensorModeBenchmarkSink final : public Operator<> {
 public:
  explicit RuntimeMultiTensorModeBenchmarkSink(const std::uint32_t sink_index = 0U)
      : sink_index_(sink_index) {}

  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    auto& builder =
        spec.input(input_, "input").queue_depth(subscriber_queue_depth(options, sink_index_));
    auto first = builder.tensor_field(MultiTensorFields::first);
    auto second = builder.tensor_field(MultiTensorFields::second);
    configure_field(first);
    configure_field(second);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{}).input(input_, DrainAll{});
    return result;
  }

  void start() override {
    if (run_state().options().scenario != Scenario::kMultiTensorLoan) {
      return;
    }
    const std::size_t warmup_slots = run_state().descriptor_warmup_slots();
    // Keep all but one hot-path backing live so the allocator must circulate the full inventory.
    warmup_retention_capacity_ = warmup_slots == 0U ? 0U : warmup_slots - 1U;
    retained_warmup_tensors_.reserve(warmup_retention_capacity_);
    for (auto& inventory : observed_backings_) {
      inventory.reserve(warmup_slots);
    }
    pool_warmup_complete_ = warmup_slots == 0U;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    BenchmarkRunState& state = run_state();
    const std::int64_t callback_timestamp =
        is_latency_workload(state.options().workload) ? now_ns() : 0;
    state.record_sink_callback(sink_index_);
    run_controlled_callback_work(state.options().callback_work_us);
    expected<SelectedBatchView<MultiTensorMessage>, Error> selected = make_unexpected(Error{});
    if (next_sequence_ == state.allocation_probe_sequence()) {
      const test::ScopedAllocationCounter allocations;
      selected = input_.selected_batch();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      selected = input_.selected_batch();
    }
    if (!selected) {
      state.fail(selected.error().code);
      return make_unexpected(std::move(selected.error()));
    }
    if (selected->size() != 1U) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    const Sample<MultiTensorMessage>& sample = selected->front();
    const std::uint64_t sequence = sample.metadata.frame_id;
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      state.record_tensor_consumer_stream(context.cuda_stream().get());
    }
    state.record_sink_callback_entry(sink_index_, sequence, callback_timestamp);
    const std::int64_t receive_timestamp = state.sink_timestamp_required(sequence) ? now_ns() : 0;
    const bool metadata_valid =
        sequence != 0U && sequence <= state.final_sequence() && sequence >= next_sequence_ &&
        sample.data.sequence == sequence && sample.metadata.sequence == sequence &&
        sample.metadata.trace_id == sequence &&
        (lossy_queue_policy(state.options().queue_policy) || sequence == next_sequence_) &&
        sample.metadata.degraded() && sample.data.first != nullptr && sample.data.second != nullptr;
    const bool descriptors_valid = metadata_valid &&
                                   tensor_descriptor_matches(*sample.data.first,
                                                             state.options().tensor_bytes,
                                                             state.options().tensor_storage,
                                                             state.options().tensor_device,
                                                             false) &&
                                   tensor_descriptor_matches(*sample.data.second,
                                                             state.options().tensor_bytes,
                                                             state.options().tensor_storage,
                                                             state.options().tensor_device,
                                                             false);
    const bool first_pointer_valid =
        descriptors_valid &&
        state.multi_tensor_pointer_matches(0U, sequence, sample.data.first->data());
    const bool second_pointer_valid =
        descriptors_valid &&
        state.multi_tensor_pointer_matches(1U, sequence, sample.data.second->data());
    const bool validate_payload = state.options().validation == PayloadValidation::kAll;
    constexpr auto expected = kMultiTensorModePayloadByte;
    const bool first_payload_valid =
        !validate_payload ||
        state.validate_tensor_bytes(*sample.data.first, expected, context.cuda_stream().get());
    const bool second_payload_valid =
        !validate_payload ||
        state.validate_tensor_bytes(*sample.data.second, expected, context.cuda_stream().get());
    if (!first_pointer_valid || !second_pointer_valid || !first_payload_valid ||
        !second_payload_valid) {
      HOLOSCAN_LOG_ERROR(
          "multi-Tensor validation failed: scenario={} sequence={} metadata={} descriptors={} "
          "first_pointer={} second_pointer={} first_payload={} second_payload={}",
          scenario_name(state.options().scenario),
          sequence,
          metadata_valid,
          descriptors_valid,
          first_pointer_valid,
          second_pointer_valid,
          first_payload_valid,
          second_payload_valid);
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    if (state.options().validation == PayloadValidation::kFinal &&
        sequence == state.final_sequence()) {
      state.retain_final_multi_tensors(*sample.data.first, *sample.data.second, expected);
    }
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (state.options().scenario == Scenario::kMultiTensorLoan && !pool_warmup_complete_ &&
        sequence <= state.options().warmup && warmup_slots != 0U) {
      const std::array backings{sample.data.first->data(), sample.data.second->data()};
      for (std::size_t field = 0U; field < backings.size(); ++field) {
        auto& inventory = observed_backings_[field];
        if (std::find(inventory.begin(), inventory.end(), backings[field]) == inventory.end()) {
          inventory.push_back(backings[field]);
        }
      }
      const bool inventory_complete = std::ranges::all_of(
          observed_backings_,
          [&](const auto& inventory) { return inventory.size() == warmup_slots; });
      if (inventory_complete) {
        retained_warmup_tensors_.clear();
        pool_warmup_complete_ = true;
      } else if (warmup_retention_capacity_ != 0U) {
        if (retained_warmup_tensors_.size() == warmup_retention_capacity_) {
          retained_warmup_tensors_.erase(retained_warmup_tensors_.begin());
        }
        retained_warmup_tensors_.push_back(
            std::array<Tensor, 2U>{*sample.data.first, *sample.data.second});
      }
    }
    state.record_delivery(sink_index_, sequence, receive_timestamp);
    next_sequence_ = sequence + 1U;
    return {};
  }

 private:
  void configure_field(TensorFieldInputSpecBuilder& builder) const {
    builder.expects_tensor(TensorInputSpec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = tensor_memory_kind(run_state().options().tensor_storage),
                .dtype = kTensorByteDtype,
                .rank = 1U,
            },
    });
  }

  Input<MultiTensorMessage> input_;
  std::uint32_t sink_index_{};
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::size_t warmup_retention_capacity_{};
  std::vector<std::array<Tensor, 2U>> retained_warmup_tensors_;
  std::array<std::vector<const void*>, 2U> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

// Sample-forward rows insert one or more metadata-preserving POD relay operators.
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
    run_controlled_callback_work(state.options().callback_work_us);

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
  const bool legacy_cover = options.partition_cover == PartitionCover::kLegacy;
  std::optional<PartitionHandle> shared_partition;
  if (legacy_cover && options.operator_lanes == OperatorLaneMode::kShared) {
    shared_partition.emplace(graph.partition("runtime-graph-shared"));
  }
  std::optional<PartitionHandle> source_partition;
  if (shared_partition) {
    source_partition = *shared_partition;
  } else if (legacy_cover || options.partition_cover == PartitionCover::kHybrid) {
    source_partition.emplace(graph.partition("runtime-graph-source"));
  }
  std::optional<PartitionHandle> common_sink_partition;
  if (legacy_cover && !shared_partition && options.operator_lanes != OperatorLaneMode::kPerSink &&
      options.scenario != Scenario::kSourceOnly && options.fanout == 1U) {
    common_sink_partition.emplace(graph.partition("runtime-graph-sink"));
  }
  std::optional<PartitionHandle> relay_partition;
  if (legacy_cover && !shared_partition && options.scenario == Scenario::kSampleForward) {
    relay_partition.emplace(graph.partition("runtime-graph-relay"));
  }
  const auto add_sink = [&](const auto& sink, const std::uint32_t index = 0U) {
    if (!legacy_cover) {
      return;
    }
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
  const auto sink_connection = [&](const std::uint32_t sink_index) {
    return ConnectionOptions{
        .queue_policy = subscriber_queue_policy(options, sink_index),
        .queue_depth = subscriber_queue_depth(options, sink_index),
    };
  };
  if (options.scenario == Scenario::kSourceOnly) {
    auto source = graph.op<RuntimeBenchmarkSourceOnly>("source");
    if (source_partition) {
      source_partition->add(source);
    }
  } else if (options.scenario == Scenario::kSignal) {
    auto source = graph.op<RuntimeBenchmarkSource<std::uint8_t>>("source");
    auto sink = graph.op<RuntimeBenchmarkSink<std::uint8_t>>("sink");
    if (source_partition) {
      source_partition->add(source);
    }
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kSampleForward) {
    auto source = graph.op<RuntimeBenchmarkSource<BenchmarkPOD64>>("source");
    auto sink = graph.op<RuntimeBenchmarkSink<BenchmarkPOD64>>("sink");
    if (source_partition) {
      source_partition->add(source);
    }
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
      if (legacy_cover) {
        if (shared_partition) {
          shared_partition->add(relay);
        } else {
          relay_partition->add(relay);
        }
      }
      relays.push_back(relay);
    }
    graph.add_flow(relays.back()->output_, sink->input_, connection);
  } else if (options.scenario == Scenario::kTensor) {
    auto source = graph.op<RuntimeTensorBenchmarkSource>("source");
    if (source_partition) {
      source_partition->add(source);
    }
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeTensorBenchmarkSink>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, sink_connection(index));
    }
  } else if (options.scenario == Scenario::kSchema) {
    auto source = graph.op<RuntimeBenchmarkSource<schema::TimeT>>("source");
    if (source_partition) {
      source_partition->add(source);
    }
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeBenchmarkSink<schema::TimeT>>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, sink_connection(index));
    }
  } else if (options.scenario == Scenario::kThirdPartySchema) {
    auto source = graph.op<RuntimeBenchmarkSource<ThirdPartyTelemetry>>("source");
    if (source_partition) {
      source_partition->add(source);
    }
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeBenchmarkSink<ThirdPartyTelemetry>>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, sink_connection(index));
    }
  } else if (options.scenario == Scenario::kImage) {
    auto source = graph.op<RuntimeImageBenchmarkSource>("source");
    if (source_partition) {
      source_partition->add(source);
    }
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeImageBenchmarkSink>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, sink_connection(index));
    }
  } else if (options.scenario == Scenario::kMultiTensor) {
    auto source = graph.op<RuntimeMultiTensorBenchmarkSource>("source");
    auto sink = graph.op<RuntimeMultiTensorBenchmarkSink>("sink");
    if (source_partition) {
      source_partition->add(source);
    }
    add_sink(sink);
    graph.add_flow(source->output_, sink->input_, connection);
  } else if (multi_tensor_scenario(options.scenario)) {
    auto source = graph.op<RuntimeMultiTensorModeBenchmarkSource>("source");
    if (source_partition) {
      source_partition->add(source);
    }
    for (std::uint32_t index = 0U; index < options.fanout; ++index) {
      const std::string name = options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
      auto sink = graph.op<RuntimeMultiTensorModeBenchmarkSink>(name, index);
      add_sink(sink, index);
      graph.add_flow(source->output_, sink->input_, sink_connection(index));
    }
  } else {
    if (options.test_gate_pod_sink_for_pressure) {
      auto source = graph.op<RuntimeBenchmarkPressuredPODDiagnosticSource>("source");
      auto sink = graph.op<RuntimeBenchmarkPressuredPODDiagnosticSink>("sink");
      if (source_partition) {
        source_partition->add(source);
      }
      add_sink(sink);
      graph.add_flow(source->output_, sink->input_, sink_connection(0U));
    } else {
      auto source = graph.op<RuntimeBenchmarkSource<BenchmarkPOD64>>("source");
      if (source_partition) {
        source_partition->add(source);
      }
      if (options.multi_input) {
        auto sink = graph.op<RuntimeBenchmarkMultiInputPODSink>("sink");
        add_sink(sink);
        graph.add_flow(source->output_, sink->first_, sink_connection(0U));
        graph.add_flow(source->second_output_, sink->second_, sink_connection(1U));
      } else {
        for (std::uint32_t index = 0U; index < options.fanout; ++index) {
          const std::string name =
              options.fanout == 1U ? "sink" : "sink-" + std::to_string(index + 1U);
          auto sink = graph.op<RuntimeBenchmarkSink<BenchmarkPOD64>>(name, index);
          add_sink(sink, index);
          if (options.multi_output && index == 1U) {
            graph.add_flow(source->second_output_, sink->input_, sink_connection(index));
          } else {
            graph.add_flow(source->output_, sink->input_, sink_connection(index));
          }
        }
      }
    }
  }
  if (options.partition_cover == PartitionCover::kIncludeAll) {
    graph.partition("runtime-graph-all").include_all();
  } else if (options.partition_cover == PartitionCover::kHybrid) {
    graph.auto_partition_remaining();
  }
  const ClockKey clock = graph.add_clock<RealtimeClock>("benchmark-clock");
  graph.set_default_clock(clock);

  CompileOptions compile_options;
  if (options.partition_cover == PartitionCover::kIncludeAll) {
    compile_options.deployment.bind_thread_placement(
        "runtime-graph-all", make_placement(options.source_cpu, options.busy_poll));
  } else if (options.partition_cover == PartitionCover::kHybrid) {
    compile_options.deployment.bind_thread_placement(
        "runtime-graph-source", make_placement(options.source_cpu, options.busy_poll));
  } else if (options.partition_cover == PartitionCover::kAutomatic) {
    // Automatic partitions have generated identities and cannot be bound by authored name.
  } else if (options.operator_lanes == OperatorLaneMode::kShared) {
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
  if ((options.scenario == Scenario::kTensor || options.scenario == Scenario::kImage ||
       options.scenario == Scenario::kMultiTensor || multi_tensor_scenario(options.scenario)) &&
      options.tensor_storage == TensorStorage::kDevice) {
    compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = DeviceId{options.tensor_device},
    });
  }
  switch (options.green_context_layout) {
    case GreenContextLayout::kNone:
      break;
    case GreenContextLayout::kSamePartition:
      compile_options.deployment.bind_cuda_green_context(
          options.partition_cover == PartitionCover::kIncludeAll ? "runtime-graph-all"
                                                                 : "runtime-graph-shared",
          {.device = DeviceId{options.tensor_device}, .sm_count = options.green_context_sm_count});
      break;
    case GreenContextLayout::kDedicatedToDedicated:
      compile_options.deployment
          .bind_cuda_green_context("runtime-graph-source",
                                   {.device = DeviceId{options.tensor_device},
                                    .sm_count = options.green_context_sm_count})
          .bind_cuda_green_context("runtime-graph-sink",
                                   {.device = DeviceId{options.tensor_device},
                                    .sm_count = options.green_context_sm_count});
      break;
    case GreenContextLayout::kDedicatedToResidual:
      compile_options.deployment.bind_cuda_green_context(
          "runtime-graph-source",
          {.device = DeviceId{options.tensor_device}, .sm_count = options.green_context_sm_count});
      break;
  }
  if (options.force_holomq_event_wake) {
    return detail::compile_with_process_local_holomq_event_wake(graph, std::move(compile_options));
  }
  if (options.force_holomq_process_local_inline) {
    return detail::compile_with_process_local_holomq(graph, std::move(compile_options));
  }
  return compile(graph, std::move(compile_options));
}

[[nodiscard]] bool process_local_event_wake(const ExecutionPlan& plan) noexcept {
  const auto* const state = detail::ExecutionPlanAccess::state(plan);
  return state != nullptr && std::any_of(state->process_local_channel_services.begin(),
                                         state->process_local_channel_services.end(),
                                         [](const auto& service) { return service.event_wake; });
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
      .p95 = percentile(0.95L),
      .p99 = percentile(0.99L),
      .p999 = include_p999 ? std::optional<std::uint64_t>{percentile(0.999L)} : std::nullopt,
      .maximum = values.back(),
  };
}

template <typename Allocation>
[[nodiscard]] bool allocation_storage_matches(const Allocation& allocation,
                                              const Options& options) noexcept {
  const DeviceId expected_device{options.tensor_device};
  const bool device_matches = options.tensor_storage != TensorStorage::kDevice
                                  ? !allocation.device.valid()
                                  : allocation.device == expected_device;
  return allocation.memory_kind == tensor_memory_kind(options.tensor_storage) && device_matches &&
         allocation.slot_count > 0U && allocation.release_capacity > 0U &&
         allocation.bounds.max_byte_span == options.tensor_bytes && !allocation.pool_id.empty();
}

[[nodiscard]] bool tensor_plan_matches_options(const ExecutionPlan& plan,
                                               const Options& options) noexcept {
  if (!tensor_backed_scenario(options.scenario)) {
    return plan.tensor_allocations().empty() && plan.tensor_field_allocations().empty();
  }
  const bool multi_tensor =
      options.scenario == Scenario::kMultiTensor || multi_tensor_scenario(options.scenario);
  if (!multi_tensor) {
    const auto allocations = plan.tensor_allocations();
    return allocations.size() == 1U && plan.tensor_field_allocations().empty() &&
           allocation_storage_matches(allocations.front(), options);
  }
  const auto fields = plan.tensor_field_allocations();
  return plan.tensor_allocations().empty() && fields.size() == 2U && fields[0].tensor_index == 0U &&
         fields[1].tensor_index == 1U && fields[0].pool_id != fields[1].pool_id &&
         allocation_storage_matches(fields[0], options) &&
         allocation_storage_matches(fields[1], options);
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

void emit_subscriber_queue_depths(std::ostream& output, const Options& options) {
  output << '[';
  const std::uint32_t count = options.scenario == Scenario::kSourceOnly ? 0U : options.fanout;
  for (std::uint32_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << subscriber_queue_depth(options, index);
  }
  output << ']';
}

void emit_subscriber_queue_policies(std::ostream& output, const Options& options) {
  output << '[';
  const std::uint32_t count = options.scenario == Scenario::kSourceOnly ? 0U : options.fanout;
  for (std::uint32_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << '"' << queue_policy_name(subscriber_queue_policy(options, index)) << '"';
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

[[nodiscard]] std::string partition_cover_signature(const ExecutionPlan& plan) {
  std::string signature;
  for (const auto& partition : plan.partitions()) {
    signature += std::to_string(partition.operator_paths().size());
    signature.push_back(':');
    for (const std::string_view path : partition.operator_paths()) {
      signature += std::to_string(path.size());
      signature.push_back(':');
      signature.append(path);
    }
    signature.push_back(';');
  }
  return signature;
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
         << ",\"p95\":" << percentiles->p95 << ",\"p99\":" << percentiles->p99 << ",\"p999\":";
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
    case RunTermination::kRequestedCompleted:
      return "requested-completed";
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

// REQ: CHAN-128
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
      << ",\"readiness_poll_contention_retries\":"
      << value(detail::RuntimePhaseCounter::kReadinessPollContentionRetries)
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
      << ",\"channel_activation_cut_inventory_scans\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutInventoryScans)
      << ",\"channel_activation_cut_subscriber_probes\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutSubscriberProbes)
      << ",\"channel_activation_cut_transport_probes\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutTransportProbes)
      << ",\"channel_activation_cut_materializations\":"
      << value(detail::RuntimePhaseCounter::kChannelActivationCutMaterializations)
      << ",\"channel_receive_backpressured\":"
      << value(detail::RuntimePhaseCounter::kChannelReceiveBackpressured)
      << ",\"channel_poll_incomplete\":"
      << value(detail::RuntimePhaseCounter::kChannelPollIncomplete)
      << ",\"channel_direct_ingress_promotion_passes\":"
      << value(detail::RuntimePhaseCounter::kChannelDirectIngressPromotionPasses)
      << ",\"channel_direct_ingress_promotions\":"
      << value(detail::RuntimePhaseCounter::kChannelDirectIngressPromotions)
      << ",\"channel_staged_evidence_sweep_visits\":"
      << value(detail::RuntimePhaseCounter::kChannelStagedEvidenceSweepVisits)
      << ",\"channel_protected_terminal_probes_with_pending_data\":"
      << value(detail::RuntimePhaseCounter::kChannelProtectedTerminalProbesWithPendingData)
      << ",\"channel_pre_witness_replacements\":"
      << value(detail::RuntimePhaseCounter::kChannelPreWitnessReplacements)
      << ",\"channel_freshness_reservation_release_deferred\":"
      << value(detail::RuntimePhaseCounter::kChannelFreshnessReservationReleaseDeferred)
      << ",\"channel_released_claim_refill_replacement_declined\":"
      << value(detail::RuntimePhaseCounter::kChannelReleasedClaimRefillReplacementDeclined)
      << ",\"channel_released_claim_refill_published\":"
      << value(detail::RuntimePhaseCounter::kChannelReleasedClaimRefillPublished)
      << ",\"temporal_retention_candidates\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionCandidates)
      << ",\"temporal_retention_evictions\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionEvictions)
      << ",\"temporal_retention_declined_not_superseded\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded)
      << ",\"temporal_retention_declined_unbounded_round\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUnboundedRound)
      << ",\"temporal_retention_declined_uncomparable_coordinate\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedUncomparableCoordinate)
      << ",\"temporal_retention_declined_above_floor\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor)
      << ",\"temporal_retention_declined_eviction_refused\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused)
      << ",\"temporal_retention_floor_resolved\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionFloorResolved)
      << ",\"temporal_retention_floor_absent\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionFloorAbsent)
      << ",\"temporal_retention_eviction_promoted\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionEvictionPromoted)
      << ",\"temporal_retention_eviction_promoted_nothing\":"
      << value(detail::RuntimePhaseCounter::kTemporalRetentionEvictionPromotedNothing)
      << ",\"temporal_reuse_last_retained_record_queries\":"
      << value(detail::RuntimePhaseCounter::kTemporalReuseLastRetainedRecordQueries)
      << ",\"temporal_reuse_last_comparator_calls\":"
      << value(detail::RuntimePhaseCounter::kTemporalReuseLastComparatorCalls)
      << ",\"candidate_source_lookup_probes\":"
      << value(detail::RuntimePhaseCounter::kCandidateSourceLookupProbes)
      << ",\"candidate_order_comparisons\":"
      << value(detail::RuntimePhaseCounter::kCandidateOrderComparisons)
      << ",\"candidate_order_moves\":" << value(detail::RuntimePhaseCounter::kCandidateOrderMoves)
      << ",\"candidate_tag_time_reservations\":"
      << value(detail::RuntimePhaseCounter::kCandidateTagTimeReservations)
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
         << ",\"terminal_frozen_candidates\":" << trace.terminal_frozen_candidates
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
    output << ",\"activations\":" << receiver.activations << ",\"previous_cut_to_candidate_ns\":";
    emit_percentiles(output, receiver.previous_cut_to_candidate, include_p999);
    output << ",\"previous_cut_to_readiness_poll_return_ns\":";
    emit_percentiles(output, receiver.previous_cut_to_readiness_poll_return, include_p999);
    output << ",\"readiness_poll_return_to_candidate_ns\":";
    emit_percentiles(output, receiver.readiness_poll_return_to_candidate, include_p999);
    output << ",\"candidate_to_selection_ns\":";
    emit_percentiles(output, receiver.candidate_to_selection, include_p999);
    output << ",\"selection_to_input_preparation_ns\":";
    emit_percentiles(output, receiver.selection_to_input_preparation, include_p999);
    output << ",\"input_preparation_to_activation_ns\":";
    emit_percentiles(output, receiver.input_preparation_to_activation, include_p999);
    output << ",\"activation_to_dispatch_start_ns\":";
    emit_percentiles(output, receiver.activation_to_dispatch_start, include_p999);
    output << ",\"candidate_to_dispatch_start_ns\":";
    emit_percentiles(output, receiver.candidate_to_dispatch_start, include_p999);
    output << ",\"dispatch_start_to_callback_entry_ns\":";
    emit_percentiles(output, receiver.dispatch_start_to_callback_entry, include_p999);
    output << ",\"callback_entry_to_return_ns\":";
    emit_percentiles(output, receiver.callback_entry_to_return, include_p999);
    output << ",\"callback_return_to_dispatch_return_ns\":";
    emit_percentiles(output, receiver.callback_return_to_dispatch_return, include_p999);
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
    output << '}';
  }
  output << "]},\n";
}

// Keep the stable evidence JSON fields together for review and downstream parsing.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_callback_thread_evidence(std::ostream& output, const BenchmarkRunState& state,
                                   const ExecutionPlan& plan,
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
  bool shared_lane_single_thread = enabled && plan.execution_partitions().size() == 1U;
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

  std::set<std::uint64_t> stable_sink_tids;
  const std::uint32_t logical_sink_count = sink_operator_count(state.options());
  for (std::uint32_t index = 0U; index < logical_sink_count; ++index) {
    const std::uint64_t sink_tid = state.sink_callback_tid(index);
    const bool multiple_threads = state.sink_callback_multiple_threads(index);
    valid = valid && sink_tid != 0U && !multiple_threads;
    stable_sink_tids.insert(sink_tid);
  }
  if (logical_sink_count != 0U) {
    if (plan.execution_partitions().size() == 1U) {
      const auto expected_owner_tid =
          callbacks_on_partition_owner ? controller_native_tid : shared_tid;
      valid = valid && stable_sink_tids.size() == 1U && expected_owner_tid.has_value() &&
              *stable_sink_tids.begin() == *expected_owner_tid;
    } else if (logical_sink_count > 1U) {
      valid = valid && stable_sink_tids.size() == logical_sink_count && shared_tid.has_value() &&
              !stable_sink_tids.contains(*shared_tid);
    }
  }

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
  output << "],\"sink_roles\":[";
  for (std::uint32_t index = 0U; index < logical_sink_count; ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"index\":" << index << ",\"callbacks\":" << state.sink_deliveries(index)
           << ",\"native_tid\":" << state.sink_callback_tid(index) << ",\"multiple_threads\":"
           << (state.sink_callback_multiple_threads(index) ? "true" : "false") << '}';
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
  // Tensor-pool retries occur before Channel emit and therefore do not contribute to either
  // Channel emit-attempt or backpressure counters.
  const std::uint64_t expected_source_emit_attempts =
      (state.publications() + state.backpressure_count()) * source_emit_count(state.options());
  const std::uint64_t expected_accepted_emits =
      state.publications() * source_emit_count(state.options()) + state.relay_publications();
  const std::uint64_t expected_input_operations =
      state.relay_callbacks() + state.sink_callbacks() * inputs_per_sink(state.options());
  const std::uint64_t frozen = value(detail::RuntimePhaseCounter::kCandidatesFrozen);
  const bool runtime_valid =
      frozen >= expected_compute_dispatches && frozen - expected_compute_dispatches <= 1U &&
      value(detail::RuntimePhaseCounter::kCandidatesSelected) == expected_compute_dispatches &&
      value(detail::RuntimePhaseCounter::kComputeDispatches) == expected_compute_dispatches &&
      value(detail::RuntimePhaseCounter::kActivationsCompleted) == expected_compute_dispatches &&
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
  const std::uint64_t activation_cuts = value(detail::RuntimePhaseCounter::kChannelActivationCuts);
  const std::uint64_t receive_backpressured =
      value(detail::RuntimePhaseCounter::kChannelReceiveBackpressured);
  const std::uint64_t retry_cut_budget =
      uses_channel_pump(state.options()) ? 0U : receive_backpressured;
  const bool activation_cut_valid =
      uses_holomq(state.options())
          ? activation_cuts == activation_cut_drains + activation_cut_skips &&
                activation_cuts >= expected_compute_dispatches &&
                activation_cuts - expected_compute_dispatches <= retry_cut_budget
          : activation_cuts == 0U && activation_cut_drains == 0U && activation_cut_skips == 0U;
  const std::uint64_t samples_received =
      value(detail::RuntimePhaseCounter::kChannelSamplesReceived);
  const std::uint64_t readiness_published =
      value(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const std::uint64_t witnesses_retired =
      value(detail::RuntimePhaseCounter::kChannelWitnessRetired);
  const std::uint64_t accepted_delivery_opportunities =
      expected_accepted_emits * subscribers_per_output(state.options());
  const bool retained_counts_valid = has_lossy_subscriber(state.options())
                                         ? samples_received == readiness_published &&
                                               samples_received >= expected_input_operations &&
                                               samples_received <= accepted_delivery_opportunities
                                         : samples_received == expected_input_operations &&
                                               readiness_published == expected_input_operations;
  const bool witness_retirement_valid = has_lossy_subscriber(state.options())
                                            ? expected_input_operations <= witnesses_retired &&
                                                  witnesses_retired <= readiness_published
                                            : witnesses_retired == expected_input_operations;
  // Each readiness revision in the gated diagnostic either publishes a retired witness or is
  // replaced before witness publication. Consumer reads count only selected payloads.
  const bool diagnostic_retirement_valid =
      !state.options().test_gate_pod_sink_for_pressure ||
      witnesses_retired + value(detail::RuntimePhaseCounter::kChannelPreWitnessReplacements) ==
          readiness_published;
  return value(detail::RuntimePhaseCounter::kChannelPrepareCalls) == expected_compute_dispatches &&
         value(detail::RuntimePhaseCounter::kChannelEmitAttempts) ==
             expected_source_emit_attempts + state.relay_callbacks() &&
         value(detail::RuntimePhaseCounter::kChannelEmitAccepted) == expected_accepted_emits &&
         value(detail::RuntimePhaseCounter::kChannelEmitBackpressured) ==
             state.backpressure_count() &&
         retained_counts_valid &&
         value(detail::RuntimePhaseCounter::kChannelInputReads) == expected_input_operations &&
         witness_retirement_valid && diagnostic_retirement_valid &&
         value(detail::RuntimePhaseCounter::kChannelRetireCalls) == expected_compute_dispatches &&
         activation_cut_valid;
}

[[nodiscard]] bool terminal_phase_state_valid(
    const RuntimePhaseTraceSummary& phase_trace,
    const detail::RuntimePhaseCounters* phase_counters) noexcept {
  if (!phase_trace.enabled || !phase_trace.activation_trace_applicable) {
    return phase_trace.terminal_frozen_candidates == 0U;
  }
  if (phase_counters == nullptr) {
    return false;
  }
  const auto snapshot = phase_counters->snapshot();
  const std::uint64_t frozen =
      snapshot[detail::RuntimePhaseCounters::index(detail::RuntimePhaseCounter::kCandidatesFrozen)];
  const std::uint64_t selected = snapshot[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kCandidatesSelected)];
  return frozen >= selected && frozen - selected == phase_trace.terminal_frozen_candidates;
}

// Keep the benchmark's stable JSON schema visible in one emitter.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_result(std::ostream& output, const BenchmarkRunState& state, const ExecutionPlan& plan,
                 const RunSession& session, const BenchmarkSupportThreadControl& thread_control,
                 const detail::RuntimePhaseCounters* phase_counters,
                 const RuntimePhaseTraceSummary& phase_trace,
                 const std::optional<std::uint64_t> final_validation_ns,
                 const BenchmarkLifecycleTiming& lifecycle_timing,
                 const std::optional<detail::RuntimeTensorOutputTestControl::PoolSnapshot>&
                     cuda_resource_counters) {
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
  const std::uint64_t delivery_opportunities = state.publications() * options.fanout;
  const double replacement_rate = delivery_opportunities == 0U
                                      ? 0.0
                                      : static_cast<double>(consumer_visible_replacements) /
                                            static_cast<double>(delivery_opportunities);
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
  const std::uint64_t device_pool_capacity_bytes =
      std::accumulate(plan.tensor_allocations().begin(),
                      plan.tensor_allocations().end(),
                      std::uint64_t{},
                      [](const std::uint64_t total, const TensorAllocationPlanRecord& allocation) {
                        return total + (allocation.memory_kind == MemoryKind::kCudaDevice
                                            ? allocation.aggregate_byte_capacity
                                            : 0U);
                      });

  output << "{\n"
         << "  \"schema\":\"holoscan.test.runtime-benchmark/v1\",\n"
         << "  \"benchmark\":\"" << benchmark_name(options) << "\",\n"
         << "  \"answers\":\"" << benchmark_answer(options.scenario) << "\",\n"
         << "  \"scenario\":\"" << scenario_name(options.scenario) << "\",\n"
         << "  \"process_local_inline_delivery\":\"" << process_local_delivery_name(options)
         << "\",\n"
         << "  \"runtime_controls\":{\"direct_continuation_elision_disabled\":"
         << (options.disable_direct_continuation_elision ? "true" : "false")
         << ",\"closed_island_composite_disabled\":"
         << (options.disable_closed_island_composite ? "true" : "false")
         << ",\"holomq_event_wake_forced\":" << (options.force_holomq_event_wake ? "true" : "false")
         << ",\"process_local_channel_pump_forced\":"
         << (options.force_process_local_channel_pump ? "true" : "false");
  if (options.test_gate_pod_sink_for_pressure) {
    output << ",\"pod_sink_pressure_gate\":{\"first_readiness_published\":"
           << state.pressure_gate_first_readiness_published()
           << ",\"first_witness_retired\":" << state.pressure_gate_first_witness_retired()
           << ",\"first_delivery_publications\":"
           << state.pressure_gate_first_delivery_publications()
           << ",\"source_completion_publications\":"
           << state.pressure_gate_source_completion_publications()
           << ",\"source_retirement_activations_completed\":"
           << state.pressure_gate_source_retirement_activations_completed()
           << ",\"notification_attempts\":" << state.pressure_gate_notification_attempts()
           << ",\"notifications_accepted\":" << state.pressure_gate_notifications_accepted()
           << ",\"notification_backpressure\":" << state.pressure_gate_notification_backpressure()
           << ",\"notification_not_ready\":" << state.pressure_gate_notification_not_ready() << '}';
  }
  output
      << "},\n"
      << "  \"queue_policy\":\"" << queue_policy_name(options.queue_policy) << "\",\n"
      << "  \"payload\":\"" << payload_name(options.scenario) << "\",\n"
      << "  \"workload\":\"" << workload_name(options.workload) << "\",\n"
      << "  \"source_trigger\":\"" << (notification_driver_used(options) ? "notification" : "clock")
      << "\",\n"
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
  const bool bulk_payload = tensor_backed_scenario(options.scenario);
  const bool multi_tensor_payload =
      options.scenario == Scenario::kMultiTensor || multi_tensor_scenario(options.scenario);
  const auto multi_tensor_capacity = [&](const auto projection) {
    std::uint64_t result{};
    for (const auto& allocation : plan.tensor_field_allocations()) {
      result += projection(allocation);
    }
    return result;
  };
  const std::uint64_t bulk_pool_slots =
      tensor_allocation != nullptr
          ? tensor_allocation->slot_count
          : (multi_tensor_payload
                 ? multi_tensor_capacity([](const auto& record) { return record.slot_count; })
                 : (tensor_field_allocation != nullptr ? tensor_field_allocation->slot_count
                                                       : (bulk_payload ? 1U : 0U)));
  const std::uint64_t bulk_release_capacity =
      tensor_allocation != nullptr
          ? tensor_allocation->release_capacity
          : (multi_tensor_payload
                 ? multi_tensor_capacity([](const auto& record) { return record.release_capacity; })
                 : (tensor_field_allocation != nullptr ? tensor_field_allocation->release_capacity
                                                       : (bulk_payload ? 1U : 0U)));
  emit_algorithm_key(output, partitioning.requested_algorithm);
  output << ",\"accepted_algorithm\":";
  emit_algorithm_key(output, partitioning.accepted_algorithm);
  output << ",\"selection_source\":\""
         << partition_selection_source_name(partitioning.selection_source) << "\",\"authoring\":\""
         << partition_cover_name(options.partition_cover) << "\",\"mode\":\""
         << partitioning_mode_name(partitioning.mode)
         << "\",\"partition_count\":" << plan.partitions().size()
         << ",\"authored_partition_count\":" << plan.partitions().authored_count()
         << ",\"automatic_partition_count\":" << plan.partitions().automatic_count()
         << ",\"cover_signature\":\"" << json_escape(partition_cover_signature(plan)) << "\"},\n"
         << "  \"geometry\":{\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
         << ",\"allocation_probe_samples\":1" << ",\"queue_depth\":" << options.queue_depth
         << ",\"subscriber_queue_depths\":";
  emit_subscriber_queue_depths(output, options);
  output << ",\"subscriber_queue_policies\":";
  emit_subscriber_queue_policies(output, options);
  output << ",\"notification_capacity\":"
         << (notification_driver_used(options) ? options.notification_capacity : 0U)
         << ",\"offered_publication_rate_per_second\":";
  if (offered_rate_mode(options)) {
    output << options.offered_rate_hz;
  } else {
    output << "null";
  }
  output << ",\"offered_publication_period_ns\":";
  if (offered_rate_mode(options)) {
    output << offered_period(options).count();
  } else {
    output << "null";
  }
  output << ",\"offered_burst_size\":" << options.offered_burst_size
         << ",\"callback_work_us\":" << options.callback_work_us
         << ",\"publication_backpressure_capacity\":" << options.queue_depth
         << ",\"fanout\":" << (options.scenario == Scenario::kSourceOnly ? 0U : options.fanout)
         << ",\"source_outputs\":" << source_output_count(options)
         << ",\"subscribers_per_output\":" << subscribers_per_output(options)
         << ",\"sink_operators\":" << sink_operator_count(options)
         << ",\"inputs_per_sink\":" << inputs_per_sink(options) << ",\"relay_count\":"
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
  if (!notification_driver_used(options)) {
    output << state.source_period().count();
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"tensor\":{";
  if (bulk_payload) {
    const bool is_multi_tensor =
        options.scenario == Scenario::kMultiTensor || multi_tensor_scenario(options.scenario);
    const bool fused_linear_stream = std::any_of(plan.cuda_compute_streams().begin(),
                                                 plan.cuda_compute_streams().end(),
                                                 [](const CudaComputeStreamPlanRecord& stream) {
                                                   return !stream.fused_linear_edges.empty();
                                                 });
    const bool plan_storage_matches = tensor_plan_matches_options(plan, options);
    output << "\"storage\":\"" << tensor_storage_name(options.tensor_storage)
           << "\",\"device\":" << options.tensor_device << ",\"direct_allocation_records\":"
           << (is_multi_tensor ? plan.tensor_field_allocations().size()
                               : plan.tensor_allocations().size())
           << ",\"memory_kind_plan_match\":" << (plan_storage_matches ? "true" : "false")
           << ",\"observed_storage_checked\":true"
           << ",\"device_resource_records\":" << plan.tensor_device_resources().size()
           << ",\"fused_linear_stream\":" << (fused_linear_stream ? "true" : "false")
           << ",\"pointer_identity_checked\":true,\"descriptor_schema\":"
           << (options.scenario == Scenario::kImage || is_multi_tensor ? "true" : "false");
    if (is_multi_tensor) {
      const auto fields = plan.tensor_field_allocations();
      output << ",\"native_type\":\"runtime_benchmark::MultiTensorMessageT\""
             << ",\"field_count\":2,\"field_pools_distinct\":"
             << (fields.size() == 2U && fields[0].pool_id != fields[1].pool_id ? "true" : "false")
             << ",\"fields\":[";
      constexpr std::array field_names{"first", "second"};
      for (std::size_t index = 0U; index < field_names.size(); ++index) {
        if (index != 0U) {
          output << ',';
        }
        output << "{\"name\":\"" << field_names[index]
               << "\",\"dtype\":\"uint8\",\"rank\":1,\"bytes\":" << options.tensor_bytes;
        if (fields.size() == field_names.size()) {
          output << ",\"tensor_index\":" << fields[index].tensor_index << ",\"pool_id\":\""
                 << json_escape(fields[index].pool_id)
                 << "\",\"pool_slots\":" << fields[index].slot_count
                 << ",\"release_capacity\":" << fields[index].release_capacity;
        }
        output << '}';
      }
      output << ']';
    }
    if (multi_tensor_scenario(options.scenario)) {
      output << ",\"emission_mode\":\""
             << (options.scenario == Scenario::kMultiTensorLoan ? "loan" : "staging")
             << "\",\"source_pointer_contract\":\"preserved\"";
    }
  }
  output << "},\n"
         << "  \"green_context\":{\"layout\":\""
         << green_context_layout_name(options.green_context_layout) << "\",\"enabled\":"
         << (options.green_context_layout == GreenContextLayout::kNone ? "false" : "true")
         << ",\"requested_sm_count\":" << options.green_context_sm_count
         << ",\"pool_records\":" << plan.cuda_green_context_pools().size()
         << ",\"resource_records\":" << plan.cuda_green_context_resources().size()
         << ",\"stream_records\":" << plan.cuda_compute_streams().size()
         << ",\"device_memory_pool_capacity_bytes\":" << device_pool_capacity_bytes;
  if (plan.cuda_green_context_pools().empty()) {
    output << ",\"deployment_device_id\":null,\"device_sm_count\":null"
           << ",\"structured_sm_quantum\":null,\"unprovisioned_sm_count\":null";
  } else {
    const auto& pool = plan.cuda_green_context_pools().front();
    output << ",\"deployment_device_id\":\"" << json_escape(pool.deployment_device_id)
           << "\",\"device_sm_count\":" << pool.device_sm_count
           << ",\"structured_sm_quantum\":" << pool.structured_sm_quantum
           << ",\"unprovisioned_sm_count\":" << pool.unprovisioned_sm_count;
  }
  output << "},\n"
         << "  \"cold_timing_ns\":{\"compile\":" << lifecycle_timing.compile_ns
         << ",\"run_async_return\":" << lifecycle_timing.run_async_return_ns
         << ",\"run_async_to_first_callback\":" << lifecycle_timing.run_async_to_first_callback_ns
         << ",\"terminal_wait\":" << lifecycle_timing.terminal_wait_ns << "},\n"
         << "  \"cuda_resource_counters\":{\"enabled\":"
         << (cuda_resource_counters.has_value() ? "true" : "false");
  if (cuda_resource_counters.has_value()) {
    const auto& counters = *cuda_resource_counters;
    output << ",\"streams_created\":" << counters.cuda_streams_created
           << ",\"streams_destroyed\":" << counters.cuda_streams_destroyed
           << ",\"events_created\":" << counters.cuda_events_created
           << ",\"events_destroyed\":" << counters.cuda_events_destroyed
           << ",\"readiness_records\":" << counters.cuda_readiness_records
           << ",\"readiness_record_elisions\":" << counters.cuda_readiness_record_elisions
           << ",\"stream_waits\":" << counters.cuda_stream_waits
           << ",\"stream_wait_elisions\":" << counters.cuda_stream_wait_elisions
           << ",\"release_records\":" << counters.cuda_release_records
           << ",\"retirement_queries\":" << counters.cuda_retirement_queries
           << ",\"stream_quiesce_ns\":" << counters.cuda_stream_quiesce_ns;
  }
  output << "},\n"
         << "  \"wake\":{\"event_wake\":" << (process_local_event_wake(plan) ? "true" : "false")
         << ",\"busy_poll\":" << (options.busy_poll ? "true" : "false")
         << ",\"receive_polling\":false},\n"
         << "  \"placement\":{\"wake_mode\":\""
         << (options.busy_poll ? "busy_poll" : "sleeping_waitset") << "\",\"operator_lanes\":\""
         << operator_lane_mode_name(options.operator_lanes) << "\",\"requested_operator_lanes\":\""
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
  output << ",\"accepted_publication_offer_rate_per_second\":";
  emit_optional_metric(
      output, offered_rate_mode(options), state.accepted_publication_offer_rate_per_second());
  output << ",\"publication_offer_start_ns\":";
  if (offered_rate_mode(options)) {
    output << state.publication_offer_start_ns();
  } else {
    output << "null";
  }
  output << ",\"publication_offer_end_ns\":";
  if (offered_rate_mode(options)) {
    output << state.publication_offer_end_ns();
  } else {
    output << "null";
  }
  output << ",\"maximum_publication_offer_lateness_ns\":";
  if (offered_rate_mode(options)) {
    output << state.maximum_publication_offer_lateness_ns();
  } else {
    output << "null";
  }
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
         << ",\"source_terminal_compute_entries\":" << state.source_terminal_compute_entries()
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
         << ",\"source_tensor_pool_retries\":" << state.source_tensor_pool_retries()
         << ",\"source_tensor_pool_in_activation_retries\":"
         << state.source_tensor_pool_in_activation_retries()
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
    const std::uint64_t sink_replacements = state.sink_replacements(index);
    const double sink_replacement_rate =
        state.publications() == 0U
            ? 0.0
            : static_cast<double>(sink_replacements) / static_cast<double>(state.publications());
    output << "{\"index\":" << index << ",\"deliveries\":" << state.sink_deliveries(index)
           << ",\"final_sequence\":" << state.sink_final_sequence(index)
           << ",\"replacements\":" << sink_replacements
           << ",\"replacement_rate\":" << sink_replacement_rate << '}';
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
  emit_runtime_thread_usage(output, phase_counters);
  emit_phase_trace(output, phase_trace, include_p999);
  emit_callback_thread_evidence(output, state, plan, thread_control);
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
         << "\",\"diagnostics\":";
  emit_runtime_diagnostics(output, session.diagnostics());
  output << "}\n}\n";
}

template <typename Integer>
[[nodiscard]] bool parse_integer(const std::string_view text, Integer& output) noexcept {
  const char* const begin = text.data();
  const char* const end = text.data() + text.size();
  const auto [position, error] = std::from_chars(begin, end, output);
  return error == std::errc() && position == end;
}

template <typename Integer>
[[nodiscard]] bool parse_integer_list(std::string_view text, std::vector<Integer>& output) {
  output.clear();
  while (!text.empty()) {
    const std::size_t separator = text.find(',');
    const std::string_view token = text.substr(0U, separator);
    Integer item{};
    if (token.empty() || !parse_integer(token, item)) {
      return false;
    }
    output.push_back(item);
    if (separator == std::string_view::npos) {
      return true;
    }
    text.remove_prefix(separator + 1U);
  }
  return false;
}

[[nodiscard]] bool parse_cpu_list(const std::string_view text, std::vector<int>& output) {
  return parse_integer_list(text, output);
}

[[nodiscard]] bool consume_value(const int argc, char** argv, int& index, std::string_view& value) {
  if (index + 1 >= argc) {
    return false;
  }
  value = argv[++index];
  return true;
}

void print_usage(std::ostream& output) {
  output << "runtime_graph_benchmark [options]\n\n"
         << "  --workload <latency|notification_latency|saturated>\n"
         << "  --scenario <source_only|pod|signal|sample_forward|schema|third_party_schema|tensor|"
            "image|multi_tensor|multi_tensor_loan|multi_tensor_staging>\n"
         << "  --operator-lanes <separate|shared|per_sink>\n"
         << "  --partition-cover <legacy|automatic|include_all|hybrid>\n"
         << "  --green-context-layout <none|same_partition|dedicated_to_dedicated|"
            "dedicated_to_residual>\n"
         << "  --green-context-sm-count <n>\n"
         << "  --tensor-storage <pageable_host|pinned_host|device>\n"
         << "  --bulk-work-profile <full_write|descriptor_only>\n"
         << "  --tensor-bytes <n>\n"
         << "  --tensor-device <n>\n"
         << "  --samples <n>\n"
         << "  --warmup <n>\n"
         << "  --queue-policy <blocking|drop_oldest>\n"
         << "  --queue-depth <n>\n"
         << "  --sink-queue-depths <n,n,...>\n"
         << "  --relay-count <n>\n"
         << "  --fanout <n>\n"
         << "  --multi-output\n"
         << "  --multi-input\n"
         << "  --mixed-policy-fanout\n"
         << "  --notification-capacity <n>\n"
         << "  --offered-rate-hz <n>\n"
         << "  --offered-burst-size <n>\n"
         << "  --publisher-period-us <n>\n"
         << "  --callback-work-us <0|1|5|20|100>\n"
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
         << "  --cuda-resource-counters\n"
         << "  --disable-direct-continuation-elision\n"
         << "  --disable-closed-island-composite\n"
         << "  --force-holomq-process-local-inline\n"
         << "  --force-holomq-event-wake\n"
         << "  --force-process-local-channel-pump\n"
         << "  --test-gate-pod-sink-for-pressure\n"
         << "  --test-source-tensor-pool-in-activation-retries <n>\n"
         << "  --timeout-ms <n>\n"
         << "  --output <path>\n"
         << "  --plan-output <path>\n";
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
    if (argument == "--cuda-resource-counters") {
      options.cuda_resource_counters = true;
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
    if (argument == "--force-holomq-event-wake") {
      options.force_holomq_event_wake = true;
      continue;
    }
    if (argument == "--force-process-local-channel-pump") {
      options.force_process_local_channel_pump = true;
      continue;
    }
    if (argument == "--multi-output") {
      options.multi_output = true;
      continue;
    }
    if (argument == "--multi-input") {
      options.multi_input = true;
      continue;
    }
    if (argument == "--mixed-policy-fanout") {
      options.mixed_policy_fanout = true;
      continue;
    }
    if (argument == "--test-fail-sink-after-relay-pressure") {
      options.test_fail_sink_after_relay_pressure = true;
      continue;
    }
    if (argument == "--test-gate-pod-sink-for-pressure") {
      options.test_gate_pod_sink_for_pressure = true;
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
      } else if (value == "third_party_schema") {
        options.scenario = Scenario::kThirdPartySchema;
      } else if (value == "tensor") {
        options.scenario = Scenario::kTensor;
      } else if (value == "image") {
        options.scenario = Scenario::kImage;
      } else if (value == "multi_tensor") {
        options.scenario = Scenario::kMultiTensor;
      } else if (value == "multi_tensor_loan") {
        options.scenario = Scenario::kMultiTensorLoan;
      } else if (value == "multi_tensor_staging") {
        options.scenario = Scenario::kMultiTensorStaging;
      } else {
        std::cerr << "--scenario must be source_only, pod, signal, sample_forward, schema, "
                     "third_party_schema, tensor, image, multi_tensor, multi_tensor_loan, or "
                     "multi_tensor_staging\n";
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
    } else if (argument == "--partition-cover") {
      if (value == "legacy") {
        options.partition_cover = PartitionCover::kLegacy;
      } else if (value == "automatic") {
        options.partition_cover = PartitionCover::kAutomatic;
      } else if (value == "include_all") {
        options.partition_cover = PartitionCover::kIncludeAll;
      } else if (value == "hybrid") {
        options.partition_cover = PartitionCover::kHybrid;
      } else {
        std::cerr << "--partition-cover must be legacy, automatic, include_all, or hybrid\n";
        return std::nullopt;
      }
    } else if (argument == "--green-context-layout") {
      if (value == "none") {
        options.green_context_layout = GreenContextLayout::kNone;
      } else if (value == "same_partition") {
        options.green_context_layout = GreenContextLayout::kSamePartition;
      } else if (value == "dedicated_to_dedicated") {
        options.green_context_layout = GreenContextLayout::kDedicatedToDedicated;
      } else if (value == "dedicated_to_residual") {
        options.green_context_layout = GreenContextLayout::kDedicatedToResidual;
      } else {
        std::cerr << "--green-context-layout has an unsupported value\n";
        return std::nullopt;
      }
    } else if (argument == "--green-context-sm-count") {
      if (!parse_integer(value, options.green_context_sm_count)) {
        std::cerr << "--green-context-sm-count must be a nonnegative integer\n";
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
    } else if (argument == "--sink-queue-depths") {
      if (!parse_integer_list(value, options.sink_queue_depths)) {
        std::cerr << "--sink-queue-depths must be a comma-separated integer list\n";
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
    } else if (argument == "--offered-rate-hz") {
      if (!parse_integer(value, options.offered_rate_hz)) {
        std::cerr << "--offered-rate-hz must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--offered-burst-size") {
      if (!parse_integer(value, options.offered_burst_size)) {
        std::cerr << "--offered-burst-size must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--publisher-period-us") {
      if (!parse_integer(value, options.publisher_period_us)) {
        std::cerr << "--publisher-period-us must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--callback-work-us") {
      if (!parse_integer(value, options.callback_work_us)) {
        std::cerr << "--callback-work-us must be an integer\n";
        return std::nullopt;
      }
    } else if (argument == "--test-source-tensor-pool-in-activation-retries") {
      if (!parse_integer(value, options.test_source_tensor_pool_in_activation_retries)) {
        std::cerr << "--test-source-tensor-pool-in-activation-retries must be an integer\n";
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
    } else if (argument == "--plan-output") {
      options.plan_output_path = value;
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
  if (std::find(kSupportedCallbackWorkUs.begin(),
                kSupportedCallbackWorkUs.end(),
                options.callback_work_us) == kSupportedCallbackWorkUs.end()) {
    std::cerr << "--callback-work-us must be 0, 1, 5, 20, or 100\n";
    return std::nullopt;
  }
  if (lossy_queue_policy(options.queue_policy) && !lossy_scenario_supported(options.scenario)) {
    std::cerr << "--queue-policy drop_oldest supports --scenario pod, schema, third_party_schema, "
                 "tensor, image, multi_tensor_loan, or multi_tensor_staging\n";
    return std::nullopt;
  }
  if (lossy_queue_policy(options.queue_policy) && options.fanout != 1U &&
      !offered_rate_mode(options)) {
    std::cerr << "--queue-policy drop_oldest fanout requires --offered-rate-hz\n";
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
  if (options.test_gate_pod_sink_for_pressure &&
      (options.partition_cover != PartitionCover::kLegacy ||
       options.workload != Workload::kSaturated || options.scenario != Scenario::kPod ||
       options.operator_lanes != OperatorLaneMode::kSeparate ||
       options.queue_policy != QueuePolicy::kDropOldest || options.queue_depth != 1U ||
       !options.sink_queue_depths.empty() || options.fanout != 1U || options.multi_output ||
       options.multi_input || options.mixed_policy_fanout || options.notification_capacity != 1U ||
       offered_rate_mode(options) || options.callback_work_us != 0U ||
       !options.force_holomq_process_local_inline || options.force_holomq_event_wake ||
       options.force_process_local_channel_pump || !options.runtime_phase_counters)) {
    std::cerr << "--test-gate-pod-sink-for-pressure requires the counter-enabled saturated "
                 "separate-lane POD DropOldest depth-one fanout-one forced-HoloMQ diagnostic row\n";
    return std::nullopt;
  }
  if (options.test_source_tensor_pool_in_activation_retries != 0U &&
      (options.test_source_tensor_pool_in_activation_retries > kMaximumTestTensorPoolRetries ||
       (options.scenario != Scenario::kTensor && options.scenario != Scenario::kMultiTensor) ||
       !notification_driver_used(options))) {
    std::cerr << "--test-source-tensor-pool-in-activation-retries requires a notification-driven "
                 "Tensor or multi-Tensor scenario and must not exceed "
              << kMaximumTestTensorPoolRetries << '\n';
    return std::nullopt;
  }
  if (options.fanout == 0U || options.fanout > kMaximumFanout) {
    std::cerr << "--fanout must be between 1 and " << kMaximumFanout << '\n';
    return std::nullopt;
  }
  if (options.fanout > 8U && options.scenario != Scenario::kPod &&
      options.scenario != Scenario::kSchema && options.scenario != Scenario::kThirdPartySchema) {
    std::cerr << "--fanout greater than 8 requires --scenario pod, schema, or third_party_schema\n";
    return std::nullopt;
  }
  if (!fanout_scenario(options.scenario) && options.fanout != 1U) {
    std::cerr << "--fanout greater than one requires --scenario pod, schema, "
                 "third_party_schema, tensor, image, multi_tensor_loan, or "
                 "multi_tensor_staging\n";
    return std::nullopt;
  }
  if (options.multi_output && options.multi_input) {
    std::cerr << "--multi-output and --multi-input are mutually exclusive\n";
    return std::nullopt;
  }
  if (options.mixed_policy_fanout &&
      (options.scenario != Scenario::kPod || options.fanout != 2U ||
       options.operator_lanes != OperatorLaneMode::kShared ||
       options.queue_policy != QueuePolicy::kBlocking || options.multi_output ||
       options.multi_input || offered_rate_mode(options))) {
    std::cerr << "--mixed-policy-fanout requires shared-lane blocking POD --fanout 2 "
                 "without multi-output, multi-input, or an offered rate\n";
    return std::nullopt;
  }
  if ((options.multi_output || options.multi_input) &&
      (options.scenario != Scenario::kPod || options.fanout != 2U ||
       options.operator_lanes != OperatorLaneMode::kShared ||
       options.queue_policy != QueuePolicy::kBlocking || offered_rate_mode(options) ||
       !options.force_holomq_process_local_inline)) {
    std::cerr << "--multi-output/--multi-input requires shared-lane blocking POD --fanout 2 "
                 "--force-holomq-process-local-inline without an offered rate\n";
    return std::nullopt;
  }
  if (!options.sink_queue_depths.empty()) {
    if (!fanout_scenario(options.scenario) || options.fanout <= 1U) {
      std::cerr << "--sink-queue-depths requires supported --fanout greater than one\n";
      return std::nullopt;
    }
    if (options.sink_queue_depths.size() != options.fanout) {
      std::cerr << "--sink-queue-depths must provide exactly one depth per fanout sink\n";
      return std::nullopt;
    }
    if (std::ranges::any_of(options.sink_queue_depths,
                            [](const std::uint32_t depth) { return depth == 0U; })) {
      std::cerr << "--sink-queue-depths values must be positive\n";
      return std::nullopt;
    }
    if (std::ranges::max(options.sink_queue_depths) != options.queue_depth) {
      std::cerr << "maximum --sink-queue-depths value must equal --queue-depth\n";
      return std::nullopt;
    }
    if ((options.multi_output || options.multi_input) &&
        std::ranges::any_of(options.sink_queue_depths, [&](const std::uint32_t depth) {
          return depth != options.sink_queue_depths.front();
        })) {
      std::cerr << "--multi-output/--multi-input requires equal sink queue depths\n";
      return std::nullopt;
    }
  }
  if (options.notification_capacity == 0U) {
    std::cerr << "--notification-capacity must be positive\n";
    return std::nullopt;
  }
  if (offered_rate_mode(options)) {
    if (options.offered_rate_hz > kMaximumOfferedRateHz) {
      std::cerr << "--offered-rate-hz must not exceed " << kMaximumOfferedRateHz << '\n';
      return std::nullopt;
    }
    if (options.workload != Workload::kSaturated || !lossy_queue_policy(options.queue_policy) ||
        options.fanout <= 1U || options.scenario != Scenario::kPod ||
        options.operator_lanes != OperatorLaneMode::kShared) {
      std::cerr << "--offered-rate-hz requires shared-lane saturated POD drop_oldest fanout\n";
      return std::nullopt;
    }
    if (options.offered_burst_size <= options.queue_depth ||
        options.offered_burst_size > kMaximumOfferedBurstSize) {
      std::cerr << "--offered-burst-size must exceed queue depth and not exceed "
                << kMaximumOfferedBurstSize << '\n';
      return std::nullopt;
    }
    if (options.samples < options.offered_burst_size * 2U) {
      std::cerr << "--offered-rate-hz requires at least two complete bursts\n";
      return std::nullopt;
    }
    if (options.samples % options.offered_burst_size != 0U) {
      std::cerr << "--offered-rate-hz requires --samples to contain only complete bursts\n";
      return std::nullopt;
    }
  } else if (options.offered_burst_size != 1U) {
    std::cerr << "--offered-burst-size requires --offered-rate-hz\n";
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
  if (options.partition_cover == PartitionCover::kAutomatic &&
      (options.operator_lanes != OperatorLaneMode::kSeparate || options.source_cpu >= 0 ||
       options.relay_cpu >= 0 || options.sink_cpu >= 0 || !options.sink_cpus.empty() ||
       options.busy_poll || options.green_context_layout != GreenContextLayout::kNone)) {
    std::cerr << "automatic partition-cover rows require --operator-lanes separate and cannot bind "
                 "generated names to placement or Green Context requests\n";
    return std::nullopt;
  }
  if (options.partition_cover == PartitionCover::kAutomatic &&
      options.force_holomq_process_local_inline) {
    std::cerr << "automatic partition-cover rows cannot force HoloMQ because compiler-derived "
                 "ownership is authoritative\n";
    return std::nullopt;
  }
  if (options.partition_cover == PartitionCover::kAutomatic && options.runtime_phase_counters &&
      options.fanout == 1U) {
    std::cerr << "automatic partition-cover rows without explicit fan-out cannot use "
                 "--runtime-phase-counters\n";
    return std::nullopt;
  }
  if (options.partition_cover == PartitionCover::kIncludeAll &&
      (options.operator_lanes != OperatorLaneMode::kShared || options.relay_cpu >= 0 ||
       options.sink_cpu >= 0 || !options.sink_cpus.empty())) {
    std::cerr << "include_all partition-cover rows require shared operator lanes and only the "
                 "source CPU placement\n";
    return std::nullopt;
  }
  if (options.partition_cover == PartitionCover::kHybrid &&
      (options.operator_lanes != OperatorLaneMode::kSeparate || options.relay_cpu >= 0 ||
       options.sink_cpu >= 0 || !options.sink_cpus.empty() ||
       (options.green_context_layout != GreenContextLayout::kNone &&
        options.green_context_layout != GreenContextLayout::kDedicatedToResidual))) {
    std::cerr << "hybrid partition-cover rows require separate lanes, source-only placement, and "
                 "at most dedicated_to_residual Green Context placement\n";
    return std::nullopt;
  }
  if (options.scenario == Scenario::kSourceOnly && options.sink_cpu >= 0) {
    std::cerr << "--sink-cpu is not used by --scenario source_only\n";
    return std::nullopt;
  }
  if (!uses_channel_pump(options) && options.channel_pump_cpu >= 0) {
    std::cerr << "--channel-pump-cpu requires an active channel pump\n";
    return std::nullopt;
  }
  if (options.force_holomq_process_local_inline && !direct_retained_scenario(options.scenario)) {
    std::cerr << "--force-holomq-process-local-inline requires pod, signal, sample_forward, "
                 "schema, third_party_schema, tensor, image, multi_tensor, multi_tensor_loan, or "
                 "multi_tensor_staging\n";
    return std::nullopt;
  }
  if (options.force_holomq_process_local_inline && options.fanout != 1U && !options.multi_output &&
      !options.multi_input) {
    std::cerr << "--force-holomq-process-local-inline requires --fanout 1, --multi-output, or "
                 "--multi-input\n";
    return std::nullopt;
  }
  const bool fanout_wake_control = fanout_scenario(options.scenario) && options.fanout > 1U;
  const bool exact_wake_control =
      options.force_holomq_process_local_inline &&
      (options.fanout == 1U || options.multi_output || options.multi_input) &&
      holomq_event_wake_control_scenario(options.scenario);
  if (options.force_holomq_event_wake && ((!exact_wake_control && !fanout_wake_control) ||
                                          options.operator_lanes != OperatorLaneMode::kShared)) {
    std::cerr << "--force-holomq-event-wake requires a shared-lane POD, schema, "
                 "third_party_schema, Tensor, Image, or multi-Tensor fanout or 1:1 POD, schema, "
                 "third_party_schema, Tensor, Image, or multi-Tensor "
                 "--force-holomq-process-local-inline row\n";
    return std::nullopt;
  }
  if (options.force_process_local_channel_pump &&
      (!uses_holomq(options) || options.operator_lanes != OperatorLaneMode::kShared ||
       options.force_holomq_event_wake)) {
    std::cerr << "--force-process-local-channel-pump requires an otherwise eventless shared-lane "
                 "HoloMQ row without --force-holomq-event-wake\n";
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
  if ((options.scenario == Scenario::kTensor || options.scenario == Scenario::kImage ||
       multi_tensor_scenario(options.scenario)) &&
      options.fanout > 1U && options.operator_lanes != OperatorLaneMode::kShared) {
    std::cerr << "Tensor, Image, and multi-Tensor fanout currently require --operator-lanes "
                 "shared\n";
    return std::nullopt;
  }
  if (options.operator_lanes == OperatorLaneMode::kPerSink &&
      (!fanout_scenario(options.scenario) || options.fanout <= 1U)) {
    std::cerr << "--operator-lanes per_sink requires supported fanout greater than one\n";
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
  if (!notification_driver_used(options) && options.notification_driver_cpu >= 0) {
    std::cerr << "--notification-driver-cpu requires a notification workload\n";
    return std::nullopt;
  }
  if (!notification_driver_used(options) && options.notification_capacity != 1U) {
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
      notification_driver_used(options) ? options.notification_driver_cpu : -1};
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
  if (options.bulk_work_profile == BulkWorkProfile::kDescriptorOnly &&
      options.scenario != Scenario::kTensor && options.scenario != Scenario::kMultiTensor) {
    std::cerr << "--bulk-work-profile descriptor_only requires --scenario tensor or multi_tensor\n";
    return std::nullopt;
  }
  if (options.green_context_layout != GreenContextLayout::kNone &&
      (options.scenario != Scenario::kTensor || options.tensor_storage != TensorStorage::kDevice)) {
    std::cerr << "--green-context-layout requires --scenario tensor --tensor-storage device\n";
    return std::nullopt;
  }
  if (options.green_context_layout == GreenContextLayout::kSamePartition &&
      options.operator_lanes != OperatorLaneMode::kShared) {
    std::cerr << "--green-context-layout same_partition requires --operator-lanes shared\n";
    return std::nullopt;
  }
  if (options.green_context_layout == GreenContextLayout::kNone &&
      options.green_context_sm_count != 0U) {
    std::cerr << "--green-context-sm-count requires a Green Context layout\n";
    return std::nullopt;
  }
  if ((options.green_context_layout == GreenContextLayout::kDedicatedToDedicated ||
       options.green_context_layout == GreenContextLayout::kDedicatedToResidual) &&
      options.operator_lanes != OperatorLaneMode::kSeparate) {
    std::cerr << "dedicated Green Context layouts require --operator-lanes separate\n";
    return std::nullopt;
  }
  if (options.cuda_resource_counters && options.scenario != Scenario::kTensor) {
    std::cerr << "--cuda-resource-counters requires --scenario tensor\n";
    return std::nullopt;
  }
  return options;
}

// Benchmark setup is intentionally linear so every failure cleans up the same shared state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_benchmark(Options options) {
  if (tensor_backed_scenario(options.scenario) &&
      (options.tensor_storage != TensorStorage::kPageableHost ||
       (options.scenario == Scenario::kTensor && uses_holomq(options)))) {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || options.tensor_device >= device_count ||
        cudaSetDevice(options.tensor_device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      std::cerr << "SKIP: requested CUDA Tensor storage is unavailable\n";
      return kSkipped;
    }
  }
  if (options.green_context_layout != GreenContextLayout::kNone) {
    auto capability = query_cuda_green_context_capability(DeviceId{options.tensor_device});
    if (!capability) {
      std::cerr << "SKIP: CUDA Green Context capability is unavailable: "
                << runtime_error_message(capability.error()) << '\n';
      return kSkipped;
    }
    const std::uint32_t quantum = capability->structured_sm_quantum;
    const std::uint32_t requested_sm_count =
        options.green_context_sm_count == 0U ? quantum : options.green_context_sm_count;
    const std::uint32_t dedicated_group_count =
        options.green_context_layout == GreenContextLayout::kDedicatedToDedicated ? 2U : 1U;
    // The Tensor benchmark always has a compiler-owned CUDA service obligation outside the
    // application partition, including the fused same-partition layout.
    const std::uint32_t residual_sm_count = quantum;
    const std::uint64_t required_sm_count =
        static_cast<std::uint64_t>(dedicated_group_count) * requested_sm_count + residual_sm_count;
    const std::uint64_t structured_sm_capacity =
        static_cast<std::uint64_t>(capability->max_structured_context_count) * quantum;
    if (quantum == 0U || requested_sm_count == 0U || requested_sm_count % quantum != 0U ||
        required_sm_count > structured_sm_capacity) {
      std::cerr << "SKIP: device cannot provide the requested structured Green Context layout\n";
      return kSkipped;
    }
    options.green_context_sm_count = requested_sm_count;
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
           .required = state.notification_driven()},
      });
  const std::int64_t compile_started_ns = now_ns();
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
      make_plan(state.options()), thread_control);
  if (state.options().force_process_local_channel_pump) {
    plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
  }
  BenchmarkLifecycleTiming lifecycle_timing{
      .compile_ns = static_cast<std::uint64_t>(now_ns() - compile_started_ns),
  };
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
  std::shared_ptr<detail::RuntimeTensorOutputTestControl> cuda_resource_control;
  if (state.options().cuda_resource_counters) {
    cuda_resource_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, cuda_resource_control);
  }
  if (!plan.ok()) {
    std::cerr << "failed to compile runtime Graph benchmark plan\n" << plan.json() << '\n';
    g_run_state = nullptr;
    return 2;
  }
  if (!tensor_plan_matches_options(plan, state.options())) {
    std::cerr << "compiled Tensor storage or field-pool evidence does not match the requested "
                 "benchmark geometry\n";
    g_run_state = nullptr;
    return 2;
  }
  if (!state.options().plan_output_path.empty()) {
    std::ofstream plan_output(state.options().plan_output_path, std::ios::out | std::ios::trunc);
    if (!plan_output) {
      std::cerr << "failed to open benchmark plan output: " << state.options().plan_output_path
                << '\n';
      g_run_state = nullptr;
      return 2;
    }
    plan_output << plan.json() << '\n';
  }
  if ((state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly &&
       (state.options().scenario == Scenario::kTensor ||
        state.options().scenario == Scenario::kMultiTensor)) ||
      state.options().scenario == Scenario::kMultiTensorLoan) {
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
      std::cerr << "descriptor warmup cannot cover the hot-path Tensor slots\n";
      g_run_state = nullptr;
      return 2;
    }
    if (state.options().scenario == Scenario::kMultiTensor &&
        warmup_slots > static_cast<std::size_t>(state.options().queue_depth) + 2U) {
      std::cerr << "descriptor warmup cannot force full multi-Tensor field-pool circulation at "
                   "this queue depth\n";
      g_run_state = nullptr;
      return 2;
    }
    state.set_descriptor_warmup_slots(warmup_slots);
  }

  const std::int64_t run_async_started_ns = now_ns();
  RunSession session = run_async(plan);
  lifecycle_timing.run_async_return_ns =
      static_cast<std::uint64_t>(now_ns() - run_async_started_ns);
  std::thread notification_driver;
  if (state.notification_driven()) {
    auto sender = state.take_notification_sender();
    auto pressure_gate_sender = state.options().test_gate_pod_sink_for_pressure
                                    ? state.take_pressure_gate_notification_sender()
                                    : std::optional<NotificationSender>{};
    if (sender.has_value() &&
        (!state.options().test_gate_pod_sink_for_pressure || pressure_gate_sender.has_value())) {
      notification_driver = std::thread([&state,
                                         thread_control,
                                         phase_counters,
                                         sender = std::move(*sender),
                                         pressure_gate_sender = std::move(pressure_gate_sender)]() {
        const detail::RuntimeThreadUsageScope thread_usage{
            phase_counters.get(), detail::RuntimeThreadUsageRole::kNotificationDriver};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        try {
#endif
          auto entered = thread_control->enter_notification_driver();
          if (!entered) {
            state.fail(entered.error().code);
            return;
          }
          if (state.options().test_gate_pod_sink_for_pressure) {
            if (!phase_counters || !pressure_gate_sender) {
              state.fail(ErrorCode::kProtocolError);
              return;
            }
            state.drive_pressured_pod_diagnostic_notifications(
                sender, *pressure_gate_sender, *phase_counters);
          } else if (state.notification_latency()) {
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
  const auto benchmark_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{state.options().timeout_ms};
  bool signaled = false;
  while (!signaled && session.termination() == RunTermination::kRunning &&
         std::chrono::steady_clock::now() < benchmark_deadline) {
    signaled = state.wait_for(std::chrono::milliseconds{10});
  }
  if (!signaled) {
    if (session.termination() == RunTermination::kFailed) {
      std::cerr << "runtime Graph benchmark session failed before completion";
      if (!session.diagnostics().empty()) {
        std::cerr << ": " << session.diagnostics().front().token;
      }
      std::cerr << '\n';
      state.fail(ErrorCode::kFailure);
    } else {
      std::cerr << "runtime Graph benchmark timeout: notifications="
                << state.notifications_accepted() << " publications=" << state.publications()
                << " deliveries=" << state.deliveries()
                << " completions=" << state.activation_completions() << '\n';
      state.fail(ErrorCode::kTimeout);
    }
  }
  if (notification_driver.joinable()) {
    notification_driver.join();
  }
  bool retained_tensor_valid = true;
  std::optional<std::uint64_t> final_validation_ns;
  if (tensor_backed_scenario(state.options().scenario) &&
      state.options().validation == PayloadValidation::kFinal && signaled && state.completed()) {
    const std::int64_t validation_start_ns = now_ns();
    if (state.options().scenario == Scenario::kMultiTensor) {
      retained_tensor_valid = state.validate_retained_multi_tensor();
    } else if (multi_tensor_scenario(state.options().scenario)) {
      retained_tensor_valid = state.validate_retained_multi_tensors();
    } else {
      retained_tensor_valid = state.validate_retained_tensor();
    }
    final_validation_ns = static_cast<std::uint64_t>(now_ns() - validation_start_ns);
    if (!retained_tensor_valid) {
      state.record_validation_error();
      state.fail(ErrorCode::kProtocolError);
    }
  }
  session.request_stop();
  const std::int64_t terminal_wait_started_ns = now_ns();
  session.wait();
  lifecycle_timing.terminal_wait_ns =
      static_cast<std::uint64_t>(now_ns() - terminal_wait_started_ns);
  const std::int64_t first_callback_entry_ns = state.first_callback_entry_ns();
  lifecycle_timing.run_async_to_first_callback_ns =
      first_callback_entry_ns > run_async_started_ns
          ? static_cast<std::uint64_t>(first_callback_entry_ns - run_async_started_ns)
          : 0U;
  const std::optional<detail::RuntimeTensorOutputTestControl::PoolSnapshot> cuda_resource_counters =
      cuda_resource_control != nullptr ? std::optional{cuda_resource_control->snapshot(0U)}
                                       : std::nullopt;
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
              final_validation_ns,
              lifecycle_timing,
              cuda_resource_counters);

  const std::uint64_t expected_total = state.final_sequence();
  const bool source_only = state.options().scenario == Scenario::kSourceOnly;
  const bool lossy = has_lossy_subscriber(state.options());
  const std::uint64_t expected_publications = source_only ? 0U : expected_total;
  const std::uint64_t expected_delivery_opportunities =
      expected_publications * state.options().fanout;
  const std::uint64_t expected_deliveries =
      source_only ? 0U : (lossy ? state.deliveries() : expected_total * state.options().fanout);
  const std::uint64_t expected_deliveries_per_sink = source_only ? 0U : expected_total;
  const std::uint64_t expected_sink_callbacks =
      state.options().multi_input ? expected_publications : expected_deliveries;
  const std::uint64_t expected_activation_completions =
      source_only || !lossy ? expected_total : state.activation_completions();
  const std::uint64_t expected_relay = state.options().scenario == Scenario::kSampleForward
                                           ? expected_total * state.options().relay_count
                                           : 0U;
  const std::uint64_t expected_tensor_emits =
      state.options().scenario == Scenario::kTensor ? expected_publications : 0U;
  const std::uint64_t expected_inline_payload_emits =
      state.options().scenario == Scenario::kSourceOnly ||
              tensor_backed_scenario(state.options().scenario)
          ? 0U
          : expected_publications * source_emit_count(state.options()) + expected_relay;
  const bool descriptor_initialization_valid =
      state.options().bulk_work_profile != BulkWorkProfile::kDescriptorOnly ||
      (state.descriptor_backing_initializations() >=
           (state.options().scenario == Scenario::kMultiTensor ? 2U : 1U) &&
       state.descriptor_backing_initializations_during_measurement() == 0U);
  const bool notification_counts_valid =
      !state.notification_driven() ||
      (offered_rate_mode(state.options())
           ? state.notifications_accepted() == expected_total &&
                 state.notification_attempts() == expected_total &&
                 state.notification_backpressure() == 0U && state.notification_not_ready() == 0U
           : (state.notification_latency() ? state.notifications_accepted() == expected_total
                                           : state.notifications_accepted() >= expected_total) &&
                 state.notification_attempts() >= state.notifications_accepted());
  const bool timing_valid =
      state.measured_start_ns() > 0 && state.measured_end_ns() > state.measured_start_ns();
  const bool replacement_counts_valid =
      !lossy || (state.deliveries() > 0U && state.deliveries() <= expected_delivery_opportunities &&
                 state.consumer_visible_replacements() + state.deliveries() ==
                     expected_delivery_opportunities &&
                 (state.options().workload == Workload::kSaturated ||
                  state.consumer_visible_replacements() == 0U));
  const double accepted_offer_rate = state.accepted_publication_offer_rate_per_second();
  const bool offered_rate_valid =
      !offered_rate_mode(state.options()) ||
      (accepted_offer_rate > 0.0 &&
       std::abs(accepted_offer_rate - static_cast<double>(state.options().offered_rate_hz)) /
               static_cast<double>(state.options().offered_rate_hz) <=
           kMaximumOfferedRateRelativeError &&
       state.every_sink_has_replacements());
  const std::uint64_t expected_source_callbacks = state.expected_source_callbacks();
  const std::uint64_t expected_measured_source_callbacks =
      state.expected_measured_source_callbacks();
  const bool measured_source_callbacks_valid =
      offered_rate_mode(state.options())
          ? state.measured_source_callbacks() == expected_measured_source_callbacks
          : state.measured_source_callbacks() >= expected_measured_source_callbacks &&
                state.measured_source_callbacks() <=
                    expected_measured_source_callbacks + state.source_backpressure_count();
  const bool valid =
      signaled && state.completed() && state.first_error() == ErrorCode::kSuccess && timing_valid &&
      state.publications() == expected_publications && state.deliveries() == expected_deliveries &&
      state.activation_completions() == expected_activation_completions &&
      state.source_callbacks() == expected_source_callbacks && measured_source_callbacks_valid &&
      state.relay_publications() == expected_relay && state.relay_callbacks() == expected_relay &&
      state.sink_callbacks() == expected_sink_callbacks && retained_tensor_valid &&
      (source_only || state.sink_evidence_valid(expected_total, expected_deliveries_per_sink)) &&
      state.lossy_activation_completions_valid() && offered_rate_valid &&
      state.validation_errors() == 0U && state.allocation_probe_valid() &&
      descriptor_initialization_valid && replacement_counts_valid && notification_counts_valid &&
      thread_control->required_roles_entered() &&
      phase_counters_valid(state, phase_counters.get()) &&
      terminal_phase_state_valid(phase_trace, phase_counters.get()) &&
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
              << " source_compute_entries=" << state.source_compute_entries()
              << " source_terminal_compute_entries=" << state.source_terminal_compute_entries()
              << " backpressured_emits=" << state.backpressure_count()
              << " source_backpressured_emits=" << state.source_backpressure_count()
              << " source_tensor_pool_retries=" << state.source_tensor_pool_retries()
              << " source_tensor_pool_in_activation_retries="
              << state.source_tensor_pool_in_activation_retries()
              << " expected_source_callbacks=" << expected_source_callbacks
              << " measured_source_callbacks=" << state.measured_source_callbacks()
              << " expected_measured_source_callbacks=" << expected_measured_source_callbacks
              << " activation_completions=" << state.activation_completions()
              << " expected_activation_completions=" << expected_activation_completions
              << " final_sequence=" << expected_total
              << " offered_rate_valid=" << offered_rate_valid
              << " accepted_offer_rate=" << accepted_offer_rate
              << " configured_offer_rate=" << state.options().offered_rate_hz
              << " start_ns=" << state.measured_start_ns() << " end_ns=" << state.measured_end_ns()
              << " termination=" << termination_name(session.termination()) << '\n';
    if (!session.diagnostics().empty()) {
      std::cerr << format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
    }
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
