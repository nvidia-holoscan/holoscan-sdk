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
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <poll.h>
#include <unistd.h>
#endif

#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime_multi_tensor_benchmark_traits.hpp"

#include "../common/benchmark_pod.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_graph_benchmark_common.hpp"

namespace holoscan {

namespace {

using benchmarking::benchmark_thread_identity;
using benchmarking::BenchmarkMonotonicClock;
using benchmarking::BenchmarkSupportThreadControl;
using benchmarking::emit_cpu_list;
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
using benchmarking::summarize_percentiles;
using benchmarking::SupportThreadBinding;
using benchmarking::SupportThreadRole;
using benchmarking::tensor_descriptor_matches;
using benchmarking::tensor_edge_bytes_match;
using benchmarking::tensor_memory_kind;
using benchmarking::tensor_storage_name;
using benchmarking::TensorStorage;
using benchmarking::TransportPercentiles;
using benchmarking::validation_timing_name;
using BenchmarkClock = BenchmarkMonotonicClock;
using MultiTensorMessage = runtime_benchmark::MultiTensorMessageT;
using MultiTensorFields = runtime_benchmark::MultiTensorMessageTensorFields;

constexpr std::uint64_t kMaximumPhaseTraceSamples = 100'000U;
// One transported sample records callback entry/return in addition to ingress, readiness,
// dispatch, retirement, and controller wake/park evidence. Leave room for the intervening
// one-millisecond controller timeout/park cycles in the documented sparse 5-ms trace row.
constexpr std::size_t kPhaseTraceRecordsPerSample = 32U;
constexpr std::size_t kDefaultTensorBytes = 1U << 20U;
constexpr std::uint8_t kDescriptorTensorFillByte = 0xA5U;
constexpr std::size_t kMaximumImageSinkComputeAllocations = 6U;
constexpr std::chrono::microseconds kDescriptorWarmupRetentionInterval{50};
constexpr std::chrono::milliseconds kMultiTensorDescriptorWarmupRetentionInterval{1};
constexpr int kSkipped = 77;

enum class Topology { kFanout, kChain };
enum class Role { kProducer, kRelay, kConsumer };
enum class Workload { kLatency, kSaturated };
enum class Payload { kPOD, kSchema, kTensor, kImage, kMultiTensorLoan, kMultiTensorStaging };
enum class BulkWorkProfile { kFullWrite, kDescriptorOnly };

struct Options {
  Topology topology{Topology::kFanout};
  Role role{Role::kProducer};
  Workload workload{Workload::kLatency};
  PayloadValidation payload_validation{PayloadValidation::kFinal};
  Payload payload{Payload::kPOD};
  TensorStorage tensor_storage{TensorStorage::kPageableHost};
  BulkWorkProfile bulk_work_profile{BulkWorkProfile::kFullWrite};
  std::string run_id;
  std::uint32_t fanout{1U};
  std::uint32_t consumer_index{};
  std::uint64_t samples{10'000U};
  std::uint64_t warmup{1'000U};
  std::uint32_t queue_depth{1U};
  std::size_t tensor_bytes{kDefaultTensorBytes};
  int tensor_device{};
  std::uint32_t publisher_period_us{1'000U};
  std::uint32_t timeout_ms{30'000U};
  std::uint32_t start_timeout_ms{90'000U};
  int producer_cpu{-1};
  int relay_cpu{-1};
  std::vector<int> consumer_cpus{-1};
  int producer_controller_cpu{-1};
  int relay_controller_cpu{-1};
  int relay_pump_cpu{-1};
  std::vector<int> consumer_controller_cpus{-1};
  std::vector<int> consumer_pump_cpus{-1};
  bool busy_poll{};
  bool runtime_phase_trace{};
  bool trace_correlation_self_test{};
  bool start_barrier_self_test{};
  int ready_fd{-1};
  int start_fd{-1};
  std::string output_path;
};

struct IngressWakeTraceSummary {
  bool applicable{};
  bool valid{};
  bool readiness_poll_elided{};
  std::size_t records{};
  std::uint64_t dropped{};
  std::uint64_t samples{};
  std::uint64_t transport_wait_samples{};
  std::uint64_t pump_wake_samples{};
  std::uint64_t parked_wake_samples{};
  std::uint64_t active_controller_samples{};
  std::vector<std::uint64_t> transport_wait_to_ingress;
  std::vector<std::uint64_t> ingress_to_pump_wake;
  std::vector<std::uint64_t> pump_wake_to_readiness_poll;
  std::vector<std::uint64_t> pump_wake_to_controller_return;
  std::vector<std::uint64_t> controller_return_to_readiness_poll;
  std::vector<std::uint64_t> controller_return_to_candidate;
  std::vector<std::uint64_t> readiness_poll_to_candidate;
  std::vector<std::uint64_t> readiness_poll_to_compute_dispatch;
  std::vector<std::uint64_t> candidate_to_compute_dispatch;
  std::vector<std::uint64_t> compute_dispatch_to_callback_entry;
  std::vector<std::uint64_t> transport_wait_to_callback_entry;
};

[[nodiscard]] std::int64_t now_ns() noexcept {
  return BenchmarkClock::now_ns();
}

[[nodiscard]] std::string consumer_role_name(const std::uint32_t fanout,
                                             const std::uint32_t index) {
  return fanout == 1U ? "consumer" : "consumer-" + std::to_string(index + 1U);
}

[[nodiscard]] std::string role_name(const Options& options) {
  switch (options.role) {
    case Role::kProducer:
      return "producer";
    case Role::kRelay:
      return "relay";
    case Role::kConsumer:
      return consumer_role_name(options.fanout, options.consumer_index);
  }
  return "unknown";
}

[[nodiscard]] std::string consumer_partition_name(const std::uint32_t fanout,
                                                  const std::uint32_t index) {
  return fanout == 1U ? "runtime-interprocess-consumer"
                      : "runtime-interprocess-consumer-" + std::to_string(index + 1U);
}

[[nodiscard]] const char* topology_name(const Topology topology) noexcept {
  return topology == Topology::kFanout ? "fanout" : "chain";
}

[[nodiscard]] const char* workload_name(const Workload workload) noexcept {
  return workload == Workload::kLatency ? "latency" : "saturated";
}

[[nodiscard]] const char* payload_name(const Payload payload) noexcept {
  switch (payload) {
    case Payload::kPOD:
      return "pod";
    case Payload::kSchema:
      return "schema";
    case Payload::kTensor:
      return "tensor";
    case Payload::kImage:
      return "image";
    case Payload::kMultiTensorLoan:
      return "multi_tensor_loan";
    case Payload::kMultiTensorStaging:
      return "multi_tensor_staging";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool multi_tensor_payload(const Payload payload) noexcept {
  return payload == Payload::kMultiTensorLoan || payload == Payload::kMultiTensorStaging;
}

[[nodiscard]] constexpr bool tensor_backed_payload(const Payload payload) noexcept {
  return payload == Payload::kTensor || payload == Payload::kImage || multi_tensor_payload(payload);
}

[[nodiscard]] constexpr bool descriptor_schema_payload(const Payload payload) noexcept {
  return payload == Payload::kImage || multi_tensor_payload(payload);
}

[[nodiscard]] constexpr bool allocation_probed_payload(const Payload payload) noexcept {
  return payload == Payload::kSchema || descriptor_schema_payload(payload);
}

[[nodiscard]] constexpr const char* allocation_probe_sink_access(const Payload payload) noexcept {
  return multi_tensor_payload(payload) ? "selected_batch" : "receive";
}

[[nodiscard]] constexpr std::size_t allocation_probe_source_limit(const Payload payload) noexcept {
  // Baseline staging may allocate once while a non-exportable Tensor falls back to its bounded
  // staging pool. Optimized implementations can bypass that failed export probe and report zero.
  return payload == Payload::kMultiTensorStaging ? 1U : 0U;
}

[[nodiscard]] constexpr std::size_t allocation_probe_sink_limit(const Payload payload) noexcept {
  return payload == Payload::kImage ? kMaximumImageSinkComputeAllocations : 0U;
}

[[nodiscard]] const char* bulk_work_profile_name(const BulkWorkProfile profile) noexcept {
  return profile == BulkWorkProfile::kFullWrite ? "full_write" : "descriptor_only";
}

[[nodiscard]] std::size_t application_payload_bytes(const Options& options) noexcept {
  switch (options.payload) {
    case Payload::kPOD:
      return sizeof(BenchmarkPOD64);
    case Payload::kSchema:
      return sizeof(schema::TimeT);
    case Payload::kTensor:
    case Payload::kImage:
      return options.tensor_bytes;
    case Payload::kMultiTensorLoan:
    case Payload::kMultiTensorStaging:
      return options.tensor_bytes * 2U;
  }
  return 0U;
}

[[nodiscard]] std::optional<std::size_t> inline_transport_bytes(const Options& options) noexcept {
  switch (options.payload) {
    case Payload::kPOD:
      return sizeof(BenchmarkPOD64);
    case Payload::kSchema:
      // Generated schemas submit a variable encoded prefix of the bounded wire capacity.
      return std::nullopt;
    case Payload::kTensor:
      return kTensorTransportPayloadBytes;
    case Payload::kImage:
      return kDescriptorBackedImagePayloadBytes;
    case Payload::kMultiTensorLoan:
    case Payload::kMultiTensorStaging:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::size_t external_buffer_bytes(const Options& options) noexcept {
  if (multi_tensor_payload(options.payload)) {
    return options.tensor_bytes * 2U;
  }
  return tensor_backed_payload(options.payload) ? options.tensor_bytes : 0U;
}

[[nodiscard]] const char* payload_write_profile(const Options& options) noexcept {
  if (options.payload == Payload::kPOD) {
    return "value_fill";
  }
  if (options.payload == Payload::kSchema) {
    return "full_payload_per_sample";
  }
  return options.bulk_work_profile == BulkWorkProfile::kFullWrite
             ? "full_buffer_write_per_sample"
             : "preinitialized_immutable_backing";
}

[[nodiscard]] const char* result_bulk_work_profile(const Options& options) noexcept {
  return tensor_backed_payload(options.payload) ? bulk_work_profile_name(options.bulk_work_profile)
                                                : "not_applicable";
}

[[nodiscard]] BenchmarkPOD64 make_payload(const std::uint64_t sequence,
                                          const std::int64_t full_start_time_ns,
                                          const std::int64_t send_time_ns) noexcept {
  BenchmarkPOD64 payload{
      .sequence = sequence,
      .source_start_time_ns = full_start_time_ns,
      .send_time_ns = send_time_ns,
  };
  payload.checksum = benchmark_pod_checksum(payload);
  return payload;
}

[[nodiscard]] bool payload_valid(const BenchmarkPOD64& payload) noexcept {
  return benchmark_pod_valid(payload);
}

[[nodiscard]] bool payload_contents_required(const Options& options,
                                             const std::uint64_t sequence) noexcept {
  return options.payload_validation == PayloadValidation::kAll ||
         (options.payload_validation == PayloadValidation::kFinal &&
          sequence == options.samples + options.warmup);
}

void emit_percentiles(std::ostream& output, const TransportPercentiles& values) {
  if (values.count == 0U) {
    output << "{\"pMin\":null,\"p50\":null,\"p99\":null,\"p999\":null,\"pMax\":null}";
    return;
  }
  output << "{\"pMin\":" << values.p_min << ",\"p50\":" << values.p50 << ",\"p99\":" << values.p99
         << ",\"p999\":";
  if (values.count >= 10'000U) {
    output << values.p999;
  } else {
    output << "null";
  }
  output << ",\"pMax\":" << values.p_max << '}';
}

[[nodiscard]] TransportPercentiles calibrate_timestamp_read_pairs() {
  constexpr std::size_t kCalibrationSamples = 4'096U;
  std::vector<std::uint64_t> samples;
  samples.reserve(kCalibrationSamples);
  for (std::size_t index = 0U; index < kCalibrationSamples; ++index) {
    const std::int64_t start_ns = now_ns();
    const std::int64_t end_ns = now_ns();
    samples.push_back(static_cast<std::uint64_t>(end_ns - start_ns));
  }
  return summarize_percentiles(std::move(samples));
}

void emit_ingress_wake_trace(std::ostream& output, const IngressWakeTraceSummary& trace) {
  output << "{\"enabled\":" << (trace.applicable ? "true" : "false");
  if (!trace.applicable) {
    output << '}';
    return;
  }
  output << ",\"valid\":" << (trace.valid ? "true" : "false")
         << ",\"readiness_poll_elided\":" << (trace.readiness_poll_elided ? "true" : "false")
         << ",\"records\":" << trace.records << ",\"dropped\":" << trace.dropped
         << ",\"samples\":" << trace.samples
         << ",\"transport_wait_samples\":" << trace.transport_wait_samples
         << ",\"pump_wake_samples\":" << trace.pump_wake_samples
         << ",\"parked_wake_samples\":" << trace.parked_wake_samples
         << ",\"active_controller_samples\":" << trace.active_controller_samples
         << ",\"transport_wait_to_ingress_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.transport_wait_to_ingress));
  output << ",\"ingress_to_pump_wake_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.ingress_to_pump_wake));
  output << ",\"pump_wake_to_readiness_poll_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.pump_wake_to_readiness_poll));
  output << ",\"pump_wake_to_controller_return_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.pump_wake_to_controller_return));
  output << ",\"controller_return_to_readiness_poll_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.controller_return_to_readiness_poll));
  output << ",\"controller_return_to_candidate_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.controller_return_to_candidate));
  output << ",\"readiness_poll_to_candidate_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.readiness_poll_to_candidate));
  output << ",\"readiness_poll_to_compute_dispatch_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.readiness_poll_to_compute_dispatch));
  output << ",\"candidate_to_compute_dispatch_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.candidate_to_compute_dispatch));
  output << ",\"compute_dispatch_to_callback_entry_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.compute_dispatch_to_callback_entry));
  output << ",\"transport_wait_to_callback_entry_ns\":";
  emit_percentiles(output, summarize_percentiles(trace.transport_wait_to_callback_entry));
  output << '}';
}

template <std::size_t Size>
[[nodiscard]] std::string hex_bytes(const std::array<std::byte, Size>& bytes) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  std::string result;
  result.reserve(Size * 2U);
  for (const std::byte value : bytes) {
    const auto byte = std::to_integer<unsigned char>(value);
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0FU]);
  }
  return result;
}

class RunState {
 public:
  explicit RunState(Options options) : options_(std::move(options)) {
    emit_ns_.reserve(static_cast<std::size_t>(options_.samples));
    producer_full_ns_.reserve(static_cast<std::size_t>(options_.samples));
    receive_ns_.reserve(static_cast<std::size_t>(options_.samples));
    end_to_end_ns_.reserve(static_cast<std::size_t>(options_.samples));
    relay_receive_ns_.reserve(static_cast<std::size_t>(options_.samples));
    relay_emit_ns_.reserve(static_cast<std::size_t>(options_.samples));
    relay_callback_ns_.reserve(static_cast<std::size_t>(options_.samples));
    first_hop_ns_.reserve(static_cast<std::size_t>(options_.samples));
    relay_residence_ns_.reserve(static_cast<std::size_t>(options_.samples));
    second_hop_ns_.reserve(static_cast<std::size_t>(options_.samples));
    receive_start_ns_by_sequence_.resize(static_cast<std::size_t>(final_sequence() + 1U));
  }

  [[nodiscard]] const Options& options() const noexcept { return options_; }

  [[nodiscard]] std::uint64_t final_sequence() const noexcept {
    return options_.warmup + options_.samples;
  }

  [[nodiscard]] bool measured(const std::uint64_t sequence) const noexcept {
    return sequence > options_.warmup && sequence <= final_sequence();
  }

  [[nodiscard]] bool allocation_probe(const std::uint64_t sequence) const noexcept {
    return allocation_probed_payload(options_.payload) && options_.warmup != 0U &&
           sequence == options_.warmup;
  }

  void set_descriptor_warmup_slots(const std::size_t slots) noexcept {
    descriptor_warmup_slots_ = slots;
  }

  [[nodiscard]] std::size_t descriptor_warmup_slots() const noexcept {
    return descriptor_warmup_slots_;
  }

  void record_start(const Role role) {
    switch (role) {
      case Role::kProducer:
        producer_starts_.fetch_add(1U, std::memory_order_release);
        break;
      case Role::kRelay:
        relay_starts_.fetch_add(1U, std::memory_order_release);
        break;
      case Role::kConsumer:
        consumer_starts_.fetch_add(1U, std::memory_order_release);
        break;
    }
    {
      const std::lock_guard lock(mutex_);
      started_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] bool wait_for_start() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
      return started_ || failed_;
    });
  }

  void publish_sender(NotificationSender sender) {
    {
      const std::lock_guard lock(mutex_);
      sender_.emplace(std::move(sender));
    }
    cv_.notify_all();
  }

  [[nodiscard]] std::optional<NotificationSender> take_sender() {
    std::unique_lock lock(mutex_);
    const bool ready = cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
      return sender_.has_value() || failed_;
    });
    if (!ready || !sender_.has_value()) {
      return std::nullopt;
    }
    return std::exchange(sender_, std::nullopt);
  }

  void record_source_callback() noexcept {
    record_callback_thread();
    source_callbacks_.fetch_add(1U, std::memory_order_release);
  }

  void record_relay_callback() noexcept {
    record_callback_thread();
    relay_callbacks_.fetch_add(1U, std::memory_order_release);
  }

  void record_backpressure() noexcept {
    backpressured_emits_.fetch_add(1U, std::memory_order_relaxed);
  }

  void record_descriptor_backing_initialization() noexcept {
    descriptor_backing_initializations_.fetch_add(1U, std::memory_order_relaxed);
    if (measured_start_ns_ != 0) {
      descriptor_backing_initializations_during_measurement_.fetch_add(1U,
                                                                       std::memory_order_relaxed);
    }
  }

  void record_source_allocation_probe(const std::size_t count) noexcept {
    source_probe_allocations_ = count;
  }

  void record_sink_allocation_probe(const std::size_t count) noexcept {
    sink_probe_allocations_ = count;
  }

  void record_publication(const std::uint64_t sequence, const std::int64_t full_start_time_ns,
                          const std::uint64_t emit_ns, const std::uint64_t producer_full_ns) {
    if (measured(sequence)) {
      if (sequence == options_.warmup + 1U) {
        measured_start_ns_ = full_start_time_ns;
      }
      emit_ns_.push_back(emit_ns);
      producer_full_ns_.push_back(producer_full_ns);
      if (sequence == final_sequence()) {
        measured_end_ns_ = full_start_time_ns + static_cast<std::int64_t>(producer_full_ns);
      }
    }
    publications_.fetch_add(1U, std::memory_order_release);
    if (sequence == final_sequence()) {
      finish();
    }
  }

  void record_delivery(const BenchmarkPOD64& payload, const std::int64_t receive_start_ns,
                       const std::int64_t receive_done_ns) {
    record_callback_thread();
    receive_start_ns_by_sequence_[static_cast<std::size_t>(payload.sequence)] = receive_start_ns;
    if (measured(payload.sequence)) {
      if (payload.sequence == options_.warmup + 1U) {
        measured_start_ns_ = payload.source_start_time_ns;
      }
      receive_ns_.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      end_to_end_ns_.push_back(static_cast<std::uint64_t>(receive_done_ns - payload.send_time_ns));
      if (options_.topology == Topology::kChain) {
        first_hop_ns_.push_back(
            static_cast<std::uint64_t>(payload.relay_receive_time_ns - payload.send_time_ns));
        relay_residence_ns_.push_back(
            static_cast<std::uint64_t>(payload.relay_send_time_ns - payload.relay_receive_time_ns));
        second_hop_ns_.push_back(
            static_cast<std::uint64_t>(receive_done_ns - payload.relay_send_time_ns));
      }
      if (payload.sequence == final_sequence()) {
        measured_end_ns_ = receive_done_ns;
      }
    }
    deliveries_.fetch_add(1U, std::memory_order_release);
    final_sequence_.store(payload.sequence, std::memory_order_release);
    if (payload.sequence == final_sequence()) {
      finish();
    }
  }

  void record_metadata_delivery(const std::uint64_t sequence,
                                const std::int64_t capture_timestamp_ns,
                                const std::int64_t receive_start_ns,
                                const std::int64_t receive_done_ns) {
    record_callback_thread();
    receive_start_ns_by_sequence_[static_cast<std::size_t>(sequence)] = receive_start_ns;
    if (measured(sequence)) {
      if (sequence == options_.warmup + 1U) {
        measured_start_ns_ = capture_timestamp_ns;
      }
      receive_ns_.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
      end_to_end_ns_.push_back(static_cast<std::uint64_t>(receive_done_ns - capture_timestamp_ns));
      if (sequence == final_sequence()) {
        measured_end_ns_ = receive_done_ns;
      }
    }
    deliveries_.fetch_add(1U, std::memory_order_release);
    final_sequence_.store(sequence, std::memory_order_release);
    if (sequence == final_sequence()) {
      finish();
    }
  }

  void record_relay_input(const BenchmarkPOD64& payload, const std::int64_t receive_start_ns,
                          const std::int64_t receive_done_ns) {
    receive_start_ns_by_sequence_[static_cast<std::size_t>(payload.sequence)] = receive_start_ns;
    if (measured(payload.sequence)) {
      relay_receive_ns_.push_back(static_cast<std::uint64_t>(receive_done_ns - receive_start_ns));
    }
    relay_inputs_.fetch_add(1U, std::memory_order_release);
    final_sequence_.store(payload.sequence, std::memory_order_release);
  }

  void record_relay_publication(const BenchmarkPOD64& payload, const std::int64_t callback_start_ns,
                                const std::int64_t emit_start_ns, const std::int64_t emit_done_ns) {
    if (measured(payload.sequence)) {
      if (payload.sequence == options_.warmup + 1U) {
        measured_start_ns_ = payload.source_start_time_ns;
      }
      relay_emit_ns_.push_back(static_cast<std::uint64_t>(emit_done_ns - emit_start_ns));
      relay_callback_ns_.push_back(static_cast<std::uint64_t>(emit_done_ns - callback_start_ns));
      if (payload.sequence == final_sequence()) {
        measured_end_ns_ = emit_done_ns;
      }
    }
    relay_publications_.fetch_add(1U, std::memory_order_release);
    if (payload.sequence == final_sequence()) {
      finish();
    }
  }

  void finish() {
    {
      const std::lock_guard lock(mutex_);
      completed_ = true;
    }
    cv_.notify_all();
  }

  void fail(const ErrorCode code) noexcept {
    ErrorCode expected = ErrorCode::kSuccess;
    static_cast<void>(first_error_.compare_exchange_strong(
        expected, code, std::memory_order_relaxed, std::memory_order_relaxed));
    {
      const std::lock_guard lock(mutex_);
      failed_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] bool wait_for_completion() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::milliseconds{options_.timeout_ms}, [this] {
      return completed_ || failed_;
    });
  }

  [[nodiscard]] bool completed() const {
    const std::lock_guard lock(mutex_);
    return completed_;
  }

  [[nodiscard]] ErrorCode first_error() const noexcept {
    return first_error_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t source_callbacks() const noexcept {
    return source_callbacks_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t publications() const noexcept {
    return publications_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t relay_callbacks() const noexcept {
    return relay_callbacks_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t relay_inputs() const noexcept {
    return relay_inputs_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t relay_publications() const noexcept {
    return relay_publications_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t deliveries() const noexcept {
    return deliveries_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t final_sequence_observed() const noexcept {
    return final_sequence_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t callback_thread_id() const noexcept {
    return callback_thread_id_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool callback_identity_is_native() const noexcept {
    return callback_identity_is_native_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool callback_native_tid_syscall_unavailable() const noexcept {
    return callback_native_tid_syscall_unavailable_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] bool callback_multiple_threads() const noexcept {
    return callback_multiple_threads_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t callback_count() const noexcept {
    switch (options_.role) {
      case Role::kProducer:
        return source_callbacks();
      case Role::kRelay:
        return relay_callbacks();
      case Role::kConsumer:
        return deliveries();
    }
    return 0U;
  }

  [[nodiscard]] std::uint64_t producer_starts() const noexcept {
    return producer_starts_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t consumer_starts() const noexcept {
    return consumer_starts_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t relay_starts() const noexcept {
    return relay_starts_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t notifications_accepted() const noexcept {
    return notifications_accepted_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t notification_attempts() const noexcept {
    return notification_attempts_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::uint64_t backpressured_emits() const noexcept {
    return backpressured_emits_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t descriptor_backing_initializations() const noexcept {
    return descriptor_backing_initializations_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t descriptor_backing_initializations_during_measurement()
      const noexcept {
    return descriptor_backing_initializations_during_measurement_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::optional<std::size_t> source_probe_allocations() const noexcept {
    return source_probe_allocations_;
  }

  [[nodiscard]] std::optional<std::size_t> sink_probe_allocations() const noexcept {
    return sink_probe_allocations_;
  }

  [[nodiscard]] std::int64_t measured_start_ns() const noexcept { return measured_start_ns_; }
  [[nodiscard]] std::int64_t measured_end_ns() const noexcept { return measured_end_ns_; }

  [[nodiscard]] TransportPercentiles emit_summary() const {
    return summarize_percentiles(emit_ns_);
  }
  [[nodiscard]] TransportPercentiles producer_full_summary() const {
    return summarize_percentiles(producer_full_ns_);
  }
  [[nodiscard]] TransportPercentiles receive_summary() const {
    return summarize_percentiles(receive_ns_);
  }
  [[nodiscard]] TransportPercentiles end_to_end_summary() const {
    return summarize_percentiles(end_to_end_ns_);
  }
  [[nodiscard]] TransportPercentiles relay_receive_summary() const {
    return summarize_percentiles(relay_receive_ns_);
  }
  [[nodiscard]] TransportPercentiles relay_emit_summary() const {
    return summarize_percentiles(relay_emit_ns_);
  }
  [[nodiscard]] TransportPercentiles relay_callback_summary() const {
    return summarize_percentiles(relay_callback_ns_);
  }
  [[nodiscard]] TransportPercentiles first_hop_summary() const {
    return summarize_percentiles(first_hop_ns_);
  }
  [[nodiscard]] TransportPercentiles relay_residence_summary() const {
    return summarize_percentiles(relay_residence_ns_);
  }
  [[nodiscard]] TransportPercentiles second_hop_summary() const {
    return summarize_percentiles(second_hop_ns_);
  }

  [[nodiscard]] std::int64_t receive_start_ns(const std::uint64_t sequence) const noexcept {
    return sequence < receive_start_ns_by_sequence_.size()
               ? receive_start_ns_by_sequence_[static_cast<std::size_t>(sequence)]
               : 0;
  }

  void drive(const NotificationSender& sender) noexcept {
    if (tensor_backed_payload(options_.payload)) {
      // The importer reaches lifecycle Start before the publisher process exists. Keep discovery
      // settlement outside the measured interval, matching the established cross-process Tensor
      // fixture, so the first terminal Tensor-loan publication is not used as an attach probe.
      std::this_thread::sleep_for(std::chrono::milliseconds{20});
    }
    while (publications() < final_sequence() && first_error() == ErrorCode::kSuccess) {
      const std::uint64_t accepted = notifications_accepted();
      const std::uint64_t callbacks = source_callbacks();
      if (accepted >= callbacks && accepted - callbacks >= options_.queue_depth) {
        std::this_thread::yield();
        continue;
      }
      if (options_.workload == Workload::kLatency) {
        std::this_thread::sleep_for(std::chrono::microseconds{options_.publisher_period_us});
      }
      notification_attempts_.fetch_add(1U, std::memory_order_relaxed);
      auto posted = sender.post();
      if (posted) {
        notifications_accepted_.fetch_add(1U, std::memory_order_release);
        continue;
      }
      if (posted.error().code == ErrorCode::kBackpressured ||
          posted.error().code == ErrorCode::kNotReady) {
        std::this_thread::yield();
        continue;
      }
      if (posted.error().code == ErrorCode::kClosed && publications() == final_sequence()) {
        return;
      }
      fail(posted.error().code);
      return;
    }
  }

 private:
  void record_callback_thread() noexcept {
    if (!options_.runtime_phase_trace) {
      return;
    }
    const benchmarking::BenchmarkThreadIdentity current = benchmark_thread_identity();
    std::uint64_t expected = 0U;
    if (!callback_thread_id_.compare_exchange_strong(
            expected, current.value, std::memory_order_relaxed, std::memory_order_relaxed)) {
      if (expected != current.value ||
          callback_identity_is_native_.load(std::memory_order_relaxed) != current.is_native ||
          callback_native_tid_syscall_unavailable_.load(std::memory_order_relaxed) !=
              current.native_tid_syscall_unavailable) {
        callback_multiple_threads_.store(true, std::memory_order_relaxed);
      }
      return;
    }
    callback_identity_is_native_.store(current.is_native, std::memory_order_relaxed);
    callback_native_tid_syscall_unavailable_.store(current.native_tid_syscall_unavailable,
                                                   std::memory_order_relaxed);
  }

  Options options_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::optional<NotificationSender> sender_;
  bool started_{};
  bool completed_{};
  bool failed_{};
  std::atomic<ErrorCode> first_error_{ErrorCode::kSuccess};
  std::atomic<std::uint64_t> source_callbacks_{};
  std::atomic<std::uint64_t> relay_callbacks_{};
  std::atomic<std::uint64_t> publications_{};
  std::atomic<std::uint64_t> relay_inputs_{};
  std::atomic<std::uint64_t> relay_publications_{};
  std::atomic<std::uint64_t> deliveries_{};
  std::atomic<std::uint64_t> final_sequence_{};
  std::atomic<std::uint64_t> producer_starts_{};
  std::atomic<std::uint64_t> relay_starts_{};
  std::atomic<std::uint64_t> consumer_starts_{};
  std::atomic<std::uint64_t> notifications_accepted_{};
  std::atomic<std::uint64_t> notification_attempts_{};
  std::atomic<std::uint64_t> backpressured_emits_{};
  std::atomic<std::uint64_t> descriptor_backing_initializations_{};
  std::atomic<std::uint64_t> descriptor_backing_initializations_during_measurement_{};
  std::atomic<std::uint64_t> callback_thread_id_{};
  std::atomic_bool callback_identity_is_native_{};
  std::atomic_bool callback_native_tid_syscall_unavailable_{};
  std::atomic_bool callback_multiple_threads_{};
  std::optional<std::size_t> source_probe_allocations_;
  std::optional<std::size_t> sink_probe_allocations_;
  std::size_t descriptor_warmup_slots_{};
  std::int64_t measured_start_ns_{};
  std::int64_t measured_end_ns_{};
  std::vector<std::uint64_t> emit_ns_;
  std::vector<std::uint64_t> producer_full_ns_;
  std::vector<std::uint64_t> receive_ns_;
  std::vector<std::uint64_t> end_to_end_ns_;
  std::vector<std::uint64_t> relay_receive_ns_;
  std::vector<std::uint64_t> relay_emit_ns_;
  std::vector<std::uint64_t> relay_callback_ns_;
  std::vector<std::uint64_t> first_hop_ns_;
  std::vector<std::uint64_t> relay_residence_ns_;
  std::vector<std::uint64_t> second_hop_ns_;
  std::vector<std::int64_t> receive_start_ns_by_sequence_;
};

RunState* g_run_state{};

[[nodiscard]] RunState& run_state() noexcept {
  return *g_run_state;
}

void append_trace_delta(std::vector<std::uint64_t>& values, const std::int64_t start,
                        const std::int64_t end, bool& valid) {
  if (start <= 0 || end < start) {
    valid = false;
    return;
  }
  values.push_back(static_cast<std::uint64_t>(end - start));
}

// The trace summarizer deliberately validates each ordered phase transition in one pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] IngressWakeTraceSummary summarize_ingress_wake_trace(
    const RunState& state, const detail::RuntimePhaseCounters& counters) {
  IngressWakeTraceSummary summary;
  summary.applicable =
      state.options().role != Role::kProducer && state.options().runtime_phase_trace;
  if (!summary.applicable) {
    return summary;
  }
  summary.valid = true;
  summary.records = counters.trace_records().size();
  summary.dropped = counters.trace_dropped();

  struct IngressRecord {
    std::uint64_t sequence{};
    std::int64_t timestamp_ns{};
  };
  struct SourceRecord {
    SourceId source{};
    std::uint64_t source_sequence{};
    std::int64_t timestamp_ns{};
  };
  std::vector<std::int64_t> transport_waits;
  std::vector<IngressRecord> ingress_records;
  std::vector<std::int64_t> pump_wakes;
  std::vector<std::int64_t> controller_returns;
  std::vector<SourceRecord> readiness_polls;
  std::vector<SourceRecord> frozen_candidates;
  std::vector<SourceRecord> compute_dispatches;
  const std::size_t expected = static_cast<std::size_t>(state.final_sequence());
  for (auto* values : {&transport_waits, &pump_wakes, &controller_returns}) {
    values->reserve(expected);
  }
  ingress_records.reserve(expected);
  readiness_polls.reserve(expected);
  frozen_candidates.reserve(expected);
  compute_dispatches.reserve(expected);

  for (const detail::RuntimePhaseTraceRecord& record : counters.trace_records()) {
    switch (record.point) {
      case detail::RuntimePhaseTracePoint::kChannelTransportWaitReturned:
        transport_waits.push_back(record.timestamp_ns);
        break;
      case detail::RuntimePhaseTracePoint::kChannelIngressRetained:
        ingress_records.push_back(
            {.sequence = record.source_sequence, .timestamp_ns = record.timestamp_ns});
        break;
      case detail::RuntimePhaseTracePoint::kChannelReadinessWakeSignaled:
        pump_wakes.push_back(record.timestamp_ns);
        break;
      case detail::RuntimePhaseTracePoint::kControllerWakeReturned:
        controller_returns.push_back(record.timestamp_ns);
        break;
      case detail::RuntimePhaseTracePoint::kReadinessPollReturned:
        readiness_polls.push_back({.source = record.source,
                                   .source_sequence = record.source_sequence,
                                   .timestamp_ns = record.timestamp_ns});
        break;
      case detail::RuntimePhaseTracePoint::kCandidateFrozen:
        frozen_candidates.push_back({.source = record.source,
                                     .source_sequence = record.source_sequence,
                                     .timestamp_ns = record.timestamp_ns});
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchStarted:
        compute_dispatches.push_back({.source = record.source,
                                      .source_sequence = record.source_sequence,
                                      .timestamp_ns = record.timestamp_ns});
        break;
      default:
        break;
    }
  }

  summary.readiness_poll_elided = readiness_polls.empty();
  const bool readiness_complete =
      summary.readiness_poll_elided || readiness_polls.size() == expected;
  const bool complete = transport_waits.size() >= expected && ingress_records.size() == expected &&
                        pump_wakes.size() <= expected && readiness_complete &&
                        frozen_candidates.size() == expected &&
                        compute_dispatches.size() == expected && summary.dropped == 0U;
  if (!complete) {
    summary.valid = false;
    return summary;
  }
  std::ranges::sort(transport_waits);
  std::ranges::sort(ingress_records, {}, &IngressRecord::timestamp_ns);
  std::ranges::sort(pump_wakes);
  std::ranges::sort(controller_returns);
  std::ranges::sort(readiness_polls, {}, &SourceRecord::timestamp_ns);
  std::ranges::sort(frozen_candidates, {}, &SourceRecord::timestamp_ns);
  std::ranges::sort(compute_dispatches, {}, &SourceRecord::timestamp_ns);

  std::size_t transport_index{};
  std::size_t pump_wake_index{};
  std::size_t controller_return_index{};
  std::size_t readiness_poll_index{};
  std::size_t frozen_candidate_index{};
  std::size_t compute_dispatch_index{};
  for (std::size_t index = 0U; index < expected; ++index) {
    const IngressRecord& ingress = ingress_records[index];
    const SourceRecord& candidate = frozen_candidates[frozen_candidate_index];
    const std::uint64_t expected_sequence = static_cast<std::uint64_t>(index) + 1U;
    const std::int64_t callback_entry = state.receive_start_ns(ingress.sequence);
    if (ingress.sequence != expected_sequence || candidate.source_sequence != ingress.sequence ||
        !candidate.source || candidate.timestamp_ns < ingress.timestamp_ns || callback_entry == 0) {
      std::cerr << "ingress trace identity mismatch: index=" << index
                << " ingress_sequence=" << ingress.sequence
                << " candidate_sequence=" << candidate.source_sequence
                << " candidate_source_valid=" << static_cast<bool>(candidate.source)
                << " ingress_ns=" << ingress.timestamp_ns
                << " candidate_ns=" << candidate.timestamp_ns << " callback_ns=" << callback_entry
                << '\n';
      summary.valid = false;
      return summary;
    }

    while (transport_index + 1U < transport_waits.size() &&
           transport_waits[transport_index + 1U] <= ingress.timestamp_ns) {
      ++transport_index;
    }
    const bool has_transport_wait = transport_index < transport_waits.size() &&
                                    transport_waits[transport_index] <= ingress.timestamp_ns;
    while (pump_wake_index < pump_wakes.size() &&
           pump_wakes[pump_wake_index] < ingress.timestamp_ns) {
      ++pump_wake_index;
    }
    const std::int64_t next_ingress_ns = index + 1U < expected
                                             ? ingress_records[index + 1U].timestamp_ns
                                             : std::numeric_limits<std::int64_t>::max();
    const bool has_pump_wake =
        pump_wake_index < pump_wakes.size() && pump_wakes[pump_wake_index] < next_ingress_ns;
    if (!summary.readiness_poll_elided) {
      while (readiness_poll_index < readiness_polls.size() &&
             (readiness_polls[readiness_poll_index].timestamp_ns < ingress.timestamp_ns ||
              readiness_polls[readiness_poll_index].source != candidate.source)) {
        ++readiness_poll_index;
      }
      if (readiness_poll_index == readiness_polls.size() ||
          readiness_polls[readiness_poll_index].timestamp_ns > candidate.timestamp_ns) {
        std::cerr << "ingress trace readiness correlation mismatch: index=" << index
                  << " ingress_ns=" << ingress.timestamp_ns
                  << " candidate_ns=" << candidate.timestamp_ns;
        if (readiness_poll_index < readiness_polls.size()) {
          std::cerr << " readiness_ns=" << readiness_polls[readiness_poll_index].timestamp_ns
                    << " readiness_sequence="
                    << readiness_polls[readiness_poll_index].source_sequence;
        }
        std::cerr << '\n';
        summary.valid = false;
        return summary;
      }
    }
    while (
        compute_dispatch_index < compute_dispatches.size() &&
        (compute_dispatches[compute_dispatch_index].timestamp_ns < candidate.timestamp_ns ||
         compute_dispatches[compute_dispatch_index].source != candidate.source ||
         compute_dispatches[compute_dispatch_index].source_sequence != candidate.source_sequence)) {
      ++compute_dispatch_index;
    }
    if (compute_dispatch_index == compute_dispatches.size() ||
        callback_entry < compute_dispatches[compute_dispatch_index].timestamp_ns) {
      std::cerr << "ingress trace compute correlation mismatch: index=" << index
                << " candidate_sequence=" << candidate.source_sequence
                << " candidate_ns=" << candidate.timestamp_ns << " callback_ns=" << callback_entry;
      if (compute_dispatch_index < compute_dispatches.size()) {
        std::cerr << " compute_sequence="
                  << compute_dispatches[compute_dispatch_index].source_sequence
                  << " compute_ns=" << compute_dispatches[compute_dispatch_index].timestamp_ns;
      }
      std::cerr << '\n';
      summary.valid = false;
      return summary;
    }
    if (!state.measured(ingress.sequence)) {
      transport_index += has_transport_wait ? 1U : 0U;
      pump_wake_index += has_pump_wake ? 1U : 0U;
      readiness_poll_index += summary.readiness_poll_elided ? 0U : 1U;
      ++frozen_candidate_index;
      ++compute_dispatch_index;
      continue;
    }
    if (has_transport_wait) {
      ++summary.transport_wait_samples;
      append_trace_delta(summary.transport_wait_to_ingress,
                         transport_waits[transport_index],
                         ingress.timestamp_ns,
                         summary.valid);
    }
    if (has_pump_wake) {
      ++summary.pump_wake_samples;
      append_trace_delta(summary.ingress_to_pump_wake,
                         ingress.timestamp_ns,
                         pump_wakes[pump_wake_index],
                         summary.valid);
    }
    const std::int64_t readiness_observed_ns =
        summary.readiness_poll_elided ? candidate.timestamp_ns
                                      : readiness_polls[readiness_poll_index].timestamp_ns;
    if (has_pump_wake && pump_wakes[pump_wake_index] <= readiness_observed_ns) {
      if (!summary.readiness_poll_elided) {
        append_trace_delta(summary.pump_wake_to_readiness_poll,
                           pump_wakes[pump_wake_index],
                           readiness_observed_ns,
                           summary.valid);
      }
      while (controller_return_index < controller_returns.size() &&
             controller_returns[controller_return_index] < pump_wakes[pump_wake_index]) {
        ++controller_return_index;
      }
      if (controller_return_index < controller_returns.size() &&
          controller_returns[controller_return_index] <= readiness_observed_ns) {
        append_trace_delta(summary.pump_wake_to_controller_return,
                           pump_wakes[pump_wake_index],
                           controller_returns[controller_return_index],
                           summary.valid);
        if (!summary.readiness_poll_elided) {
          append_trace_delta(summary.controller_return_to_readiness_poll,
                             controller_returns[controller_return_index],
                             readiness_observed_ns,
                             summary.valid);
        }
        append_trace_delta(summary.controller_return_to_candidate,
                           controller_returns[controller_return_index],
                           candidate.timestamp_ns,
                           summary.valid);
        ++summary.parked_wake_samples;
        ++controller_return_index;
      } else {
        ++summary.active_controller_samples;
      }
    } else {
      // The controller can observe the published source revision before the pump emits its
      // redundant wake. That valid active-controller route has no wake-to-poll latency.
      ++summary.active_controller_samples;
    }
    if (!summary.readiness_poll_elided) {
      append_trace_delta(summary.readiness_poll_to_candidate,
                         readiness_observed_ns,
                         candidate.timestamp_ns,
                         summary.valid);
      append_trace_delta(summary.readiness_poll_to_compute_dispatch,
                         readiness_observed_ns,
                         compute_dispatches[compute_dispatch_index].timestamp_ns,
                         summary.valid);
    }
    append_trace_delta(summary.candidate_to_compute_dispatch,
                       candidate.timestamp_ns,
                       compute_dispatches[compute_dispatch_index].timestamp_ns,
                       summary.valid);
    append_trace_delta(summary.compute_dispatch_to_callback_entry,
                       compute_dispatches[compute_dispatch_index].timestamp_ns,
                       callback_entry,
                       summary.valid);
    if (has_transport_wait) {
      append_trace_delta(summary.transport_wait_to_callback_entry,
                         transport_waits[transport_index],
                         callback_entry,
                         summary.valid);
    }
    ++summary.samples;
    transport_index += has_transport_wait ? 1U : 0U;
    pump_wake_index += has_pump_wake ? 1U : 0U;
    readiness_poll_index += summary.readiness_poll_elided ? 0U : 1U;
    ++frozen_candidate_index;
    ++compute_dispatch_index;
  }
  summary.valid = summary.valid && summary.samples == state.options().samples &&
                  summary.transport_wait_samples == state.options().samples;
  return summary;
}

[[nodiscard]] bool trace_correlation_self_test_case(const bool include_transport_wait,
                                                    const bool include_late_pump_wake,
                                                    const bool include_readiness_poll = true) {
  Options options;
  options.role = Role::kConsumer;
  options.samples = 1U;
  options.warmup = 0U;
  options.runtime_phase_trace = true;
  RunState state{std::move(options)};

  std::array<detail::RuntimePhaseTraceRecord, 6U> trace_storage{};
  detail::RuntimePhaseCounters counters{trace_storage};
  constexpr SourceId kSource = detail::make_source_id(1U);
  const auto trace = [&](const detail::RuntimePhaseTracePoint point,
                         const SourceId source,
                         const std::uint64_t source_sequence,
                         const std::int64_t timestamp_ns) {
    counters.trace(point, 0U, source, source_sequence);
    trace_storage[counters.trace_records().size() - 1U].timestamp_ns = timestamp_ns;
  };
  if (include_transport_wait) {
    trace(detail::RuntimePhaseTracePoint::kChannelTransportWaitReturned, {}, 0U, 10);
  }
  trace(detail::RuntimePhaseTracePoint::kChannelIngressRetained, {}, 1U, 20);
  if (include_readiness_poll) {
    trace(detail::RuntimePhaseTracePoint::kReadinessPollReturned, kSource, 0U, 30);
  }
  trace(detail::RuntimePhaseTracePoint::kCandidateFrozen, kSource, 1U, 35);
  trace(detail::RuntimePhaseTracePoint::kComputeDispatchStarted, kSource, 1U, 40);
  if (include_late_pump_wake) {
    trace(detail::RuntimePhaseTracePoint::kChannelReadinessWakeSignaled, {}, 0U, 60);
  }

  state.record_delivery(
      BenchmarkPOD64{
          .sequence = 1U,
          .source_start_time_ns = 1,
          .send_time_ns = 1,
      },
      50,
      51);
  const IngressWakeTraceSummary summary = summarize_ingress_wake_trace(state, counters);
  if (!include_transport_wait) {
    return !summary.valid && summary.samples == 0U && summary.transport_wait_samples == 0U;
  }
  return summary.valid && summary.readiness_poll_elided == !include_readiness_poll &&
         summary.samples == 1U && summary.parked_wake_samples == 0U &&
         summary.transport_wait_samples == 1U &&
         summary.pump_wake_samples == (include_late_pump_wake ? 1U : 0U) &&
         summary.active_controller_samples == 1U && summary.pump_wake_to_readiness_poll.empty() &&
         summary.ingress_to_pump_wake == (include_late_pump_wake ? std::vector<std::uint64_t>{40U}
                                                                 : std::vector<std::uint64_t>{}) &&
         summary.transport_wait_to_ingress == (include_transport_wait
                                                   ? std::vector<std::uint64_t>{10U}
                                                   : std::vector<std::uint64_t>{}) &&
         summary.readiness_poll_to_compute_dispatch == (include_readiness_poll
                                                            ? std::vector<std::uint64_t>{10U}
                                                            : std::vector<std::uint64_t>{}) &&
         summary.readiness_poll_to_candidate == (include_readiness_poll
                                                     ? std::vector<std::uint64_t>{5U}
                                                     : std::vector<std::uint64_t>{}) &&
         summary.candidate_to_compute_dispatch == std::vector<std::uint64_t>{5U} &&
         summary.compute_dispatch_to_callback_entry == std::vector<std::uint64_t>{10U};
}

[[nodiscard]] bool trace_correlation_self_test() {
  return trace_correlation_self_test_case(false, false) &&
         trace_correlation_self_test_case(true, false) &&
         trace_correlation_self_test_case(true, true) &&
         trace_correlation_self_test_case(true, false, false);
}

class BenchmarkProducer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output").max_emits_per_compute(1U);
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(run_state().options().queue_depth)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &BenchmarkProducer::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override { run_state().record_start(Role::kProducer); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    state.record_source_callback();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }

    const std::int64_t full_start_time_ns = now_ns();
    const std::int64_t send_time_ns = now_ns();
    BenchmarkPOD64 payload = make_payload(next_sequence_, full_start_time_ns, send_time_ns);
    if (!payload_contents_required(state.options(), next_sequence_)) {
      payload.checksum = 0U;
    }
    const std::int64_t emit_start_ns = now_ns();
    auto emitted = output_.emit(payload,
                                EmitOptions{
                                    .frame_id = next_sequence_,
                                    .trace = TraceContext::root(next_sequence_),
                                });
    const std::int64_t emit_done_ns = now_ns();
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady) {
        state.record_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }

    state.record_publication(next_sequence_,
                             full_start_time_ns,
                             static_cast<std::uint64_t>(emit_done_ns - emit_start_ns),
                             static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
    ++next_sequence_;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      run_state().fail(sender.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    run_state().publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

 private:
  Output<BenchmarkPOD64> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkSchemaProducer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output").max_emits_per_compute(1U);
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(run_state().options().queue_depth)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &BenchmarkSchemaProducer::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override { run_state().record_start(Role::kProducer); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RunState& state = run_state();
    state.record_source_callback();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }

    const std::int64_t full_start_time_ns = now_ns();
    schema::TimeT payload{
        .time_ns = static_cast<std::int64_t>(next_sequence_),
        .acqtime_ns = payload_contents_required(state.options(), next_sequence_)
                          ? static_cast<std::int64_t>(next_sequence_ ^ 0x5A5A'5A5AU)
                          : 0,
    };
    const std::int64_t send_time_ns = now_ns();
    const std::int64_t emit_start_ns = now_ns();
    const EmitOptions emit_options{
        .capture_time =
            TimePoint{
                .timestamp_ns = send_time_ns,
                .clock_id = context.activation_time().clock_id,
            },
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
    };
    expected<void, Error> emitted;
    if (state.allocation_probe(next_sequence_)) {
      const test::ScopedAllocationCounter allocations;
      emitted = output_.emit(std::move(payload), emit_options);
      state.record_source_allocation_probe(allocations.count());
    } else {
      emitted = output_.emit(std::move(payload), emit_options);
    }
    const std::int64_t emit_done_ns = now_ns();
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady) {
        state.record_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }

    state.record_publication(next_sequence_,
                             full_start_time_ns,
                             static_cast<std::uint64_t>(emit_done_ns - emit_start_ns),
                             static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
    ++next_sequence_;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      run_state().fail(sender.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    run_state().publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

 private:
  Output<schema::TimeT> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

[[nodiscard]] bool initialize_pinned_device_source(const Options& options, void*& source,
                                                   const std::string_view payload_name) noexcept {
  if (options.tensor_storage != TensorStorage::kPinnedHost) {
    return true;
  }
  const cudaError_t selected = cudaSetDevice(options.tensor_device);
  if (selected != cudaSuccess) {
    std::cerr << payload_name << " pinned-host staging failed to select CUDA device "
              << options.tensor_device << ": " << cudaGetErrorName(selected) << " ("
              << cudaGetErrorString(selected) << ")\n";
    return false;
  }
  const cudaError_t allocated = cudaMalloc(&source, options.tensor_bytes);
  if (allocated != cudaSuccess) {
    std::cerr << payload_name << " pinned-host staging failed to allocate " << options.tensor_bytes
              << " bytes on CUDA device " << options.tensor_device << ": "
              << cudaGetErrorName(allocated) << " (" << cudaGetErrorString(allocated) << ")\n";
    source = nullptr;
    return false;
  }
  return true;
}

class BenchmarkTensorProducer final : public Operator<> {
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
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(options.queue_depth)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &BenchmarkTensorProducer::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override {
    RunState& state = run_state();
    initialized_backings_.reserve(state.descriptor_warmup_slots());
    if (!initialize_pinned_device_source(state.options(), pinned_device_source_, "Tensor")) {
      state.fail(ErrorCode::kFailure);
    }
    state.record_start(Role::kProducer);
  }

  void stop() override {
    if (pinned_device_source_ != nullptr) {
      static_cast<void>(cudaFree(pinned_device_source_));
      pinned_device_source_ = nullptr;
    }
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RunState& state = run_state();
    state.record_source_callback();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }

    const std::int64_t full_start_time_ns = now_ns();
    const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
    auto loan =
        output_.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kTensorByteDtype});
    if (!loan) {
      if (loan.error().code == ErrorCode::kBackpressured ||
          loan.error().code == ErrorCode::kNotReady ||
          loan.error().code == ErrorCode::kResourceExhausted) {
        state.record_backpressure();
        return {};
      }
      state.fail(loan.error().code);
      return make_unexpected(std::move(loan).error());
    }

    const bool descriptor_only =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
    const std::uint8_t fill =
        descriptor_only ? kDescriptorTensorFillByte : static_cast<std::uint8_t>(next_sequence_);
    void* const pinned_device_source = pinned_device_source_;
    bool initializes_descriptor_backing{};
    const auto should_write_payload = [this, descriptor_only, &initializes_descriptor_backing](
                                          const TensorOutputWriteGuard& writer) noexcept {
      initializes_descriptor_backing = descriptor_only && !backing_initialized(writer.data());
      return !descriptor_only || initializes_descriptor_backing;
    };
    expected<void, Error> committed;
    if (state.options().tensor_storage != TensorStorage::kPageableHost) {
      const ::cuda::stream_ref stream = context.cuda_stream();
      if (stream.get() == nullptr) {
        state.fail(ErrorCode::kNotReady);
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      committed = loan->write_and_commit(
          stream,
          [fill, pinned_device_source, stream, &state, &should_write_payload](
              const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (!should_write_payload(writer)) {
              return {};
            }
            if (state.options().tensor_storage == TensorStorage::kDevice) {
              if (cudaMemsetAsync(
                      writer.data(), static_cast<int>(fill), writer.byte_size(), stream.get()) !=
                  cudaSuccess) {
                return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
              }
            } else {
              if (pinned_device_source == nullptr ||
                  cudaMemsetAsync(pinned_device_source,
                                  static_cast<int>(fill),
                                  writer.byte_size(),
                                  stream.get()) != cudaSuccess ||
                  cudaMemcpyAsync(writer.data(),
                                  pinned_device_source,
                                  writer.byte_size(),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()) != cudaSuccess) {
                return make_unexpected(Error{ErrorCode::kFailure, "pinned-host CUDA write failed"});
              }
            }
            return {};
          });
    } else {
      committed = loan->write_host_and_commit(
          [fill, &should_write_payload](const TensorOutputWriteGuard& writer) {
            if (should_write_payload(writer)) {
              std::memset(writer.data(), static_cast<int>(fill), writer.byte_size());
            }
          });
    }
    if (!committed) {
      std::cerr << "Tensor write-and-commit failed for " << state.options().tensor_bytes
                << " bytes in " << tensor_storage_name(state.options().tensor_storage)
                << " storage on CUDA device " << state.options().tensor_device << ": "
                << runtime_error_message(committed.error()) << '\n';
      state.fail(committed.error().code);
      return make_unexpected(std::move(committed).error());
    }
    if (initializes_descriptor_backing) {
      remember_initialized_backing(loan_backing_);
      state.record_descriptor_backing_initialization();
      auto canceled = std::move(*loan).cancel();
      if (!canceled) {
        state.fail(canceled.error().code);
        return make_unexpected(std::move(canceled).error());
      }
      return {};
    }
    if (descriptor_only && state.options().workload == Workload::kSaturated &&
        next_sequence_ > state.options().warmup &&
        initialized_backings_.size() < state.descriptor_warmup_slots()) {
      // Do not start the throughput clock until every steady-state backing is conditioned.
      auto canceled = std::move(*loan).cancel();
      if (!canceled) {
        state.fail(canceled.error().code);
        return make_unexpected(std::move(canceled).error());
      }
      return {};
    }

    // The synthetic frame is complete here, so capture time is also the common
    // publication-to-receive timing anchor without touching Tensor payload bytes.
    const std::int64_t send_time_ns = now_ns();
    const std::int64_t emit_start_ns = now_ns();
    auto emitted = output_.emit(std::move(*loan),
                                EmitOptions{
                                    .capture_time =
                                        TimePoint{
                                            .timestamp_ns = send_time_ns,
                                            .clock_id = context.activation_time().clock_id,
                                        },
                                    .frame_id = next_sequence_,
                                    .trace = TraceContext::root(next_sequence_),
                                });
    const std::int64_t emit_done_ns = now_ns();
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady) {
        state.record_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }

    state.record_publication(next_sequence_,
                             full_start_time_ns,
                             static_cast<std::uint64_t>(emit_done_ns - emit_start_ns),
                             static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
    ++next_sequence_;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      run_state().fail(sender.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    run_state().publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

 private:
  [[nodiscard]] bool backing_initialized(void* const backing) noexcept {
    loan_backing_ = backing;
    if (last_initialized_backing_ == backing) {
      return true;
    }
    const bool found =
        std::find(initialized_backings_.begin(), initialized_backings_.end(), backing) !=
        initialized_backings_.end();
    if (found) {
      last_initialized_backing_ = backing;
    }
    return found;
  }

  void remember_initialized_backing(void* const backing) {
    initialized_backings_.push_back(backing);
    last_initialized_backing_ = backing;
  }

  Output<Tensor> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  void* pinned_device_source_{};
  void* loan_backing_{};
  void* last_initialized_backing_{};
  std::vector<void*> initialized_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkImageProducer final : public Operator<> {
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
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(options.queue_depth)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &BenchmarkImageProducer::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override {
    RunState& state = run_state();
    initialized_backings_.reserve(state.descriptor_warmup_slots());
    state.record_start(Role::kProducer);
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RunState& state = run_state();
    state.record_source_callback();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }

    std::optional<test::ScopedAllocationCounter> allocation_probe;
    if (state.allocation_probe(next_sequence_)) {
      allocation_probe.emplace();
    }

    const std::int64_t full_start_time_ns = now_ns();
    const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
    auto loan =
        output_.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kTensorByteDtype});
    if (!loan) {
      if (loan.error().code == ErrorCode::kBackpressured ||
          loan.error().code == ErrorCode::kNotReady ||
          loan.error().code == ErrorCode::kResourceExhausted) {
        state.record_backpressure();
        return {};
      }
      state.fail(loan.error().code);
      return make_unexpected(std::move(loan).error());
    }

    const bool descriptor_only =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
    const std::uint8_t fill =
        descriptor_only ? kDescriptorTensorFillByte : static_cast<std::uint8_t>(next_sequence_);
    bool initializes_descriptor_backing{};
    const auto should_write_payload = [this, descriptor_only, &initializes_descriptor_backing](
                                          const TensorOutputWriteGuard& writer) noexcept {
      initializes_descriptor_backing = descriptor_only && !backing_initialized(writer.data());
      return !descriptor_only || initializes_descriptor_backing;
    };
    expected<void, Error> committed;
    // Descriptor-schema pinned-host services are host ordered and deliberately do not admit a
    // CUDA service stream. Use the host writer for all host-addressable Image storage; only device
    // storage requires CUDA readiness.
    if (state.options().tensor_storage == TensorStorage::kDevice) {
      const ::cuda::stream_ref stream = context.cuda_stream();
      if (stream.get() == nullptr) {
        state.fail(ErrorCode::kNotReady);
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      committed = loan->write_and_commit(
          stream,
          [fill, stream, &should_write_payload](
              const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (!should_write_payload(writer)) {
              return {};
            }
            if (cudaMemsetAsync(
                    writer.data(), static_cast<int>(fill), writer.byte_size(), stream.get()) !=
                cudaSuccess) {
              return make_unexpected(Error{ErrorCode::kFailure, "cudaMemsetAsync failed"});
            }
            return {};
          });
    } else {
      committed = loan->write_host_and_commit(
          [fill, &should_write_payload](const TensorOutputWriteGuard& writer) {
            if (should_write_payload(writer)) {
              std::memset(writer.data(), static_cast<int>(fill), writer.byte_size());
            }
          });
    }
    if (!committed) {
      std::cerr << "Image Tensor write-and-commit failed for " << state.options().tensor_bytes
                << " bytes in " << tensor_storage_name(state.options().tensor_storage)
                << " storage on CUDA device " << state.options().tensor_device << ": "
                << runtime_error_message(committed.error()) << '\n';
      state.fail(committed.error().code);
      return make_unexpected(std::move(committed).error());
    }
    if (initializes_descriptor_backing) {
      remember_initialized_backing(loan_backing_);
      state.record_descriptor_backing_initialization();
      auto canceled = std::move(*loan).cancel();
      if (!canceled) {
        state.fail(canceled.error().code);
        return make_unexpected(std::move(canceled).error());
      }
      if (allocation_probe) {
        state.record_source_allocation_probe(allocation_probe->count());
      }
      return {};
    }
    if (descriptor_only && state.options().workload == Workload::kSaturated &&
        next_sequence_ > state.options().warmup &&
        initialized_backings_.size() < state.descriptor_warmup_slots()) {
      // Do not start the throughput clock until every steady-state backing is conditioned.
      auto canceled = std::move(*loan).cancel();
      if (!canceled) {
        state.fail(canceled.error().code);
        return make_unexpected(std::move(canceled).error());
      }
      if (allocation_probe) {
        state.record_source_allocation_probe(allocation_probe->count());
      }
      return {};
    }

    schema::ImageT image;
    image.height = 1;
    image.width = static_cast<std::int32_t>(state.options().tensor_bytes);
    image.encoding = schema::ImageEncoding_CUSTOM;

    const std::int64_t send_time_ns = now_ns();
    const std::int64_t emit_start_ns = now_ns();
    auto emitted = output_.emit_tensor(std::move(*loan),
                                       image,
                                       EmitOptions{
                                           .capture_time =
                                               TimePoint{
                                                   .timestamp_ns = send_time_ns,
                                                   .clock_id = context.activation_time().clock_id,
                                               },
                                           .frame_id = next_sequence_,
                                           .trace = TraceContext::root(next_sequence_),
                                           .flags = EmitFlags::kDegraded,
                                       });
    const std::int64_t emit_done_ns = now_ns();
    if (!emitted) {
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady) {
        state.record_backpressure();
        return {};
      }
      std::cerr << "Image publication failed: " << runtime_error_message(emitted.error()) << '\n';
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }

    if (allocation_probe) {
      state.record_source_allocation_probe(allocation_probe->count());
    }

    state.record_publication(next_sequence_,
                             full_start_time_ns,
                             static_cast<std::uint64_t>(emit_done_ns - emit_start_ns),
                             static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
    ++next_sequence_;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      run_state().fail(sender.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    run_state().publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

 private:
  [[nodiscard]] bool backing_initialized(void* const backing) noexcept {
    loan_backing_ = backing;
    if (last_initialized_backing_ == backing) {
      return true;
    }
    const bool found =
        std::find(initialized_backings_.begin(), initialized_backings_.end(), backing) !=
        initialized_backings_.end();
    if (found) {
      last_initialized_backing_ = backing;
    }
    return found;
  }

  void remember_initialized_backing(void* const backing) {
    initialized_backings_.push_back(backing);
    last_initialized_backing_ = backing;
  }

  Output<schema::ImageT> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  void* loan_backing_{};
  void* last_initialized_backing_{};
  std::vector<void*> initialized_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkMultiTensorProducer final : public Operator<> {
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
    spec.notification_source(notification_, "benchmark-notification")
        .capacity(options.queue_depth)
        .sender_reference_capacity(1U);
    spec.lifecycle().stage(LifecycleStage::kArm, &BenchmarkMultiTensorProducer::arm);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override {
    RunState& state = run_state();
    for (auto& inventory : initialized_backings_) {
      inventory.reserve(state.descriptor_warmup_slots());
    }
    if (state.options().payload == Payload::kMultiTensorStaging) {
      if (!initialize_foreign_tensors()) {
        state.fail(ErrorCode::kResourceExhausted);
      } else if (state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly) {
        state.record_descriptor_backing_initialization();
        state.record_descriptor_backing_initialization();
      }
    }
    state.record_start(Role::kProducer);
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RunState& state = run_state();
    state.record_source_callback();
    if (next_sequence_ > state.final_sequence()) {
      return {};
    }
    const std::int64_t full_start_time_ns = now_ns();
    const auto publish = [&]() {
      return state.options().payload == Payload::kMultiTensorLoan ? publish_direct_bundle(context)
                                                                  : publish_foreign_value(context);
    };
    expected<void, Error> emitted;
    if (state.allocation_probe(next_sequence_)) {
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
        state.record_backpressure();
        return {};
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }
    if (!publication_completed_) {
      return {};
    }
    const std::int64_t emit_done_ns = now_ns();
    state.record_publication(next_sequence_,
                             full_start_time_ns,
                             static_cast<std::uint64_t>(emit_done_ns - emit_start_ns_),
                             static_cast<std::uint64_t>(emit_done_ns - full_start_time_ns));
    ++next_sequence_;
    publication_completed_ = false;
    return {};
  }

  [[nodiscard]] LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      run_state().fail(sender.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    run_state().publish_sender(std::move(*sender));
    return LifecycleStatus::kOk;
  }

 private:
  [[nodiscard]] bool initialize_foreign_tensors() {
    try {
      const std::size_t bytes = run_state().options().tensor_bytes;
      auto first = make_preinitialized_byte_tensor(bytes,
                                                   run_state().options().tensor_storage,
                                                   run_state().options().tensor_device,
                                                   kDescriptorTensorFillByte,
                                                   first_foreign_storage_);
      auto second = make_preinitialized_byte_tensor(bytes,
                                                    run_state().options().tensor_storage,
                                                    run_state().options().tensor_device,
                                                    kDescriptorTensorFillByte,
                                                    second_foreign_storage_);
      if (!first || !second) {
        return false;
      }
      first_foreign_ = std::move(*first);
      second_foreign_ = std::move(*second);
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }

  [[nodiscard]] bool backing_initialized(const std::size_t field, void* const backing) const {
    const auto& inventory = initialized_backings_[field];
    return std::find(inventory.begin(), inventory.end(), backing) != inventory.end();
  }

  [[nodiscard]] expected<void, Error> publish_direct_bundle(ExecutionContext& context) {
    RunState& state = run_state();
    const std::array shape{static_cast<std::int64_t>(state.options().tensor_bytes)};
    const std::array requests{
        tensor_shape(MultiTensorFields::first, shape),
        tensor_shape(MultiTensorFields::second, shape),
    };
    auto bundle = output_.allocate_tensors(requests);
    if (!bundle) {
      return make_unexpected(std::move(bundle).error());
    }
    const bool descriptor_only =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly;
    const std::uint8_t fill =
        descriptor_only ? kDescriptorTensorFillByte : static_cast<std::uint8_t>(next_sequence_);
    std::array<void*, 2U> new_backings{};
    const auto write_field = [&](const std::size_t index) {
      return [&, index](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
        const bool initialize = descriptor_only && !backing_initialized(index, writer.data());
        if (!descriptor_only || initialize) {
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
        if (initialize) {
          new_backings[index] = writer.data();
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
      return make_unexpected(std::move(committed).error());
    }
    for (std::size_t index = 0U; index < new_backings.size(); ++index) {
      if (new_backings[index] != nullptr) {
        initialized_backings_[index].push_back(new_backings[index]);
        state.record_descriptor_backing_initialization();
      }
    }
    if (descriptor_only && state.options().workload == Workload::kSaturated &&
        next_sequence_ > state.options().warmup &&
        std::any_of(
            initialized_backings_.begin(), initialized_backings_.end(), [&](const auto& field) {
              return field.size() < state.descriptor_warmup_slots();
            })) {
      // Abandon the unpublished committed bundle until both field inventories are conditioned.
      return {};
    }
    MultiTensorMessage message;
    message.sequence = next_sequence_;
    emit_start_ns_ = now_ns();
    auto emitted = output_.emit_tensors(
        std::move(*bundle), message, make_emit_options(context.activation_time().clock_id));
    publication_completed_ = emitted.has_value();
    return emitted;
  }

  [[nodiscard]] expected<void, Error> publish_foreign_value(ExecutionContext& context) {
    if (first_foreign_ == nullptr || second_foreign_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (run_state().options().bulk_work_profile == BulkWorkProfile::kFullWrite) {
      const auto fill = static_cast<std::uint8_t>(next_sequence_);
      if (run_state().options().tensor_storage == TensorStorage::kDevice) {
        if (cudaMemset(first_foreign_storage_->data(), fill, run_state().options().tensor_bytes) !=
                cudaSuccess ||
            cudaMemset(second_foreign_storage_->data(), fill, run_state().options().tensor_bytes) !=
                cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure, "device Tensor write failed"});
        }
      } else {
        std::memset(first_foreign_storage_->data(), fill, run_state().options().tensor_bytes);
        std::memset(second_foreign_storage_->data(), fill, run_state().options().tensor_bytes);
      }
    }
    MultiTensorMessage message;
    message.sequence = next_sequence_;
    message.first = first_foreign_;
    message.second = second_foreign_;
    emit_start_ns_ = now_ns();
    auto emitted =
        output_.emit(std::move(message), make_emit_options(context.activation_time().clock_id));
    publication_completed_ = emitted.has_value();
    return emitted;
  }

  [[nodiscard]] EmitOptions make_emit_options(const ClockId clock) const noexcept {
    return EmitOptions{
        .capture_time = TimePoint{.timestamp_ns = now_ns(), .clock_id = clock},
        .frame_id = next_sequence_,
        .trace = TraceContext::root(next_sequence_),
        .flags = EmitFlags::kDegraded,
    };
  }

  Output<MultiTensorMessage> output_;
  NotificationSource notification_;
  std::uint64_t next_sequence_{1U};
  std::int64_t emit_start_ns_{};
  bool publication_completed_{};
  std::array<std::vector<void*>, 2U> initialized_backings_;
  std::shared_ptr<PreinitializedTensorStorage> first_foreign_storage_;
  std::shared_ptr<PreinitializedTensorStorage> second_foreign_storage_;
  std::shared_ptr<Tensor> first_foreign_;
  std::shared_ptr<Tensor> second_foreign_;

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkMultiTensorConsumer final : public Operator<> {
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
    result.trigger(OnAnyInput{}).input(input_, DrainAll{});
    return result;
  }

  void start() override {
    RunState& state = run_state();
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    warmup_retention_capacity_ = std::min(warmup_slots == 0U ? 0U : warmup_slots - 1U,
                                          std::max<std::size_t>(2U, state.options().queue_depth));
    retained_warmup_tensors_.reserve(warmup_retention_capacity_);
    observed_first_backings_.reserve(warmup_slots);
    observed_second_backings_.reserve(warmup_slots);
    pool_warmup_complete_ = warmup_slots == 0U;
    state.record_start(Role::kConsumer);
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    if (!pool_warmup_complete_ && next_sequence_ > state.options().warmup) {
      // A first-touch backing may be initialized and canceled without ever reaching this process.
      // Do not carry warmup-only catalog borrows into the measured interval while waiting for it.
      retained_warmup_tensors_.clear();
      pool_warmup_complete_ = true;
    }
    const std::int64_t receive_start_ns = now_ns();
    expected<SelectedBatchView<MultiTensorMessage>, Error> selected = make_unexpected(Error{});
    if (state.allocation_probe(next_sequence_)) {
      const test::ScopedAllocationCounter allocations;
      selected = input_.selected_batch();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      selected = input_.selected_batch();
    }
    const std::int64_t receive_done_ns = now_ns();
    if (!selected) {
      state.fail(selected.error().code);
      return make_unexpected(std::move(selected).error());
    }
    if (selected->size() != 1U) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const Sample<MultiTensorMessage>& sample = selected->front();
    const std::uint64_t sequence = sample.metadata.frame_id;
    const bool metadata_valid =
        sequence == next_sequence_ && sample.data.sequence == sequence &&
        sample.metadata.sequence == sequence && sample.metadata.trace_id == sequence &&
        sample.metadata.capture_timestamp_ns > 0 &&
        sample.metadata.capture_timestamp_ns <= receive_done_ns &&
        sample.metadata.source_clock_id.valid() && sample.metadata.degraded() &&
        sample.data.first != nullptr && sample.data.second != nullptr;
    const bool tensors_valid = metadata_valid &&
                               tensor_descriptor_matches(*sample.data.first,
                                                         state.options().tensor_bytes,
                                                         state.options().tensor_storage,
                                                         state.options().tensor_device,
                                                         true) &&
                               tensor_descriptor_matches(*sample.data.second,
                                                         state.options().tensor_bytes,
                                                         state.options().tensor_storage,
                                                         state.options().tensor_device,
                                                         true);
    const bool validate_payload =
        state.options().payload_validation == PayloadValidation::kAll ||
        (state.options().payload_validation == PayloadValidation::kFinal &&
         sequence == state.final_sequence());
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kDescriptorTensorFillByte
            : static_cast<std::uint8_t>(sequence);
    const bool payload_valid =
        !validate_payload ||
        (tensor_edge_bytes_match(*sample.data.first, expected, state.options().tensor_storage) &&
         tensor_edge_bytes_match(*sample.data.second, expected, state.options().tensor_storage));
    if (!tensors_valid || !payload_valid) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (!pool_warmup_complete_ && sequence <= state.options().warmup && warmup_slots != 0U) {
      const auto remember = [](std::vector<const void*>& inventory, const void* backing) {
        if (std::find(inventory.begin(), inventory.end(), backing) == inventory.end()) {
          inventory.push_back(backing);
        }
      };
      remember(observed_first_backings_, sample.data.first->data());
      remember(observed_second_backings_, sample.data.second->data());
      const bool inventory_complete = observed_first_backings_.size() == warmup_slots &&
                                      observed_second_backings_.size() == warmup_slots;
      if (inventory_complete) {
        retained_warmup_tensors_.clear();
        pool_warmup_complete_ = true;
      } else if (warmup_retention_capacity_ != 0U) {
        // Keep backing ownership without retaining the finite cross-process Sample read claim.
        // The current activation temporarily adds one backing before the oldest retained copy is
        // released, forcing pool rotation. Trim before returning so post-compute ingress retains
        // one free catalog-borrow slot.
        if (retained_warmup_tensors_.size() == warmup_retention_capacity_) {
          std::this_thread::sleep_for(kMultiTensorDescriptorWarmupRetentionInterval);
          retained_warmup_tensors_.erase(retained_warmup_tensors_.begin());
        }
        retained_warmup_tensors_.push_back(
            std::array<Tensor, 2U>{*sample.data.first, *sample.data.second});
      }
    }
    state.record_metadata_delivery(
        sequence, sample.metadata.capture_timestamp_ns, receive_start_ns, receive_done_ns);
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
  std::size_t warmup_retention_capacity_{};
  std::vector<std::array<Tensor, 2U>> retained_warmup_tensors_;
  std::vector<const void*> observed_first_backings_;
  std::vector<const void*> observed_second_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
    spec.output(output_, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override { run_state().record_start(Role::kRelay); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    state.record_relay_callback();
    const std::int64_t callback_start_ns = now_ns();
    const std::int64_t receive_start_ns = now_ns();
    auto sample = input_.receive();
    const std::int64_t receive_done_ns = now_ns();
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    if (sample->data.sequence != next_sequence_ ||
        (payload_contents_required(state.options(), next_sequence_) &&
         !payload_valid(sample->data)) ||
        sample->metadata.frame_id != next_sequence_ ||
        sample->metadata.trace_id != next_sequence_) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    BenchmarkPOD64 payload = sample->data;
    payload.relay_receive_time_ns = receive_done_ns;
    state.record_relay_input(payload, receive_start_ns, receive_done_ns);

    std::int64_t emit_start_ns{};
    std::int64_t emit_done_ns{};
    while (true) {
      payload.relay_send_time_ns = now_ns();
      payload.checksum = payload_contents_required(state.options(), next_sequence_)
                             ? benchmark_pod_checksum(payload)
                             : 0U;
      emit_start_ns = now_ns();
      auto emitted = output_.emit(payload,
                                  EmitOptions{
                                      .frame_id = next_sequence_,
                                      .trace = TraceContext::root(next_sequence_),
                                  });
      emit_done_ns = now_ns();
      if (emitted) {
        break;
      }
      if (emitted.error().code == ErrorCode::kBackpressured ||
          emitted.error().code == ErrorCode::kNotReady) {
        state.record_backpressure();
        std::this_thread::yield();
        continue;
      }
      state.fail(emitted.error().code);
      return make_unexpected(std::move(emitted).error());
    }
    state.record_relay_publication(payload, callback_start_ns, emit_start_ns, emit_done_ns);
    ++next_sequence_;
    return {};
  }

 private:
  Input<BenchmarkPOD64> input_;
  Output<BenchmarkPOD64> output_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override { run_state().record_start(Role::kConsumer); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    const std::int64_t receive_start_ns = now_ns();
    auto sample = input_.receive();
    const std::int64_t receive_done_ns = now_ns();
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    const bool receive_timing_valid = receive_start_ns <= receive_done_ns;
    const bool relay_timing_valid =
        state.options().topology == Topology::kFanout
            ? sample->data.source_start_time_ns <= sample->data.send_time_ns &&
                  sample->data.send_time_ns <= receive_done_ns &&
                  sample->data.relay_receive_time_ns == 0 && sample->data.relay_send_time_ns == 0
            : sample->data.source_start_time_ns <= sample->data.send_time_ns &&
                  sample->data.send_time_ns <= sample->data.relay_receive_time_ns &&
                  sample->data.relay_receive_time_ns <= sample->data.relay_send_time_ns &&
                  sample->data.relay_send_time_ns <= receive_done_ns;
    const bool valid = sample->data.sequence == next_sequence_ &&
                       (!payload_contents_required(state.options(), next_sequence_) ||
                        payload_valid(sample->data)) &&
                       receive_timing_valid && relay_timing_valid &&
                       sample->metadata.frame_id == next_sequence_ &&
                       sample->metadata.trace_id == next_sequence_;
    if (!valid) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    state.record_delivery(sample->data, receive_start_ns, receive_done_ns);
    ++next_sequence_;
    return {};
  }

 private:
  Input<BenchmarkPOD64> input_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkSchemaConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input").queue_depth(run_state().options().queue_depth);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override { run_state().record_start(Role::kConsumer); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    const std::int64_t receive_start_ns = now_ns();
    expected<Sample<schema::TimeT>, Error> sample = make_unexpected(Error{});
    if (state.allocation_probe(next_sequence_)) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    const std::int64_t receive_done_ns = now_ns();
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }

    const bool payload_valid =
        sample->data.time_ns == static_cast<std::int64_t>(next_sequence_) &&
        (!payload_contents_required(state.options(), next_sequence_) ||
         sample->data.acqtime_ns == static_cast<std::int64_t>(next_sequence_ ^ 0x5A5A'5A5AU));
    const bool metadata_valid = sample->metadata.frame_id == next_sequence_ &&
                                sample->metadata.trace_id == next_sequence_ &&
                                sample->metadata.sequence == next_sequence_ &&
                                sample->metadata.capture_timestamp_ns > 0 &&
                                sample->metadata.capture_timestamp_ns <= receive_done_ns &&
                                sample->metadata.source_clock_id.valid();
    if (!payload_valid || !metadata_valid) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    state.record_metadata_delivery(
        next_sequence_, sample->metadata.capture_timestamp_ns, receive_start_ns, receive_done_ns);
    ++next_sequence_;
    return {};
  }

 private:
  Input<schema::TimeT> input_;
  std::uint64_t next_sequence_{1U};

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkTensorConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input")
        .queue_depth(options.queue_depth)
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
    RunState& state = run_state();
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    warmup_retention_capacity_ = std::min(warmup_slots == 0U ? 0U : warmup_slots - 1U,
                                          std::max<std::size_t>(2U, state.options().queue_depth));
    observed_backings_.reserve(warmup_slots);
    retained_warmup_payloads_.reserve(warmup_retention_capacity_);
    pool_warmup_complete_ = warmup_slots == 0U;
    state.record_start(Role::kConsumer);
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    if (!pool_warmup_complete_ && next_sequence_ > state.options().warmup) {
      // A first-touch backing may be initialized and canceled without ever reaching this process.
      // Do not carry warmup-only catalog borrows into the measured interval while waiting for it.
      retained_warmup_payloads_.clear();
      pool_warmup_complete_ = true;
    }
    const std::int64_t receive_start_ns = now_ns();
    auto sample = input_.receive();
    const std::int64_t receive_done_ns = now_ns();
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }

    const bool metadata_valid = sample->metadata.frame_id == next_sequence_ &&
                                sample->metadata.trace_id == next_sequence_ &&
                                sample->metadata.sequence == next_sequence_ &&
                                sample->metadata.capture_timestamp_ns > 0 &&
                                sample->metadata.capture_timestamp_ns <= receive_done_ns &&
                                sample->metadata.source_clock_id.valid();
    const bool descriptor_valid = tensor_descriptor_matches(sample->data,
                                                            state.options().tensor_bytes,
                                                            state.options().tensor_storage,
                                                            state.options().tensor_device,
                                                            true);
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kDescriptorTensorFillByte
            : static_cast<std::uint8_t>(next_sequence_);
    if (!metadata_valid || !descriptor_valid ||
        (payload_contents_required(state.options(), next_sequence_) &&
         !tensor_edge_bytes_match(sample->data, expected, state.options().tensor_storage))) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (!pool_warmup_complete_ && next_sequence_ <= state.options().warmup && warmup_slots != 0U) {
      const void* const backing = sample->data.data();
      if (std::find(observed_backings_.begin(), observed_backings_.end(), backing) ==
          observed_backings_.end()) {
        observed_backings_.push_back(backing);
      }
      if (observed_backings_.size() == warmup_slots) {
        retained_warmup_payloads_.clear();
        pool_warmup_complete_ = true;
      } else if (warmup_retention_capacity_ != 0U) {
        // Keep backing ownership without retaining the finite cross-process Sample read claim.
        // Trim the retained set before returning so the Runtime's post-compute ingress drain keeps
        // one free catalog-borrow slot.
        if (retained_warmup_payloads_.size() == warmup_retention_capacity_) {
          std::this_thread::sleep_for(kDescriptorWarmupRetentionInterval);
          retained_warmup_payloads_.erase(retained_warmup_payloads_.begin());
        }
        retained_warmup_payloads_.push_back(sample->data);
      }
    }

    state.record_metadata_delivery(
        next_sequence_, sample->metadata.capture_timestamp_ns, receive_start_ns, receive_done_ns);
    ++next_sequence_;
    return {};
  }

 private:
  Input<Tensor> input_;
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::size_t warmup_retention_capacity_{};
  std::vector<Tensor> retained_warmup_payloads_;
  std::vector<const void*> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

class BenchmarkImageConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const Options& options = run_state().options();
    spec.input(input_, "input")
        .queue_depth(options.queue_depth)
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
    RunState& state = run_state();
    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    warmup_retention_capacity_ = std::min(warmup_slots == 0U ? 0U : warmup_slots - 1U,
                                          std::max<std::size_t>(2U, state.options().queue_depth));
    observed_backings_.reserve(warmup_slots);
    retained_warmup_payloads_.reserve(warmup_retention_capacity_);
    pool_warmup_complete_ = warmup_slots == 0U;
    state.record_start(Role::kConsumer);
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RunState& state = run_state();
    if (!pool_warmup_complete_ && next_sequence_ > state.options().warmup) {
      // A first-touch backing may be initialized and canceled without ever reaching this process.
      // Do not carry warmup-only catalog borrows into the measured interval while waiting for it.
      retained_warmup_payloads_.clear();
      pool_warmup_complete_ = true;
    }
    const std::int64_t receive_start_ns = now_ns();
    expected<Sample<schema::ImageT>, Error> sample = make_unexpected(Error{});
    if (state.allocation_probe(next_sequence_)) {
      const test::ScopedAllocationCounter allocations;
      sample = input_.receive();
      state.record_sink_allocation_probe(allocations.count());
    } else {
      sample = input_.receive();
    }
    const std::int64_t receive_done_ns = now_ns();
    if (!sample) {
      state.fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }

    const Tensor* const tensor = sample->data.data.get();
    const bool metadata_valid =
        sample->metadata.frame_id == next_sequence_ &&
        sample->metadata.trace_id == next_sequence_ &&
        sample->metadata.sequence == next_sequence_ && sample->metadata.capture_timestamp_ns > 0 &&
        sample->metadata.capture_timestamp_ns <= receive_done_ns &&
        sample->metadata.source_clock_id.valid() && sample->metadata.degraded();
    const bool image_valid =
        sample->data.height == 1 &&
        sample->data.width == static_cast<std::int32_t>(state.options().tensor_bytes) &&
        sample->data.encoding == schema::ImageEncoding_CUSTOM && tensor != nullptr;
    bool descriptor_valid = image_valid;
    if (descriptor_valid) {
      descriptor_valid = tensor_descriptor_matches(*tensor,
                                                   state.options().tensor_bytes,
                                                   state.options().tensor_storage,
                                                   state.options().tensor_device,
                                                   true);
    }
    const std::uint8_t expected =
        state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly
            ? kDescriptorTensorFillByte
            : static_cast<std::uint8_t>(next_sequence_);
    if (!metadata_valid || !descriptor_valid ||
        (payload_contents_required(state.options(), next_sequence_) &&
         !tensor_edge_bytes_match(*tensor, expected, state.options().tensor_storage))) {
      state.fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    const std::size_t warmup_slots = state.descriptor_warmup_slots();
    if (!pool_warmup_complete_ && next_sequence_ <= state.options().warmup && warmup_slots != 0U) {
      const void* const backing = tensor->data();
      if (std::find(observed_backings_.begin(), observed_backings_.end(), backing) ==
          observed_backings_.end()) {
        observed_backings_.push_back(backing);
      }
      if (observed_backings_.size() == warmup_slots) {
        retained_warmup_payloads_.clear();
        pool_warmup_complete_ = true;
      } else if (warmup_retention_capacity_ != 0U) {
        // Keep backing ownership without retaining the finite cross-process Sample read claim.
        // Trim the retained set before returning so the Runtime's post-compute ingress drain keeps
        // one free catalog-borrow slot.
        if (retained_warmup_payloads_.size() == warmup_retention_capacity_) {
          std::this_thread::sleep_for(kDescriptorWarmupRetentionInterval);
          retained_warmup_payloads_.erase(retained_warmup_payloads_.begin());
        }
        retained_warmup_payloads_.push_back(sample->data);
      }
    }

    state.record_metadata_delivery(
        next_sequence_, sample->metadata.capture_timestamp_ns, receive_start_ns, receive_done_ns);
    ++next_sequence_;
    return {};
  }

 private:
  Input<schema::ImageT> input_;
  std::uint64_t next_sequence_{1U};
  bool pool_warmup_complete_{};
  std::size_t warmup_retention_capacity_{};
  std::vector<schema::ImageT> retained_warmup_payloads_;
  std::vector<const void*> observed_backings_;

  friend ExecutionPlan make_plan(const Options& options);
};

[[nodiscard]] ExecutionPlan make_plan(const Options& options) {
  Graph graph{"runtime-interprocess-" + options.run_id};
  const ClockKey clock = graph.add_clock<RealtimeClock>("benchmark-clock");
  graph.set_default_clock(clock);

  constexpr std::array producer_selector{detail::GraphObjectSelector{.graph_path = "producer"}};
  CompileOptions compile_options;
  detail::ProcessPlacementBinding process_placement;
  compile_options.deployment.bind_thread_placement(
      "runtime-interprocess-producer", make_placement(options.producer_cpu, options.busy_poll));
  process_placement.bind_process_role("producer",
                                      detail::ProcessPlacement{
                                          .members = producer_selector,
                                          .launch_group = "producer-process",
                                      });
  if (options.payload == Payload::kSchema) {
    auto producer = graph.op<BenchmarkSchemaProducer>("producer");
    auto consumer = graph.op<BenchmarkSchemaConsumer>("consumer");
    const std::string consumer_partition = consumer_partition_name(options.fanout, 0U);
    graph.partition("runtime-interprocess-producer").add(producer);
    graph.partition(consumer_partition).add(consumer);
    graph.add_flow(producer->output_,
                   consumer->input_,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kBlocking,
                       .queue_depth = options.queue_depth,
                   });
    compile_options.deployment.bind_thread_placement(
        consumer_partition, make_placement(options.consumer_cpus[0], options.busy_poll));
    constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
    process_placement.bind_process_role("consumer",
                                        detail::ProcessPlacement{
                                            .members = consumer_selector,
                                            .launch_group = "consumer-process",
                                        });
    return detail::compile_with_process_placement(
        graph, std::move(compile_options), std::move(process_placement));
  }
  if (multi_tensor_payload(options.payload)) {
    auto producer = graph.op<BenchmarkMultiTensorProducer>("producer");
    auto consumer = graph.op<BenchmarkMultiTensorConsumer>("consumer");
    const std::string consumer_partition = consumer_partition_name(options.fanout, 0U);
    graph.partition("runtime-interprocess-producer").add(producer);
    graph.partition(consumer_partition).add(consumer);
    graph.add_flow(producer->output_,
                   consumer->input_,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kBlocking,
                       .queue_depth = options.queue_depth,
                   });
    compile_options.deployment.bind_thread_placement(
        consumer_partition, make_placement(options.consumer_cpus[0], options.busy_poll));
    constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
    process_placement.bind_process_role("consumer",
                                        detail::ProcessPlacement{
                                            .members = consumer_selector,
                                            .launch_group = "consumer-process",
                                        });
    if (options.tensor_storage != TensorStorage::kPageableHost) {
      compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "producer",
          .output_port = "output",
          .device = DeviceId{options.tensor_device},
      });
    }
    return detail::compile_with_process_placement(
        graph, std::move(compile_options), std::move(process_placement));
  }
  if (options.payload == Payload::kTensor) {
    auto producer = graph.op<BenchmarkTensorProducer>("producer");
    auto consumer = graph.op<BenchmarkTensorConsumer>("consumer");
    const std::string consumer_partition = consumer_partition_name(options.fanout, 0U);
    graph.partition("runtime-interprocess-producer").add(producer);
    graph.partition(consumer_partition).add(consumer);
    graph.add_flow(producer->output_,
                   consumer->input_,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kBlocking,
                       .queue_depth = options.queue_depth,
                   });
    compile_options.deployment.bind_thread_placement(
        consumer_partition, make_placement(options.consumer_cpus[0], options.busy_poll));
    constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
    process_placement.bind_process_role("consumer",
                                        detail::ProcessPlacement{
                                            .members = consumer_selector,
                                            .launch_group = "consumer-process",
                                        });
    if (options.tensor_storage != TensorStorage::kPageableHost) {
      compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "producer",
          .output_port = "output",
          .device = DeviceId{options.tensor_device},
      });
    }
    return detail::compile_with_process_placement(
        graph, std::move(compile_options), std::move(process_placement));
  }
  if (options.payload == Payload::kImage) {
    auto producer = graph.op<BenchmarkImageProducer>("producer");
    auto consumer = graph.op<BenchmarkImageConsumer>("consumer");
    const std::string consumer_partition = consumer_partition_name(options.fanout, 0U);
    graph.partition("runtime-interprocess-producer").add(producer);
    graph.partition(consumer_partition).add(consumer);
    graph.add_flow(producer->output_,
                   consumer->input_,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kBlocking,
                       .queue_depth = options.queue_depth,
                   });
    compile_options.deployment.bind_thread_placement(
        consumer_partition, make_placement(options.consumer_cpus[0], options.busy_poll));
    constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
    process_placement.bind_process_role("consumer",
                                        detail::ProcessPlacement{
                                            .members = consumer_selector,
                                            .launch_group = "consumer-process",
                                        });
    if (options.tensor_storage != TensorStorage::kPageableHost) {
      compile_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "producer",
          .output_port = "output",
          .device = DeviceId{options.tensor_device},
      });
    }
    return detail::compile_with_process_placement(
        graph, std::move(compile_options), std::move(process_placement));
  }

  auto producer = graph.op<BenchmarkProducer>("producer");
  graph.partition("runtime-interprocess-producer").add(producer);
  if (options.topology == Topology::kChain) {
    auto relay = graph.op<BenchmarkRelay>("relay");
    auto consumer = graph.op<BenchmarkConsumer>("consumer");
    const std::string consumer_partition = consumer_partition_name(options.fanout, 0U);
    graph.partition("runtime-interprocess-relay").add(relay);
    graph.partition(consumer_partition).add(consumer);
    const ConnectionOptions connection{
        .queue_policy = QueuePolicy::kBlocking,
        .queue_depth = options.queue_depth,
    };
    graph.add_flow(producer->output_, relay->input_, connection);
    graph.add_flow(relay->output_, consumer->input_, connection);
    compile_options.deployment
        .bind_thread_placement("runtime-interprocess-relay",
                               make_placement(options.relay_cpu, options.busy_poll))
        .bind_thread_placement(consumer_partition,
                               make_placement(options.consumer_cpus[0], options.busy_poll));
    const std::array relay_selector{detail::GraphObjectSelector{.graph_path = "relay"}};
    const std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
    process_placement
        .bind_process_role("relay",
                           detail::ProcessPlacement{
                               .members = relay_selector,
                               .launch_group = "relay-process",
                           })
        .bind_process_role("consumer",
                           detail::ProcessPlacement{
                               .members = consumer_selector,
                               .launch_group = "consumer-process",
                           });
    return detail::compile_with_process_placement(
        graph, std::move(compile_options), std::move(process_placement));
  }
  for (std::uint32_t index = 0U; index < options.fanout; ++index) {
    const std::string role = consumer_role_name(options.fanout, index);
    const std::string partition_name = consumer_partition_name(options.fanout, index);
    auto consumer = graph.op<BenchmarkConsumer>(role);
    graph.partition(partition_name).add(consumer);
    graph.add_flow(producer->output_,
                   consumer->input_,
                   ConnectionOptions{
                       .queue_policy = QueuePolicy::kBlocking,
                       .queue_depth = options.queue_depth,
                   });
    compile_options.deployment.bind_thread_placement(
        partition_name, make_placement(options.consumer_cpus[index], options.busy_poll));
    const std::array consumer_selector{detail::GraphObjectSelector{.graph_path = role}};
    process_placement.bind_process_role(role,
                                        detail::ProcessPlacement{
                                            .members = consumer_selector,
                                            .launch_group = role + "-process",
                                        });
  }
  return detail::compile_with_process_placement(
      graph, std::move(compile_options), std::move(process_placement));
}

[[nodiscard]] std::optional<ProcessGroupId> process_group_for_role(const ExecutionPlan& plan,
                                                                   const std::string_view role) {
  const auto process_roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
  const auto role_record =
      std::find_if(process_roles.begin(), process_roles.end(), [&](const auto& record) {
        return record.role == role && record.disposition == PlanDisposition::kAccepted;
      });
  if (role_record == process_roles.end() || role_record->members.size() != 1U) {
    return std::nullopt;
  }
  const VertexId vertex = role_record->members.front();
  const auto vertex_process_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto placement = std::find_if(vertex_process_placements.begin(),
                                      vertex_process_placements.end(),
                                      [&](const auto& record) { return record.vertex == vertex; });
  if (placement == vertex_process_placements.end() || !placement->process_group) {
    return std::nullopt;
  }
  return placement->process_group;
}

[[nodiscard]] bool report_ready(const int fd) noexcept {
  if (fd < 0) {
    return true;
  }
#if defined(__linux__)
  constexpr std::byte kReady{1U};
  const std::byte* next = &kReady;
  std::size_t remaining = sizeof(kReady);
  while (remaining != 0U) {
    const ssize_t written = ::write(fd, next, remaining);
    if (written > 0) {
      next += written;
      remaining -= static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
#else
  static_cast<void>(fd);
  return false;
#endif
}

[[nodiscard]] bool wait_for_benchmark_start(const int fd, const std::uint32_t timeout_ms) noexcept {
  if (fd < 0) {
    return true;
  }
#if defined(__linux__)
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  const int bounded_timeout =
      timeout_ms > static_cast<std::uint32_t>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(timeout_ms);
  int ready{};
  do {
    ready = ::poll(&descriptor, 1U, bounded_timeout);
  } while (ready < 0 && errno == EINTR);
  if (ready != 1 || (descriptor.revents & POLLIN) == 0) {
    return false;
  }
  std::byte signal{};
  while (true) {
    const ssize_t received = ::read(fd, &signal, sizeof(signal));
    if (received == static_cast<ssize_t>(sizeof(signal))) {
      return signal == std::byte{1U};
    }
    if (received < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
#else
  static_cast<void>(fd);
  static_cast<void>(timeout_ms);
  return false;
#endif
}

[[nodiscard]] bool start_barrier_self_test() noexcept {
#if defined(__linux__)
  const auto exercise = [](const std::optional<std::byte> signal,
                           const std::uint32_t timeout_ms) noexcept {
    std::array<int, 2U> descriptors{-1, -1};
    if (::pipe(descriptors.data()) != 0) {
      return false;
    }
    bool write_succeeded = true;
    if (signal.has_value()) {
      write_succeeded = ::write(descriptors[1], &*signal, sizeof(*signal)) ==
                        static_cast<ssize_t>(sizeof(*signal));
    }
    const bool accepted = write_succeeded && wait_for_benchmark_start(descriptors[0], timeout_ms);
    static_cast<void>(::close(descriptors[0]));
    static_cast<void>(::close(descriptors[1]));
    return accepted;
  };
  return exercise(std::byte{1U}, 50U) && !exercise(std::byte{2U}, 50U) &&
         !exercise(std::nullopt, 1U);
#else
  return true;
#endif
}

void emit_callback_thread_evidence(std::ostream& output, const RunState& state,
                                   const BenchmarkSupportThreadControl& thread_control) {
  const bool enabled = state.options().runtime_phase_trace;
  output << "  \"callback_thread_evidence\":{";
  if (!enabled) {
    output << "\"enabled\":false},\n";
    return;
  }
  std::optional<benchmarking::SupportThreadEvidence> controller;
  for (const benchmarking::SupportThreadEvidence& evidence : thread_control.evidence()) {
    if (evidence.role == SupportThreadRole::kController) {
      controller = evidence;
      break;
    }
  }
  const std::uint64_t callback_tid = state.callback_thread_id();
  const bool identity_kind_matches =
      controller.has_value() && controller->native_tid.has_value() &&
      controller->thread_identity_is_native == state.callback_identity_is_native() &&
      controller->native_tid_syscall_unavailable == state.callback_native_tid_syscall_unavailable();
  const bool callbacks_on_partition_owner = identity_kind_matches &&
                                            !state.callback_multiple_threads() &&
                                            *controller->native_tid == callback_tid;
  const bool valid = controller.has_value() && controller->entered && callback_tid != 0U &&
                     state.callback_count() != 0U && !state.callback_multiple_threads() &&
                     identity_kind_matches && callbacks_on_partition_owner;
  const std::string_view identity_kind =
      identity_kind_matches && state.callback_identity_is_native() ? "native_tid"
      : identity_kind_matches                                      ? "std_thread_id_hash"
                                                                   : "mixed";
  output << "\"enabled\":true,\"valid\":" << (valid ? "true" : "false")
         << ",\"thread_identity_kind\":\"" << identity_kind
         << "\",\"native_tid_syscall_unavailable\":"
         << (state.callback_native_tid_syscall_unavailable() ? "true" : "false")
         << ",\"controller_native_tid\":";
  if (controller.has_value() && controller->native_tid.has_value()) {
    output << *controller->native_tid;
  } else {
    output << "null";
  }
  output << ",\"callback_role\":\"" << role_name(state.options())
         << "\",\"callback_count\":" << state.callback_count()
         << ",\"callback_native_tid\":" << callback_tid
         << ",\"multiple_threads\":" << (state.callback_multiple_threads() ? "true" : "false")
         << ",\"callbacks_on_partition_owner\":"
         << (callbacks_on_partition_owner ? "true" : "false") << "},\n";
}

void emit_runtime_phases(std::ostream& output,
                         const detail::RuntimePhaseCounters* const phase_counters) {
  output << "  \"runtime_phases\":{";
  if (phase_counters == nullptr) {
    output << "\"enabled\":false},\n";
    return;
  }
  const auto snapshot = phase_counters->snapshot();
  const auto value = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  output << "\"enabled\":true,\"readiness_scans\":"
         << value(detail::RuntimePhaseCounter::kReadinessScans)
         << ",\"no_progress_scans\":" << value(detail::RuntimePhaseCounter::kNoProgressScans)
         << ",\"wait_calls\":" << value(detail::RuntimePhaseCounter::kWaitCalls)
         << ",\"wake_signals\":" << value(detail::RuntimePhaseCounter::kWakeSignals)
         << ",\"wake_returns\":" << value(detail::RuntimePhaseCounter::kWakeReturns)
         << ",\"timeout_returns\":" << value(detail::RuntimePhaseCounter::kTimeoutReturns)
         << ",\"channel_samples_received\":"
         << value(detail::RuntimePhaseCounter::kChannelSamplesReceived)
         << ",\"channel_readiness_published\":"
         << value(detail::RuntimePhaseCounter::kChannelReadinessPublished)
         << ",\"readiness_poll_dispatches\":"
         << value(detail::RuntimePhaseCounter::kReadinessPollDispatches)
         << ",\"compute_dispatches\":" << value(detail::RuntimePhaseCounter::kComputeDispatches)
         << "},\n";
}

// Keep the benchmark's stable JSON schema visible in one emitter.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_result(std::ostream& output, const RunState& state, const ExecutionPlan& plan,
                 const RunSession& session, const ProcessGroupId process_group,
                 const detail::RuntimePhaseCounters* const phase_counters,
                 const BenchmarkSupportThreadControl& thread_control,
                 const IngressWakeTraceSummary& ingress_wake_trace) {
  const Options& options = state.options();
  const auto* internal = detail::ExecutionPlanAccess::state(plan);
  const std::string service_id =
      internal != nullptr && internal->cross_process_channel_services.size() == 1U
          ? internal->cross_process_channel_services.front().service_id
          : std::string{};
  const std::string benchmark_name =
      options.payload == Payload::kTensor
          ? "runtime_graph_interprocess_tensor_1to1_" +
                std::string(tensor_storage_name(options.tensor_storage)) + "_" +
                bulk_work_profile_name(options.bulk_work_profile)
          : (options.payload == Payload::kImage
                 ? "runtime_graph_interprocess_image_1to1_" +
                       std::string(tensor_storage_name(options.tensor_storage)) + "_" +
                       bulk_work_profile_name(options.bulk_work_profile)
                 : (multi_tensor_payload(options.payload)
                        ? "runtime_graph_interprocess_" +
                              std::string(payload_name(options.payload)) + "_1to1_" +
                              std::string(tensor_storage_name(options.tensor_storage)) + "_" +
                              bulk_work_profile_name(options.bulk_work_profile)
                        : (options.payload == Payload::kSchema
                               ? "runtime_graph_interprocess_schema_time_1to1"
                               : (options.topology == Topology::kChain
                                      ? "runtime_graph_interprocess_pod_source_relay_sink"
                                      : "runtime_graph_interprocess_pod_1to" +
                                            std::to_string(options.fanout)))));
  const std::int64_t start_ns = state.measured_start_ns();
  const std::int64_t end_ns = state.measured_end_ns();
  const std::uint64_t duration_ns =
      start_ns > 0 && end_ns > start_ns ? static_cast<std::uint64_t>(end_ns - start_ns) : 0U;
  const double messages_per_second =
      duration_ns == 0U
          ? 0.0
          : static_cast<double>(options.samples) * 1.0e9 / static_cast<double>(duration_ns);
  const TransportPercentiles timestamp_read_pair_overhead = calibrate_timestamp_read_pairs();
  const TensorAllocationPlanRecord* tensor_allocation = nullptr;
  const TensorFieldAllocationPlanRecord* tensor_field_allocation = nullptr;
  if (internal != nullptr && internal->cross_process_channel_services.size() == 1U) {
    const auto& service = internal->cross_process_channel_services.front();
    const std::string_view pool_id = service.tensor_pool_id;
    const bool raw_tensor = service.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema = detail::descriptor_tensor_schema_service(service);
    const auto allocations = plan.tensor_allocations();
    const auto allocation = std::find_if(
        allocations.begin(), allocations.end(), [&](const TensorAllocationPlanRecord& record) {
          return service.tensor_payload && (raw_tensor || descriptor_tensor_schema) &&
                 record.output == service.source &&
                 (!raw_tensor || (!pool_id.empty() && record.pool_id == pool_id)) &&
                 service.tensor_max_byte_span == record.bounds.max_byte_span &&
                 service.tensor_memory_kind == record.memory_kind &&
                 service.tensor_device == record.device;
        });
    if (allocation != allocations.end()) {
      tensor_allocation = &*allocation;
    }
    if (multi_tensor_payload(options.payload)) {
      const auto field_allocations = plan.tensor_field_allocations();
      const auto field_allocation =
          std::find_if(field_allocations.begin(), field_allocations.end(), [&](const auto& record) {
            return record.output == service.source && record.tensor_index == 0U;
          });
      if (field_allocation != field_allocations.end()) {
        tensor_field_allocation = &*field_allocation;
      }
    }
  }

  output
      << "{\n"
      << "  \"schema\":\"holoscan.test.runtime-interprocess-benchmark/v1\",\n"
      << "  \"benchmark\":\"" << benchmark_name << "\",\n"
      << "  \"answers\":\""
      << (options.payload == Payload::kTensor
              ? "Operator Graph Tensor cost across one same-host process boundary"
              : (options.payload == Payload::kImage
                     ? "Operator Graph Image metadata and embedded Tensor cost across one "
                       "same-host process boundary"
                     : (multi_tensor_payload(options.payload)
                            ? "Operator Graph multi-Tensor schema cost across one same-host "
                              "process boundary"
                            : (options.payload == Payload::kSchema
                                   ? "generated-schema encode, verify, materialize, and activation "
                                     "cost across one same-host process boundary"
                                   : (options.topology == Topology::kChain
                                          ? "Operator Graph cost across two sequential same-host "
                                            "process boundaries"
                                          : "Operator Graph cost across same-host process "
                                            "fanout")))))
      << "\",\n"
      << "  \"payload\":\"" << payload_name(options.payload) << "\",\n"
      << "  \"role\":\"" << role_name(options) << "\",\n"
      << "  \"consumer_index\":";
  if (options.role == Role::kConsumer) {
    output << options.consumer_index + 1U;
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"run_id\":\"" << json_escape(options.run_id) << "\",\n"
         << "  \"workload\":\"" << workload_name(options.workload) << "\",\n"
         << "  \"process_id\":"
         << static_cast<std::uint64_t>(
#if defined(__linux__)
                ::getpid()
#else
                0
#endif
                    )
         << ",\n"
         << "  \"process_group\":" << process_group.value << ",\n"
         << "  \"graph_hash\":\"" << hex_bytes(plan.graph_hash().value) << "\",\n"
         << "  \"plan_digest\":\"" << hex_bytes(plan.digest().value) << "\",\n"
         << "  \"service_id\":\"" << json_escape(service_id) << "\",\n"
         << "  \"service_ids\":[";
  if (internal != nullptr) {
    for (std::size_t index = 0U; index < internal->cross_process_channel_services.size(); ++index) {
      if (index != 0U) {
        output << ',';
      }
      output << '"' << json_escape(internal->cross_process_channel_services[index].service_id)
             << '"';
    }
  }
  output << "],\n"
         << "  \"clock\":{\"source\":\"" << BenchmarkClock::source()
         << "\",\"steady\":true,\"scope\":\"" << BenchmarkClock::scope() << "\"},\n"
         << "  \"timestamp_instrumentation\":{\"calibration\":"
            "\"post_measurement_back_to_back_pair\""
         << ",\"clock_reads_per_sample\":2,\"pair_overhead_ns\":";
  emit_percentiles(output, timestamp_read_pair_overhead);
  output << "},\n"
         << "  \"payload_validation\":\"" << payload_validation_name(options.payload_validation)
         << "\",\n"
         << "  \"validation_timing\":\"" << validation_timing_name(options.payload_validation)
         << "\",\n"
         << "  \"geometry\":{\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
         << ",\"queue_depth\":" << options.queue_depth
         << ",\"notification_capacity\":" << options.queue_depth
         << ",\"publication_backpressure_capacity\":" << options.queue_depth
         << ",\"payload_bytes\":" << application_payload_bytes(options)
         << ",\"application_payload_bytes\":" << application_payload_bytes(options)
         << ",\"inline_transport_bytes\":";
  const std::optional<std::size_t> transport_bytes = inline_transport_bytes(options);
  if (transport_bytes) {
    output << *transport_bytes;
  } else {
    output << "null";
  }
  output << ",\"external_buffer_bytes\":" << external_buffer_bytes(options) << ",\"pool_slots\":";
  if (tensor_allocation != nullptr) {
    output << tensor_allocation->slot_count;
  } else if (tensor_field_allocation != nullptr) {
    const std::uint64_t slots = std::accumulate(
        plan.tensor_field_allocations().begin(),
        plan.tensor_field_allocations().end(),
        std::uint64_t{},
        [](const std::uint64_t total, const auto& field) { return total + field.slot_count; });
    output << slots;
  } else {
    output << "null";
  }
  output << ",\"release_capacity\":";
  if (tensor_allocation != nullptr) {
    output << tensor_allocation->release_capacity;
  } else if (tensor_field_allocation != nullptr) {
    const std::uint64_t capacity =
        std::accumulate(plan.tensor_field_allocations().begin(),
                        plan.tensor_field_allocations().end(),
                        std::uint64_t{},
                        [](const std::uint64_t total, const auto& field) {
                          return total + field.release_capacity;
                        });
    output << capacity;
  } else {
    output << "null";
  }
  output << ",\"payload_write_profile\":\"" << payload_write_profile(options)
         << "\",\"bulk_work_profile\":\"" << result_bulk_work_profile(options)
         << "\",\"topology\":\"" << topology_name(options.topology)
         << "\",\"fanout\":" << options.fanout
         << ",\"processes\":" << (options.topology == Topology::kChain ? 3U : options.fanout + 1U)
         << ",\"transport_hops\":" << (options.topology == Topology::kChain ? 2U : 1U)
         << ",\"publisher_period_us\":";
  if (options.workload == Workload::kLatency) {
    output << options.publisher_period_us;
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"generated_schema\":{";
  if (options.payload == Payload::kSchema) {
    output << "\"native_type\":\"schema::TimeT\",\"wire_capacity\":"
           << schema_package<schema::TimeT>().max_serialized_size();
  } else if (options.payload == Payload::kImage) {
    output << "\"native_type\":\"schema::ImageT\",\"wire_capacity\":"
           << kDescriptorBackedImagePayloadBytes;
  } else if (multi_tensor_payload(options.payload)) {
    output << "\"native_type\":\"runtime_benchmark::MultiTensorMessageT\",\"wire_capacity\":"
           << detail::runtime_port_type_ops<MultiTensorMessage>()->wire_capacity;
  }
  output << "},\n"
         << "  \"tensor\":{";
  if (tensor_backed_payload(options.payload)) {
    const auto service_fields_match = [&](const auto& service) {
      return service.tensor_fields.size() == 2U &&
             std::ranges::all_of(service.tensor_fields, [&](const auto& field) {
               return field.memory_kind == tensor_memory_kind(options.tensor_storage);
             });
    };
    const bool plan_storage_matches =
        internal != nullptr && internal->cross_process_channel_services.size() == 1U &&
        (multi_tensor_payload(options.payload)
             ? service_fields_match(internal->cross_process_channel_services.front())
             : internal->cross_process_channel_services.front().tensor_memory_kind ==
                   tensor_memory_kind(options.tensor_storage));
    const bool service_pool_bound =
        internal != nullptr && internal->cross_process_channel_services.size() == 1U &&
        (multi_tensor_payload(options.payload)
             ? internal->cross_process_channel_services.front().tensor_fields.size() == 2U &&
                   internal->cross_process_channel_services.front().tensor_fields[0].pool_id !=
                       internal->cross_process_channel_services.front().tensor_fields[1].pool_id &&
                   std::ranges::all_of(
                       internal->cross_process_channel_services.front().tensor_fields,
                       [](const auto& field) { return !field.pool_id.empty(); })
             : !internal->cross_process_channel_services.front().tensor_pool_id.empty());
    output << "\"storage\":\"" << tensor_storage_name(options.tensor_storage)
           << "\",\"device\":" << options.tensor_device << ",\"descriptor_schema\":"
           << (descriptor_schema_payload(options.payload) ? "true" : "false")
           << ",\"read_only_import_checked\":true"
           << ",\"capture_timestamp_in_sample_metadata\":true" << ",\"direct_allocation_records\":"
           << (multi_tensor_payload(options.payload) ? plan.tensor_field_allocations().size()
                                                     : plan.tensor_allocations().size())
           << ",\"memory_kind_plan_match\":" << (plan_storage_matches ? "true" : "false")
           << ",\"service_pool_bound\":" << (service_pool_bound ? "true" : "false");
    if (multi_tensor_payload(options.payload)) {
      const auto fields = plan.tensor_field_allocations();
      output << ",\"field_pools_distinct\":"
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
  }
  output << "},\n"
         << "  \"allocation_probe\":{\"sequence\":";
  const bool allocation_probe_enabled =
      allocation_probed_payload(options.payload) && options.warmup != 0U;
  if (allocation_probe_enabled) {
    output << options.warmup;
  } else {
    output << "null";
  }
  output << ",\"excluded_from_measurement\":" << (allocation_probe_enabled ? "true" : "false")
         << ",\"sink_access\":\"" << allocation_probe_sink_access(options.payload) << "\""
         << ",\"source_compute_allocation_limit\":";
  if (allocation_probe_enabled) {
    output << allocation_probe_source_limit(options.payload);
  } else {
    output << "null";
  }
  output << ",\"sink_compute_allocation_limit\":";
  if (allocation_probe_enabled) {
    output << allocation_probe_sink_limit(options.payload);
  } else {
    output << "null";
  }
  output << ",\"source_compute_allocations\":";
  if (state.source_probe_allocations()) {
    output << *state.source_probe_allocations();
  } else {
    output << "null";
  }
  output << ",\"sink_compute_allocations\":";
  if (state.sink_probe_allocations()) {
    output << *state.sink_probe_allocations();
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"wake\":{\"event_wake\":" << (options.role == Role::kProducer ? "false" : "true")
         << ",\"busy_poll\":" << (options.busy_poll ? "true" : "false")
         << ",\"receive_polling\":false},\n"
         << "  \"placement\":{\"producer_cpu\":";
  if (options.producer_cpu >= 0) {
    output << options.producer_cpu;
  } else {
    output << "null";
  }
  output << ",\"relay_cpu\":";
  emit_optional_cpu(output, options.relay_cpu);
  output << ",\"consumer_cpus\":";
  emit_cpu_list(output, options.consumer_cpus);
  output << ",\"producer_controller_cpu\":";
  emit_optional_cpu(output, options.producer_controller_cpu);
  output << ",\"relay_controller_cpu\":";
  emit_optional_cpu(output, options.relay_controller_cpu);
  output << ",\"relay_pump_cpu\":";
  emit_optional_cpu(output, options.relay_pump_cpu);
  output << ",\"consumer_controller_cpus\":";
  emit_cpu_list(output, options.consumer_controller_cpus);
  output << ",\"consumer_pump_cpus\":";
  emit_cpu_list(output, options.consumer_pump_cpus);
  output << ",\"lane_wake_mode\":\"" << (options.busy_poll ? "busy_poll" : "sleeping_waitset")
         << "\"},\n"
         << "  \"support_threads\":";
  emit_support_thread_evidence(output, thread_control);
  output << ",\n";
  emit_runtime_thread_usage(output, phase_counters);
  emit_callback_thread_evidence(output, state, thread_control);
  output << "  \"timing\":{\"measured_start_ns\":" << start_ns << ",\"measured_end_ns\":" << end_ns
         << ",\"completion_window_ns\":" << duration_ns
         << ",\"messages_per_second\":" << messages_per_second << ",\"emit_ns\":";
  emit_percentiles(output, state.emit_summary());
  output << ",\"producer_full_ns\":";
  emit_percentiles(output, state.producer_full_summary());
  output << ",\"receive_ns\":";
  emit_percentiles(output, state.receive_summary());
  output << ",\"end_to_end_ns\":";
  emit_percentiles(output, state.end_to_end_summary());
  output << ",\"relay_receive_ns\":";
  emit_percentiles(output, state.relay_receive_summary());
  output << ",\"relay_emit_ns\":";
  emit_percentiles(output, state.relay_emit_summary());
  output << ",\"relay_callback_ns\":";
  emit_percentiles(output, state.relay_callback_summary());
  output << ",\"first_hop_ns\":";
  emit_percentiles(output, state.first_hop_summary());
  output << ",\"relay_residence_ns\":";
  emit_percentiles(output, state.relay_residence_summary());
  output << ",\"second_hop_ns\":";
  emit_percentiles(output, state.second_hop_summary());
  output << "},\n"
         << "  \"counts\":{\"source_starts\":" << state.producer_starts()
         << ",\"relay_starts\":" << state.relay_starts()
         << ",\"consumer_starts\":" << state.consumer_starts()
         << ",\"source_callbacks\":" << state.source_callbacks()
         << ",\"relay_callbacks\":" << state.relay_callbacks()
         << ",\"publications\":" << state.publications()
         << ",\"relay_inputs\":" << state.relay_inputs()
         << ",\"relay_publications\":" << state.relay_publications()
         << ",\"deliveries\":" << state.deliveries()
         << ",\"final_sequence\":" << state.final_sequence_observed()
         << ",\"notification_attempts\":" << state.notification_attempts()
         << ",\"notifications_accepted\":" << state.notifications_accepted()
         << ",\"backpressured_emits\":" << state.backpressured_emits()
         << ",\"descriptor_backing_initializations\":" << state.descriptor_backing_initializations()
         << ",\"descriptor_backing_initializations_during_measurement\":"
         << state.descriptor_backing_initializations_during_measurement() << "},\n";
  emit_runtime_phases(output, phase_counters);
  output << "  \"ingress_wake_trace\":";
  emit_ingress_wake_trace(output, ingress_wake_trace);
  output << ",\n"
         << "  \"termination\":\""
         << (session.termination() == RunTermination::kStopped ? "stopped" : "other") << "\",\n"
         << "  \"diagnostic_count\":" << session.diagnostics().size() << ",\n"
         << "  \"diagnostics\":";
  emit_runtime_diagnostics(output, session.diagnostics());
  output << "\n}\n";
}

[[nodiscard]] bool parse_unsigned(const std::string_view text, std::uint64_t& result) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_signed(const std::string_view text, int& result) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_cpu_list(std::string_view text, std::vector<int>& result) {
  std::vector<int> parsed;
  while (!text.empty()) {
    const std::size_t separator = text.find(',');
    const std::string_view item = text.substr(0U, separator);
    int cpu{};
    if (item.empty() || !parse_signed(item, cpu)) {
      return false;
    }
    parsed.push_back(cpu);
    if (separator == std::string_view::npos) {
      text = {};
    } else {
      text.remove_prefix(separator + 1U);
    }
  }
  if (parsed.empty()) {
    return false;
  }
  result = std::move(parsed);
  return true;
}

void print_help(std::ostream& output) {
  output << "runtime_graph_interprocess_benchmark\n\n"
         << "Usage:\n"
         << "  runtime_graph_interprocess_benchmark --role <producer|relay|consumer> "
            "--run-id <id> [options]\n\n"
         << "Options:\n"
         << "  --topology <fanout|chain>\n"
         << "  --workload <latency|saturated>\n"
         << "  --payload <pod|schema|tensor|image|multi_tensor_loan|multi_tensor_staging>\n"
         << "  --tensor-storage <pageable_host|pinned_host|device>\n"
         << "  --tensor-bytes <count>\n"
         << "  --tensor-device <ordinal>\n"
         << "  --bulk-work-profile <full_write|descriptor_only>\n"
         << "  --samples <count>\n"
         << "  --warmup <count>\n"
         << "  --fanout <1..3>\n"
         << "  --consumer-index <one-based index>\n"
         << "  --queue-depth <count>\n"
         << "  --publisher-period-us <microseconds>\n"
         << "  --payload-validation <none|final|all>\n"
         << "  --timeout-ms <milliseconds>\n"
         << "  --start-timeout-ms <milliseconds>\n"
         << "  --producer-cpu <ordinal|-1>\n"
         << "  --relay-cpu <ordinal|-1>\n"
         << "  --consumer-cpus <ordinal|-1,...>\n"
         << "  --producer-controller-cpu <ordinal|-1>\n"
         << "  --relay-controller-cpu <ordinal|-1>\n"
         << "  --relay-pump-cpu <ordinal|-1>\n"
         << "  --consumer-controller-cpus <ordinal|-1,...>\n"
         << "  --consumer-pump-cpus <ordinal|-1,...>\n"
         << "  --busy-poll <0|1>\n"
         << "  --runtime-phase-trace <0|1>\n"
         << "  --trace-correlation-self-test <0|1>\n"
         << "  --start-barrier-self-test <0|1>\n"
         << "  --ready-fd <descriptor>\n"
         << "  --start-fd <descriptor>\n"
         << "  --output <path>\n";
}

// Keep command-line validation adjacent to each option's parsing branch.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<Options> parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help") {
      print_help(std::cout);
      return std::nullopt;
    }
    if (index + 1 >= argc) {
      std::cerr << "missing value for " << argument << '\n';
      return std::nullopt;
    }
    const std::string_view value{argv[++index]};
    if (argument == "--topology") {
      if (value == "fanout") {
        options.topology = Topology::kFanout;
      } else if (value == "chain") {
        options.topology = Topology::kChain;
      } else {
        std::cerr << "--topology must be fanout or chain\n";
        return std::nullopt;
      }
    } else if (argument == "--role") {
      if (value == "producer") {
        options.role = Role::kProducer;
      } else if (value == "relay") {
        options.role = Role::kRelay;
      } else if (value == "consumer") {
        options.role = Role::kConsumer;
      } else {
        std::cerr << "--role must be producer, relay, or consumer\n";
        return std::nullopt;
      }
    } else if (argument == "--workload") {
      if (value == "latency") {
        options.workload = Workload::kLatency;
      } else if (value == "saturated") {
        options.workload = Workload::kSaturated;
      } else {
        std::cerr << "--workload must be latency or saturated\n";
        return std::nullopt;
      }
    } else if (argument == "--payload") {
      if (value == "pod") {
        options.payload = Payload::kPOD;
      } else if (value == "schema") {
        options.payload = Payload::kSchema;
      } else if (value == "tensor") {
        options.payload = Payload::kTensor;
      } else if (value == "image") {
        options.payload = Payload::kImage;
      } else if (value == "multi_tensor_loan") {
        options.payload = Payload::kMultiTensorLoan;
      } else if (value == "multi_tensor_staging") {
        options.payload = Payload::kMultiTensorStaging;
      } else {
        std::cerr << "--payload must be pod, schema, tensor, image, multi_tensor_loan, or "
                     "multi_tensor_staging\n";
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
    } else if (argument == "--tensor-bytes") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "invalid --tensor-bytes\n";
        return std::nullopt;
      }
      options.tensor_bytes = static_cast<std::size_t>(parsed);
    } else if (argument == "--tensor-device") {
      if (!parse_signed(value, options.tensor_device)) {
        std::cerr << "invalid --tensor-device\n";
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
    } else if (argument == "--run-id") {
      options.run_id = value;
    } else if (argument == "--fanout") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --fanout\n";
        return std::nullopt;
      }
      options.fanout = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--consumer-index") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed == 0U ||
          parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --consumer-index\n";
        return std::nullopt;
      }
      options.consumer_index = static_cast<std::uint32_t>(parsed - 1U);
    } else if (argument == "--samples") {
      if (!parse_unsigned(value, options.samples)) {
        std::cerr << "invalid --samples\n";
        return std::nullopt;
      }
    } else if (argument == "--warmup") {
      if (!parse_unsigned(value, options.warmup)) {
        std::cerr << "invalid --warmup\n";
        return std::nullopt;
      }
    } else if (argument == "--queue-depth") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --queue-depth\n";
        return std::nullopt;
      }
      options.queue_depth = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--publisher-period-us") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --publisher-period-us\n";
        return std::nullopt;
      }
      options.publisher_period_us = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--payload-validation") {
      if (value == "none") {
        options.payload_validation = PayloadValidation::kNone;
      } else if (value == "final") {
        options.payload_validation = PayloadValidation::kFinal;
      } else if (value == "all") {
        options.payload_validation = PayloadValidation::kAll;
      } else {
        std::cerr << "--payload-validation must be none, final, or all\n";
        return std::nullopt;
      }
    } else if (argument == "--timeout-ms") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --timeout-ms\n";
        return std::nullopt;
      }
      options.timeout_ms = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--start-timeout-ms") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "invalid --start-timeout-ms\n";
        return std::nullopt;
      }
      options.start_timeout_ms = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--producer-cpu") {
      if (!parse_signed(value, options.producer_cpu)) {
        std::cerr << "invalid --producer-cpu\n";
        return std::nullopt;
      }
    } else if (argument == "--relay-cpu") {
      if (!parse_signed(value, options.relay_cpu)) {
        std::cerr << "invalid --relay-cpu\n";
        return std::nullopt;
      }
    } else if (argument == "--consumer-cpus") {
      if (!parse_cpu_list(value, options.consumer_cpus)) {
        std::cerr << "invalid --consumer-cpus\n";
        return std::nullopt;
      }
    } else if (argument == "--producer-controller-cpu") {
      if (!parse_signed(value, options.producer_controller_cpu)) {
        std::cerr << "invalid --producer-controller-cpu\n";
        return std::nullopt;
      }
    } else if (argument == "--relay-controller-cpu") {
      if (!parse_signed(value, options.relay_controller_cpu)) {
        std::cerr << "invalid --relay-controller-cpu\n";
        return std::nullopt;
      }
    } else if (argument == "--relay-pump-cpu") {
      if (!parse_signed(value, options.relay_pump_cpu)) {
        std::cerr << "invalid --relay-pump-cpu\n";
        return std::nullopt;
      }
    } else if (argument == "--consumer-controller-cpus") {
      if (!parse_cpu_list(value, options.consumer_controller_cpus)) {
        std::cerr << "invalid --consumer-controller-cpus\n";
        return std::nullopt;
      }
    } else if (argument == "--consumer-pump-cpus") {
      if (!parse_cpu_list(value, options.consumer_pump_cpus)) {
        std::cerr << "invalid --consumer-pump-cpus\n";
        return std::nullopt;
      }
    } else if (argument == "--busy-poll") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > 1U) {
        std::cerr << "invalid --busy-poll\n";
        return std::nullopt;
      }
      options.busy_poll = parsed == 1U;
    } else if (argument == "--runtime-phase-trace") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > 1U) {
        std::cerr << "invalid --runtime-phase-trace\n";
        return std::nullopt;
      }
      options.runtime_phase_trace = parsed == 1U;
    } else if (argument == "--trace-correlation-self-test") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > 1U) {
        std::cerr << "invalid --trace-correlation-self-test\n";
        return std::nullopt;
      }
      options.trace_correlation_self_test = parsed == 1U;
    } else if (argument == "--start-barrier-self-test") {
      std::uint64_t parsed{};
      if (!parse_unsigned(value, parsed) || parsed > 1U) {
        std::cerr << "invalid --start-barrier-self-test\n";
        return std::nullopt;
      }
      options.start_barrier_self_test = parsed == 1U;
    } else if (argument == "--ready-fd") {
      if (!parse_signed(value, options.ready_fd)) {
        std::cerr << "invalid --ready-fd\n";
        return std::nullopt;
      }
    } else if (argument == "--start-fd") {
      if (!parse_signed(value, options.start_fd)) {
        std::cerr << "invalid --start-fd\n";
        return std::nullopt;
      }
    } else if (argument == "--output") {
      options.output_path = value;
    } else {
      std::cerr << "unknown option: " << argument << '\n';
      return std::nullopt;
    }
  }

  if (options.run_id.empty()) {
    std::cerr << "--run-id is required\n";
    return std::nullopt;
  }
  if (options.samples == 0U || options.samples > kMaximumSamples ||
      options.warmup > kMaximumSamples ||
      options.warmup > std::numeric_limits<std::uint64_t>::max() - options.samples) {
    std::cerr << "sample geometry exceeds the bounded benchmark range\n";
    return std::nullopt;
  }
  if (options.fanout == 0U || options.fanout > 3U ||
      options.consumer_cpus.size() != options.fanout ||
      options.consumer_controller_cpus.size() != options.fanout ||
      options.consumer_pump_cpus.size() != options.fanout) {
    std::cerr << "--fanout must be 1..3 and match every consumer CPU list\n";
    return std::nullopt;
  }
  if (options.topology == Topology::kChain && options.fanout != 1U) {
    std::cerr << "chain topology requires --fanout 1\n";
    return std::nullopt;
  }
  if (tensor_backed_payload(options.payload) &&
      (options.topology != Topology::kFanout || options.fanout != 1U)) {
    std::cerr << "Tensor-backed payloads currently require --topology fanout and --fanout 1\n";
    return std::nullopt;
  }
  if (options.payload == Payload::kSchema &&
      (options.topology != Topology::kFanout || options.fanout != 1U)) {
    std::cerr << "schema payload currently requires --topology fanout and --fanout 1\n";
    return std::nullopt;
  }
  if (tensor_backed_payload(options.payload) &&
      (options.tensor_bytes == 0U || options.tensor_bytes > kMaximumTensorBytes)) {
    std::cerr << "--tensor-bytes exceeds the bounded Tensor range\n";
    return std::nullopt;
  }
  if (options.tensor_device < 0) {
    std::cerr << "--tensor-device must be nonnegative\n";
    return std::nullopt;
  }
  if (!tensor_backed_payload(options.payload) &&
      (options.tensor_storage != TensorStorage::kPageableHost ||
       options.tensor_bytes != kDefaultTensorBytes || options.tensor_device != 0 ||
       options.bulk_work_profile != BulkWorkProfile::kFullWrite)) {
    std::cerr << "Tensor storage options require a Tensor-backed payload\n";
    return std::nullopt;
  }
  if (options.payload == Payload::kMultiTensorStaging &&
      options.bulk_work_profile == BulkWorkProfile::kFullWrite) {
    std::cerr << "multi-Tensor staging requires --bulk-work-profile descriptor_only\n";
    return std::nullopt;
  }
  if (options.topology == Topology::kFanout && options.role == Role::kRelay) {
    std::cerr << "relay role requires chain topology\n";
    return std::nullopt;
  }
  if (options.role != Role::kConsumer && options.consumer_index != 0U) {
    std::cerr << "--consumer-index is consumer-only\n";
    return std::nullopt;
  }
  if (options.role == Role::kConsumer && options.consumer_index >= options.fanout) {
    std::cerr << "--consumer-index exceeds --fanout\n";
    return std::nullopt;
  }
  if (options.runtime_phase_trace && options.samples + options.warmup > kMaximumPhaseTraceSamples) {
    std::cerr << "runtime phase trace sample count exceeds the bounded diagnostic range\n";
    return std::nullopt;
  }
  if (options.queue_depth == 0U || options.timeout_ms == 0U || options.start_timeout_ms == 0U) {
    std::cerr << "--queue-depth, --timeout-ms, and --start-timeout-ms must be positive\n";
    return std::nullopt;
  }
  if (options.workload == Workload::kLatency && options.queue_depth > 2U) {
    std::cerr << "latency workload requires --queue-depth 1 or 2\n";
    return std::nullopt;
  }
  if (options.workload == Workload::kLatency && options.publisher_period_us == 0U) {
    std::cerr << "latency workload requires a positive --publisher-period-us\n";
    return std::nullopt;
  }
  if (options.workload == Workload::kSaturated && options.publisher_period_us != 0U) {
    std::cerr << "saturated workload requires --publisher-period-us 0\n";
    return std::nullopt;
  }
  std::vector<int> requested_cpus{options.producer_cpu, options.producer_controller_cpu};
  if (options.topology == Topology::kChain) {
    requested_cpus.push_back(options.relay_cpu);
    requested_cpus.push_back(options.relay_controller_cpu);
    requested_cpus.push_back(options.relay_pump_cpu);
  } else if (options.relay_cpu >= 0 || options.relay_controller_cpu >= 0 ||
             options.relay_pump_cpu >= 0) {
    std::cerr << "relay CPU placement requires chain topology\n";
    return std::nullopt;
  }
  requested_cpus.insert(
      requested_cpus.end(), options.consumer_cpus.begin(), options.consumer_cpus.end());
  requested_cpus.insert(requested_cpus.end(),
                        options.consumer_controller_cpus.begin(),
                        options.consumer_controller_cpus.end());
  requested_cpus.insert(
      requested_cpus.end(), options.consumer_pump_cpus.begin(), options.consumer_pump_cpus.end());
  if (std::any_of(
          requested_cpus.begin(), requested_cpus.end(), [](const int cpu) { return cpu < -1; })) {
    std::cerr << "CPU values must be nonnegative ordinals or -1\n";
    return std::nullopt;
  }
  for (std::size_t lhs = 0U; lhs < requested_cpus.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1U; rhs < requested_cpus.size(); ++rhs) {
      if (requested_cpus[lhs] >= 0 && requested_cpus[lhs] == requested_cpus[rhs]) {
        std::cerr << "all requested lane and support-thread CPUs must be distinct\n";
        return std::nullopt;
      }
    }
  }
  if (options.role != Role::kProducer && options.start_fd >= 0) {
    std::cerr << "--start-fd is producer-only\n";
    return std::nullopt;
  }
  if (options.role == Role::kProducer && ((options.ready_fd >= 0) != (options.start_fd >= 0))) {
    std::cerr << "producer --ready-fd and --start-fd must be provided together\n";
    return std::nullopt;
  }
  return options;
}

// Benchmark setup is intentionally linear so every failure cleans up the same shared state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_benchmark(Options options) {
#if !defined(__linux__)
  static_cast<void>(options);
  std::cerr << "SKIP: graph interprocess benchmark currently requires Linux\n";
  return kSkipped;
#else
  if (options.trace_correlation_self_test) {
    return trace_correlation_self_test() ? 0 : 1;
  }
  if (options.start_barrier_self_test) {
    return start_barrier_self_test() ? 0 : 1;
  }
  if (tensor_backed_payload(options.payload) &&
      options.tensor_storage != TensorStorage::kPageableHost) {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || options.tensor_device >= device_count) {
      std::cerr << "SKIP: requested Tensor CUDA device is unavailable\n";
      return kSkipped;
    }
  }
  RunState state{std::move(options)};
  g_run_state = &state;
  std::vector<detail::RuntimePhaseTraceRecord> phase_trace_storage;
  std::shared_ptr<detail::RuntimePhaseCounters> phase_counters;
  if (state.options().runtime_phase_trace) {
    phase_trace_storage.resize(static_cast<std::size_t>(state.final_sequence()) *
                               kPhaseTraceRecordsPerSample);
    phase_counters = std::make_shared<detail::RuntimePhaseCounters>(std::span{phase_trace_storage});
  }
  const Options& benchmark_options = state.options();
  const int controller_cpu =
      benchmark_options.role == Role::kProducer
          ? benchmark_options.producer_controller_cpu
          : (benchmark_options.role == Role::kRelay
                 ? benchmark_options.relay_controller_cpu
                 : benchmark_options.consumer_controller_cpus[benchmark_options.consumer_index]);
  const int pump_cpu =
      benchmark_options.role == Role::kProducer
          ? -1
          : (benchmark_options.role == Role::kRelay
                 ? benchmark_options.relay_pump_cpu
                 : benchmark_options.consumer_pump_cpus[benchmark_options.consumer_index]);
  const auto thread_control =
      std::make_shared<BenchmarkSupportThreadControl>(std::vector<SupportThreadBinding>{
          {.role = SupportThreadRole::kController,
           .requested_cpu = controller_cpu,
           .required = true},
          {.role = SupportThreadRole::kChannelPump,
           .requested_cpu = pump_cpu,
           .required = benchmark_options.role != Role::kProducer,
           .forbidden = benchmark_options.role == Role::kProducer},
      });
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
      make_plan(state.options()), thread_control);
  if (phase_counters) {
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, phase_counters);
  }
  if (!plan.ok()) {
    std::cerr << "failed to compile interprocess Runtime graph\n" << plan.json() << '\n';
    g_run_state = nullptr;
    return 2;
  }
  const auto* plan_state = detail::ExecutionPlanAccess::state(plan);
  if (plan_state == nullptr || static_cast<bool>(plan_state->executable_factories.phase_counters) !=
                                   static_cast<bool>(phase_counters)) {
    std::cerr << "interprocess Runtime phase-counter attachment does not match diagnostics\n";
    g_run_state = nullptr;
    return 2;
  }
  if ((state.options().payload == Payload::kTensor || state.options().payload == Payload::kImage) &&
      state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly) {
    const auto allocations = plan.tensor_allocations();
    if (allocations.size() != 1U || allocations.front().slot_count == 0U) {
      std::cerr << "single-Tensor compiled pool inventory is invalid\n";
      g_run_state = nullptr;
      return 2;
    }
    // The pool includes failure-recovery reserves that ordinary steady-state traffic does not
    // touch. Warm the producer loan, queued and borrowed transport-retention sets, and retirement
    // slot; forcing every recovery slot into circulation would exhaust the readiness contract.
    const std::size_t warmup_slots =
        std::min(static_cast<std::size_t>(allocations.front().slot_count),
                 static_cast<std::size_t>(allocations.front().max_outstanding_loans) +
                     2U * static_cast<std::size_t>(state.options().queue_depth) + 1U);
    if (state.options().warmup < warmup_slots) {
      std::cerr << "single-Tensor warmup cannot cover the compiled pool inventory\n";
      g_run_state = nullptr;
      return 2;
    }
    state.set_descriptor_warmup_slots(warmup_slots);
  }
  if (state.options().payload == Payload::kMultiTensorLoan &&
      state.options().bulk_work_profile == BulkWorkProfile::kDescriptorOnly) {
    const auto fields = plan.tensor_field_allocations();
    if (fields.size() != 2U || fields.front().slot_count == 0U ||
        fields.front().slot_count != fields.back().slot_count ||
        fields.front().max_outstanding_loans != fields.back().max_outstanding_loans) {
      std::cerr << "multi-Tensor compiled field-pool inventory is invalid\n";
      g_run_state = nullptr;
      return 2;
    }
    // Every publication draws one slot from each independent field pool. Saturated cross-process
    // release timing can rotate every compiled slot into the hot path, so condition both complete
    // field inventories instead of relying on a scheduler-dependent retention estimate.
    const std::size_t compiled_slots = static_cast<std::size_t>(fields.front().slot_count);
    const std::size_t warmup_slots_per_field =
        state.options().workload == Workload::kSaturated
            ? compiled_slots
            : static_cast<std::size_t>(fields.front().max_outstanding_loans) +
                  static_cast<std::size_t>(state.options().queue_depth) + 1U;
    if (warmup_slots_per_field == 0U || warmup_slots_per_field > compiled_slots) {
      std::cerr << "multi-Tensor hot-path field inventory is invalid\n";
      g_run_state = nullptr;
      return 2;
    }
    const std::size_t required_warmup = state.options().workload == Workload::kSaturated
                                            ? 2U * warmup_slots_per_field
                                            : warmup_slots_per_field;
    if (state.options().warmup < required_warmup) {
      std::cerr << "multi-Tensor warmup cannot cover the hot-path field slots\n";
      g_run_state = nullptr;
      return 2;
    }
    state.set_descriptor_warmup_slots(warmup_slots_per_field);
  }
  const std::optional<ProcessGroupId> process_group =
      process_group_for_role(plan, role_name(state.options()));
  const auto* internal = detail::ExecutionPlanAccess::state(plan);
  // Compatible fanout arms share one compiler-owned broadcast service. A chain still needs one
  // service per cross-process hop.
  const std::size_t expected_service_count = state.options().topology == Topology::kChain ? 2U : 1U;
  if (!process_group.has_value() || internal == nullptr ||
      internal->cross_process_channel_services.size() != expected_service_count) {
    std::cerr << "compiled plan did not contain the expected role-isolated cross-process edges\n";
    g_run_state = nullptr;
    return 2;
  }

  RunSession session = detail::run_async_worker(plan, *process_group);
  const bool started = state.wait_for_start();
  std::optional<NotificationSender> producer_sender;
  if (state.options().role == Role::kProducer && started) {
    producer_sender = state.take_sender();
    if (!producer_sender.has_value()) {
      state.fail(ErrorCode::kTimeout);
    }
  }
  const bool sender_ready = state.options().role != Role::kProducer || producer_sender.has_value();
  const bool ready_reported = started && sender_ready && report_ready(state.options().ready_fd);
  const bool start_permitted =
      state.options().role != Role::kProducer ||
      (ready_reported &&
       wait_for_benchmark_start(state.options().start_fd, state.options().start_timeout_ms));

  std::thread notification_driver;
  if (state.options().role == Role::kProducer && started && start_permitted) {
    notification_driver =
        std::thread([&state, sender = std::move(*producer_sender)]() { state.drive(sender); });
  }

  const bool signaled = started && ready_reported && start_permitted && state.wait_for_completion();
  if (!signaled) {
    state.fail(ErrorCode::kTimeout);
  }
  if (notification_driver.joinable()) {
    notification_driver.join();
  }
  session.request_stop();
  session.wait();
  const IngressWakeTraceSummary ingress_wake_trace =
      phase_counters ? summarize_ingress_wake_trace(state, *phase_counters)
                     : IngressWakeTraceSummary{};

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
              *process_group,
              phase_counters.get(),
              *thread_control,
              ingress_wake_trace);

  bool role_counts_valid{};
  switch (state.options().role) {
    case Role::kProducer:
      role_counts_valid = state.producer_starts() == 1U && state.relay_starts() == 0U &&
                          state.consumer_starts() == 0U &&
                          state.publications() == state.final_sequence() &&
                          state.relay_inputs() == 0U && state.relay_publications() == 0U &&
                          state.deliveries() == 0U;
      break;
    case Role::kRelay:
      role_counts_valid = state.producer_starts() == 0U && state.relay_starts() == 1U &&
                          state.consumer_starts() == 0U && state.publications() == 0U &&
                          state.relay_inputs() == state.final_sequence() &&
                          state.relay_publications() == state.final_sequence() &&
                          state.deliveries() == 0U &&
                          state.final_sequence_observed() == state.final_sequence();
      break;
    case Role::kConsumer:
      role_counts_valid = state.producer_starts() == 0U && state.relay_starts() == 0U &&
                          state.consumer_starts() == 1U && state.publications() == 0U &&
                          state.relay_inputs() == 0U && state.relay_publications() == 0U &&
                          state.deliveries() == state.final_sequence() &&
                          state.final_sequence_observed() == state.final_sequence();
      break;
  }
  const bool valid = signaled && state.completed() && state.first_error() == ErrorCode::kSuccess &&
                     role_counts_valid && thread_control->required_roles_entered() &&
                     (!ingress_wake_trace.applicable || ingress_wake_trace.valid) &&
                     session.termination() == RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    std::cerr << "interprocess Runtime graph benchmark failed: role=" << role_name(state.options())
              << " started=" << started << " completed=" << state.completed()
              << " error=" << static_cast<std::uint32_t>(state.first_error())
              << " publications=" << state.publications() << " deliveries=" << state.deliveries()
              << " diagnostics=";
    for (const Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << ' ';
    }
    std::cerr << '\n';
    if (!session.diagnostics().empty()) {
      std::cerr << format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
    }
  }
  g_run_state = nullptr;
  return valid ? 0 : 1;
#endif
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
    std::cerr << "interprocess Runtime graph benchmark setup failed: " << error.what() << '\n';
    return 2;
  }
}
