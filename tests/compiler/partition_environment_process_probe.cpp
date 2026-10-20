/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"
#include "interprocess_test_operators.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"

namespace holoscan {
namespace {

constexpr std::string_view kSelectorName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";
constexpr std::string_view kSourceChainAlgorithm = "holoscan.partition.source_chain";
constexpr std::string_view kExperimentalWarning = "PARTITION_PLANNER_EXPERIMENTAL_SELECTED";
constexpr std::string_view kAbsentSelectorArgument = "-";
constexpr std::string_view kProbeInstance = "HOLOSCAN_TEST_PARTITION_PROBE_INSTANCE";

using namespace std::chrono_literals;

[[nodiscard]] bool ensure_probe_instance(const std::string_view scenario) {
  if (std::getenv(kProbeInstance.data()) != nullptr) {
    return true;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string instance = std::string(scenario) + "-" +
                               std::to_string(static_cast<std::uint64_t>(getpid())) + "-" +
                               std::to_string(tick);
  return setenv(kProbeInstance.data(), instance.c_str(), 1) == 0;
}

[[nodiscard]] std::string probe_graph_name(const std::string_view base) {
  const char* const instance = std::getenv(kProbeInstance.data());
  return instance == nullptr ? std::string{base} : std::string{base} + "-" + std::string{instance};
}

// Select the CUDA device on the Runtime controller thread, where Tensor pools are materialized.
class CUDADeviceSupportThreadControl final : public detail::RuntimeSupportThreadControl {
 public:
  explicit CUDADeviceSupportThreadControl(DeviceId device) : device_(device) {}

  [[nodiscard]] expected<void, Error> enter(SupportThreadRole role) override {
    if (role == SupportThreadRole::kController &&
        (cudaSetDevice(device_.value) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return {};
  }

 private:
  DeviceId device_;
};

struct WorkerProbeState {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    last.store(0U, std::memory_order_release);
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
    read_only.store(false, std::memory_order_release);
    device_type.store(static_cast<std::int32_t>(kDLExtDev), std::memory_order_release);
    device_id.store(-1, std::memory_order_release);
  }

  static void record(const std::uint32_t value) noexcept {
    {
      const std::lock_guard lock{mutex};
      last.store(value, std::memory_order_release);
      count.fetch_add(1U, std::memory_order_acq_rel);
    }
    ready.notify_all();
  }

  static void record_source_start() noexcept {
    source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  static void record_sink_start() noexcept {
    sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_starts(const std::size_t expected_source,
                                            const std::size_t expected_sink) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [=] {
      return source_starts.load(std::memory_order_acquire) >= expected_source &&
             sink_starts.load(std::memory_order_acquire) >= expected_sink;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint32_t> last{};
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
  static inline std::atomic_bool read_only{};
  static inline std::atomic<std::int32_t> device_type{};
  static inline std::atomic<std::int32_t> device_id{-1};
};

[[nodiscard]] test::ScalarProcessBoundaryHooks worker_scalar_hooks() noexcept {
  return {
      .payload = 0x15U,
      .source_started = &WorkerProbeState::record_source_start,
      .sink_started = &WorkerProbeState::record_sink_start,
      .sample_received = &WorkerProbeState::record,
  };
}

constexpr std::uint32_t kPublicationPayload = 71U;
constexpr std::int64_t kCaptureInstant = 17;

class PublicationCoordinateSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms});
    return result;
  }

  void start() override { WorkerProbeState::record_source_start(); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint capture{.timestamp_ns = kCaptureInstant,
                            .clock_id = context.activation_time().clock_id};
    auto emitted = output.emit(kPublicationPayload, EmitOptions{.capture_time = capture});
    if (!emitted && emitted.error().code != ErrorCode::kNotReady &&
        emitted.error().code != ErrorCode::kBackpressured &&
        emitted.error().code != ErrorCode::kPublicationIndeterminate) {
      return make_unexpected(std::move(emitted).error());
    }
    return {};
  }

  Output<std::uint32_t> output;
};

class PublicationCoordinateSink final : public Operator<> {
 public:
  explicit PublicationCoordinateSink(bool process_local) : process_local_(process_local) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  void start() override { WorkerProbeState::record_sink_start(); }

  // REQ: RUNTIME-057
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const auto& metadata = sample->metadata;
    const ClockId clock = context.activation_time().clock_id;
    const bool capture_retained = clock.valid() && metadata.capture_time().valid() &&
                                  metadata.capture_timestamp_ns == kCaptureInstant &&
                                  metadata.source_clock_id == clock;
    const bool publication_valid =
        process_local_ ? metadata.publish_time().valid() && metadata.publish_timestamp_ns > 0 &&
                             metadata.publish_clock_id == clock
                       : !metadata.publish_time().valid() && metadata.publish_timestamp_ns == 0 &&
                             !metadata.publish_clock_id.valid();
    if (sample->data != kPublicationPayload || !capture_retained || !publication_valid) {
      return make_unexpected(Error{ErrorCode::kProtocolError,
                                   "publication-coordinate metadata did not match locality"});
    }
    WorkerProbeState::record(sample->data);
    return {};
  }

  Input<std::uint32_t> input;

 private:
  bool process_local_;
};

struct WorkerFanoutProbeState {
  static constexpr std::size_t kSampleCount = 8U;

  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    source_starts = 0U;
    sink_starts = 0U;
    emitted = 0U;
    count = 0U;
    values = {};
  }

  static void record_source_start() {
    {
      const std::lock_guard lock{mutex};
      ++source_starts;
    }
    ready.notify_all();
  }

  static void record_sink_start(std::size_t) {
    {
      const std::lock_guard lock{mutex};
      ++sink_starts;
    }
    ready.notify_all();
  }

  static void record_emit() {
    {
      const std::lock_guard lock{mutex};
      ++emitted;
    }
    ready.notify_all();
  }

  static void record(std::size_t, const std::uint32_t value) {
    {
      const std::lock_guard lock{mutex};
      if (count < values.size()) {
        values[count] = value;
      }
      ++count;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_source() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return source_starts == 1U; });
  }

  [[nodiscard]] static bool wait_for_sink() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return sink_starts == 1U; });
  }

  [[nodiscard]] static bool wait_for_emitted() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return emitted == kSampleCount; });
  }

  [[nodiscard]] static bool wait_for_received() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return count == kSampleCount; });
  }

  [[nodiscard]] static bool exact_received_sequence() {
    const std::lock_guard lock{mutex};
    for (std::size_t index = 0U; index < values.size(); ++index) {
      if (values[index] != index + 1U) {
        return false;
      }
    }
    return source_starts == 0U && sink_starts == 1U && count == kSampleCount;
  }

  [[nodiscard]] static bool exact_producer_state() {
    const std::lock_guard lock{mutex};
    return source_starts == 1U && sink_starts == 0U && emitted == kSampleCount;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t source_starts{};
  static inline std::size_t sink_starts{};
  static inline std::size_t emitted{};
  static inline std::size_t count{};
  static inline std::array<std::uint32_t, kSampleCount> values{};
};

[[nodiscard]] test::FanoutProcessBoundaryHooks worker_fanout_hooks() noexcept {
  return {
      .sample_count = WorkerFanoutProbeState::kSampleCount,
      .source_started = &WorkerFanoutProbeState::record_source_start,
      .sink_started = &WorkerFanoutProbeState::record_sink_start,
      .sample_emitted = &WorkerFanoutProbeState::record_emit,
      .sample_received = &WorkerFanoutProbeState::record,
  };
}

template <MemoryKind Storage>
class WorkerHostTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = Storage,
                    .dtype = DLDataType{kDLUInt, 32U, 1U},
                    .rank = 0U,
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint32_t)},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms});
    return result;
  }

  void start() override {
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      if (cudaSetDevice(0) != cudaSuccess ||
          cudaMalloc(&device_value_, sizeof(std::uint32_t)) != cudaSuccess) {
        if (device_value_ != nullptr) {
          static_cast<void>(cudaFree(device_value_));
          device_value_ = nullptr;
        }
      }
    }
    WorkerProbeState::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    WorkerProbeState::ready.notify_all();
  }

  void stop() override {
    if (device_value_ != nullptr) {
      static_cast<void>(cudaFree(device_value_));
      device_value_ = nullptr;
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (published_) {
      return {};
    }
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 32U, 1U},
    });
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan).error());
    }
    auto writer = [&]() {
      if constexpr (Storage == MemoryKind::kHost) {
        return loan->write_host();
      } else {
        return loan->write(context.cuda_stream());
      }
    }();
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      // The exported HoloMQ slot is the async D2H destination; publication carries its fence.
      if (device_value_ == nullptr || context.cuda_stream().get() == nullptr ||
          cudaMemsetAsync(
              device_value_, kFillByte, sizeof(std::uint32_t), context.cuda_stream().get()) !=
              cudaSuccess ||
          cudaMemcpyAsync(writer->data(),
                          device_value_,
                          sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    } else if constexpr (Storage == MemoryKind::kCudaDevice) {
      if (context.cuda_stream().get() == nullptr ||
          cudaMemsetAsync(
              writer->data(), kFillByte, writer->byte_size(), context.cuda_stream().get()) !=
              cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    } else {
      auto* value = writer->template data_as<std::uint32_t>();
      if (value == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      *value = kValue;
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted && (emitted.error().code == ErrorCode::kNotReady ||
                     emitted.error().code == ErrorCode::kBackpressured ||
                     emitted.error().code == ErrorCode::kPublicationIndeterminate)) {
      return {};
    }
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    published_ = true;
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint8_t kFillByte = 0x13U;
  static constexpr std::uint32_t kValue = 0x13131313U;

 private:
  bool published_{};
  void* device_value_{};
};

template <MemoryKind Storage>
class WorkerHostTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = Storage,
                    .dtype = DLDataType{kDLUInt, 32U, 1U},
                    .rank = 0U,
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  void start() override {
    WorkerProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    WorkerProbeState::ready.notify_all();
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    std::uint32_t observed{};
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      // A CPU reader must wait for the producer's imported generation fence.
      auto ready = sample->data.synchronize_for_host();
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
      const auto* value = sample->data.data_as<std::uint32_t>();
      if (value == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      observed = *value;
    } else if constexpr (Storage == MemoryKind::kCudaDevice) {
      // Runtime has already ordered the imported generation onto this activation stream. Keep the
      // verification copy on that compiler-selected stream so a nonzero placement never falls
      // back to the callback thread's ambient CUDA device.
      int callback_device = -1;
      if (cudaGetDevice(&callback_device) != cudaSuccess ||
          callback_device != sample->data.device().device_id ||
          context.cuda_stream().get() == nullptr ||
          cudaMemcpyAsync(&observed,
                          sample->data.data(),
                          sizeof(observed),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    } else {
      const auto* value = sample->data.data_as<std::uint32_t>();
      if (value == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      observed = *value;
    }
    WorkerProbeState::read_only.store(sample->data.is_read_only(), std::memory_order_release);
    WorkerProbeState::device_type.store(
        static_cast<std::int32_t>(sample->data.device().device_type), std::memory_order_release);
    WorkerProbeState::device_id.store(sample->data.device().device_id, std::memory_order_release);
    WorkerProbeState::record(observed);
    return {};
  }

  Input<Tensor> input;
};

using WorkerTensorSource = WorkerHostTensorSource<MemoryKind::kHost>;
using WorkerTensorSink = WorkerHostTensorSink<MemoryKind::kHost>;
using WorkerPinnedTensorSource = WorkerHostTensorSource<MemoryKind::kPinnedHost>;
using WorkerPinnedTensorSink = WorkerHostTensorSink<MemoryKind::kPinnedHost>;
using WorkerDeviceTensorSource = WorkerHostTensorSource<MemoryKind::kCudaDevice>;
using WorkerDeviceTensorSink = WorkerHostTensorSink<MemoryKind::kCudaDevice>;

[[nodiscard]] ExecutionPlan compile_linear(Profile profile = Profile::kStandard) {
  Graph graph{probe_graph_name("partition-environment-fresh-process")};
  test::add_linear_graph(graph);
  CompileOptions options;
  options.profile = profile;
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan compile_worker_plan(bool publication_coordinate = false,
                                                bool process_local = false) {
  Graph graph{probe_graph_name("partition-worker-environment")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  if (publication_coordinate) {
    auto source = graph.op<PublicationCoordinateSource>("source");
    auto target = graph.op<PublicationCoordinateSink>("target", process_local);
    graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 4U});
  } else {
    auto source = graph.op<test::ScalarProcessBoundarySource>("source", worker_scalar_hooks());
    auto target = graph.op<test::ScalarProcessBoundarySink>("target", worker_scalar_hooks());
    graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 4U});
  }
  if (process_local) {
    return compile(graph);
  }

  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array target_selector{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = target_selector,
                             .launch_group = "consumer-process",
                         });
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

[[nodiscard]] ExecutionPlan compile_fanout_worker_plan() {
  Graph graph{probe_graph_name("partition-fanout-worker-environment")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<test::FanoutProcessBoundarySource>("source", worker_fanout_hooks());
  auto first = graph.op<test::FanoutProcessBoundarySink>("first_target", worker_fanout_hooks(), 0U);
  auto second =
      graph.op<test::FanoutProcessBoundarySink>("second_target", worker_fanout_hooks(), 1U);
  graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 4U});

  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array first_selector{detail::GraphObjectSelector{.graph_path = "first_target"}};
  constexpr std::array second_selector{detail::GraphObjectSelector{.graph_path = "second_target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("first-consumer",
                         detail::ProcessPlacement{
                             .members = first_selector,
                             .launch_group = "first-consumer-process",
                         })
      .bind_process_role("second-consumer",
                         detail::ProcessPlacement{
                             .members = second_selector,
                             .launch_group = "second-consumer-process",
                         });
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

template <MemoryKind Storage>
[[nodiscard]] ExecutionPlan compile_tensor_worker_plan(const DeviceId device = DeviceId{0}) {
  constexpr std::string_view graph_name = [] {
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      return "partition-pinned-tensor-worker-environment";
    }
    if constexpr (Storage == MemoryKind::kCudaDevice) {
      return "partition-device-tensor-worker-environment";
    }
    return "partition-tensor-worker-environment";
  }();
  Graph graph{probe_graph_name(graph_name)};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<WorkerHostTensorSource<Storage>>("source");
  auto target = graph.op<WorkerHostTensorSink<Storage>>("target");
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 2U});

  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array target_selector{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = target_selector,
                             .launch_group = "consumer-process",
                         });
  if constexpr (Storage == MemoryKind::kPinnedHost || Storage == MemoryKind::kCudaDevice) {
    const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
    const auto output =
        std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
          return port.direction == PortDirection::kOutput && port.name == "output";
        });
    if (output == snapshot->ports.end()) {
      throw std::logic_error{"Tensor worker fixture output port is missing"};
    }
    const std::array device_bindings{
        detail::TensorDeviceBindingRecord{
            .port = PortId{output->id.value},
            .device = Storage == MemoryKind::kCudaDevice ? device : DeviceId{0},
        },
    };
    // Pinned storage needs a private synchronization device; device storage uses its exact GPU.
    return detail::compile_with_runtime_evidence_and_process_placement(
        graph,
        CompileOptions{},
        detail::RuntimeCompileEvidenceView{
            .package_capabilities = detail::runtime_package_capability_view(),
            .tensor_device_bindings = device_bindings,
        },
        std::move(placement));
  }
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

[[nodiscard]] std::size_t diagnostic_count(const ExecutionPlan& plan, std::string_view token,
                                           DiagnosticSeverity severity) {
  return static_cast<std::size_t>(
      std::count_if(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
        return item.token == token && item.severity == severity;
      }));
}

[[nodiscard]] const Diagnostic* find_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  const auto found = std::find_if(plan.diagnostics().begin(),
                                  plan.diagnostics().end(),
                                  [&](const auto& item) { return item.token == token; });
  return found == plan.diagnostics().end() ? nullptr : &*found;
}

[[nodiscard]] bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool check_selector(std::optional<std::string_view> expected) {
  const char* const observed = std::getenv(kSelectorName.data());
  if (!expected.has_value()) {
    return check(observed == nullptr, "selector was expected to be absent");
  }
  return check(observed != nullptr && std::string_view(observed) == *expected,
               "selector did not match the fresh-process case");
}

[[nodiscard]] bool set_selector(std::optional<std::string_view> value) noexcept {
#if defined(_WIN32)
  return _putenv_s(kSelectorName.data(), value.has_value() ? std::string{*value}.c_str() : "") == 0;
#else
  return value.has_value() ? setenv(kSelectorName.data(), std::string{*value}.c_str(), 1) == 0
                           : unsetenv(kSelectorName.data()) == 0;
#endif
}

[[nodiscard]] bool check_valid_plan(const ExecutionPlan& plan, std::string_view algorithm,
                                    PlanSelectionSource source, AlgorithmMaturity maturity,
                                    std::size_t warning_count) {
  const auto& record = plan.partitioning_plan();
  bool valid = true;
  valid &= check(plan.ok(), "expected an accepted plan");
  valid &= check(record.disposition == PlanDisposition::kAccepted,
                 "expected an accepted partition disposition");
  valid &= check(record.selection_source == source, "partition selection source mismatch");
  valid &= check(record.requested_algorithm.has_value(), "requested algorithm is absent");
  valid &= check(record.accepted_algorithm.has_value(), "accepted algorithm is absent");
  if (record.requested_algorithm.has_value()) {
    valid &= check(record.requested_algorithm->id == algorithm,
                   "requested algorithm identifier mismatch");
    valid &=
        check(record.requested_algorithm->version == 1U, "requested algorithm version mismatch");
  }
  if (record.accepted_algorithm.has_value()) {
    valid &=
        check(record.accepted_algorithm->id == algorithm, "accepted algorithm identifier mismatch");
    valid &= check(record.accepted_algorithm->version == 1U, "accepted algorithm version mismatch");
  }
  valid &= check(record.maturity == maturity, "partition maturity mismatch");
  valid &= check(record.config_digest != PlanDigest{}, "partition config digest is zero");
  valid &= check(
      diagnostic_count(plan, kExperimentalWarning, DiagnosticSeverity::kWarning) == warning_count,
      "experimental warning count mismatch");
  return valid;
}

[[nodiscard]] bool check_removed_selector_plan(const ExecutionPlan& plan) {
  const auto& record = plan.partitioning_plan();
  constexpr std::string_view kToken = "PARTITION_ENVIRONMENT_SELECTOR_REMOVED";
  const Diagnostic* const diagnostic = find_diagnostic(plan, kToken);
  bool valid = true;
  valid &= check(!plan.ok(), "expected a rejected plan");
  valid &= check(record.disposition == PlanDisposition::kRejected,
                 "expected a rejected partition disposition");
  valid &= check(record.selection_source == PlanSelectionSource::kCoreDefault,
                 "removed selector acquired selection provenance");
  valid &=
      check(!record.accepted_algorithm.has_value(), "rejected plan retained an accepted algorithm");
  valid &= check(record.maturity == AlgorithmMaturity::kProduction,
                 "removed selector changed default planner maturity");
  valid &= check(record.config_digest != PlanDigest{}, "rejected config digest is zero");
  valid &= check(diagnostic != nullptr, "expected rejection diagnostic is absent");
  valid &= check(diagnostic_count(plan, kToken, DiagnosticSeverity::kError) == 1U,
                 "rejection diagnostic count mismatch");
  if (diagnostic != nullptr) {
    valid &= check(record.reason == diagnostic->id,
                   "partition rejection reason does not identify its diagnostic");
  }
  valid &= check(record.requested_algorithm.has_value(),
                 "removed selector erased the production default attempt");
  if (record.requested_algorithm.has_value()) {
    valid &= check(record.requested_algorithm->id == kSourceChainAlgorithm,
                   "removed selector changed the production default attempt");
    valid &= check(record.requested_algorithm->version == 1U,
                   "production default algorithm version mismatch");
  }
  valid &= check(diagnostic_count(plan, kExperimentalWarning, DiagnosticSeverity::kWarning) == 0U,
                 "rejected selector emitted an experimental acceptance warning");
  return valid;
}

#if defined(__linux__)
[[nodiscard]] std::optional<std::string_view> selector_from_argument(
    const std::string_view argument) {
  return argument == kAbsentSelectorArgument ? std::nullopt
                                             : std::optional<std::string_view>{argument};
}

[[nodiscard]] bool run_consumer_worker(const int result_fd,
                                       const std::optional<std::string_view> compile_selector,
                                       const std::optional<std::string_view> runtime_selector,
                                       bool publication_coordinate) {
  WorkerProbeState::reset();
  bool valid = set_selector(compile_selector);
  const ExecutionPlan plan = compile_worker_plan(publication_coordinate);
  valid &= check(!compile_selector.has_value(),
                 "worker compile must not use the removed environment selector");
  valid &= check_valid_plan(plan,
                            kSourceChainAlgorithm,
                            PlanSelectionSource::kCoreDefault,
                            AlgorithmMaturity::kProduction,
                            0U);
  valid &= set_selector(runtime_selector);
  valid &= check_selector(runtime_selector);

  const auto* state = detail::ExecutionPlanAccess::state(plan);
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "consumer exec has no unique cross-process service");
  valid &=
      check(state != nullptr && state->clocks.size() == 1U && state->root_default_clock.valid(),
            "consumer exec has no plan-owned default clock");
  if (!valid) {
    const char ready = 0;
    if (write(result_fd, &ready, sizeof(ready)) != sizeof(ready)) {
      std::cerr << "consumer exec could not report startup failure\n";
    }
    return false;
  }

  const auto& service = state->cross_process_channel_services.front();
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  const bool started = WorkerProbeState::wait_for_starts(0U, 1U) &&
                       WorkerProbeState::source_starts.load(std::memory_order_acquire) == 0U;
  const char ready = started ? 1 : 0;
  const bool ready_written = write(result_fd, &ready, sizeof(ready)) == sizeof(ready);
  const bool received = started && ready_written && WorkerProbeState::wait_for_count(2U);
  consumer.request_stop();
  consumer.wait();
  const bool child_valid = received &&
                           WorkerProbeState::last.load(std::memory_order_acquire) ==
                               (publication_coordinate ? kPublicationPayload : 0x15U) &&
                           WorkerProbeState::sink_starts.load(std::memory_order_acquire) == 1U &&
                           WorkerProbeState::source_starts.load(std::memory_order_acquire) == 0U &&
                           consumer.termination() == RunTermination::kStopped &&
                           consumer.diagnostics().empty();
  const char result = child_valid ? 1 : 0;
  const bool result_written = write(result_fd, &result, sizeof(result)) == sizeof(result);
  return child_valid && result_written;
}

[[nodiscard]] bool run_fanout_consumer_worker(const int result_fd, const std::size_t target_index) {
  WorkerFanoutProbeState::reset();
  // Key setup: reconstruct the complete plan after exec, then materialize only this consumer group.
  const ExecutionPlan plan = compile_fanout_worker_plan();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  bool valid = check(plan.ok(), "consumer exec rejected the fanout worker plan");
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "consumer exec has no unique fanout service");
  if (state != nullptr) {
    valid &= check(state->cross_process_channel_publishers.size() == 1U,
                   "consumer exec did not reconstruct one fanout publisher");
    valid &= check(state->cross_process_channel_subscribers.size() == 2U,
                   "consumer exec did not reconstruct both fanout subscribers");
  }
  if (!valid) {
    const char ready = 0;
    if (write(result_fd, &ready, sizeof(ready)) != sizeof(ready)) {
      std::cerr << "fanout consumer exec could not report startup failure\n";
    }
    return false;
  }

  const auto& service = state->cross_process_channel_services.front();
  valid &= check(service.max_subscribers == 2U && service.target_process_groups.size() == 2U &&
                     target_index < service.target_process_groups.size(),
                 "consumer exec reconstructed invalid fanout service geometry");
  if (!valid) {
    const char ready = 0;
    if (write(result_fd, &ready, sizeof(ready)) != sizeof(ready)) {
      std::cerr << "fanout consumer exec could not report service failure\n";
    }
    return false;
  }

  RunSession consumer = detail::run_async_worker(plan, service.target_process_groups[target_index]);
  const bool started = WorkerFanoutProbeState::wait_for_sink();
  const char ready = started ? 1 : 0;
  const bool ready_written = write(result_fd, &ready, sizeof(ready)) == sizeof(ready);
  const bool received = started && ready_written && WorkerFanoutProbeState::wait_for_received();
  consumer.request_stop();
  consumer.wait();

  // Key assertion: this isolated consumer sees every value exactly once and in source order.
  const bool child_valid = received && WorkerFanoutProbeState::exact_received_sequence() &&
                           consumer.termination() == RunTermination::kStopped &&
                           consumer.diagnostics().empty();
  const char result = child_valid ? 1 : 0;
  const bool result_written = write(result_fd, &result, sizeof(result)) == sizeof(result);
  return child_valid && result_written;
}

[[nodiscard]] bool run_tensor_consumer_worker(const int result_fd, const MemoryKind storage,
                                              const DeviceId device) {
  WorkerProbeState::reset();
  // Recompile after exec so matching service identity is not inherited from the producer.
  const ExecutionPlan plan = [&] {
    switch (storage) {
      case MemoryKind::kPinnedHost:
        return compile_tensor_worker_plan<MemoryKind::kPinnedHost>();
      case MemoryKind::kCudaDevice:
        return compile_tensor_worker_plan<MemoryKind::kCudaDevice>(device);
      default:
        return compile_tensor_worker_plan<MemoryKind::kHost>();
    }
  }();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  bool valid = check(plan.ok(), "consumer exec rejected the Tensor worker plan");
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "consumer exec has no unique Tensor service");
  if (!valid) {
    const char ready = 0;
    if (write(result_fd, &ready, sizeof(ready)) != sizeof(ready)) {
      std::cerr << "Tensor consumer exec could not report startup failure\n";
    }
    return false;
  }

  const auto& service = state->cross_process_channel_services.front();
  valid &= check(storage != MemoryKind::kCudaDevice ||
                     (service.tensor_memory_kind == MemoryKind::kCudaDevice &&
                      service.tensor_device == device),
                 "consumer exec did not retain the requested CUDA device");
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  // Arm the independently reconstructed subscriber before the producer creates its pool.
  const bool started = WorkerProbeState::wait_for_starts(0U, 1U) &&
                       WorkerProbeState::source_starts.load(std::memory_order_acquire) == 0U;
  const char ready = started ? 1 : 0;
  const bool ready_written = write(result_fd, &ready, sizeof(ready)) == sizeof(ready);
  const bool received = started && ready_written && WorkerProbeState::wait_for_count(1U);
  consumer.request_stop();
  consumer.wait();

  // The child must see the exact bytes through an imported read-only HoloMQ buffer lease.
  const bool child_valid =
      valid && received &&
      WorkerProbeState::last.load(std::memory_order_acquire) == WorkerTensorSource::kValue &&
      WorkerProbeState::read_only.load(std::memory_order_acquire) &&
      WorkerProbeState::device_type.load(std::memory_order_acquire) ==
          (storage == MemoryKind::kPinnedHost
               ? kDLCUDAHost
               : (storage == MemoryKind::kCudaDevice ? kDLCUDA : kDLCPU)) &&
      (storage != MemoryKind::kCudaDevice ||
       WorkerProbeState::device_id.load(std::memory_order_acquire) == device.value) &&
      WorkerProbeState::sink_starts.load(std::memory_order_acquire) == 1U &&
      WorkerProbeState::source_starts.load(std::memory_order_acquire) == 0U &&
      consumer.termination() == RunTermination::kStopped && consumer.diagnostics().empty();
  const char result = child_valid ? 1 : 0;
  const bool result_written = write(result_fd, &result, sizeof(result)) == sizeof(result);
  return child_valid && result_written;
}
#endif

// REQ: CHAN-085
[[nodiscard]] bool run_fanout_worker_plan() {
#if defined(__linux__)
  WorkerFanoutProbeState::reset();

  // Key setup: two exec'd consumers independently reconstruct and arm the same shared service.
  const ExecutionPlan plan = compile_fanout_worker_plan();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  bool valid = check(plan.ok(), "producer rejected the fanout worker plan");
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "producer plan has no unique fanout service");
  if (state != nullptr) {
    valid &= check(state->cross_process_channel_publishers.size() == 1U,
                   "producer plan did not retain one fanout publisher");
    valid &= check(state->cross_process_channel_subscribers.size() == 2U,
                   "producer plan did not retain both fanout subscribers");
  }
  if (!valid) {
    return false;
  }
  const auto& service = state->cross_process_channel_services.front();
  valid &= check(service.max_subscribers == 2U && service.target_process_groups.size() == 2U,
                 "producer plan has invalid fanout service geometry");
  if (!valid) {
    return false;
  }

  std::array<std::array<int, 2U>, 2U> child_results{};
  for (auto& child_result : child_results) {
    if (pipe(child_result.data()) != 0) {
      for (auto& opened : child_results) {
        if (opened[0] != 0) {
          close(opened[0]);
        }
        if (opened[1] != 0) {
          close(opened[1]);
        }
      }
      return check(false, "failed to create fanout worker result pipe");
    }
  }

  std::array<pid_t, 2U> consumer_pids{};
  for (std::size_t index = 0U; index < consumer_pids.size(); ++index) {
    const pid_t consumer_pid = fork();
    if (consumer_pid < 0) {
      valid = check(false, "failed to fork fanout consumer worker");
      break;
    }
    if (consumer_pid == 0) {
      for (std::size_t pipe_index = 0U; pipe_index < child_results.size(); ++pipe_index) {
        close(child_results[pipe_index][0]);
        if (pipe_index != index) {
          close(child_results[pipe_index][1]);
        }
      }
      const std::string result_fd_argument = std::to_string(child_results[index][1]);
      const std::string target_argument = std::to_string(index);
      execl("/proc/self/exe",
            "partition_environment_process_probe",
            "fanout-worker-consumer",
            result_fd_argument.c_str(),
            target_argument.c_str(),
            static_cast<char*>(nullptr));
      std::_Exit(127);
    }
    consumer_pids[index] = consumer_pid;
  }

  for (auto& child_result : child_results) {
    close(child_result[1]);
  }
  std::array<char, 2U> consumer_ready{};
  if (valid) {
    for (std::size_t index = 0U; index < consumer_ready.size(); ++index) {
      valid &= check(
          read(child_results[index][0], &consumer_ready[index], sizeof(consumer_ready[index])) ==
                  sizeof(consumer_ready[index]) &&
              consumer_ready[index] == 1,
          "fanout consumer did not arm its independently reconstructed endpoint");
    }
  }

  if (valid) {
    RunSession producer = detail::run_async_worker(plan, service.source_process_group);
    const bool producer_started = WorkerFanoutProbeState::wait_for_source();
    const bool emitted = producer_started && WorkerFanoutProbeState::wait_for_emitted();
    std::array<char, 2U> consumer_results{};
    for (std::size_t index = 0U; index < consumer_results.size(); ++index) {
      valid &= check(read(child_results[index][0],
                          &consumer_results[index],
                          sizeof(consumer_results[index])) == sizeof(consumer_results[index]) &&
                         consumer_results[index] == 1,
                     "fanout consumer did not receive the exact sequence");
    }
    producer.request_stop();
    producer.wait();

    // Key assertion: the producer published once per value and both process peers accepted all
    // values.
    valid &= check(emitted && WorkerFanoutProbeState::exact_producer_state(),
                   "fanout producer did not emit the exact sequence");
    valid &=
        check(producer.termination() == RunTermination::kStopped && producer.diagnostics().empty(),
              "fanout producer did not stop cleanly");
  }

  for (auto& child_result : child_results) {
    close(child_result[0]);
  }
  for (const pid_t consumer_pid : consumer_pids) {
    if (consumer_pid <= 0) {
      continue;
    }
    int child_status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(consumer_pid, &child_status, 0);
    } while (waited < 0 && errno == EINTR);
    valid &=
        check(waited == consumer_pid && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
              "exec-isolated fanout consumer failed");
  }
  return valid;
#else
  // The cross-address-space HoloMQ proof currently exercises the Linux process API.
  return true;
#endif
}

// REQ: CHAN-093
// REQ: CHAN-103
[[nodiscard]] bool run_tensor_worker_plan(const MemoryKind storage = MemoryKind::kHost,
                                          const DeviceId device = DeviceId{0},
                                          const DeviceId producer_ambient_device = {}) {
#if defined(__linux__)
  if (storage == MemoryKind::kCudaDevice || producer_ambient_device.valid()) {
    int device_count = 0;
    const DeviceId required_device =
        producer_ambient_device.valid() ? producer_ambient_device : device;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !required_device.valid() ||
        required_device.value >= device_count) {
      if (required_device.value == 1) {
        std::cout << "SKIP: independent-process CUDA Tensor test requires GPU ordinal 1\n";
        return true;
      }
      return check(false, "requested CUDA Tensor test device is unavailable");
    }
  }
  WorkerProbeState::reset();
  const ExecutionPlan compiled = [&] {
    switch (storage) {
      case MemoryKind::kPinnedHost:
        return compile_tensor_worker_plan<MemoryKind::kPinnedHost>();
      case MemoryKind::kCudaDevice:
        return compile_tensor_worker_plan<MemoryKind::kCudaDevice>(device);
      default:
        return compile_tensor_worker_plan<MemoryKind::kHost>();
    }
  }();
  std::shared_ptr<detail::RuntimeTensorOutputTestControl> tensor_output_control;
  ExecutionPlan plan = compiled;
  if (producer_ambient_device.valid()) {
    tensor_output_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, tensor_output_control);
    plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
        plan, std::make_shared<CUDADeviceSupportThreadControl>(producer_ambient_device));
  }
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  bool valid = check(plan.ok(), "producer rejected the Tensor worker plan");
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "producer plan has no unique Tensor service");
  if (!valid) {
    return false;
  }
  const auto& service = state->cross_process_channel_services.front();
  valid &= check(storage != MemoryKind::kCudaDevice ||
                     (service.tensor_memory_kind == MemoryKind::kCudaDevice &&
                      service.tensor_device == device),
                 "producer plan did not retain the requested CUDA device");

  std::array<int, 2U> child_result{};
  if (pipe(child_result.data()) != 0) {
    return check(false, "failed to create Tensor worker result pipe");
  }
  const std::string result_fd_argument = std::to_string(child_result[1]);
  const std::string device_argument = std::to_string(device.value);
  const pid_t consumer_pid = fork();
  if (consumer_pid < 0) {
    close(child_result[0]);
    close(child_result[1]);
    return check(false, "failed to fork Tensor consumer worker");
  }
  if (consumer_pid == 0) {
    close(child_result[0]);
    execl("/proc/self/exe",
          "partition_environment_process_probe",
          storage == MemoryKind::kPinnedHost
              ? "pinned-tensor-worker-consumer"
              : (storage == MemoryKind::kCudaDevice ? "device-tensor-worker-consumer"
                                                    : "tensor-worker-consumer"),
          result_fd_argument.c_str(),
          device_argument.c_str(),
          static_cast<char*>(nullptr));
    std::_Exit(127);
  }

  close(child_result[1]);
  char consumer_ready = 0;
  valid &= check(
      read(child_result[0], &consumer_ready, sizeof(consumer_ready)) == sizeof(consumer_ready) &&
          consumer_ready == 1,
      "Tensor consumer did not arm its independently reconstructed endpoint");
  if (valid) {
    // The source writes directly into the final exportable loan; publication sends its descriptor.
    RunSession producer = detail::run_async_worker(plan, service.source_process_group);
    const bool producer_started =
        WorkerProbeState::wait_for_starts(1U, 0U) &&
        WorkerProbeState::sink_starts.load(std::memory_order_acquire) == 0U;
    char consumer_result = 0;
    const bool result_read =
        read(child_result[0], &consumer_result, sizeof(consumer_result)) == sizeof(consumer_result);
    producer.request_stop();
    producer.wait();
    valid &= check(producer_started, "Tensor producer reconstructed the wrong process group");
    valid &= check(result_read && consumer_result == 1,
                   "exec-isolated Tensor consumer did not import the direct loan");
    valid &=
        check(producer.termination() == RunTermination::kStopped && producer.diagnostics().empty(),
              "Tensor producer did not stop cleanly");
    if (tensor_output_control != nullptr) {
      const auto snapshot = tensor_output_control->snapshot(0U);
      valid &= check(snapshot.holomq_generation_vmm_device_ordinal == 0,
                     "host generation VMM was not created on CUDA device 0");
    }
  }
  close(child_result[0]);

  int child_status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(consumer_pid, &child_status, 0);
  } while (waited < 0 && errno == EINTR);
  valid &=
      check(waited == consumer_pid && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "exec-isolated Tensor consumer failed");
  return valid;
#else
  // The cross-address-space HoloMQ proof currently exercises the Linux process API.
  return true;
#endif
}

[[nodiscard]] bool run_worker_plan(const ExecutionPlan& plan,
                                   std::optional<std::string_view> worker_selector,
                                   std::string_view expected_algorithm,
                                   PlanSelectionSource expected_source,
                                   bool publication_coordinate = false) {
  WorkerProbeState::reset();
  bool valid = set_selector(worker_selector);
  valid &= check_selector(worker_selector);
  valid &= check(plan.ok(), "worker inherited a rejected plan");
  valid &= check(plan.partitioning_plan().disposition == PlanDisposition::kAccepted,
                 "worker changed the accepted partition disposition");
  std::string_view accepted_algorithm;
  if (const auto& accepted = plan.partitioning_plan().accepted_algorithm; accepted.has_value()) {
    accepted_algorithm = accepted.value().id;
  }
  valid &= check(accepted_algorithm == expected_algorithm,
                 "worker reinterpreted the compile-site partition algorithm");
  valid &= check(plan.partitioning_plan().selection_source == expected_source,
                 "worker reinterpreted the compile-site selection provenance");
  valid &= check(plan.scheduler_plan().ready_policy.id == "holoscan.ready.deterministic_serial",
                 "worker changed the accepted READY policy");
  valid &=
      check(plan.scheduler_plan().max_concurrent_activations == plan.execution_partitions().size(),
            "worker changed the accepted activation-concurrency bound");
  if (!valid) {
    return false;
  }

  const auto* state = detail::ExecutionPlanAccess::state(plan);
  valid &= check(state != nullptr && state->cross_process_channel_services.size() == 1U,
                 "worker plan has no unique cross-process service");
  valid &=
      check(state != nullptr && state->clocks.size() == 1U && state->root_default_clock.valid(),
            "producer has no plan-owned default clock");
  if (!valid) {
    return false;
  }
  const auto& service = state->cross_process_channel_services.front();

#if defined(__linux__)
  std::array<int, 2U> child_result{};
  if (pipe(child_result.data()) != 0) {
    return check(false, "failed to create process-group worker result pipe");
  }
  const std::string result_fd_argument = std::to_string(child_result[1]);
  const std::string compile_selector_argument =
      expected_source == PlanSelectionSource::kCoreDefault
          ? std::string{kAbsentSelectorArgument}
          : std::string{expected_algorithm == kSourceChainAlgorithm ? "source_chain/v1"
                                                                    : "serial_baseline/v1"};
  const std::string runtime_selector_argument = worker_selector.has_value()
                                                    ? std::string{*worker_selector}
                                                    : std::string{kAbsentSelectorArgument};
  const pid_t consumer_pid = fork();
  if (consumer_pid < 0) {
    close(child_result[0]);
    close(child_result[1]);
    return check(false, "failed to fork process-group consumer worker");
  }
  if (consumer_pid == 0) {
    close(child_result[0]);
    execl("/proc/self/exe",
          "partition_environment_process_probe",
          "worker-consumer",
          result_fd_argument.c_str(),
          compile_selector_argument.c_str(),
          runtime_selector_argument.c_str(),
          publication_coordinate ? "publication" : "scalar",
          static_cast<char*>(nullptr));
    std::_Exit(127);
  }

  close(child_result[1]);
  char consumer_ready = 0;
  valid &= check(
      read(child_result[0], &consumer_ready, sizeof(consumer_ready)) == sizeof(consumer_ready) &&
          consumer_ready == 1,
      "consumer process-group worker did not reach start");
  if (valid) {
    RunSession producer = detail::run_async_worker(plan, service.source_process_group);
    const bool producer_started =
        WorkerProbeState::wait_for_starts(1U, 0U) &&
        WorkerProbeState::sink_starts.load(std::memory_order_acquire) == 0U;
    char consumer_result = 0;
    const bool result_read =
        read(child_result[0], &consumer_result, sizeof(consumer_result)) == sizeof(consumer_result);
    producer.request_stop();
    producer.wait();
    valid &= check(producer_started, "producer process-group worker reconstructed the wrong role");
    valid &= check(result_read && consumer_result == 1,
                   "consumer process-group worker did not receive the producer sample");
    valid &=
        check(producer.termination() == RunTermination::kStopped && producer.diagnostics().empty(),
              "producer process-group worker did not stop cleanly");
  }
  close(child_result[0]);
  int child_status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(consumer_pid, &child_status, 0);
  } while (waited < 0 && errno == EINTR);
  valid &=
      check(waited == consumer_pid && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "consumer process-group worker failed");
  return valid;
#else
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool received = WorkerProbeState::wait_for_count(2U);
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();
  valid &= check(received, "worker did not receive data through the accepted cross-process edge");
  valid &= check(WorkerProbeState::last.load(std::memory_order_acquire) ==
                     (publication_coordinate ? kPublicationPayload : 0x15U),
                 "worker observed the wrong payload");
  valid &= check(producer.termination() == RunTermination::kStopped &&
                     consumer.termination() == RunTermination::kStopped,
                 "worker termination changed with its local selector");
  valid &= check(producer.diagnostics().empty() && consumer.diagnostics().empty(),
                 "worker produced a diagnostic after the plan was accepted");
  return valid;
#endif
}

#if defined(__linux__)
[[nodiscard]] bool run_worker_child(const ExecutionPlan& plan, std::string_view case_name,
                                    std::optional<std::string_view> worker_selector,
                                    std::string_view expected_algorithm,
                                    PlanSelectionSource expected_source) {
  const pid_t child = fork();
  if (child < 0) {
    return check(false, "failed to fork worker child");
  }
  if (child == 0) {
    const bool valid = run_worker_plan(plan, worker_selector, expected_algorithm, expected_source);
    if (!valid) {
      std::cerr << "worker environment case failed: " << case_name << '\n';
    }
    std::_Exit(valid ? 0 : 1);
  }

  int status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child) {
    return check(false, "failed to reap worker child");
  }
  return check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "worker child failed accepted-plan execution");
}
#endif

// REQ: COMPILE-068
[[nodiscard]] bool run_worker_environment_matrix(std::optional<std::string_view> compile_selector) {
#if defined(__linux__)
  if (!check_selector(compile_selector)) {
    return false;
  }
  if (!check(!compile_selector.has_value(),
             "worker compile must not use the removed environment selector")) {
    return false;
  }
  const ExecutionPlan plan = compile_worker_plan();
  if (!check_valid_plan(plan,
                        kSourceChainAlgorithm,
                        PlanSelectionSource::kCoreDefault,
                        AlgorithmMaturity::kProduction,
                        0U)) {
    return false;
  }

  bool valid = true;
  valid &= run_worker_child(
      plan, "absent", std::nullopt, kSourceChainAlgorithm, PlanSelectionSource::kCoreDefault);
  valid &= run_worker_child(plan,
                            "retired-source-chain",
                            "source_chain/v1",
                            kSourceChainAlgorithm,
                            PlanSelectionSource::kCoreDefault);
  valid &= run_worker_child(plan,
                            "retired-serial-baseline",
                            "serial_baseline/v1",
                            kSourceChainAlgorithm,
                            PlanSelectionSource::kCoreDefault);
  valid &= run_worker_child(plan,
                            "retired-invalid",
                            "definitely-not-an-algorithm",
                            kSourceChainAlgorithm,
                            PlanSelectionSource::kCoreDefault);
  return valid;
#else
  // The installed SDK remains portable; this process-inheritance proof is a
  // Linux HoloMQ integration test because ExecutionPlan has no wire format.
  static_cast<void>(compile_selector);
  return true;
#endif
}

// REQ: RUNTIME-057
[[nodiscard]] bool run_publication_coordinate_plan(bool process_local) {
  if (!set_selector(std::nullopt)) {
    return false;
  }
  // Both cases compile the same clock declaration and Operators. Only placement changes.
  const ExecutionPlan plan = compile_worker_plan(true, process_local);
  if (!process_local) {
    return run_worker_plan(
        plan, std::nullopt, kSourceChainAlgorithm, PlanSelectionSource::kCoreDefault, true);
  }

  WorkerProbeState::reset();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  if (!check(plan.ok() && state != nullptr && state->clocks.size() == 1U &&
                 state->root_default_clock.valid() && state->cross_process_channel_services.empty(),
             "local graph has no plan-owned default clock or crosses processes")) {
    return false;
  }
  RunSession session = run_async(plan);
  const bool received = WorkerProbeState::wait_for_count(2U);
  session.request_stop();
  session.wait();
  return check(received &&
                   WorkerProbeState::last.load(std::memory_order_acquire) == kPublicationPayload &&
                   WorkerProbeState::source_starts.load(std::memory_order_acquire) == 1U &&
                   WorkerProbeState::sink_starts.load(std::memory_order_acquire) == 1U,
               "local compiled graph did not retain publication and capture coordinates") &&
         check(session.termination() == RunTermination::kStopped && session.diagnostics().empty(),
               "local publication-coordinate graph did not stop cleanly");
}

[[nodiscard]] bool run_case(std::string_view scenario) {
  if (scenario == "worker-publication-coordinate") {
    return run_publication_coordinate_plan(false);
  }
  if (scenario == "local-publication-coordinate") {
    return run_publication_coordinate_plan(true);
  }
  if (scenario == "absent") {
    return check_selector(std::nullopt) && check_valid_plan(compile_linear(),
                                                            kSourceChainAlgorithm,
                                                            PlanSelectionSource::kCoreDefault,
                                                            AlgorithmMaturity::kProduction,
                                                            0U);
  }
  if (scenario == "serial-baseline") {
    return check_selector("serial_baseline/v1") && check_removed_selector_plan(compile_linear());
  }
  if (scenario == "source-chain") {
    return check_selector("source_chain/v1") && check_removed_selector_plan(compile_linear());
  }
  if (scenario == "invalid") {
    return check_selector("source_chain") && check_removed_selector_plan(compile_linear());
  }
  if (scenario == "forbidden-profile") {
    return check_selector("source_chain/v1") &&
           check_removed_selector_plan(compile_linear(Profile::kRobotics));
  }
  if (scenario == "worker-default") {
    return run_worker_environment_matrix(std::nullopt);
  }
  if (scenario == "worker-fanout") {
    return run_fanout_worker_plan();
  }
  if (scenario == "worker-tensor") {
    return run_tensor_worker_plan();
  }
  if (scenario == "worker-pinned-tensor") {
    return run_tensor_worker_plan(MemoryKind::kPinnedHost);
  }
  if (scenario == "worker-device-tensor") {
    return run_tensor_worker_plan(MemoryKind::kCudaDevice, DeviceId{0});
  }
  if (scenario == "worker-device-tensor-nonzero") {
    return run_tensor_worker_plan(MemoryKind::kCudaDevice, DeviceId{1});
  }
  if (scenario == "worker-host-tensor-nonzero-ambient") {
    return run_tensor_worker_plan(MemoryKind::kHost, DeviceId{0}, DeviceId{1});
  }
  return check(false, "unknown fresh-process partition environment scenario");
}

}  // namespace
}  // namespace holoscan

int main(int argc, char** argv) {
  if (argc >= 2 && !holoscan::ensure_probe_instance(argv[1])) {
    std::cerr << "failed to set the partition probe instance\n";
    return 2;
  }
#if defined(__linux__)
  if (argc == 4 && std::string_view(argv[1]) == "fanout-worker-consumer") {
    char* end = nullptr;
    const long parsed_fd = std::strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed_fd < 0 ||
        parsed_fd > std::numeric_limits<int>::max()) {
      return 2;
    }
    const int result_fd = static_cast<int>(parsed_fd);
    end = nullptr;
    const long parsed_target = std::strtol(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || parsed_target < 0 || parsed_target >= 2) {
      close(result_fd);
      return 2;
    }
    const bool valid =
        holoscan::run_fanout_consumer_worker(result_fd, static_cast<std::size_t>(parsed_target));
    close(result_fd);
    return valid ? 0 : 1;
  }
  if (argc == 4 && (std::string_view(argv[1]) == "tensor-worker-consumer" ||
                    std::string_view(argv[1]) == "pinned-tensor-worker-consumer" ||
                    std::string_view(argv[1]) == "device-tensor-worker-consumer")) {
    char* end = nullptr;
    const long parsed_fd = std::strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed_fd < 0 ||
        parsed_fd > std::numeric_limits<int>::max()) {
      return 2;
    }
    const int result_fd = static_cast<int>(parsed_fd);
    end = nullptr;
    const long parsed_device = std::strtol(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || parsed_device < 0 ||
        parsed_device > std::numeric_limits<std::int32_t>::max()) {
      close(result_fd);
      return 2;
    }
    const holoscan::MemoryKind storage =
        std::string_view(argv[1]) == "pinned-tensor-worker-consumer"
            ? holoscan::MemoryKind::kPinnedHost
            : (std::string_view(argv[1]) == "device-tensor-worker-consumer"
                   ? holoscan::MemoryKind::kCudaDevice
                   : holoscan::MemoryKind::kHost);
    const bool valid = holoscan::run_tensor_consumer_worker(
        result_fd, storage, holoscan::DeviceId{static_cast<std::int32_t>(parsed_device)});
    close(result_fd);
    return valid ? 0 : 1;
  }
  if (argc == 6 && std::string_view(argv[1]) == "worker-consumer") {
    char* end = nullptr;
    const long parsed_fd = std::strtol(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed_fd < 0 ||
        parsed_fd > std::numeric_limits<int>::max()) {
      return 2;
    }
    const int result_fd = static_cast<int>(parsed_fd);
    const bool valid = holoscan::run_consumer_worker(result_fd,
                                                     holoscan::selector_from_argument(argv[3]),
                                                     holoscan::selector_from_argument(argv[4]),
                                                     std::string_view(argv[5]) == "publication");
    close(result_fd);
    return valid ? 0 : 1;
  }
#endif
  if (argc != 2) {
    std::cerr << "expected exactly one scenario argument\n";
    return 2;
  }
  return holoscan::run_case(argv[1]) ? 0 : 1;
}
