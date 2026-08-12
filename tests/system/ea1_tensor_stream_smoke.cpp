/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for end-to-end Tensor delivery through a three-operator in-process graph.
// Verifies that the runtime allocates, routes, and streams device Tensors correctly:
// a source fills a device buffer, a filter copies it device-to-device on the shared
// runtime stream, and a sink transfers to host on a separate stream and checks the bytes.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kTensorBytes = 256U;
constexpr std::uint8_t kExpectedByte = 0x2AU;
constexpr int kSkipped = 77;

constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr holoscan::TensorPortLayout kDeviceByteVector{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kByteDtype,
    .rank = 1U,
};

// SmokeState: static rendez-vous shared across all operator invocations within one run.
// Operators write stream handles and received bytes here; main() reads after the session stops.
struct SmokeState {
  static void reset(int requested_device) {
    const std::lock_guard lock(mutex);
    received.fill(0U);
    failure.reset();
    sink_complete = false;
    expected_device.store(requested_device, std::memory_order_relaxed);
    producer_device.store(-1, std::memory_order_relaxed);
    filter_device.store(-1, std::memory_order_relaxed);
    consumer_device.store(-1, std::memory_order_relaxed);
    producer_stream.store(0U, std::memory_order_relaxed);
    filter_stream.store(0U, std::memory_order_relaxed);
    consumer_stream.store(0U, std::memory_order_relaxed);
  }

  static void fail(holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      if (!failure.has_value()) {
        failure = code;
      }
    }
    cv.notify_all();
  }

  static void record(std::span<const std::uint8_t, kTensorBytes> values) {
    {
      const std::lock_guard lock(mutex);
      std::copy(values.begin(), values.end(), received.begin());
      sink_complete = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return sink_complete || failure.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::array<std::uint8_t, kTensorBytes> received{};
  static inline std::optional<holoscan::ErrorCode> failure;
  static inline bool sink_complete{};
  static inline std::atomic<int> expected_device{};
  static inline std::atomic<int> producer_device{-1};
  static inline std::atomic<int> filter_device{-1};
  static inline std::atomic<int> consumer_device{-1};
  static inline std::atomic<std::uintptr_t> producer_stream{};
  static inline std::atomic<std::uintptr_t> filter_stream{};
  static inline std::atomic<std::uintptr_t> consumer_stream{};
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> fail(
    holoscan::ErrorCode code = holoscan::ErrorCode::kFailure) {
  SmokeState::fail(code);
  return holoscan::make_unexpected(holoscan::Error{code});
}

// Test-local equivalent of the source-private CUDA device guards used by the
// Tensor runtime. It keeps stream creation, use, and destruction in the
// stream's owning CUDA context without changing the caller's active device.
class ScopedCUDADevice final {
 public:
  explicit ScopedCUDADevice(int requested) noexcept {
    if (requested < 0 || cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested) {
      if (cudaSetDevice(requested) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ScopedCUDADevice(const ScopedCUDADevice&) = delete;
  ScopedCUDADevice& operator=(const ScopedCUDADevice&) = delete;

  ~ScopedCUDADevice() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  bool active_{};
  bool changed_{};
};

// TensorSource: fires once on a 1 ms clock tick, allocates a kTensorBytes device Tensor,
// fills it with kExpectedByte via cudaMemsetAsync on the runtime stream, and emits it.
// Records the runtime CUDA stream handle so main() can verify stream identity later.
class TensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(kDeviceByteVector)
        .tensor_allocation(
            holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kTensorBytes});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;

    int active_device = -1;
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr || cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != SmokeState::expected_device.load(std::memory_order_acquire)) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    SmokeState::producer_device.store(active_device, std::memory_order_release);
    SmokeState::producer_stream.store(reinterpret_cast<std::uintptr_t>(stream),
                                      std::memory_order_release);

    constexpr std::array<std::int64_t, 1U> kShape{static_cast<std::int64_t>(kTensorBytes)};
    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = kShape, .dtype = kByteDtype});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto committed = loan->write_and_commit(
        context.cuda_stream(),
        [stream](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          if (writer.byte_size() != kTensorBytes ||
              cudaMemsetAsync(writer.data(), kExpectedByte, kTensorBytes, stream) != cudaSuccess) {
            return fail();
          }
          return {};
        });
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan),
                               holoscan::EmitOptions{.capture_time = context.activation_time()});
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  holoscan::Output<holoscan::Tensor> output;

 private:
  bool emitted_{};
};

// TensorFilter: receives the source Tensor, validates its shape and dtype, then copies
// device-to-device into a freshly allocated output Tensor on the same runtime stream.
// Propagates the upstream capture time so the downstream sink can correlate timestamps.
class TensorFilter final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(kDeviceByteVector);
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(kDeviceByteVector)
        .tensor_allocation(
            holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kTensorBytes});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    int active_device = -1;
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr || cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != SmokeState::expected_device.load(std::memory_order_acquire)) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    SmokeState::filter_device.store(active_device, std::memory_order_release);
    SmokeState::filter_stream.store(reinterpret_cast<std::uintptr_t>(stream),
                                    std::memory_order_release);

    auto sample = input.receive();
    if (!sample) {
      return fail(sample.error().code);
    }
    const auto shape = sample->data.shape_span();
    const DLDataType dtype = sample->data.dtype();
    if (shape.size() != 1U || shape.front() != static_cast<std::int64_t>(kTensorBytes) ||
        dtype.code != kByteDtype.code || dtype.bits != kByteDtype.bits ||
        dtype.lanes != kByteDtype.lanes || sample->data.nbytes() != kTensorBytes ||
        sample->data.data() == nullptr) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }

    constexpr std::array<std::int64_t, 1U> kShape{static_cast<std::int64_t>(kTensorBytes)};
    auto loan =
        output.allocate_tensor(holoscan::TensorLoanRequest{.shape = kShape, .dtype = kByteDtype});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return fail(writer.error().code);
    }
    if (cudaMemcpyAsync(
            writer->data(), sample->data.data(), kTensorBytes, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess) {
      return fail();
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(
        std::move(*loan), holoscan::EmitOptions{.capture_time = sample->metadata.capture_time()});
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;
};

// TensorSink: receives the filtered Tensor and copies it to host using its own non-blocking
// CUDA stream (distinct from the runtime stream) after verifying the readiness-stream
// contract. Synchronizes the stream then stores the bytes in SmokeState for validation.
class TensorSink final : public holoscan::Operator<> {
 public:
  ~TensorSink() override {
    if (consumer_stream_ != nullptr) {
      const ScopedCUDADevice selected{consumer_device_};
      if (selected.active()) {
        static_cast<void>(cudaStreamDestroy(consumer_stream_));
      }
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(kDeviceByteVector);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    cudaStream_t runtime_stream = context.cuda_stream().get();
    if (runtime_stream == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }

    auto sample = input.receive();
    if (!sample) {
      return fail(sample.error().code);
    }
    const DLDevice device = sample->data.device();
    int active_device = -1;
    if (device.device_type != kDLCUDA || device.device_id < 0 ||
        cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != SmokeState::expected_device.load(std::memory_order_acquire) ||
        device.device_id != active_device) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    if ((consumer_stream_ == nullptr &&
         cudaStreamCreateWithFlags(&consumer_stream_, cudaStreamNonBlocking) != cudaSuccess) ||
        runtime_stream == consumer_stream_) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    consumer_device_ = active_device;
    SmokeState::consumer_stream.store(reinterpret_cast<std::uintptr_t>(consumer_stream_),
                                      std::memory_order_release);
    SmokeState::consumer_device.store(active_device, std::memory_order_release);
    if (!sample->data.has_readiness_stream_contract()) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }

    std::array<std::uint8_t, kTensorBytes> host{};
    auto copied = sample->data.copy_to_host_async(host.data(), host.size(), consumer_stream_);
    if (!copied) {
      return fail(copied.error().code);
    }
    if (cudaStreamSynchronize(consumer_stream_) != cudaSuccess) {
      return fail();
    }
    SmokeState::record(host);
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;

 private:
  cudaStream_t consumer_stream_{};
  int consumer_device_{-1};
};

// Reusable device-Tensor pipeline. Its temporal contracts intentionally depend
// on an imported default clock so each embedding must bind the exact child
// instance to an explicit parent clock.
class EmbeddedTensorPipeline final : public holoscan::Graph {
 public:
  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    auto source = op<TensorSource>("source");
    auto filter = op<TensorFilter>("filter");
    auto sink = op<TensorSink>("sink");
    add_flow(source->output, filter->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    add_flow(filter->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  }

 private:
  holoscan::ClockKey clock_import_{};
};

[[nodiscard]] holoscan::ExecutionPlan compile_tensor_graph(bool embedded, int device,
                                                           holoscan::Profile profile) {
  holoscan::Graph graph{embedded ? "ea1-embedded-tensor-stream" : "ea1-flat-tensor-stream"};
  holoscan::CompileOptions options;
  options.profile = profile;
  const holoscan::DeviceId placement{device};

  if (embedded) {
    auto child = graph.embed<EmbeddedTensorPipeline>("pipeline");
    const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    graph.bind_clock(child, child->clock_import(), clock);
    options.deployment
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "/pipeline/source", .output_port = "output", .device = placement})
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "/pipeline/filter", .output_port = "output", .device = placement});
  } else {
    const auto source = graph.op<TensorSource>("source");
    const auto filter = graph.op<TensorFilter>("filter");
    const auto sink = graph.op<TensorSink>("sink");
    graph.add_flow(source->output, filter->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(filter->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
    graph.set_default_clock(clock);
    options.deployment
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "source", .output_port = "output", .device = placement})
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "filter", .output_port = "output", .device = placement});
  }
  return holoscan::compile(graph, options);
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " vertex=" << diagnostic.source.vertex.value
              << " port=" << diagnostic.source.port.value() << " service=" << diagnostic.service
              << " required=" << diagnostic.required << " provided=" << diagnostic.provided << '\n';
  }
}

[[nodiscard]] int run_tensor_plan(const holoscan::ExecutionPlan& plan, int device) {
  SmokeState::reset(device);
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool completed = SmokeState::wait();
  session.request_stop();
  session.wait();

  if (session.termination() != holoscan::RunTermination::kStopped) {
    print_diagnostics(session);
    return 2;
  }
  if (!session.diagnostics().empty()) {
    print_diagnostics(session);
    return 3;
  }

  const std::lock_guard lock(SmokeState::mutex);
  const bool correct = std::all_of(SmokeState::received.begin(),
                                   SmokeState::received.end(),
                                   [](std::uint8_t value) { return value == kExpectedByte; });
  const std::uintptr_t producer_stream =
      SmokeState::producer_stream.load(std::memory_order_acquire);
  const std::uintptr_t filter_stream = SmokeState::filter_stream.load(std::memory_order_acquire);
  const std::uintptr_t consumer_stream =
      SmokeState::consumer_stream.load(std::memory_order_acquire);
  const bool content_and_streams_match =
      completed && !SmokeState::failure.has_value() && SmokeState::sink_complete && correct &&
      producer_stream != 0U && filter_stream != 0U && consumer_stream != 0U &&
      producer_stream == filter_stream && producer_stream != consumer_stream &&
      SmokeState::producer_device.load(std::memory_order_acquire) == device &&
      SmokeState::filter_device.load(std::memory_order_acquire) == device &&
      SmokeState::consumer_device.load(std::memory_order_acquire) == device;
  return content_and_streams_match ? 0 : 4;
}

}  // namespace

// REQ: OPERATOR-010
// REQ: COMPILE-053
// REQ: COMPILE-054
// REQ: RUNTIME-027
// REQ: COMPILE-048
// REQ: OPERATOR-025
// REQ: TENSOR-CORE-013
//
// Graph topology:
//
//   TensorSource ──[Tensor]──> TensorFilter ──[Tensor]──> TensorSink
//   (cudaMemset,               (D→D copy,                 (D→H copy,
//    runtime stream)            runtime stream)            own stream)
//
// The runtime stream must be shared between Source and Filter (same handle), while
// the Sink must use a distinct non-blocking stream to prove stream-contract isolation.
int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    std::cout << "SKIP: no usable CUDA device\n";
    return kSkipped;  // exit 77: no usable CUDA device present in this environment
  }

  const int tested_device_count = std::min(device_count, 2);
  for (int device = 0; device < tested_device_count; ++device) {
    for (const holoscan::Profile profile :
         {holoscan::Profile::kStandard, holoscan::Profile::kRobotics}) {
      for (const bool embedded : {false, true}) {
        const holoscan::ExecutionPlan plan = compile_tensor_graph(embedded, device, profile);
        const std::string_view plan_json = plan.json();
        const bool embedded_identity =
            !embedded || (plan_json.find("\"clock_import_resolutions\"") != std::string::npos &&
                          plan_json.find("pipeline") != std::string::npos);
        if (!plan.ok() || plan.tensor_allocations().size() != 2U ||
            plan.tensor_device_resources().size() != 1U || !embedded_identity ||
            plan.digest() == holoscan::PlanDigest{}) {
          std::cerr << plan_json << '\n';
          return 1;
        }
        // One immutable plan must reconstruct session-local Operator instances
        // and materialize CUDA resources without retaining authoring declarations.
        int run_result = run_tensor_plan(plan, device);
        if (run_result == 0) {
          run_result = run_tensor_plan(plan, device);
        }
        if (run_result != 0) {
          std::cerr << "tensor replay failed: embedded=" << embedded
                    << " profile=" << static_cast<unsigned int>(profile) << " device=" << device
                    << '\n';
          return run_result;
        }
      }
    }
  }
  return 0;
}
