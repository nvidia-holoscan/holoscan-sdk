/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// tensors_in_depth — the device-pool tensor story: stream-ordered writes, pre-ordered
// receive with zero synchronization calls, fanout release-stream behavior, one explicit
// host-inspection boundary at the edge, and immediate drop on pool exhaustion.
//
//                          +--[kBlocking, qd 8]--> device_consumer  (device work on the
//   source --(fanout)------+                        runtime stream, NO sync calls)
//   (device pool,          +--[kDropOldest, qd 1]--> inspector     (copy_to_host_async on a
//    cudaMemsetAsync)                               private stream, ONE final sync)
//
// Requires a CUDA device; without one the program prints a skip marker and exits 77.

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr std::size_t kFrameBytes = 256U;
constexpr std::array<std::int64_t, 1U> kFrameShape{static_cast<std::int64_t>(kFrameBytes)};
// Logical facts every consumer may rely on: CUDA device memory, uint8, rank 1.
constexpr holoscan::TensorRepresentation kDeviceByteFrame{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kByteDtype,
    .rank = 1U,
};
constexpr std::size_t kFrameGoal = 6U;    // frames the device consumer must see
constexpr std::size_t kInspectGoal = 3U;  // frames the edge inspector stages to host
constexpr std::size_t kDropGoal = 2U;     // kResourceExhausted refusals to demonstrate
constexpr std::size_t kMaxHeldLoans = 64U;
constexpr int kNoCudaDeviceReturnCode = 77;

struct InspectionResources;

// Operators record what happened here; main() prints the summary only after the
// session has fully stopped. graph.op<> takes no pointers, so shared observation
// state is static by design (same pattern as tensors_101 / slow_fast_fusion).
struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    device_frames = 0U;
    readiness_ok = true;
    inspection_done = false;
    release_registration_ok = false;
    inspected_values.clear();
    drops.clear();
    producer_stream.store(0U, std::memory_order_relaxed);
    device_consumer_stream.store(0U, std::memory_order_relaxed);
    inspector_private_stream.store(0U, std::memory_order_relaxed);
  }

  static void record_device_frame(bool readiness_contract) {
    {
      const std::lock_guard lock(mutex);
      ++device_frames;
      readiness_ok = readiness_ok && readiness_contract;
    }
    ready.notify_all();
  }

  static void record_inspection(std::vector<std::uint8_t> values, bool release_registered) {
    {
      const std::lock_guard lock(mutex);
      inspected_values = std::move(values);
      release_registration_ok = release_registered;
      inspection_done = true;
    }
    ready.notify_all();
  }

  static void record_drop(std::uint32_t frame_id, holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      drops.emplace_back(frame_id, code);
    }
    ready.notify_all();
  }

  // The producer switches from streaming to the exhaustion demonstration only
  // after both consumers have proven their part of the lesson.
  [[nodiscard]] static bool streaming_complete() {
    const std::lock_guard lock(mutex);
    return device_frames >= kFrameGoal && inspection_done;
  }

  [[nodiscard]] static bool wait_for_all() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] {
      return device_frames >= kFrameGoal && inspection_done && drops.size() >= kDropGoal;
    });
  }

  // Published before run_async and cleared after the session has joined.
  static inline InspectionResources* inspection_resources{};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t device_frames{};
  static inline bool readiness_ok{true};
  static inline bool inspection_done{};
  static inline bool release_registration_ok{};
  static inline std::vector<std::uint8_t> inspected_values;
  static inline std::vector<std::pair<std::uint32_t, holoscan::ErrorCode>> drops;
  static inline std::atomic<std::uintptr_t> producer_stream{};
  static inline std::atomic<std::uintptr_t> device_consumer_stream{};
  static inline std::atomic<std::uintptr_t> inspector_private_stream{};
};

// Release-stream registration borrows the stream; it does not transfer ownership.
// Keep both the stream and asynchronous-copy destinations alive until session
// cleanup has released Tensor handles and enqueued their deferred deallocation.
struct InspectionResources {
  InspectionResources() = default;
  InspectionResources(const InspectionResources&) = delete;
  InspectionResources& operator=(const InspectionResources&) = delete;
  InspectionResources(InspectionResources&&) = delete;
  InspectionResources& operator=(InspectionResources&&) = delete;

  ~InspectionResources() {
    Observation::inspection_resources = nullptr;
    if (private_stream != nullptr) {
      // Early termination may precede the inspector's final copy/synchronization.
      if (!synchronization_completed) {
        static_cast<void>(cudaStreamSynchronize(private_stream));
      }
      static_cast<void>(cudaStreamDestroy(private_stream));
    }
  }

  cudaStream_t private_stream{};
  std::array<std::array<std::uint8_t, kFrameBytes>, kInspectGoal> staging{};
  bool synchronization_completed{};
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> fail(
    holoscan::ErrorCode code = holoscan::ErrorCode::kFailure) {
  return holoscan::make_unexpected(holoscan::Error{code});
}

// PatternSource: fills one device frame per tick with cudaMemsetAsync on the
// runtime compute stream. Commit records the readiness event on that stream —
// no flush, no sync — and emit publishes. After streaming completes it pins pool
// slots on purpose to show that an exhausted pool refuses immediately.
class PatternSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // The output spec requests a bounded plan-owned CUDA device pool. The
    // device itself is bound at compile time — never inferred.
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kDeviceByteFrame,
            .bounds = holoscan::tensor_bounds(kFrameBytes),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 2ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (exhaust_demo_done_) {
      return {};  // both lessons shown; idle until main() stops the session
    }

    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    Observation::producer_stream.store(reinterpret_cast<std::uintptr_t>(stream),
                                       std::memory_order_release);

    if (!Observation::streaming_complete()) {
      return emit_one_frame(context, stream);
    }
    return exhaust_pool_once();
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_one_frame(
      holoscan::ExecutionContext& context, cudaStream_t stream) {
    const std::uint32_t frame_id = frame_id_ + 1U;
    const int fill_byte = 1 + static_cast<int>((frame_id - 1U) % 255U);  // never zero

    // Step 1: borrow one preallocated device slot. Never blocks, never mallocs.
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
        return {};  // retry this logical frame after earlier slots finish retiring
      }
      return holoscan::make_unexpected(std::move(loan).error());
    }

    // Step 2: a device write phase is stream-ordered. The runtime requires the
    // loan's write stream to be this operator's runtime compute stream.
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != kFrameBytes ||
        cudaMemsetAsync(writer->data(), fill_byte, kFrameBytes, stream) != cudaSuccess) {
      return fail();
    }

    // Step 3: commit records the readiness event on the write stream and nothing
    // else — it never synchronizes and never publishes.
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }

    // Step 4: publication stays explicit and separate from commit.
    auto published = frame.emit(std::move(*loan),
                                holoscan::EmitOptions{.capture_time = context.activation_time()});
    if (!published) {
      if (published.error().code == holoscan::ErrorCode::kBackpressured) {
        // A blocking edge refuses immediately when full. This terminal attempt
        // consumed the loan; retry the logical frame with a fresh loan next tick.
        return {};
      }
      return holoscan::make_unexpected(std::move(published).error());
    }
    frame_id_ = frame_id;
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> exhaust_pool_once() {
    const std::uint32_t frame_id = ++frame_id_;

    // Model a backlog of in-flight frames: hold borrowed loans instead of
    // publishing them. Each live loan pins one pool slot.
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
        // Drop-and-continue: the refusal is immediate — there is no blocking
        // loan variant to accidentally stall a real-time pipeline on.
        Observation::record_drop(frame_id, loan.error().code);
        if (++drops_seen_ >= kDropGoal) {
          exhaust_demo_done_ = true;
        }
        return {};
      }
      return holoscan::make_unexpected(std::move(loan).error());
    }
    if (held_loans_.size() >= kMaxHeldLoans) {
      return fail(holoscan::ErrorCode::kResourceExhausted);  // pool larger than the demo expects
    }
    held_loans_.push_back(std::move(*loan));
    return {};
  }

  std::uint32_t frame_id_{};
  std::size_t drops_seen_{};
  bool exhaust_demo_done_{};
  std::vector<holoscan::TensorOutputLoan> held_loans_;
};

// DeviceConsumer: proves pre-ordered receive. Before compute() runs, the runtime
// has already enqueued a wait on this operator's compute stream against the
// producer's readiness event — so device work enqueued here simply queues behind
// the producer's writes. This callback contains ZERO synchronization calls.
class DeviceConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame")
        .queue_depth(8U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kDeviceByteFrame});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    // One stream per process group and device: this must be the same handle the
    // producer wrote on. main() verifies the recorded identities after the run.
    Observation::device_consumer_stream.store(reinterpret_cast<std::uintptr_t>(stream),
                                              std::memory_order_release);

    auto sample = frame.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = sample->data;
    const auto shape = tensor.shape_span();
    const DLDataType dtype = tensor.dtype();
    const DLDevice device = tensor.device();
    if (shape.size() != 1U || shape.front() != kFrameShape.front() ||
        dtype.code != kByteDtype.code || dtype.bits != kByteDtype.bits ||
        dtype.lanes != kByteDtype.lanes ||
        tensor.nbytes() != static_cast<std::int64_t>(kFrameBytes) || tensor.data() == nullptr ||
        device.device_type != kDLCUDA) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }

    // Stand-in for a kernel launch: stream-ordered scratch + device-to-device
    // copy + stream-ordered free. All three enqueue and return — no host block.
    void* scratch = nullptr;
    if (cudaMallocAsync(&scratch, kFrameBytes, stream) != cudaSuccess) {
      return fail();
    }
    const bool copied =
        cudaMemcpyAsync(scratch, tensor.data(), kFrameBytes, cudaMemcpyDeviceToDevice, stream) ==
        cudaSuccess;
    if (cudaFreeAsync(scratch, stream) != cudaSuccess || !copied) {
      return fail();
    }

    Observation::record_device_frame(tensor.has_readiness_stream_contract());
    return {};
  }

  holoscan::Input<holoscan::Tensor> frame;
};

// EdgeInspector: the one place host access is allowed. It stages asynchronous
// device-to-host copies on its own non-blocking stream, registers that stream
// for release tracking, and synchronizes after the LAST staged copy — one final
// sync on the normal path, never per frame. Resources outlive session cleanup.
class EdgeInspector final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kDeviceByteFrame});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    // Freshest-only tap idiom: trigger on the data itself. The kDropOldest depth-1
    // edge already provides the tap semantics -- if this operator falls
    // behind, newer frames replace queued ones instead of backpressuring the
    // device path. OnEach is intentional because every retained arrival may
    // start inspection; OnClock with a selected input is also supported when
    // periodic sampling is the desired contract.
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = frame.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (staged_ >= kInspectGoal) {
      return {};  // inspection finished; keep draining the freshest-only edge quietly
    }

    auto& resources = *Observation::inspection_resources;
    cudaStream_t runtime_stream = context.cuda_stream().get();
    if (resources.private_stream == nullptr &&
        cudaStreamCreateWithFlags(&resources.private_stream, cudaStreamNonBlocking) !=
            cudaSuccess) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    if (resources.private_stream == runtime_stream) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    Observation::inspector_private_stream.store(
        reinterpret_cast<std::uintptr_t>(resources.private_stream), std::memory_order_release);

    holoscan::Tensor& tensor = sample->data;
    if (tensor.device().device_type != kDLCUDA ||
        tensor.nbytes() != static_cast<std::int64_t>(kFrameBytes)) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }

    // Enqueue: copy_to_host_async first makes the private stream wait on the
    // tensor's readiness event, then enqueues the D2H copy. No host block.
    auto& slot = resources.staging[staged_];
    auto copied = tensor.copy_to_host_async(slot.data(), slot.size(), resources.private_stream);
    if (!copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    // Release-stream registration: the pool may recycle this slot only after
    // work on the registered stream — including our pending copy — completes.
    release_registered_ = tensor.set_deallocation_stream(resources.private_stream) &&
                          (staged_ == 0U || release_registered_);
    ++staged_;

    if (staged_ < kInspectGoal) {
      return {};
    }

    // The normal path has one synchronization: one blocking wait on the
    // private stream, after the last staged copy, off the steady-state path.
    if (cudaStreamSynchronize(resources.private_stream) != cudaSuccess) {
      return fail();
    }
    resources.synchronization_completed = true;
    std::vector<std::uint8_t> values;
    values.reserve(kInspectGoal);
    for (const auto& buffer : resources.staging) {
      const std::uint8_t first = buffer.front();
      for (const std::uint8_t byte : buffer) {
        if (byte != first) {
          return fail(holoscan::ErrorCode::kInvalidArgument);  // torn frame
        }
      }
      if (first == 0U || (!values.empty() && first <= values.back())) {
        return fail(holoscan::ErrorCode::kInvalidArgument);  // stale or out of order
      }
      values.push_back(first);
    }
    Observation::record_inspection(std::move(values), release_registered_);
    return {};
  }

  holoscan::Input<holoscan::Tensor> frame;

 private:
  std::size_t staged_{};
  bool release_registered_{};
};

}  // namespace

int main() {
  // CUDA gate: this example teaches the device-pool path, so it needs a device.
  // Without one it reports CTest's conventional skipped return code.
  int device_count = 0;
  const cudaError_t probe = cudaGetDeviceCount(&device_count);
  if (probe == cudaErrorNoDevice || probe == cudaErrorInsufficientDriver ||
      (probe == cudaSuccess && device_count == 0)) {
    std::cout << "tensors_in_depth: skipped: no CUDA device\n";
    return kNoCudaDeviceReturnCode;
  }
  if (probe != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(probe) << '\n';
    return 1;
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    std::cout << "tensors_in_depth: skipped: no CUDA device\n";  // device present but unusable
    return kNoCudaDeviceReturnCode;
  }

  Observation::reset();
  InspectionResources inspection_resources;
  Observation::inspection_resources = &inspection_resources;

  holoscan::Graph graph{"tensors-in-depth"};
  const auto source = graph.op<PatternSource>("source");
  const auto device_consumer = graph.op<DeviceConsumer>("device_consumer");
  const auto inspector = graph.op<EdgeInspector>("inspector");
  // Fanout: one emitted tensor, two consumers, one release stream per consumer.
  // The device path keeps every accepted frame (kBlocking, deep queue); the observational
  // inspection edge keeps only the newest sample (kDropOldest, depth 1) so the
  // slower host-staging tap can never backpressure the device pipeline.
  graph.add_flow(
      source->frame, device_consumer->frame, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->frame,
                 inspector->frame,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // CUDA-backed pools require an explicit device binding: the compiler never
  // silently infers device zero.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "frame",
      .device = holoscan::DeviceId{0},
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = Observation::wait_for_all();
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "tensors_in_depth: run did not complete cleanly\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic: " << diagnostic.token << '\n';
    }
    return 2;
  }

  // The session is fully stopped: observation state is stable and these lines
  // cannot interleave with operator activity.
  const std::lock_guard lock(Observation::mutex);
  const std::uintptr_t producer = Observation::producer_stream.load(std::memory_order_acquire);
  const std::uintptr_t consumer =
      Observation::device_consumer_stream.load(std::memory_order_acquire);
  const std::uintptr_t private_stream =
      Observation::inspector_private_stream.load(std::memory_order_acquire);
  if (producer == 0U || producer != consumer || private_stream == 0U ||
      private_stream == producer) {
    std::cerr << "tensors_in_depth: stream identity check failed\n";
    return 3;
  }
  std::cout << "shared runtime stream: producer == device consumer (one per process group + "
               "device)\n";
  if (!Observation::readiness_ok || Observation::device_frames < kFrameGoal) {
    std::cerr << "tensors_in_depth: device consumer verification failed\n";
    return 3;
  }
  std::cout << "device consumer: " << Observation::device_frames
            << " frames received pre-ordered, 0 synchronization calls\n";
  if (!Observation::release_registration_ok ||
      Observation::inspected_values.size() != kInspectGoal) {
    std::cerr << "tensors_in_depth: edge inspection failed\n";
    return 3;
  }
  std::cout << "inspector: staged bytes";
  for (const std::uint8_t value : Observation::inspected_values) {
    std::cout << ' ' << static_cast<unsigned>(value);
  }
  std::cout << " (uniform, increasing) via copy_to_host_async + one final sync\n";
  std::cout << "inspector: private release stream registered with the pool\n";
  for (std::size_t index = 0; index < kDropGoal; ++index) {
    const auto& [frame_id, code] = Observation::drops[index];
    if (code != holoscan::ErrorCode::kResourceExhausted) {
      std::cerr << "tensors_in_depth: unexpected drop code " << holoscan::error_code_name(code)
                << '\n';
      return 3;
    }
    std::cout << "frame " << frame_id << " dropped: " << holoscan::error_code_name(code) << '\n';
  }
  std::cout << "tensors_in_depth: device tour complete (frames=" << Observation::device_frames
            << ", inspected=" << kInspectGoal << ", drops=" << kDropGoal << ")\n";
  return 0;
}
