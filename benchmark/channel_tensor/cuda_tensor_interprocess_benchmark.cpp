/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Matched, role-aware native graph probe: manual publisher -> IPC -> graph source -> GPU readers.
#include <cuda_runtime_api.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <holomq/context_builder.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/holomq_tensor.hpp"

namespace {
using namespace holoscan;
using SteadyClock = std::chrono::steady_clock;

// A pinned payload is registered host memory, whose device address equals its host address only
// where cudaDevAttrCanUseHostPointerForRegisteredMem is set. Device allocations resolve to
// themselves.
[[nodiscard]] void* device_address(void* pointer) noexcept {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return pointer;
  }
  return attributes.devicePointer != nullptr ? attributes.devicePointer : pointer;
}
struct Probe {
  std::atomic_size_t completed{};
  std::atomic_bool failed{};
};
struct BufferOwner {
  void* pointer{};
  ~BufferOwner() {
    if (pointer != nullptr) {
      static_cast<void>(cudaFreeHost(pointer));
    }
  }
};
struct StreamOwner {
  cudaStream_t stream{};
  ~StreamOwner() {
    if (stream != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream));
      static_cast<void>(cudaStreamDestroy(stream));
    }
  }
};
class Source final : public Operator<> {
 public:
  Source(std::shared_ptr<ChannelSubscriber<Tensor>> subscriber, MemoryKind memory)
      : subscriber_(std::move(subscriber)),
        memory_(memory),
        synchronize_(std::getenv("HOLOSCAN_BENCHMARK_SYNC_CONSUMERS") != nullptr) {}
  void setup(OperatorSpec& spec) override {
    spec.output(out, "out")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = memory_, .dtype = DLDataType{kDLUInt, 8U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 64U},
            .storage = TensorOutputStorage::kPassThrough});
    if (memory_ == MemoryKind::kPinnedHost) {
      spec.output(placement, "placement")
          .produces_tensor(
              TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                                  .dtype = DLDataType{kDLUInt, 8U, 1U}},
                               .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 64U},
                               .storage = TensorOutputStorage::kRuntimePool});
    }
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnClock{.period = std::chrono::microseconds{1}});
    return c;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    // HoloMQ pinned imports use driver APIs; initialize the callback thread's primary context.
    if (!initialized_) {
      if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      initialized_ = true;
    }
    if (!pending_) {
      auto received = subscriber_->receive_sample();
      if (!received) {
        return received.error().code == ErrorCode::kNotReady
                   ? expected<void, Error>{}
                   : make_unexpected(std::move(received.error()));
      }
      pending_.emplace(std::move(*received));
      // The control also proves producer readiness before fanout. Older imports
      // enqueue the readiness wait only on their first consumer's stream.
      if (synchronize_ &&
          (!pending_->data.synchronize_for_consumer_stream(context.cuda_stream().get()) ||
           cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess)) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    auto sent = out.emit(*pending_);
    if (!sent && sent.error().code == ErrorCode::kBackpressured) {
      return {};
    }
    if (sent) {
      pending_.reset();
    }
    return sent;
  }
  Output<Tensor> out;
  Output<Tensor> placement;

 private:
  std::shared_ptr<ChannelSubscriber<Tensor>> subscriber_;
  MemoryKind memory_;
  bool initialized_{};
  bool synchronize_{};
  std::optional<Sample<Tensor>> pending_;
};
class Reader final : public Operator<> {
 public:
  Reader(std::shared_ptr<Probe> probe, std::shared_ptr<BufferOwner> output,
         std::shared_ptr<StreamOwner> caller, bool pinned)
      : probe_(std::move(probe)),
        output_(std::move(output)),
        caller_(std::move(caller)),
        pinned_(pinned),
        synchronize_(std::getenv("HOLOSCAN_BENCHMARK_SYNC_CONSUMERS") != nullptr) {
    if (cudaMalloc(&device_output_, 64U) != cudaSuccess) {
      throw std::runtime_error("GPU output allocation failed");
    }
  }
  ~Reader() override { static_cast<void>(cudaFree(device_output_)); }
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(3U);
    if (pinned_) {
      // Pinned input metadata carries no CUDA placement. Declare one explicitly for the reader.
      spec.output(placement, "placement")
          .produces_tensor(
              TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                                  .dtype = DLDataType{kDLUInt, 8U, 1U}},
                               .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 64U},
                               .storage = TensorOutputStorage::kRuntimePool});
    }
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnEach{in});
    return c;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    auto stream = caller_ != nullptr ? caller_->stream : context.cuda_stream().get();
    if (!sample->data.synchronize_for_consumer_stream(stream) ||
        !sample->data.set_deallocation_stream(stream) ||
        cudaMemcpyAsync(device_output_, sample->data.data(), 64U, cudaMemcpyDefault, stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(output_->pointer, device_output_, 64U, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    struct Record {
      Probe* probe;
      const std::uint8_t* data;
      std::uint8_t expected;
    };
    auto record = std::make_unique<Record>(Record{probe_.get(),
                                                  static_cast<std::uint8_t*>(output_->pointer),
                                                  static_cast<std::uint8_t>(++frame_ % 251U + 1U)});
    auto status = cudaLaunchHostFunc(
        stream,
        [](void* pointer) {
          const std::unique_ptr<Record> owner{static_cast<Record*>(pointer)};
          if (owner->data[0] != owner->expected || owner->data[63] != owner->expected) {
            owner->probe->failed.store(true);
          }
          owner->probe->completed.fetch_add(1U, std::memory_order_release);
        },
        record.get());
    if (status != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    // The accepted CUDA host callback now owns the record passed through record.get().
    static_cast<void>(record.release());  // NOLINT(bugprone-unused-return-value)
    // A defensive application baseline keeps its input lease until GPU work completes.
    if (synchronize_ && cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return {};
  }
  Input<Tensor> in;
  Output<Tensor> placement;

 private:
  void* device_output_{};
  std::shared_ptr<Probe> probe_;
  std::shared_ptr<BufferOwner> output_;
  std::shared_ptr<StreamOwner> caller_;
  std::size_t frame_{};
  bool pinned_{};
  bool synchronize_{};
};
bool byte_write(int fd, std::uint8_t value) {
  return ::write(fd, &value, 1U) == 1;
}
int byte_read(int fd) {
  pollfd descriptor{.fd = fd, .events = POLLIN, .revents = 0};
  if (::poll(&descriptor, 1U, 30000) <= 0) {
    return -1;
  }
  std::uint8_t byte{};
  return ::read(fd, &byte, 1U) == 1 ? byte : -1;
}
int consumer(const std::string& name, int readers, bool pinned, bool caller, std::size_t frames,
             int ready, int finish) {
  auto runtime = ChannelRuntime::create();
  if (!runtime || cudaFree(nullptr) != cudaSuccess) {
    return 1;
  }
  ChannelOptions options;
  options.tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle;
  options.expected_recipient_count = 1U;
  options.subscriber_queue_depth = 3U;
  auto subscriber = runtime->create_subscriber<Tensor>(name, options);
  if (!subscriber) {
    return 2;
  }
  auto probe = std::make_shared<Probe>();
  std::vector<std::shared_ptr<BufferOwner>> outputs;
  std::vector<std::shared_ptr<StreamOwner>> streams;
  for (int i = 0; i < readers; ++i) {
    outputs.push_back(std::make_shared<BufferOwner>());
    streams.push_back(std::make_shared<StreamOwner>());
  }
  Graph graph{name + "_gpu"};
  auto source =
      graph.op<Source>("source",
                       std::make_shared<ChannelSubscriber<Tensor>>(std::move(*subscriber)),
                       pinned ? MemoryKind::kPinnedHost : MemoryKind::kCudaDevice);
  auto source_partition = graph.partition("source");
  source_partition.add(source);
  for (int i = 0; i < readers; ++i) {
    if (cudaMallocHost(&outputs[i]->pointer, 64U) != cudaSuccess ||
        (caller &&
         cudaStreamCreateWithFlags(&streams[i]->stream, cudaStreamNonBlocking) != cudaSuccess)) {
      return 3;
    }
    auto sink = graph.op<Reader>("reader" + std::to_string(i),
                                 probe,
                                 outputs[i],
                                 caller ? streams[i] : std::shared_ptr<StreamOwner>{},
                                 pinned);
    graph.add_flow(source->out, sink->in);
    auto partition = graph.partition("reader" + std::to_string(i));
    partition.add(sink);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions compile_options;
  compile_options.cuda_execution.compute_stream_policy =
      CudaComputeStreamPolicy::kPerExecutionPartition;
  if (!pinned) {
    compile_options.deployment.bind_tensor_output_device(
        {.operator_path = "source", .output_port = "out", .device = DeviceId{0}});
  } else {
    compile_options.deployment.bind_tensor_output_device(
        {.operator_path = "source", .output_port = "placement", .device = DeviceId{0}});
    for (int i = 0; i < readers; ++i) {
      compile_options.deployment.bind_tensor_output_device(
          {.operator_path = "reader" + std::to_string(i),
           .output_port = "placement",
           .device = DeviceId{0}});
    }
  }
  auto plan = compile(graph, compile_options);
  if (!plan.ok()) {
    std::cerr << plan.json();
    return 4;
  }
  auto session = run_async(plan);
  if (!byte_write(ready, 1U)) {
    return 5;
  }
  const auto deadline = SteadyClock::now() + std::chrono::seconds{30};
  while (probe->completed.load(std::memory_order_acquire) < frames * readers &&
         SteadyClock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds{100});
  }
  if (probe->failed.load() || probe->completed.load() != frames * readers) {
    std::cerr << "consumer GPU data/completion check failed; completed=" << probe->completed.load()
              << " expected=" << frames * readers << " data_failed=" << probe->failed.load()
              << '\n';
    return 6;
  }
  if (!byte_write(finish, 1U)) {
    return 7;
  }
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    return 6;
  }
  return byte_write(finish, 0U) ? 0 : 7;
}
// Keep child-process coordination, publisher-slot reclamation, and failure cleanup in
// execution order so every early exit remains under the same process and native owners.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int run(int argc, char** argv) {
  if (argc == 10 && std::string(argv[1]) == "consumer") {
    return consumer(argv[2],
                    std::atoi(argv[3]),
                    std::atoi(argv[4]) != 0,
                    std::atoi(argv[5]) != 0,
                    std::strtoull(argv[6], nullptr, 10),
                    std::atoi(argv[7]),
                    std::atoi(argv[8]));
  }
  if (argc != 6) {
    std::cerr << "usage: <readers 1|2|4> <slots 2|3> <pinned 0|1> <caller 0|1> <frames>\n";
    return 1;
  }
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return 77;
  }
  const int readers = std::atoi(argv[1]);
  const int slots = std::atoi(argv[2]);
  const bool pinned = std::atoi(argv[3]) != 0;
  const bool caller = std::atoi(argv[4]) != 0;
  const std::size_t frames = std::strtoull(argv[5], nullptr, 10);
  constexpr std::size_t kWarmup = 64U;
  const auto total = frames + kWarmup;
  if ((readers != 1 && readers != 2 && readers != 4) || (slots != 2 && slots != 3) ||
      frames == 0U) {
    return 1;
  }
  const std::string name = "hsdk_retire_graph_" + std::to_string(getpid());
  std::array<int, 2U> ready{}, finish{};
  if (::pipe(ready.data()) != 0 || ::pipe(finish.data()) != 0) {
    return 2;
  }
  std::vector<std::string> args{argv[0],
                                "consumer",
                                name,
                                argv[1],
                                argv[3],
                                argv[4],
                                std::to_string(total),
                                std::to_string(ready[1]),
                                std::to_string(finish[1]),
                                "end"};
  std::vector<char*> child_args;
  child_args.reserve(args.size() + 1U);
  for (auto& a : args) {
    child_args.push_back(a.data());
  }
  child_args.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addclose(&actions, ready[0]);
  posix_spawn_file_actions_addclose(&actions, finish[0]);
  pid_t child{};
  const int spawned = posix_spawn(&child, argv[0], &actions, nullptr, child_args.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawned != 0) {
    return 3;
  }
  ::close(ready[1]);
  ::close(finish[1]);
  struct ChildCleanup {
    pid_t child;
    int ready;
    int finish;
    ~ChildCleanup() {
      ::close(ready);
      ::close(finish);
      if (child > 0) {
        ::kill(child, SIGKILL);
        ::waitpid(child, nullptr, 0);
      }
    }
  } cleanup{child, ready[0], finish[0]};
  if (cudaFree(nullptr) != cudaSuccess) {
    return 4;
  }
  StreamOwner producer;
  if (cudaStreamCreateWithFlags(&producer.stream, cudaStreamNonBlocking) != cudaSuccess) {
    return 4;
  }
  auto context = holomq::ContextBuilder()
                     .with_vmm_fd_export(true)
                     .with_fd_pass_socket("/tmp/" + name + ".sock")
                     .create();
  if (!context) {
    return 5;
  }
  auto pool =
      pinned ? context->create_host_pinned_buffer_pool(64U, slots, CU_MEMHOSTREGISTER_DEVICEMAP)
             : context->create_device_buffer_pool(64U, slots);
  if (!pool) {
    return 6;
  }
  auto runtime = ChannelRuntime::create();
  if (!runtime) {
    return 7;
  }
  ChannelOptions options;
  options.tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle;
  options.expected_recipient_count = 1U;
  options.subscriber_queue_depth = 3U;
  auto publisher = channel_internal::TensorPublisherTestAccess::create(*runtime, name, options);
  if (!publisher || byte_read(ready[0]) != 1) {
    return 8;
  }
  const auto deadline = SteadyClock::now() + std::chrono::seconds{30};
  const std::array<std::int64_t, 1U> shape{64};
  std::vector<std::pair<const void*, SteadyClock::time_point>> outstanding;
  std::vector<double> recycle_us;
  auto start = SteadyClock::now();
  auto cpu_start = std::clock();
  for (std::size_t frame = 1U; frame <= total; ++frame) {
    if (frame == kWarmup + 1U) {
      start = SteadyClock::now();
      cpu_start = std::clock();
      recycle_us.clear();
    }
    std::optional<holomq::Buffer> owner;
    while (!owner && SteadyClock::now() < deadline) {
      static_cast<void>(publisher->update_connections());
      auto available = pool->acquire();
      if (available) {
        owner.emplace(std::move(*available));
      }
    }
    if (!owner) {
      return 9;
    }
    const auto now = SteadyClock::now();
    auto previous = std::find_if(outstanding.begin(), outstanding.end(), [&](const auto& row) {
      return row.first == owner->ptr();
    });
    if (previous != outstanding.end()) {
      recycle_us.push_back(
          std::chrono::duration<double, std::micro>(now - previous->second).count());
    }
    if (const auto filled = cudaMemsetAsync(device_address(owner->ptr()),
                                            static_cast<int>(frame % 251U + 1U),
                                            64U,
                                            producer.stream);
        filled != cudaSuccess) {
      std::cerr << "producer frame fill failed: " << cudaGetErrorName(filled) << '\n';
      return 10;
    }
    auto tensor = channel_internal::make_tensor_from_holomq_buffer(
        shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, std::move(*owner));
    if (!tensor || !tensor->set_readiness_stream(producer.stream)) {
      return 11;
    }
    const Sample<Tensor> sample{std::move(*tensor), {.sequence = frame}};
    bool emitted{};
    while (!emitted && SteadyClock::now() < deadline) {
      static_cast<void>(publisher->update_connections());
      auto sent = publisher->emit_sample(sample);
      if (!sent && sent.error().code != ErrorCode::kBackpressured &&
          sent.error().code != ErrorCode::kNotReady) {
        return 12;
      }
      emitted = sent && *sent == 1U;
    }
    if (!emitted) {
      return 13;
    }
    if (previous == outstanding.end()) {
      outstanding.emplace_back(sample.data.data(), SteadyClock::now());
    } else {
      previous->second = SteadyClock::now();
    }
  }
  if (byte_read(finish[0]) != 1) {
    return 14;
  }
  const auto gpu_complete = SteadyClock::now();
  const auto cpu_complete = std::clock();
  if (byte_read(finish[0]) != 0) {
    return 14;
  }
  std::size_t reclaimed{};
  std::vector<holomq::Buffer> final_slots;
  while (reclaimed < static_cast<std::size_t>(slots) && SteadyClock::now() < deadline) {
    static_cast<void>(publisher->update_connections());
    auto available = pool->acquire();
    if (available) {
      final_slots.push_back(std::move(*available));
      ++reclaimed;
    }
  }
  if (reclaimed != static_cast<std::size_t>(slots)) {
    return 15;
  }
  int status{};
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    return 16;
  }
  cleanup.child = -1;
  const double seconds = std::chrono::duration<double>(gpu_complete - start).count();
  std::sort(recycle_us.begin(), recycle_us.end());
  const auto percentile = [&](double p) {
    return recycle_us.empty() ? 0.0
                              : recycle_us[static_cast<std::size_t>(
                                    p * static_cast<double>(recycle_us.size() - 1U))];
  };
  std::cout << "{\"frames\":" << frames << ",\"readers\":" << readers << ",\"slots\":" << slots
            << ",\"warmup\":" << kWarmup << ",\"pinned\":" << pinned << ",\"caller\":" << caller
            << ",\"seconds\":" << seconds
            << ",\"frames_per_second\":" << static_cast<double>(frames) / seconds
            << ",\"publisher_cpu_seconds\":"
            << static_cast<double>(cpu_complete - cpu_start) / static_cast<double>(CLOCKS_PER_SEC)
            << ",\"recycle_p50_us\":" << percentile(.5) << ",\"recycle_p95_us\":" << percentile(.95)
            << ",\"recycle_p99_us\":" << percentile(.99) << ",\"full_seconds\":"
            << std::chrono::duration<double>(SteadyClock::now() - start).count()
            << ",\"synchronized_consumers\":"
            << (std::getenv("HOLOSCAN_BENCHMARK_SYNC_CONSUMERS") != nullptr)
            << ",\"all_slots_returned\":true}\n";
  static_cast<void>(::unlink(("/tmp/" + name + ".sock").c_str()));
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const int status = run(argc, argv);
  // Most failure paths return a bare step code; name it so a CTest failure is not silent.
  if (status != 0 && status != 77) {
    std::cerr << "cuda_tensor_interprocess_benchmark " << (argc == 10 ? "consumer" : "producer")
              << " failed with status " << status << '\n';
  }
  return status;
}
