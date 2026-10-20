/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <holomq/context_builder.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/holomq_tensor.hpp"
#include "common/cuda_test_utils.hpp"
#include "support/host_unique_test_name.hpp"
#include "support/optional_assertions.hpp"

namespace holoscan {
namespace {
using namespace std::chrono_literals;
constexpr std::size_t kBytes = 64U;
constexpr TensorRepresentation kRepresentation{
    .memory_kind = MemoryKind::kCudaDevice, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U};

struct ImportedTensorFanoutProbe {
  Tensor first;
  Tensor replacement;
  const void* first_address{};
  CUdeviceptr gate_address{};
  void* copied{};
  std::atomic_bool slow_queued{};
  std::atomic_bool fast_completed{};
  std::atomic_uint32_t slow_callbacks{};
  std::atomic_uint32_t fast_callbacks{};
};

// Blocking edges report a full queue as kBackpressured from emit(); propagating it fails the run.
[[nodiscard]] expected<void, Error> backpressure_is_not_failure(expected<void, Error> emitted) {
  if (!emitted && emitted.error().code == ErrorCode::kBackpressured) {
    return {};
  }
  return emitted;
}

class ImportedFanoutSource final : public Operator<> {
 public:
  explicit ImportedFanoutSource(std::shared_ptr<ImportedTensorFanoutProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kRepresentation,
            .bounds = tensor_bounds(kBytes),
            .storage = TensorOutputStorage::kPassThrough,
        });
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    // A loaded host can leave the relay a tick behind. Retry the first frame on the next tick and
    // skip a replacement rather than failing the run on backpressure.
    if (!first_emitted_) {
      auto emitted = output.emit(probe_->first);
      if (emitted) {
        first_emitted_ = true;
        probe_->first = Tensor{};
      }
      return backpressure_is_not_failure(std::move(emitted));
    }
    return backpressure_is_not_failure(output.emit(probe_->replacement));
  }
  Output<Tensor> output;

 private:
  std::shared_ptr<ImportedTensorFanoutProbe> probe_;
  bool first_emitted_{};
};

class ImportedFanoutRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kRepresentation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kRepresentation,
            .bounds = tensor_bounds(kBytes),
            .storage = TensorOutputStorage::kPassThrough,
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
    // A sink a frame behind drops this replacement; the first frame always finds empty queues.
    return backpressure_is_not_failure(output.emit(std::move(*sample)));
  }
  Input<Tensor> input;
  Output<Tensor> output;
};

class ImportedFanoutSink final : public Operator<> {
 public:
  ImportedFanoutSink(std::shared_ptr<ImportedTensorFanoutProbe> probe, bool slow)
      : probe_(std::move(probe)), slow_(slow) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kRepresentation});
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const auto stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (sample->data.data() == probe_->first_address) {
      if (slow_) {
        if (cuStreamWaitValue32(stream, probe_->gate_address, 1U, CU_STREAM_WAIT_VALUE_EQ) !=
                CUDA_SUCCESS ||
            cudaMemcpyAsync(
                probe_->copied, sample->data.data(), kBytes, cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        probe_->slow_queued.store(true, std::memory_order_release);
      } else if (cudaLaunchHostFunc(
                     stream,
                     [](void* data) {
                       static_cast<ImportedTensorFanoutProbe*>(data)->fast_completed.store(
                           true, std::memory_order_release);
                     },
                     probe_.get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    (slow_ ? probe_->slow_callbacks : probe_->fast_callbacks)
        .fetch_add(1U, std::memory_order_release);
    return {};
  }
  Input<Tensor> input;

 private:
  std::shared_ptr<ImportedTensorFanoutProbe> probe_;
  bool slow_{};
};

TEST(HoloMQTensorLifetimeTest, ImportedTensorForwardedAcrossPartitionsWaitsForSlowGPUReader) {
  test::skip_without_cuda_device();
  if (testing::Test::IsSkipped()) {
    return;
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto primary = test::ScopedCudaPrimaryContext::create();
  ASSERT_TRUE(primary);
  const std::string name = test::make_host_unique_test_name("imported-fanout");
  const std::string socket = "/tmp/hmq_graph_" + std::to_string(::getpid()) + ".sock";
  struct SocketCleanup {
    const std::string& path;
    ~SocketCleanup() { static_cast<void>(::unlink(path.c_str())); }
  };
  const SocketCleanup socket_cleanup{socket};
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket).create();
  ASSERT_TRUE(context);
  auto pool = context->create_device_buffer_pool(kBytes, 1U);
  auto replacement_pool = context->create_device_buffer_pool(kBytes, 1U);
  ASSERT_TRUE(pool);
  ASSERT_TRUE(replacement_pool);
  auto owner = pool->acquire();
  auto replacement_owner = replacement_pool->acquire();
  ASSERT_TRUE(owner);
  ASSERT_TRUE(replacement_owner);
  ASSERT_EQ(cudaMemset(owner->ptr(), 0x7B, kBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(replacement_owner->ptr(), 0, kBytes), cudaSuccess);
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle);
  auto imported = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported);
  const std::array<std::int64_t, 1> shape{kBytes};
  const auto& dtype = kRepresentation.dtype;
  HOLOSCAN_ASSERT_HAS_VALUE(dtype) << "Expected Tensor dtype";
  auto first = channel_internal::make_tensor_from_holomq_const_buffer(
      shape, *dtype, std::nullopt, std::move(*imported));
  auto replacement = channel_internal::make_tensor_from_holomq_buffer(
      shape, *dtype, std::nullopt, std::move(*replacement_owner));
  ASSERT_TRUE(first);
  ASSERT_TRUE(replacement);
  owner = holomq::Buffer{};
  auto probe = std::make_shared<ImportedTensorFanoutProbe>();
  probe->first_address = first->data();
  probe->first = std::move(*first);
  probe->replacement = std::move(*replacement);
  std::uint32_t* gate{};
  ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&gate), sizeof(*gate), cudaHostAllocMapped),
            cudaSuccess);
  *gate = 0U;
  void* copied{};
  ASSERT_EQ(cudaMallocHost(&copied, kBytes), cudaSuccess);
  struct StorageGuard {
    std::uint32_t* gate{};
    void* copied{};
    ~StorageGuard() {
      static_cast<void>(cudaFreeHost(copied));
      static_cast<void>(cudaFreeHost(gate));
    }
  };
  const StorageGuard storage{gate, copied};
  probe->copied = copied;
  ASSERT_EQ(cuMemHostGetDevicePointer(&probe->gate_address, gate, 0U), CUDA_SUCCESS);
  Graph graph{name};
  auto source = graph.op<ImportedFanoutSource>("source", probe);
  auto relay = graph.op<ImportedFanoutRelay>("relay");
  auto fast = graph.op<ImportedFanoutSink>("fast", probe, false);
  auto slow = graph.op<ImportedFanoutSink>("slow", probe, true);
  graph.partition("source").add(source);
  graph.partition("relay").add(relay);
  graph.partition("fast").add(fast);
  graph.partition("slow").add(slow);
  graph.add_flow(source->output, relay->input);
  graph.add_flow(relay->output, fast->input);
  graph.add_flow(relay->output, slow->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = CudaComputeStreamPolicy::kPerExecutionPartition;
  for (const std::string& path : {std::string{"source"}, std::string{"relay"}}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = path, .output_port = "output", .device = DeviceId{0}});
  }
  const auto plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  struct SessionGuard {
    RunSession& session;
    std::uint32_t* gate;
    ~SessionGuard() {
      std::atomic_ref<std::uint32_t>(*gate).store(1U, std::memory_order_release);
      session.request_stop();
      session.wait();
    }
  };
  const SessionGuard cleanup{session, gate};
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while ((!probe->slow_queued.load(std::memory_order_acquire) ||
          !probe->fast_completed.load(std::memory_order_acquire) ||
          probe->slow_callbacks.load(std::memory_order_acquire) < 32U ||
          probe->fast_callbacks.load(std::memory_order_acquire) < 32U) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(probe->slow_queued.load());
  ASSERT_TRUE(probe->fast_completed.load());
  ASSERT_GE(probe->slow_callbacks.load(), 32U);
  ASSERT_GE(probe->fast_callbacks.load(), 32U);
  // Replacement frames evict the original channel handles while its GPU reader stays gated.
  EXPECT_FALSE(pool->acquire());
  std::atomic_ref<std::uint32_t>(*gate).store(1U, std::memory_order_release);
  session.request_stop();
  session.wait();
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(*static_cast<const std::uint8_t*>(copied), 0x7B);
  bool released{};
  const auto release_deadline = std::chrono::steady_clock::now() + 2s;
  while (!(released = static_cast<bool>(pool->acquire())) &&
         std::chrono::steady_clock::now() < release_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(released);
}

}  // namespace
}  // namespace holoscan
