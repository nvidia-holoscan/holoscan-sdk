/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <array>
#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <holomq/context_builder.hpp>

#include "../../src/channel/holomq_tensor.hpp"
#include "../common/channel_benchmark_common.hpp"

#include "../../src/core/domain/tensor_access.hpp"

namespace {
double process_cpu_seconds() {
  timespec time{};
  static_cast<void>(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time));
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) * 1e-9;
}

struct StreamOwner {
  cudaStream_t stream{};
  ~StreamOwner() { static_cast<void>(cudaStreamDestroy(stream)); }
};

// Keep import, retirement submission, and reacquisition in one benchmark transaction so the
// timing boundaries and publisher-address proof describe the same batch.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void run_imported_tensor_retirement(benchmark::State& state, bool sdk_owned) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    state.SkipWithMessage("requires a CUDA device");
    return;
  }
  StreamOwner stream;
  if (cudaSetDevice(0) != cudaSuccess ||
      cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking) != cudaSuccess) {
    state.SkipWithError("failed to create consumer stream");
    return;
  }
  const auto socket = "/tmp/hmq_retire_" + std::to_string(getpid()) + ".sock";
  struct SocketCleanup {
    const std::string& path;
    ~SocketCleanup() { holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(path); }
  };
  const SocketCleanup socket_cleanup{socket};
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket).create();
  if (!context) {
    state.SkipWithError("failed to create HoloMQ context");
    return;
  }
  constexpr std::size_t kBytes = 64U;
  auto pool = context->create_device_buffer_pool(kBytes, 1U);
  if (!pool) {
    state.SkipWithError("failed to create one-slot publisher pool");
    return;
  }
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kBytes)};
  const DLDataType dtype{kDLUInt, 8U, 1U};
  std::vector<holoscan::Tensor> tensors;
  const auto batch = static_cast<std::size_t>(state.range(0));
  tensors.reserve(batch);
  auto owner = pool->acquire();
  if (!owner) {
    state.SkipWithError("failed to acquire publisher slot");
    return;
  }
  const void* const publisher_address = owner->ptr();
  // Establish the mapping cache before timing; each measured import still creates a fresh lease.
  auto warm_handle = owner->export_handle_unsynchronized();
  if (!warm_handle) {
    state.SkipWithError("failed to export publisher slot");
    return;
  }
  {
    auto warm_import = context->import_buffer_unsynchronized(*warm_handle);
    if (!warm_import) {
      state.SkipWithError("failed to warm import cache");
      return;
    }
  }
  auto owned_lifetime = std::make_shared<holomq::cuda::StreamLifetime>(stream.stream, 0);
  const std::shared_ptr<const holomq::cuda::StreamLifetime> lifetime =
      sdk_owned ? owned_lifetime : nullptr;
  const holoscan::detail::TensorConsumerScope scope{stream.stream, &lifetime};
  double submit_seconds{};
  double drain_seconds{};
  double cpu_seconds{};
  std::int64_t batches{};
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start = process_cpu_seconds();
    auto handle = owner->export_handle_unsynchronized();
    bool succeeded = handle.has_value();
    for (std::size_t i = 0U; succeeded && i < batch; ++i) {
      auto buffer = context->import_buffer_unsynchronized(*handle);
      if (!buffer) {
        succeeded = false;
        break;
      }
      auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
          shape, dtype, std::nullopt, std::move(*buffer));
      if (!tensor || !tensor->synchronize_for_consumer_stream(stream.stream) ||
          !tensor->set_deallocation_stream(stream.stream)) {
        succeeded = false;
        break;
      }
      tensors.emplace_back(std::move(*tensor));
    }
    // No payload kernels: this isolates import, registration and native retirement machinery.
    // Drop the publisher too, so successful reacquisition proves that every import retired.
    owner = holomq::Buffer{};
    tensors.clear();
    const auto submitted = std::chrono::steady_clock::now();
    succeeded = cudaStreamSynchronize(stream.stream) == cudaSuccess && succeeded;
    const auto deadline = submitted + std::chrono::seconds{2};
    while (succeeded && !(owner = pool->acquire())) {
      if (std::chrono::steady_clock::now() >= deadline) {
        succeeded = false;
        break;
      }
      std::this_thread::yield();
    }
    const auto end = std::chrono::steady_clock::now();
    if (!succeeded || owner->ptr() != publisher_address) {
      state.SkipWithError("import failed or publisher slot did not retire");
      break;
    }
    const double elapsed = std::chrono::duration<double>(end - start).count();
    state.SetIterationTime(elapsed);
    submit_seconds += std::chrono::duration<double>(submitted - start).count();
    drain_seconds += std::chrono::duration<double>(end - submitted).count();
    cpu_seconds += process_cpu_seconds() - cpu_start;
    ++batches;
  }
  owned_lifetime->close();
  if (cudaStreamSynchronize(stream.stream) == cudaSuccess) {
    static_cast<void>(owned_lifetime->publish_retired());
  } else {
    static_cast<void>(owned_lifetime->publish_quarantined());
  }
  const double items = static_cast<double>(batches) * static_cast<double>(batch);
  if (items > 0.0) {
    state.counters["submit_us_per_tensor"] = submit_seconds * 1e6 / items;
    state.counters["drain_us_per_tensor"] = drain_seconds * 1e6 / items;
    state.counters["process_cpu_us_per_tensor"] = cpu_seconds * 1e6 / items;
  }
  state.counters["sdk_owned_stream"] = static_cast<double>(sdk_owned);
  state.SetItemsProcessed(batches * static_cast<std::int64_t>(batch));
}

void BM_ImportedTensorCallerStreamRetirement(benchmark::State& state) {
  run_imported_tensor_retirement(state, false);
}
BENCHMARK(BM_ImportedTensorCallerStreamRetirement)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();

void BM_ImportedTensorSDKStreamRetirement(benchmark::State& state) {
  run_imported_tensor_retirement(state, true);
}
BENCHMARK(BM_ImportedTensorSDKStreamRetirement)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
}  // namespace
