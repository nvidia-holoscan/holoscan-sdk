/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda.h>

#include <holomq/buffer_pool.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include "buffer_pool/sync/cuda_stream_generation.hpp"

namespace {

[[nodiscard]] holomq::expected<void, holomq::Error> check_cuda_result(
    const CUresult result, const std::string_view operation) {
  if (result == CUDA_SUCCESS) {
    return {};
  }

  const char* error_name = nullptr;
  const char* error_description = nullptr;
  static_cast<void>(cuGetErrorName(result, &error_name));
  static_cast<void>(cuGetErrorString(result, &error_description));
  std::string message(operation);
  message.append(" failed");
  if (error_name != nullptr) {
    message.append(": ").append(error_name);
  }
  if (error_description != nullptr) {
    message.append(" (").append(error_description).push_back(')');
  }
  return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kFailure, std::move(message)});
}

class CUDABenchmarkResources {
 public:
  CUDABenchmarkResources() = default;
  CUDABenchmarkResources(const CUDABenchmarkResources&) = delete;
  CUDABenchmarkResources& operator=(const CUDABenchmarkResources&) = delete;

  ~CUDABenchmarkResources() {
    if (context_ != nullptr) {
      static_cast<void>(cuCtxSetCurrent(context_));
    }
    if (wait_stream_ != nullptr) {
      static_cast<void>(cuStreamDestroy(wait_stream_));
    }
    if (stream_ != nullptr) {
      static_cast<void>(cuStreamDestroy(stream_));
    }
    if (generation_word_ != 0U) {
      static_cast<void>(cuMemFree(generation_word_));
    }
    if (context_ != nullptr) {
      static_cast<void>(cuCtxSetCurrent(nullptr));
      static_cast<void>(cuDevicePrimaryCtxRelease(device_));
    }
  }

  [[nodiscard]] holomq::expected<void, holomq::Error> initialize(
      const bool create_wait_stream, const unsigned char generation_byte = 0U) {
    if (auto status = check_cuda_result(cuInit(0), "cuInit"); !status.has_value()) {
      return status;
    }
    if (auto status = check_cuda_result(cuDeviceGet(&device_, 0), "cuDeviceGet(device=0)");
        !status.has_value()) {
      return status;
    }
    if (auto status = check_cuda_result(cuDevicePrimaryCtxRetain(&context_, device_),
                                        "cuDevicePrimaryCtxRetain(device=0)");
        !status.has_value()) {
      return status;
    }
    if (auto status = check_cuda_result(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
        !status.has_value()) {
      return status;
    }
    if (auto status = check_cuda_result(cuMemAlloc(&generation_word_, sizeof(std::uint64_t)),
                                        "cuMemAlloc(generation_word)");
        !status.has_value()) {
      return status;
    }
    if (auto status =
            check_cuda_result(cuMemsetD8(generation_word_, generation_byte, sizeof(std::uint64_t)),
                              "cuMemsetD8(generation_word)");
        !status.has_value()) {
      return status;
    }
    if (auto status = check_cuda_result(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                                        "cuStreamCreate(benchmark_stream)");
        !status.has_value()) {
      return status;
    }
    if (create_wait_stream) {
      if (auto status = check_cuda_result(cuStreamCreate(&wait_stream_, CU_STREAM_NON_BLOCKING),
                                          "cuStreamCreate(wait_stream)");
          !status.has_value()) {
        return status;
      }
    }
    return {};
  }

  [[nodiscard]] CUstream stream() const noexcept { return stream_; }
  [[nodiscard]] CUstream wait_stream() const noexcept { return wait_stream_; }
  [[nodiscard]] CUdeviceptr generation_word() const noexcept { return generation_word_; }

 private:
  CUdevice device_{};
  CUcontext context_{nullptr};
  CUstream stream_{nullptr};
  CUstream wait_stream_{nullptr};
  CUdeviceptr generation_word_{0U};
};

class CUDAEventBatch {
 public:
  CUDAEventBatch() = default;
  CUDAEventBatch(const CUDAEventBatch&) = delete;
  CUDAEventBatch& operator=(const CUDAEventBatch&) = delete;

  ~CUDAEventBatch() {
    for (const CUevent event : events_) {
      if (event != nullptr) {
        static_cast<void>(cuEventDestroy(event));
      }
    }
  }

  [[nodiscard]] holomq::expected<void, holomq::Error> initialize(const std::size_t count) {
    events_.resize(count);
    for (std::size_t index = 0U; index < events_.size(); ++index) {
      const std::string operation = "cuEventCreate(index=" + std::to_string(index) + ")";
      if (auto status =
              check_cuda_result(cuEventCreate(&events_[index], CU_EVENT_DISABLE_TIMING), operation);
          !status.has_value()) {
        return status;
      }
    }
    return {};
  }

  [[nodiscard]] std::vector<CUevent>& events() noexcept { return events_; }

 private:
  std::vector<CUevent> events_;
};

void run_cuda_generation_write_batch(benchmark::State& state) {
  const auto batch_size = static_cast<std::uint64_t>(state.range(0));
  CUDABenchmarkResources resources;
  auto initialized = resources.initialize(false);
  if (!initialized.has_value()) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }

  std::uint64_t generation = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0U; index < batch_size; ++index) {
      ++generation;
      if (!holomq::stream_write_generation(
              resources.stream(), resources.generation_word(), generation)) {
        succeeded = false;
        break;
      }
    }
    if (succeeded && cuStreamSynchronize(resources.stream()) != CUDA_SUCCESS) {
      succeeded = false;
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    if (!succeeded) {
      state.SkipWithError("CUDA generation operation failed");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(generation);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["stream_commands_per_item"] = 1.0;
}

void run_cuda_satisfied_generation_wait_batch(benchmark::State& state) {
  const auto batch_size = static_cast<std::uint64_t>(state.range(0));
  CUDABenchmarkResources resources;
  auto initialized = resources.initialize(true, 0x01U);
  if (!initialized.has_value()) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }

  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0U; index < batch_size; ++index) {
      if (!holomq::stream_wait_generation(
              resources.wait_stream(), resources.generation_word(), 1U)) {
        succeeded = false;
        break;
      }
    }
    if (succeeded && cuStreamSynchronize(resources.wait_stream()) != CUDA_SUCCESS) {
      succeeded = false;
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    if (!succeeded) {
      state.SkipWithError("CUDA satisfied generation wait failed");
      break;
    }
    ++completed_batches;
  }

  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["stream_commands_per_item"] = 1.0;
}

void CUDAAPI count_host_callback(void* const user_data) {
  static_cast<std::atomic<std::uint64_t>*>(user_data)->fetch_add(1U, std::memory_order_relaxed);
}

void bm_cuda_launch_host_func_completion(benchmark::State& state) {
  const auto batch_size = static_cast<std::uint64_t>(state.range(0));
  CUDABenchmarkResources resources;
  auto initialized = resources.initialize(false);
  if (!initialized.has_value()) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }

  std::atomic<std::uint64_t> completed_callbacks{0U};
  std::uint64_t expected_callbacks = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0U; index < batch_size; ++index) {
      if (cuLaunchHostFunc(resources.stream(), count_host_callback, &completed_callbacks) !=
          CUDA_SUCCESS) {
        succeeded = false;
        break;
      }
      ++expected_callbacks;
    }
    if (succeeded && cuStreamSynchronize(resources.stream()) != CUDA_SUCCESS) {
      succeeded = false;
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    if (!succeeded || completed_callbacks.load(std::memory_order_relaxed) != expected_callbacks) {
      state.SkipWithError("CUDA host callback completion failed");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(completed_callbacks.load(std::memory_order_relaxed));
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["heap_payload"] = 0.0;
}

void bm_device_buffer_ready_release_completion(benchmark::State& state) {
  constexpr std::size_t kBufferBytes = 4096U;
  const auto batch_size = static_cast<std::uint32_t>(state.range(0));
  CUDABenchmarkResources resources;
  auto initialized = resources.initialize(false);
  if (!initialized.has_value()) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }
  auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  if (!context.has_value()) {
    state.SkipWithError("failed to create the HoloMQ BufferPool benchmark context");
    return;
  }
  auto pool = context->create_device_buffer_pool(kBufferBytes, batch_size);
  if (!pool.has_value()) {
    state.SkipWithError("failed to create the HoloMQ device BufferPool");
    return;
  }

  std::vector<holomq::Buffer> buffers;
  buffers.reserve(batch_size);
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      auto buffer = pool->acquire();
      if (!buffer.has_value()) {
        succeeded = false;
        break;
      }
      buffers.emplace_back(std::move(buffer).value());
    }
    for (auto& buffer : buffers) {
      if (!buffer.enqueue_ready_set(resources.stream())) {
        succeeded = false;
        break;
      }
    }
    buffers.clear();
    if (cuStreamSynchronize(resources.stream()) != CUDA_SUCCESS) {
      succeeded = false;
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    if (!succeeded) {
      state.SkipWithError("HoloMQ device Buffer ready/release batch failed");
      break;
    }
    ++completed_batches;
  }

  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["buffer_bytes"] = static_cast<double>(kBufferBytes);
  state.counters["vmm_fd_export"] = 1.0;
}

void run_cuda_event_record_query_completion(benchmark::State& state,
                                            const bool include_generation_write) {
  const auto batch_size = static_cast<std::size_t>(state.range(0));
  CUDABenchmarkResources resources;
  auto initialized = resources.initialize(false);
  if (!initialized.has_value()) {
    state.SkipWithError(initialized.error().message.c_str());
    return;
  }
  CUDAEventBatch event_batch;
  auto events_initialized = event_batch.initialize(batch_size);
  if (!events_initialized.has_value()) {
    state.SkipWithError(events_initialized.error().message.c_str());
    return;
  }

  std::uint64_t not_ready_queries = 0U;
  std::uint64_t generation = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (const CUevent event : event_batch.events()) {
      if (include_generation_write &&
          !holomq::stream_write_generation(
              resources.stream(), resources.generation_word(), ++generation)) {
        succeeded = false;
        break;
      }
      if (cuEventRecord(event, resources.stream()) != CUDA_SUCCESS) {
        succeeded = false;
        break;
      }
    }
    for (const CUevent event : event_batch.events()) {
      while (succeeded) {
        const CUresult query_result = cuEventQuery(event);
        if (query_result == CUDA_SUCCESS) {
          break;
        }
        if (query_result != CUDA_ERROR_NOT_READY) {
          succeeded = false;
          break;
        }
        ++not_ready_queries;
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    if (!succeeded) {
      state.SkipWithError("CUDA release-event record/query batch failed");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(not_ready_queries);
  const std::int64_t completed_items = completed_batches * static_cast<std::int64_t>(batch_size);
  state.SetItemsProcessed(completed_items);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["not_ready_queries_per_item"] =
      completed_items > 0
          ? static_cast<double>(not_ready_queries) / static_cast<double>(completed_items)
          : 0.0;
  state.counters["preallocated_events"] = 1.0;
  state.counters["stream_commands_per_item"] = include_generation_write ? 2.0 : 1.0;
}

void bm_cuda_event_record_query_completion(benchmark::State& state) {
  run_cuda_event_record_query_completion(state, false);
}

void bm_cuda_generation_event_record_query_completion(benchmark::State& state) {
  run_cuda_event_record_query_completion(state, true);
}

void run_host_buffer_control_lifecycle(benchmark::State& state, const bool reuse_controls) {
  const auto batch_size = static_cast<std::size_t>(state.range(0));
  CUDABenchmarkResources cuda_resources;
  auto cuda_initialized = cuda_resources.initialize(false);
  if (!cuda_initialized.has_value()) {
    state.SkipWithError(cuda_initialized.error().message.c_str());
    return;
  }
  auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  if (!context.has_value()) {
    state.SkipWithError(context.error().message.c_str());
    return;
  }
  auto pool = context->create_host_buffer_pool(64U, static_cast<std::uint32_t>(batch_size));
  if (!pool.has_value()) {
    state.SkipWithError(pool.error().message.c_str());
    return;
  }

  std::vector<holomq::Buffer> owners;
  std::vector<holomq::ConstBuffer> imports(batch_size);
  if (reuse_controls) {
    owners.reserve(batch_size);
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto acquired = pool->acquire();
      if (!acquired.has_value()) {
        state.SkipWithError(acquired.error().message.c_str());
        return;
      }
      owners.push_back(std::move(*acquired));
      auto handle = owners.back().export_handle_unsynchronized();
      if (!handle.has_value() ||
          !context->rebind_buffer_unsynchronized(*handle, imports[index]).has_value()) {
        state.SkipWithError("failed to prime reusable host buffer controls");
        return;
      }
      imports[index].release();
    }
  }

  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    for (std::size_t index = 0U; index < batch_size; ++index) {
      if (reuse_controls) {
        if (!owners[index].prepare_for_reuse()) {
          succeeded = false;
          break;
        }
        auto handle = owners[index].export_handle_unsynchronized();
        if (!handle.has_value() ||
            !context->rebind_buffer_unsynchronized(*handle, imports[index]).has_value()) {
          succeeded = false;
          break;
        }
        benchmark::DoNotOptimize(imports[index].ptr());
        imports[index].release();
      } else {
        auto owner = pool->acquire();
        if (!owner.has_value()) {
          succeeded = false;
          break;
        }
        auto handle = owner->export_handle_unsynchronized();
        if (!handle.has_value()) {
          succeeded = false;
          break;
        }
        auto imported = context->import_buffer_unsynchronized(*handle);
        if (!imported.has_value()) {
          succeeded = false;
          break;
        }
        benchmark::DoNotOptimize(imported->ptr());
      }
    }
    if (!succeeded) {
      state.SkipWithError("host buffer control lifecycle failed");
      break;
    }
    ++completed_batches;
  }

  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["reused_controls"] = reuse_controls ? 1.0 : 0.0;
}

void bm_host_buffer_control_lifecycle_cold(benchmark::State& state) {
  run_host_buffer_control_lifecycle(state, false);
}

void bm_host_buffer_control_lifecycle_reused(benchmark::State& state) {
  run_host_buffer_control_lifecycle(state, true);
}

void bm_cuda_stream_write_value64_completion(benchmark::State& state) {
  run_cuda_generation_write_batch(state);
}

void bm_cuda_stream_satisfied_wait_value64_completion(benchmark::State& state) {
  run_cuda_satisfied_generation_wait_batch(state);
}

BENCHMARK(bm_cuda_stream_write_value64_completion)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(bm_cuda_stream_satisfied_wait_value64_completion)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(bm_cuda_launch_host_func_completion)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(bm_device_buffer_ready_release_completion)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(bm_cuda_event_record_query_completion)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(bm_cuda_generation_event_record_query_completion)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(bm_host_buffer_control_lifecycle_cold)->Arg(1)->Arg(8)->Arg(64);
BENCHMARK(bm_host_buffer_control_lifecycle_reused)->Arg(1)->Arg(8)->Arg(64);

}  // namespace
