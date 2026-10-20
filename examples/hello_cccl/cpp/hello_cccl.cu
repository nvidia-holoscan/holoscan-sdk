/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/holoscan.hpp>

#include <cub/block/block_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>

#include <cuda_runtime_api.h>
#include <cuda/atomic>
#include <cuda/stream>

#include <cstddef>
#include <iostream>

namespace {

constexpr int kBlockSize = 128;
constexpr int kNoCudaDeviceReturnCode = 77;

__global__ void sum_kernel(const int* data, int* result, std::size_t count) {
  using BlockReduce = cub::BlockReduce<int, kBlockSize>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  const std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_sum = 0;
  if (index < count) {
    thread_sum = data[index];
  }

  const int block_sum = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
    atomic_result.fetch_add(block_sum, cuda::memory_order_relaxed);
  }
}

int skip_if_no_cuda_device() {
  int device_count = 0;
  const cudaError_t result = cudaGetDeviceCount(&device_count);
  if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver || device_count == 0) {
    std::cout << "No CUDA-capable device is available; skipping hello_cccl.\n";
    return kNoCudaDeviceReturnCode;
  }
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(result) << "\n";
    return 1;
  }
  return 0;
}

bool get_current_cuda_device(int* device_id) {
  const cudaError_t result = cudaGetDevice(device_id);
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(result) << "\n";
    return false;
  }
  return true;
}

bool synchronize_completion_event(cuda::event_ref event, const char* operation_name) {
  event.sync();
  if (!event.is_done()) {
    std::cerr << operation_name << " completion event did not finish.\n";
    return false;
  }
  return true;
}

bool run_sum(cuda::stream_ref stream, thrust::device_vector<int>* data,
             thrust::device_vector<int>* result) {
  auto stream_policy = thrust::cuda::par.on(stream.get());
  thrust::fill(stream_policy, data->begin(), data->end(), 1);
  thrust::fill(stream_policy, result->begin(), result->end(), 0);

  const int block_count = static_cast<int>((data->size() + kBlockSize - 1) / kBlockSize);
  sum_kernel<<<block_count, kBlockSize, 0, stream.get()>>>(thrust::raw_pointer_cast(data->data()),
                                                           thrust::raw_pointer_cast(result->data()),
                                                           data->size());

  const cudaError_t cuda_result = cudaGetLastError();
  if (cuda_result != cudaSuccess) {
    std::cerr << "sum_kernel launch failed: " << cudaGetErrorString(cuda_result) << "\n";
    return false;
  }

  cuda::event completion_event = stream.record_event();
  cuda::event_ref completion = completion_event;
  return synchronize_completion_event(completion, "CCCL CUDA sum");
}

}  // namespace

int main() {
  const int cuda_check = skip_if_no_cuda_device();
  if (cuda_check != 0) {
    return cuda_check;
  }

  holoscan::hello();

  int device_id = 0;
  if (!get_current_cuda_device(&device_id)) {
    return 1;
  }

  try {
    cuda::stream owning_stream{cuda::device_ref{device_id}};
    cuda::stream_ref stream = owning_stream;

    constexpr std::size_t kItemCount = 1024;
    thrust::device_vector<int> data(kItemCount);
    thrust::device_vector<int> result(1);

    if (!run_sum(stream, &data, &result)) {
      return 1;
    }

    const int sum = result[0];
    if (sum != static_cast<int>(kItemCount)) {
      std::cerr << "CCCL result verification failed: " << sum << " != " << kItemCount << "\n";
      return 1;
    }

    std::cout << "CCCL CUDA sum: " << sum << "\n";
    std::cout << "CCCL CUDA sample complete\n";
  } catch (const cuda::cuda_error& error) {
    std::cerr << "CCCL CUDA stream/event operation failed: " << error.what() << "\n";
    return 1;
  }

  return 0;
}
