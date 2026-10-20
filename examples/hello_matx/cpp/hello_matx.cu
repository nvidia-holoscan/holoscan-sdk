/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/holoscan.hpp>

#include <cuda_runtime_api.h>
#include <matx.h>
#include <cuda/stream>

#include <iostream>

namespace {

constexpr int kNoCudaDeviceReturnCode = 77;

int skip_if_no_cuda_device() {
  int device_count = 0;
  const cudaError_t result = cudaGetDeviceCount(&device_count);
  if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver) {
    std::cout << "No CUDA-capable device is available; skipping hello_matx.\n";
    return kNoCudaDeviceReturnCode;
  }
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(result) << "\n";
    return 1;
  }
  if (device_count == 0) {
    std::cout << "No CUDA-capable device is available; skipping hello_matx.\n";
    return kNoCudaDeviceReturnCode;
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

bool is_close(float actual, float expected) {
  const float difference = actual > expected ? actual - expected : expected - actual;
  return difference < 0.001F;
}

bool synchronize_completion_event(cuda::event_ref event, const char* operation_name) {
  event.sync();
  if (!event.is_done()) {
    std::cerr << operation_name << " completion event did not finish.\n";
    return false;
  }
  return true;
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

    matx::cudaExecutor executor{stream.get()};
    auto input = matx::linspace(0.0F, 1.0F, static_cast<matx::index_t>(4));
    auto output = matx::make_tensor<float>({4});

    (output = input + 1.0F).run(executor);
    cuda::event completion_event = stream.record_event();
    cuda::event_ref completion = completion_event;
    if (!synchronize_completion_event(completion, "MatX CUDA expression")) {
      return 1;
    }

    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      std::cerr << "MatX CUDA execution failed: " << cudaGetErrorString(result) << "\n";
      return 1;
    }

    const float last_value = output(3);

    if (!is_close(last_value, 2.0F)) {
      std::cerr << "MatX result verification failed: " << last_value << " != 2\n";
      return 1;
    }

    std::cout << "MatX CUDA last value: " << last_value << "\n";
    std::cout << "MatX CUDA sample complete\n";
  } catch (const cuda::cuda_error& error) {
    std::cerr << "MatX CUDA stream/event operation failed: " << error.what() << "\n";
    return 1;
  }

  return 0;
}
