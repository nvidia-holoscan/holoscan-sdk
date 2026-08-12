/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace holoscan::test {

inline bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

inline int invalid_cuda_device_id() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess ? device_count + 1024 : 1024;
}

inline void skip_without_cuda_device() {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA devices available";
  }
}

// Some HoloMQ buffer-pool tests call CUDA driver APIs internally. Retain the primary context so
// those calls share the same context family as CUDA runtime API calls made by the surrounding test.
class ScopedCudaPrimaryContext {
 public:
  ScopedCudaPrimaryContext() noexcept = default;

  ScopedCudaPrimaryContext(const ScopedCudaPrimaryContext&) = delete;
  ScopedCudaPrimaryContext& operator=(const ScopedCudaPrimaryContext&) = delete;

  ScopedCudaPrimaryContext(ScopedCudaPrimaryContext&& other) noexcept
      : device_(other.device_), context_(other.context_) {
    other.context_ = nullptr;
  }

  ScopedCudaPrimaryContext& operator=(ScopedCudaPrimaryContext&& other) noexcept {
    if (this != &other) {
      release();
      device_ = other.device_;
      context_ = other.context_;
      other.context_ = nullptr;
    }
    return *this;
  }

  ~ScopedCudaPrimaryContext() { release(); }

  [[nodiscard]] static std::optional<ScopedCudaPrimaryContext> create(const int ordinal = 0) {
    if (cuInit(0) != CUDA_SUCCESS) {
      return std::nullopt;
    }
    CUdevice device{};
    if (cuDeviceGet(&device, ordinal) != CUDA_SUCCESS) {
      return std::nullopt;
    }
    CUcontext context{};
    if (cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
      return std::nullopt;
    }
    if (cuCtxSetCurrent(context) != CUDA_SUCCESS) {
      static_cast<void>(cuDevicePrimaryCtxRelease(device));
      return std::nullopt;
    }
    return ScopedCudaPrimaryContext(device, context);
  }

 private:
  ScopedCudaPrimaryContext(const CUdevice device, CUcontext context) noexcept
      : device_(device), context_(context) {}

  void release() noexcept {
    if (context_ == nullptr) {
      return;
    }
    CUcontext current{};
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == context_) {
      static_cast<void>(cuCtxSetCurrent(nullptr));
    }
    static_cast<void>(cuDevicePrimaryCtxRelease(device_));
    context_ = nullptr;
  }

  CUdevice device_{};
  CUcontext context_{nullptr};
};

struct CudaPinnedHostDeleter {
  void operator()(void* pointer) const {
    if (pointer) {
      cudaFreeHost(pointer);
    }
  }
};

struct CudaStreamDeleter {
  void operator()(cudaStream_t stream) const {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

using UniqueCudaStream = std::unique_ptr<CUstream_st, CudaStreamDeleter>;

struct CudaEventDeleter {
  void operator()(cudaEvent_t event) const {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }
};

using UniqueCudaEvent = std::unique_ptr<CUevent_st, CudaEventDeleter>;

inline cudaError_t create_non_blocking_cuda_stream(UniqueCudaStream& stream) {
  cudaStream_t raw_stream = nullptr;
  const cudaError_t error = cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    stream.reset(raw_stream);
  }
  return error;
}

inline cudaError_t create_cuda_event(UniqueCudaEvent& event) {
  cudaEvent_t raw_event = nullptr;
  const cudaError_t error = cudaEventCreateWithFlags(&raw_event, cudaEventDisableTiming);
  if (error == cudaSuccess) {
    event.reset(raw_event);
  }
  return error;
}

struct BlockingHostState {
  std::mutex mutex;
  std::condition_variable cv;
  bool started = false;
  bool release = false;
};

inline void CUDART_CB blocking_host_callback(void* user_data) {
  auto& state = *static_cast<BlockingHostState*>(user_data);
  std::unique_lock<std::mutex> lock(state.mutex);
  state.started = true;
  state.cv.notify_all();
  state.cv.wait(lock, [&state] { return state.release; });
}

inline bool wait_until_started(BlockingHostState& state) {
  std::unique_lock<std::mutex> lock(state.mutex);
  return state.cv.wait_for(lock, std::chrono::seconds(5), [&state] { return state.started; });
}

inline void release_blocking_host(BlockingHostState& state) {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.release = true;
  state.cv.notify_all();
}

struct BlockingHostReleaseGuard {
  explicit BlockingHostReleaseGuard(BlockingHostState& state_arg) : state(&state_arg) {}

  ~BlockingHostReleaseGuard() {
    if (state) {
      release_blocking_host(*state);
    }
  }

  BlockingHostReleaseGuard(const BlockingHostReleaseGuard&) = delete;
  BlockingHostReleaseGuard& operator=(const BlockingHostReleaseGuard&) = delete;

  void dismiss() { state = nullptr; }

  BlockingHostState* state = nullptr;
};

}  // namespace holoscan::test
