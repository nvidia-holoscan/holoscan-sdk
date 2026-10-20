/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace {
std::uint64_t& event_records() noexcept {
  static thread_local std::uint64_t records{};
  return records;
}

// Process-wide: imports create events on callback threads and retirement destroys them on a
// worker thread.
std::atomic<std::uint64_t> event_creations{};
std::atomic<std::uint64_t> event_destructions{};

template <typename Function>
Function next_symbol(const char* name) noexcept {
  const auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
  if (function == nullptr) {
    std::abort();
  }
  return function;
}
}  // namespace

// Test-only preload shim: count the real CUDA calls, without counters in SDK hot paths.
extern "C" std::uint64_t hsdk_test_thread_event_records() noexcept {
  return event_records();
}

extern "C" std::uint64_t hsdk_test_event_creations() noexcept {
  return event_creations.load();
}

extern "C" std::uint64_t hsdk_test_event_destructions() noexcept {
  return event_destructions.load();
}

extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event) {
  using Create = cudaError_t(CUDARTAPI*)(cudaEvent_t*);
  static const auto create = next_symbol<Create>("cudaEventCreate");
  const auto result = create(event);
  if (result == cudaSuccess) {
    ++event_creations;
  }
  return result;
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  using Create = cudaError_t(CUDARTAPI*)(cudaEvent_t*, unsigned int);
  static const auto create = next_symbol<Create>("cudaEventCreateWithFlags");
  const auto result = create(event, flags);
  if (result == cudaSuccess) {
    ++event_creations;
  }
  return result;
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  using Destroy = cudaError_t(CUDARTAPI*)(cudaEvent_t);
  static const auto destroy = next_symbol<Destroy>("cudaEventDestroy");
  const auto result = destroy(event);
  if (result == cudaSuccess) {
    ++event_destructions;
  }
  return result;
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  using Record = cudaError_t(CUDARTAPI*)(cudaEvent_t, cudaStream_t);
  static const auto record = next_symbol<Record>("cudaEventRecord");
  const auto result = record(event, stream);
  if (result == cudaSuccess) {
    ++event_records();
  }
  return result;
}
