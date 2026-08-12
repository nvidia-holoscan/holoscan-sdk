/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/**
 * @brief Native CUDA event pool using CCCL cuda::event ownership.
 *
 * This is an experimental counterpart to CudaStreamPool. Runtime-facing code can own events through
 * bounded pool leases while passing cuda::event_ref views to CUDA-facing APIs. A production runtime
 * should attach lightweight CudaEventToken metadata to messages rather than exposing raw event
 * handles or cuda::event_ref values as serialized payload.
 *
 * If a lease outlives the owning pool during experimental teardown, release drops the event instead
 * of dereferencing the destroyed pool. A production version may still move full shared pool state
 * behind the lease if cross-owner reuse is required.
 *
 * The pool intentionally owns untimed cuda::event objects. Timing-capable cuda::timed_event objects
 * are useful for benchmarks or observability, but the normal runtime synchronization path should
 * not pay timing overhead unless a concrete timing use case opts into a separate pool or mode.
 */
class CudaEventPool {
 public:
  struct Config {
    int32_t device_id = 0;
    cuda::event_flags event_flags = cuda::event_flags::none;
    uint32_t reserved_size = 1;
    uint32_t max_size = 0;  // 0 means unlimited
    std::string nvtx_identifier = "cuda_event_pool";
    // When non-null, events are created in this application-provided CUDA context: the context is
    // made current (via the driver's cuCtxPushCurrent) for the duration of event creation and the
    // previous context is restored afterward. The pool only borrows the context and never destroys
    // it; the caller owns its lifetime and must keep it valid for as long as the pool exists. Use
    // this to keep an operator's events in the same context as its streams (e.g. a context shared
    // with a graphics API for CUDA/graphics interop). Requires CUDA 12.3+ at runtime. When null,
    // events are created on the device normally.
    CUcontext cuda_context = nullptr;
  };

  struct EventEntry {
    EventEntry(cuda::event event_arg, std::string name_arg)
        : event(std::move(event_arg)), name(std::move(name_arg)) {}

    cuda::event event;
    std::string name;
  };

  class EventLease {
   public:
    EventLease() = default;
    EventLease(const EventLease&) = delete;
    EventLease& operator=(const EventLease&) = delete;
    HOLOSCAN_CORE_EXPORT EventLease(EventLease&& other) noexcept;
    HOLOSCAN_CORE_EXPORT EventLease& operator=(EventLease&& other) noexcept;
    HOLOSCAN_CORE_EXPORT ~EventLease();

    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> release();

    [[nodiscard]] HOLOSCAN_CORE_EXPORT cuda::event_ref event() const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT cudaEvent_t native_handle() const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> record(
        ::cuda::stream_ref stream) const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wait(::cuda::stream_ref stream) const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> synchronize() const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<bool, Error> is_done() const;
    [[nodiscard]] explicit operator bool() const noexcept { return entry_ != nullptr; }

   private:
    friend class CudaEventPool;
    EventLease(CudaEventPool* pool, std::shared_ptr<EventEntry> entry,
               std::weak_ptr<bool> pool_alive);

    CudaEventPool* pool_ = nullptr;
    std::shared_ptr<EventEntry> entry_;
    std::weak_ptr<bool> pool_alive_;
  };

  CudaEventPool() = default;
  HOLOSCAN_CORE_EXPORT explicit CudaEventPool(Config config);
  HOLOSCAN_CORE_EXPORT CudaEventPool(int32_t device_id, cuda::event_flags event_flags,
                                     uint32_t reserved_size, uint32_t max_size,
                                     std::string nvtx_identifier = "cuda_event_pool");
  HOLOSCAN_CORE_EXPORT ~CudaEventPool();

  HOLOSCAN_CORE_EXPORT void initialize();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<EventLease, Error> acquire();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> release(
      std::shared_ptr<EventEntry> entry);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t available() const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t allocated() const;
  [[nodiscard]] size_t capacity() const noexcept { return config_.max_size; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  enum struct AllocatorStage : uint8_t {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::shared_ptr<EventEntry>, Error>
  create_event_entry(std::string name);
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reserve_events();

  Config config_;
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<EventEntry>> reserved_events_;
  std::vector<EventEntry*> allocated_events_;
  std::shared_ptr<bool> alive_token_ = std::make_shared<bool>(true);
  AllocatorStage stage_ = AllocatorStage::kUninitialized;
  uint32_t next_event_index_ = 0;
};

}  // namespace holoscan
