/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/lifecycle_types.hpp>

#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"

namespace holoscan::test::run_session_tensor {

template <typename T>
[[nodiscard]] const T* optional_value(const std::optional<T>& value) noexcept {
  if (!value.has_value()) {
    return nullptr;
  }
  return std::addressof(value.value());
}

// Runtime-materializing tests publish into the host-wide HoloMQ namespace. Give each invocation
// its own graph identity so concurrent CTest runs and parallel SDK containers cannot derive the
// same Channel service names.
[[nodiscard]] std::string unique_runtime_graph_name(std::string_view base);

[[nodiscard]] bool wait_for_runtime_phase_count(
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    detail::RuntimePhaseCounter counter, std::uint64_t minimum);

// Shared probe for lifecycle restart tests. Records the first and second sink
// values across a restart boundary and optionally retains the escaped Tensor to
// race session teardown against backing-claim release.
// Used by: DirectTensorPoolDeepRestart*, DirectTensorPoolShallowRestart*,
//          DirectTensorSameStreamElision*, DirectTensorMixedCatalogFanout*
struct RuntimeDirectTensorRestartProbeState {
  static void reset(LifecycleStage stage = LifecycleStage::kAllocate,
                    cudaStream_t consumer = nullptr, bool retain_without_host_wait = false) {
    const std::lock_guard lock(mutex);
    sink_count = 0U;
    first_value = 0U;
    second_value = 0U;
    source_completed = false;
    restart_accepted = false;
    stale_guard_rejected = false;
    restart_allocate_count = 0U;
    restart_stage = stage;
    consumer_stream = consumer;
    skip_host_wait = retain_without_host_wait;
    predecessor_source_stream = nullptr;
    successor_source_stream = nullptr;
    escaped_tensor.reset();
    error.reset();
  }

  static void record_source_stream(cudaStream_t stream, bool successor) {
    const std::lock_guard lock(mutex);
    (successor ? successor_source_stream : predecessor_source_stream) = stream;
  }

  static void record_sink(const Tensor& tensor, std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      if (sink_count == 0U) {
        first_value = value;
      } else if (sink_count == 1U) {
        second_value = value;
      }
      escaped_tensor = tensor;
      ++sink_count;
    }
    cv.notify_all();
  }

  static void record_source_complete(bool accepted, bool stale_rejected) {
    {
      const std::lock_guard lock(mutex);
      restart_accepted = accepted;
      stale_guard_rejected = stale_rejected;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, std::chrono::seconds{5}, [] { return source_completed && sink_count >= 2U; });
  }

  [[nodiscard]] static bool first_sample_observed() {
    const std::lock_guard lock(mutex);
    return sink_count != 0U;
  }

  [[nodiscard]] static LifecycleStage requested_restart_stage() {
    const std::lock_guard lock(mutex);
    return restart_stage;
  }

  [[nodiscard]] static cudaStream_t selected_consumer_stream() {
    const std::lock_guard lock(mutex);
    return consumer_stream;
  }

  static void release_escaped_tensor() {
    const std::lock_guard lock(mutex);
    escaped_tensor.reset();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::size_t sink_count{};
  static inline std::uint64_t first_value{};
  static inline std::uint64_t second_value{};
  static inline bool source_completed{};
  static inline bool restart_accepted{};
  static inline bool stale_guard_rejected{};
  static inline std::uint32_t restart_allocate_count{};
  static inline LifecycleStage restart_stage{LifecycleStage::kAllocate};
  static inline bool skip_host_wait{};
  static inline cudaStream_t consumer_stream{};
  static inline cudaStream_t predecessor_source_stream{};
  static inline cudaStream_t successor_source_stream{};
  static inline std::optional<Tensor> escaped_tensor;
  static inline std::optional<ErrorCode> error;
};

// Graph topology:
//   source --[queue_depth=2]--> sink
//          --[queue_depth=2]--> exported  (optional, when add_catalog_export=true)
// Self-restart is permitted on source; lifecycle transaction capacity is expanded for
// the restart fence. Device binding is injected via compile-with-runtime-evidence.
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_restart_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control = {},
    bool add_catalog_export = false, std::string_view graph_name = "runtime-direct-tensor-restart");

}  // namespace holoscan::test::run_session_tensor
