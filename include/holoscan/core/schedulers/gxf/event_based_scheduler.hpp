/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP
#define HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/event_based_scheduler.hpp>
#include "../../gxf/gxf_scheduler.hpp"
#include "../../resources/gxf/clock.hpp"

namespace holoscan {

/**
 * @brief Event-based scheduler.
 *
 * This is a multi-thread scheduler that uses an event-based design. Unlike the
 * `MultiThreadScheduler`, it does not utilize a dedicated polling thread that is constantly
 * polling operators to check which are ready to execute. Instead, certain events in the underlying
 * framework will indicate that the scheduling status of an operator should be checked.
 *
 * ==Parameters==
 *
 * - **worker_thread_number** (int64_t): The number of (CPU) worker threads to use for executing
 * operators. Defaults to 1. This creates a default thread pool. Operators not explicitly assigned
 * to a user-defined thread pool (via make_thread_pool) will use this default pool.
 * - **pin_cores** (list of int, optional): CPU core IDs to pin the default thread pool's worker
 * threads to (empty means no core pinning). Note: This only affects the default pool; to control
 * CPU affinity for user-defined thread pools, use the pin_cores parameter in ThreadPool::add().
 * - **stop_on_deadlock** (bool): If True, the application will terminate if a deadlock state is
 * reached. Defaults to true.
 * - **stop_on_deadlock_timeout** (int64_t): The amount of time (in ms) before an application is
 * considered to be in deadlock. Defaults to 0.
 * - **max_duration_ms** (int64_t, optional): Terminate the application after the specified
 * duration even if deadlock does not occur. If unspecified, the application can run indefinitely.
 * - **enable_queue_stealing** (bool): If true, default worker threads attempt to steal ready jobs
 * from other default worker queues before blocking on their own queue. Defaults to false.
 * - **steal_scan_limit** (int64_t): Maximum number of victim queues scanned per steal attempt
 * (0 means scan all queues). Defaults to 0.
 * - **enable_worker_postcheck_fastpath** (bool): If true, workers perform a fresh checkEntity()
 * after executeEntity() and directly update READY/WAIT_TIME conditions without routing that entity
 * through the dispatcher. Defaults to false.
 * - **postcheck_fallback_notify_interval** (int64_t): When worker postcheck returns a non-ready
 * state, send a periodic dispatcher wake-up every N fallbacks per worker. Set to 0 to only notify
 * when no workers are running. Defaults to 256.
 * - **postcheck_fallback_notify_min_workers** (int64_t): Periodic fallback notify is enabled only
 * when worker_thread_number is at least this value. Defaults to 8.
 * - **postcheck_fallback_notify_min_period_ns** (int64_t): Minimum global time spacing (in
 * nanoseconds) between periodic fallback dispatcher wake-ups. Defaults to 100000.
 * - **internal_event_shard_count** (int64_t): Number of internal notification shards used by
 * notifyDispatcher (0 = auto = worker_thread_number). Defaults to 0.
 * - **dispatcher_internal_pop_batch_size** (int64_t): Maximum number of internal notifications
 * drained from one shard per dispatcher pop step. Defaults to 32.
 * - **wait_state_shard_count** (int64_t): Number of shards used for WAIT_EVENT and WAIT tracking
 * lists. Defaults to 1.
 * - **log_perf_stats** (bool): If true, logs scheduler instrumentation counters during
 * deinitialize(). Defaults to false.
 */
class EventBasedScheduler : public gxf::GXFScheduler {
 public:
  HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(EventBasedScheduler, gxf::GXFScheduler)
  EventBasedScheduler() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::EventBasedScheduler"; }

  std::shared_ptr<Clock> clock() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // Parameter getters used for printing scheduler description (e.g. for Python __repr__)
  int64_t worker_thread_number() const { return worker_thread_number_; }
  bool stop_on_deadlock() const { return stop_on_deadlock_; }
  int64_t stop_on_deadlock_timeout() const { return stop_on_deadlock_timeout_; }
  int64_t network_connection_timeout() const { return network_connection_timeout_; }
  // could return std::optional<int64_t>, but just using int64_t simplifies the Python bindings
  int64_t max_duration_ms() const {
    return max_duration_ms_.has_value() ? max_duration_ms_.get() : -1;
  }
  std::vector<uint32_t> pin_cores() const {
    return pin_cores_.has_value() ? pin_cores_.get() : std::vector<uint32_t>{};
  }
  bool enable_queue_stealing() { return enable_queue_stealing_; }
  int64_t steal_scan_limit() { return steal_scan_limit_; }
  bool enable_worker_postcheck_fastpath() { return enable_worker_postcheck_fastpath_; }
  int64_t postcheck_fallback_notify_interval() { return postcheck_fallback_notify_interval_; }
  int64_t postcheck_fallback_notify_min_workers() { return postcheck_fallback_notify_min_workers_; }
  int64_t postcheck_fallback_notify_min_period_ns() {
    return postcheck_fallback_notify_min_period_ns_;
  }
  int64_t internal_event_shard_count() { return internal_event_shard_count_; }
  int64_t dispatcher_internal_pop_batch_size() { return dispatcher_internal_pop_batch_size_; }
  int64_t wait_state_shard_count() { return wait_state_shard_count_; }
  bool log_perf_stats() { return log_perf_stats_; }

  nvidia::gxf::EventBasedScheduler* get() const;

 private:
  Parameter<std::shared_ptr<gxf::Clock>> clock_;
  Parameter<int64_t> worker_thread_number_;
  Parameter<bool> stop_on_deadlock_;
  Parameter<int64_t> max_duration_ms_;
  Parameter<int64_t> stop_on_deadlock_timeout_;    // in ms
  Parameter<int64_t> network_connection_timeout_;  // in ms
  Parameter<std::vector<uint32_t>> pin_cores_;     // CPU core IDs to pin the worker threads to
  Parameter<bool> enable_queue_stealing_;
  Parameter<int64_t> steal_scan_limit_;
  Parameter<bool> enable_worker_postcheck_fastpath_;
  Parameter<int64_t> postcheck_fallback_notify_interval_;
  Parameter<int64_t> postcheck_fallback_notify_min_workers_;
  Parameter<int64_t> postcheck_fallback_notify_min_period_ns_;
  Parameter<int64_t> internal_event_shard_count_;
  Parameter<int64_t> dispatcher_internal_pop_batch_size_;
  Parameter<int64_t> wait_state_shard_count_;
  Parameter<bool> log_perf_stats_;
  // The following parameter needs to wait on ThreadPool support
  // Parameter<bool> thread_pool_allocation_auto_;

  void* clock_gxf_cptr() const override;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP */
