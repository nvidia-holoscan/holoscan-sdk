/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <holoscan/core/schedulers/gxf/event_based_scheduler.hpp>

#include <memory>

#include <holoscan/core/clock.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/resources/gxf/realtime_clock.hpp>

namespace holoscan {

void EventBasedScheduler::setup(ComponentSpec& spec) {
  spec.param(clock_,
             "clock",
             "Clock",
             "The clock used by the scheduler to define flow of time. Typically this "
             "would be a std::shared_ptr<RealtimeClock>.");
  spec.param(
      worker_thread_number_, "worker_thread_number", "Thread Number", "Number of threads", 1L);
  spec.param(stop_on_deadlock_,
             "stop_on_deadlock",
             "Stop on dead end",
             "If enabled the scheduler will stop when all entities are in a waiting state, but "
             "no periodic entity exists to break the dead end. Should be disabled when "
             "scheduling conditions can be changed by external actors, for example by clearing "
             "queues manually.",
             true);
  spec.param(max_duration_ms_,
             "max_duration_ms",
             "Max Duration [ms]",
             "The maximum duration for which the scheduler will execute (in ms). If not "
             "specified the scheduler will run until all work is done. If periodic terms are "
             "present this means the  application will run indefinitely",
             ParameterFlag::kOptional);
  spec.param(stop_on_deadlock_timeout_,
             "stop_on_deadlock_timeout",
             "Delay (in ms) until stop_on_deadlock kicks in",
             "Scheduler will wait this amount of time (in ms) before determining that it is in "
             "deadlock and should stop. It will reset if a job comes in during the wait. A "
             "negative value means not stop on deadlock. This parameter only applies when  "
             "stop_on_deadlock=true",
             int64_t(0));
  spec.param(network_connection_timeout_,
             "network_connection_timeout",
             "Timeout for network connection establishment (in ms)",
             "During the initial phase when network connections are being established, this longer "
             "timeout is used instead of stop_on_deadlock_timeout. This allows sufficient time for "
             "UCX connections to be established without triggering false deadlock detection. "
             "This parameter has no effect on single fragment (non-distributed) applications. "
             "Defaults to 5000ms (5 seconds).",
             int64_t(5000));
  spec.param(pin_cores_,
             "pin_cores",
             "Pin Cores",
             "CPU core IDs to pin the worker threads to. If specified, all the worker threads "
             "created based on the parameter `worker_thread_number` will be pinned to the same "
             "set of specified cores. If not specified, the worker threads will not be pinned to "
             "any cores.",
             ParameterFlag::kOptional);
  spec.param(enable_queue_stealing_,
             "enable_queue_stealing",
             "Enable Queue Stealing",
             "If true, default worker threads attempt to steal ready jobs from other default "
             "worker queues before blocking on their own queue.",
             false);
  spec.param(steal_scan_limit_,
             "steal_scan_limit",
             "Steal Scan Limit",
             "Maximum number of victim queues scanned per steal attempt (0 means scan all "
             "queues).",
             int64_t(0));
  spec.param(enable_worker_postcheck_fastpath_,
             "enable_worker_postcheck_fastpath",
             "Enable Worker Postcheck Fast Path",
             "If true, workers perform a fresh checkEntity() after executeEntity() and directly "
             "update READY/WAIT_TIME conditions without routing that entity through dispatcher.",
             false);
  spec.param(postcheck_fallback_notify_interval_,
             "postcheck_fallback_notify_interval",
             "Postcheck Fallback Notify Interval",
             "When worker postcheck returns a non-ready state, send a periodic dispatcher "
             "wake-up every N fallbacks per worker. Set to 0 to only notify when no workers are "
             "running.",
             int64_t(256));
  spec.param(postcheck_fallback_notify_min_workers_,
             "postcheck_fallback_notify_min_workers",
             "Postcheck Fallback Notify Min Workers",
             "Periodic fallback notify is enabled only when worker_thread_number is at least "
             "this value.",
             int64_t(8));
  spec.param(postcheck_fallback_notify_min_period_ns_,
             "postcheck_fallback_notify_min_period_ns",
             "Postcheck Fallback Notify Min Period [ns]",
             "Minimum global time spacing between periodic fallback dispatcher wake-ups.",
             int64_t(100000));
  spec.param(internal_event_shard_count_,
             "internal_event_shard_count",
             "Internal Event Shard Count",
             "Number of internal notification shards used by notifyDispatcher (0 = auto = "
             "worker_thread_number).",
             int64_t(0));
  spec.param(dispatcher_internal_pop_batch_size_,
             "dispatcher_internal_pop_batch_size",
             "Dispatcher Internal Pop Batch Size",
             "Maximum number of internal notifications drained from one shard per dispatcher pop "
             "step.",
             int64_t(32));
  spec.param(wait_state_shard_count_,
             "wait_state_shard_count",
             "Wait State Shard Count",
             "Number of shards used for WAIT_EVENT and WAIT tracking lists.",
             int64_t(1));
  spec.param(log_perf_stats_,
             "log_perf_stats",
             "Log Perf Stats",
             "If true, logs scheduler instrumentation counters during deinitialize().",
             false);
}

nvidia::gxf::EventBasedScheduler* EventBasedScheduler::get() const {
  return static_cast<nvidia::gxf::EventBasedScheduler*>(gxf_cptr_);
}

std::shared_ptr<Clock> EventBasedScheduler::clock() {
  if (clock_.has_value()) {
    // Create a Clock resource that wraps the gxf::Clock implementation
    return std::make_shared<Clock>(std::static_pointer_cast<ClockInterface>(clock_.get()));
  }
  return nullptr;
}

void EventBasedScheduler::initialize() {
  // Set up prerequisite parameters before calling Scheduler::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'clock'
  auto has_clock = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "clock"); });
  // Create the clock if there was no argument provided.
  if (has_clock == args().end()) {
    clock_ = frag->make_resource<holoscan::RealtimeClock>("event_based_scheduler__realtime_clock");
    clock_->gxf_cname(clock_->name().c_str());
    if (gxf_eid_ != 0) {
      clock_->gxf_eid(gxf_eid_);
    }
    add_arg(clock_.get());
  }

  // parent class initialize() call must be after the argument additions above
  Scheduler::initialize();
}

void* EventBasedScheduler::clock_gxf_cptr() const {
  if (clock_.has_value() && clock_.get() != nullptr) {
    return clock_.get()->gxf_cptr();
  }
  return nullptr;
}

}  // namespace holoscan
