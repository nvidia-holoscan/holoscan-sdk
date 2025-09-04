/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/schedulers/gxf/event_based_scheduler.hpp"

#include <memory>

#include "holoscan/core/clock.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"

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
             0L);
  spec.param(pin_cores_,
             "pin_cores",
             "Pin Cores",
             "CPU core IDs to pin the worker threads to. If specified, all the worker threads "
             "created based on the parameter `worker_thread_number` will be pinned to the same "
             "set of specified cores. If not specified, the worker threads will not be pinned to "
             "any cores.",
             ParameterFlag::kOptional);
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
