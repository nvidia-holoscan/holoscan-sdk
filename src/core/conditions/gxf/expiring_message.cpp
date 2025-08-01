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

#include "holoscan/core/conditions/gxf/expiring_message.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

void ExpiringMessageAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Queue channel",
             "The scheduling term permits execution if this channel has at least a given number of "
             "messages available.");
  spec.param(clock_, "clock", "Clock", "The clock to be used to get the time from.");
  spec.param(max_batch_size_,
             "max_batch_size",
             "Maximum Batch Size",
             "The maximum number of messages to be batched together");
  spec.param(max_delay_ns_,
             "max_delay_ns",
             "Maximum delay in nano seconds.",
             "The maximum delay from first message to wait before submitting workload");
}

nvidia::gxf::ExpiringMessageAvailableSchedulingTerm* ExpiringMessageAvailableCondition::get()
    const {
  return static_cast<nvidia::gxf::ExpiringMessageAvailableSchedulingTerm*>(gxf_cptr_);
}

void ExpiringMessageAvailableCondition::initialize() {
  // Set up prerequisite parameters before calling Scheduler::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'clock'
  auto has_clock = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "clock"); });
  // Create the clock if there was no argument provided.
  if (has_clock == args().end()) {
    clock_ = frag->make_resource<holoscan::RealtimeClock>("expiring_message__realtime_clock");
    clock_->gxf_cname(clock_->name().c_str());
    if (gxf_eid_ != 0) {
      clock_->gxf_eid(gxf_eid_);
    }
    add_arg(clock_.get());
  }

  // parent class initialize() call must be after the argument additions above
  GXFCondition::initialize();
}

void ExpiringMessageAvailableCondition::max_batch_size(int64_t max_batch_size) {
  auto expiring_message = get();
  if (expiring_message) {
    // expiring_message->setMaxBatchSize(max_batch_size);
    GxfParameterSetInt64(gxf_context_, gxf_cid_, "max_batch_size", max_batch_size);
  }
  max_batch_size_ = max_batch_size;
}

void ExpiringMessageAvailableCondition::max_delay(int64_t max_delay_ns) {
  auto expiring_message = get();

  if (expiring_message) {
    HOLOSCAN_LOG_INFO("Setting max delay to {}", max_delay_ns);
    // expiring_message->setMaxDelayNs(max_delay_ns);
    GxfParameterSetInt64(gxf_context_, gxf_cid_, "max_delay_ns", max_delay_ns);
  }
  max_delay_ns_ = max_delay_ns;
}

int64_t ExpiringMessageAvailableCondition::max_delay_ns() {
  auto expiring_message = get();
  if (expiring_message) {
    int64_t max_delay_ns = 0;
    // int64_t max_delay_ns = expiring_message->maxDelayNs();
    GxfParameterGetInt64(gxf_context_, gxf_cid_, "max_delay_ns", &max_delay_ns);
    if (max_delay_ns != max_delay_ns_) {
      max_delay_ns_ = max_delay_ns;
    }
  }
  return max_delay_ns_;
}

}  // namespace holoscan
