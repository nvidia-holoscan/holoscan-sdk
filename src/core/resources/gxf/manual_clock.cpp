/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/manual_clock.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

ManualClock::ManualClock(const std::string& name, nvidia::gxf::ManualClock* component)
    : Clock(name, component) {
  uint64_t initial_timestamp = 0L;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfParameterGetUInt64(gxf_context_, gxf_cid_, "initial_timestamp", &initial_timestamp));
  initial_timestamp_ = initial_timestamp;
}

double ManualClock::time() const {
  if (gxf_cptr_) {
    nvidia::gxf::ManualClock* clock = static_cast<nvidia::gxf::ManualClock*>(gxf_cptr_);
    return clock->time();
  }
  return 0.0;
}

int64_t ManualClock::timestamp() const {
  if (gxf_cptr_) {
    nvidia::gxf::ManualClock* clock = static_cast<nvidia::gxf::ManualClock*>(gxf_cptr_);
    return clock->timestamp();
  }
  return 0;
}

void ManualClock::sleep_for(int64_t duration_ns) {
  if (gxf_cptr_) {
    nvidia::gxf::ManualClock* clock = static_cast<nvidia::gxf::ManualClock*>(gxf_cptr_);
    clock->sleepFor(duration_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void ManualClock::sleep_until(int64_t target_time_ns) {
  if (gxf_cptr_) {
    nvidia::gxf::ManualClock* clock = static_cast<nvidia::gxf::ManualClock*>(gxf_cptr_);
    clock->sleepUntil(target_time_ns);
  } else {
    HOLOSCAN_LOG_ERROR("RealtimeClock component not yet registered with GXF");
  }
}

void ManualClock::setup(ComponentSpec& spec) {
  spec.param(initial_timestamp_,
             "initial_timestamp",
             "Initial timestamp",
             "The initial timestamp on the clock (in nanoseconds).",
             0L);
}

}  // namespace holoscan
