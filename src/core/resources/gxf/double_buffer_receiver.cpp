/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/resources/gxf/annotated_double_buffer_receiver.hpp"

namespace holoscan {

DoubleBufferReceiver::DoubleBufferReceiver(const std::string& name,
                                           nvidia::gxf::DoubleBufferReceiver* component)
    : Receiver(name, component) {
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (!maybe_capacity) {
    throw std::runtime_error("Failed to get capacity");
  }
  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (!maybe_policy) {
    throw std::runtime_error("Failed to get policy");
  }
  capacity_ = maybe_capacity.value();
  policy_ = maybe_policy.value();
}

DoubleBufferReceiver::DoubleBufferReceiver(const std::string& name,
                                           AnnotatedDoubleBufferReceiver* component)
    : Receiver(name, component) {
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (!maybe_capacity) {
    throw std::runtime_error("Failed to get capacity");
  }
  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (!maybe_policy) {
    throw std::runtime_error("Failed to get policy");
  }
  capacity_ = maybe_capacity.value();
  policy_ = maybe_policy.value();
  tracking_ = true;
}

nvidia::gxf::DoubleBufferReceiver* DoubleBufferReceiver::get() const {
  return static_cast<nvidia::gxf::DoubleBufferReceiver*>(gxf_cptr_);
}

const char* DoubleBufferReceiver::gxf_typename() const {
  if (tracking_) {
    return "holoscan::AnnotatedDoubleBufferReceiver";
  } else {
    return "nvidia::gxf::DoubleBufferReceiver";
  }
}

void DoubleBufferReceiver::setup(ComponentSpec& spec) {
  spec.param(capacity_, "capacity", "Capacity", "", 1UL);
  auto default_policy = holoscan::gxf::get_default_queue_policy();
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", default_policy);
}

void DoubleBufferReceiver::track() {
  tracking_ = true;
}

}  // namespace holoscan
