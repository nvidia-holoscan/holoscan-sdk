/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ping_tx.hpp"

namespace nvidia {
namespace holoscan {
namespace sample {

gxf_result_t PingTx::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal");
  result &= registrar->parameter(clock_,
                                 "clock",
                                 "Clock",
                                 "Clock for Timestamp",
                                 gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(signal_vector_, "signal_vector");
  result &= registrar->parameter(signal_data_, "signal_data", "Signal data", "Signal data", {});
  result &= registrar->parameter(pool_, "pool", "Pool", "Allocator instance for output tensors.");
  return gxf::ToResultCode(result);
}

gxf_result_t PingTx::tick() {
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failure creating message entity.");
    return message.error();
  }
  auto maybe_clock = clock_.try_get();
  int64_t now;
  if (maybe_clock) {
    now = maybe_clock.value()->timestamp();
  } else {
    now = 0;
  }
  GXF_LOG_INFO("Signal vector size: %llu", signal_vector_.get().size());

  signal_->publish(message.value(), now);
  GXF_LOG_INFO("Message Sent: %d", this->count);
  GXF_LOG_INFO("Message data size: %llu  [0] = %d",
               this->signal_data_.get().size(),
               this->signal_data_.get()[0]);
  this->count = this->count + 1;
  return ToResultCode(message);
}

}  // namespace sample
}  // namespace holoscan
}  // namespace nvidia
