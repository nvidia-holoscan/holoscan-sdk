/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./ping_rx.hpp"

namespace nvidia {
namespace holoscan {
namespace sample {

gxf_result_t PingRx::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal");
  return gxf::ToResultCode(result);
}

gxf_result_t PingRx::tick() {
  auto message = signal_->receive();
  GXF_LOG_INFO("Message Received: %d", this->count);
  this->count = this->count + 1;
  if (!message || message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }
  return GXF_SUCCESS;
}

}  // namespace sample
}  // namespace holoscan
}  // namespace nvidia