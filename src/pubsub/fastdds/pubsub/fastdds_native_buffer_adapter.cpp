/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/pubsub/fastdds/pubsub/fastdds_native_buffer_adapter.hpp>

#include <utility>

#include <holoscan/logger/logger.hpp>

namespace holoscan {

using nvidia::gxf::Expected;
using nvidia::gxf::Unexpected;

FastDdsNativeBufferAdapter::FastDdsNativeBufferAdapter()
    : HoloIpcCudaNativeBufferAdapterBase("FastDdsNativeBufferAdapter") {}

Expected<void> FastDdsNativeBufferAdapter::initialize(
    eprosima::fastdds::dds::DomainParticipant* participant,
    nvidia::gxf::NativeBufferPolicy policy) {
  if (policy == nvidia::gxf::NativeBufferPolicy::kDisabled) {
    return initialize_common(nullptr, policy);
  }

  if (!participant) {
    HOLOSCAN_LOG_ERROR("FastDdsNativeBufferAdapter::initialize: null participant");
    return Unexpected(GXF_ARGUMENT_NULL);
  }

  auto ipc_context = ipc::make_context<ipc::transport::fastdds::FastDdsTransport>(participant);
  return initialize_common(std::move(ipc_context), policy);
}

}  // namespace holoscan
