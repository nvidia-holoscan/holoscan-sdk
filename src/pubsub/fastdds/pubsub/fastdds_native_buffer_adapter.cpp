/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
