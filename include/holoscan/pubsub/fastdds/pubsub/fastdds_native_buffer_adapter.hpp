/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_NATIVE_BUFFER_ADAPTER_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_NATIVE_BUFFER_ADAPTER_HPP

#include <fastdds/dds/domain/DomainParticipant.hpp>

#include <holoscan/ipc/transport/fastdds/fast_dds_transport.hpp>

#include <holoscan/pubsub/common/holoipc_cuda_native_buffer_adapter_base.hpp>

namespace holoscan {

/// Fast-DDS native-buffer adapter: thin wrapper that creates ipc::Context<FastDdsTransport>.
class FastDdsNativeBufferAdapter
    : public HoloIpcCudaNativeBufferAdapterBase<ipc::transport::fastdds::FastDdsTransport> {
 public:
  FastDdsNativeBufferAdapter();

  /// Initialize with the DomainParticipant from FastDdsPubSubContext.
  ///
  /// The @p participant is retained indirectly through the holoipc Fast-DDS transport
  /// and must outlive this adapter (including its shutdown). When @p policy is
  /// `NativeBufferPolicy::kDisabled`, @p participant is ignored and may be null.
  ///
  /// This method is idempotent: repeated calls after a successful initialization are
  /// no-ops that log a warning and return success, keeping the original policy and
  /// context.
  nvidia::gxf::Expected<void> initialize(eprosima::fastdds::dds::DomainParticipant* participant,
                                         nvidia::gxf::NativeBufferPolicy policy);
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_NATIVE_BUFFER_ADAPTER_HPP */
