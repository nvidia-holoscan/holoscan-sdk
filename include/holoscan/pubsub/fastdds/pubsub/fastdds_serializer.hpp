/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_SERIALIZER_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_SERIALIZER_HPP

#include <holoscan/pubsub/common/holo_entity_serializer_base.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_endpoint.hpp>

namespace holoscan {

/// Fast-DDS entity serializer: thin wrapper over HoloEntitySerializerBase<FastDdsEndpoint>.
class FastDdsSerializer : public HoloEntitySerializerBase<FastDdsEndpoint> {
 public:
  explicit FastDdsSerializer(cudaStream_t cuda_stream = nullptr);
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_SERIALIZER_HPP
