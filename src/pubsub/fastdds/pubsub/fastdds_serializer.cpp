/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/fastdds/pubsub/fastdds_serializer.hpp>

namespace holoscan {

FastDdsSerializer::FastDdsSerializer(cudaStream_t cuda_stream)
    : HoloEntitySerializerBase("FastDdsSerializer", cuda_stream) {}

}  // namespace holoscan
