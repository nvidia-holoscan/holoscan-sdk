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
