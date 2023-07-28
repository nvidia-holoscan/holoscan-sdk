/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_VIDEO_STREAM_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_VIDEO_STREAM_SERIALIZER_HPP

#include <memory>
#include <vector>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Video stream entity serializer.
 *
 * Used by VideoStreamReplayerOp to deserialize video streams and by VideoStreamRecorderOp to
 * serialize video streams.
 */
class VideoStreamSerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(VideoStreamSerializer, GXFResource)
  VideoStreamSerializer() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::stream_playback::VideoStreamSerializer";
  }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

 private:
  Parameter<std::vector<std::shared_ptr<holoscan::Resource>>> component_serializers_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_VIDEO_STREAM_SERIALIZER_HPP */
