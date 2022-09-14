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

#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"

namespace holoscan {

void VideoStreamSerializer::setup(ComponentSpec& spec) {
  spec.param(component_serializers_,
             "component_serializers",
             "Component serializers",
             "List of serializers for serializing and deserializing components");
}

void VideoStreamSerializer::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto component_serializer =
      frag->make_resource<holoscan::StdComponentSerializer>("component_serializer");

  add_arg(Arg("component_serializers") =
              std::vector<std::shared_ptr<Resource>>{component_serializer});

  GXFResource::initialize();
}

}  // namespace holoscan
