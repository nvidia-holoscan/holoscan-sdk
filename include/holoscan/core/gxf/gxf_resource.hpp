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

#ifndef HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP
#define HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP

#include <string>

#include <gxf/core/component.hpp>

#include "../resource.hpp"

#include "./gxf_component.hpp"
#include "./gxf_utils.hpp"

namespace holoscan::gxf {

class GXFResource : public holoscan::Resource, public gxf::GXFComponent {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(GXFResource, holoscan::Resource)
  GXFResource() = default;
  GXFResource(const std::string& name, nvidia::gxf::Component* component);

  void initialize() override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_RESOURCE_HPP */
