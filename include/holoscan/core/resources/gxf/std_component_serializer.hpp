/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_STD_COMPONENT_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_STD_COMPONENT_SERIALIZER_HPP

#include <memory>
#include <vector>

// TODO(unknown): provide get() method once upstream issue with missing GXF header is resolved
// #include <gxf/serialization/std_component_serializer.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./unbounded_allocator.hpp"

namespace holoscan {

/**
 * @brief Standard GXF component serializer.
 *
 * This class is capable of serializing and deserialing various basic numeric types as well as
 * `nvidia::gxf::Tensor` and `nvidia::gxf::Timestamp`.
 *
 * - **allocator** (std::shared_ptr<holoscan::Allocator>, optional): The allocator used for
 * deserialization of Tensor components. Defaults to an `UnboundedAllocator` if none is provided.
 */
class StdComponentSerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(StdComponentSerializer, GXFResource)
  StdComponentSerializer() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::StdComponentSerializer"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  //  nvidia::gxf::StdComponentSerializer* get() const;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_STD_COMPONENT_SERIALIZER_HPP */
