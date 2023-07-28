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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP

#include <string>

#include <gxf/std/transmitter.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base transmitter class.
 *
 * Transmitters are used by output ports to emit messages.
 */
class Transmitter : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Transmitter, GXFResource)
  Transmitter() = default;
  Transmitter(const std::string& name, nvidia::gxf::Transmitter* component);

  const char* gxf_typename() const override { return "nvidia::gxf::Transmitter"; }
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP */
