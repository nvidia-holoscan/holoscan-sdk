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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP

#include "./transmitter.hpp"

namespace holoscan {

class DoubleBufferTransmitter : public Transmitter {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(DoubleBufferTransmitter, Transmitter)
  DoubleBufferTransmitter() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::DoubleBufferTransmitter"; }

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP */
