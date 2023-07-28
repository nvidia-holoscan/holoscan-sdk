/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_UCX_RECEIVER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_UCX_RECEIVER_HPP

#include <memory>
#include <string>

#include "./receiver.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

namespace holoscan {

constexpr int32_t kDefaultUcxPort = 13337;

/**
 * @brief UCX-based double buffer receiver class.
 *
 * The UcxReceiver class is used to receive messages from an operator within another
 * fragment of a distributed application.
 */
class UcxReceiver : public Receiver {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UcxReceiver, Receiver)
  UcxReceiver() = default;
  UcxReceiver(const std::string& name, nvidia::gxf::Receiver* component);

  const char* gxf_typename() const override { return "nvidia::gxf::UcxReceiver"; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /// @brief The IPv4 network address used by the receiver.
  std::string address();

  /// @brief The network port used by the receiver.
  int32_t port();

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  Parameter<std::string> address_;
  Parameter<int32_t> port_;  // just <int> in the GXF extension
  Parameter<std::shared_ptr<UcxSerializationBuffer>> buffer_;
  // TODO: support GPUDevice nvidia::gxf::Resource
  // nvidia::gxf::Resource<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>> gpu_device_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_UCX_RECEIVER_HPP */
