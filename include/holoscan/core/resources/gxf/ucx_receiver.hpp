/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gxf/ucx/ucx_receiver.hpp>

#include "./receiver.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

namespace holoscan {

constexpr uint32_t kDefaultUcxPort = 13337;

/**
 * @brief UCX-based double buffer receiver class.
 *
 * The UcxReceiver class is used to receive messages from an operator within another
 * fragment of a distributed application. This is based on the same double-buffer queue as the
 * DoubleBufferReceiver, but also handles serialization/deserialization of data for sending
 * it over the network via UCX's active message APIs.
 *
 * Application authors are not expected to use this class directly. It will be automatically
 * configured for input ports specified via `Operator::setup` when `Application::add_flow` has been
 * used to make a connection across fragments of a distributed application.
 *
 * ==Parameters==
 *
 * - **capacity** (uint64_t, optional): The capacity of the double-buffer queue used by the
 * receiver. Defaults to 1.
 * - **policy** (uint64_t, optional): The policy to use when a message arrives, but there is no
 * space in the receiver. The possible values are 0: pop, 1: reject, 2: fault (Default: 2).
 * - **address** (std::string, optional): The (IPv4) address of the receiver.
 * Default of "0.0.0.0" corresponds to INADDR_ANY.
 * - **port** (uint32_t, optional): The receiver port (Default: holoscan::kDefaultUcxPort).
 * - **buffer** (std::shared_ptr<holoscan::UcxSerializationBuffer>, optional): The serialization
 * buffer that should be used. This defaults to `UcxSerializationBuffer` and should not need to be
 * set by the application author.
 */
class UcxReceiver : public Receiver {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UcxReceiver, Receiver)
  UcxReceiver() = default;
  UcxReceiver(const std::string& name, nvidia::gxf::Receiver* component);

  const char* gxf_typename() const override { return "holoscan::HoloscanUcxReceiver"; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /// @brief The IPv4 network address used by the receiver.
  std::string address();

  /// @brief The network port used by the receiver.
  uint32_t port();

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

  nvidia::gxf::UcxReceiver* get() const;

  /// @brief Enable tracking in the underlying holoscan::HoloscanUcxReceiver class
  void track();

 private:
  Parameter<std::string> address_;
  Parameter<uint32_t> port_;
  Parameter<std::shared_ptr<holoscan::UcxSerializationBuffer>> buffer_;
  // TODO(unknown): support GPUDevice nvidia::gxf::Resource
  // // nvidia::gxf::Resource<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>> gpu_device_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_UCX_RECEIVER_HPP */
