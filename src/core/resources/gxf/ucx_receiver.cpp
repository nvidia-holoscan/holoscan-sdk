/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/ucx_receiver.hpp"

#include <memory>
#include <string>

#include "gxf/ucx/ucx_serialization_buffer.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

namespace holoscan {

UcxReceiver::UcxReceiver(const std::string& name, nvidia::gxf::Receiver* component)
    : Receiver(name, component) {
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (!maybe_capacity) { throw std::runtime_error("Failed to get capacity"); }
  capacity_ = maybe_capacity.value();

  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (!maybe_policy) { throw std::runtime_error("Failed to get policy"); }
  policy_ = maybe_policy.value();

  auto maybe_address = component->getParameter<std::string>("address");
  if (!maybe_address) { throw std::runtime_error("Failed to get address"); }
  address_ = maybe_address.value();

  auto maybe_port = component->getParameter<uint32_t>("port");
  if (!maybe_port) { throw std::runtime_error("Failed to get port"); }
  port_ = maybe_port.value();

  // get the serialization buffer object
  auto maybe_buffer =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::UcxSerializationBuffer>>("buffer");
  if (!maybe_buffer) { throw std::runtime_error("Failed to get buffer"); }
  auto buffer_handle = maybe_buffer.value();
  buffer_ = std::make_shared<holoscan::UcxSerializationBuffer>(std::string{buffer_handle->name()},
                                                               buffer_handle.get());
}

void UcxReceiver::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxReceiver::setup");
  spec.param(capacity_, "capacity", "Capacity", "", 1UL);
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", 2UL);
  spec.param(address_, "address", "RX address", "RX address", std::string("0.0.0.0"));
  spec.param(port_, "port", "rx_port", "RX port", kDefaultUcxPort);
  spec.param(buffer_, "buffer", "Serialization Buffer", "");

  // TODO: implement OperatorSpec::resource for managing nvidia::gxf:Resource types
  // spec.resource(gpu_device_, "Optional GPU device resource");
}

nvidia::gxf::UcxReceiver* UcxReceiver::get() const {
  return static_cast<nvidia::gxf::UcxReceiver*>(gxf_cptr_);
}

void UcxReceiver::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxReceiver::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'buffer'
  auto has_buffer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "buffer"); });
  // Create an UcxSerializationBuffer if no buffer was provided
  if (has_buffer == args().end()) {
    auto buffer =
        frag->make_resource<holoscan::UcxSerializationBuffer>("ucx_rx_serialization_buffer");
    add_arg(Arg("buffer") = buffer);
    buffer->gxf_cname(buffer->name().c_str());
    if (gxf_eid_ != 0) { buffer->gxf_eid(gxf_eid_); }
  }
  GXFResource::initialize();
}

std::string UcxReceiver::address() {
  return address_.get();
}

uint32_t UcxReceiver::port() {
  return port_.get();
}

}  // namespace holoscan
