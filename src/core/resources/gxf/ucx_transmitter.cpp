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

#include "holoscan/core/resources/gxf/ucx_transmitter.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "gxf/ucx/ucx_serialization_buffer.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/resources/gxf/holoscan_ucx_transmitter.hpp"
#include "holoscan/core/resources/gxf/ucx_receiver.hpp"  // for kDefaultUcxPort
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

namespace holoscan {

UcxTransmitter::UcxTransmitter(const std::string& name, nvidia::gxf::Transmitter* component)
    : Transmitter(name, component) {
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (!maybe_capacity) {
    throw std::runtime_error("Failed to get capacity");
  }
  capacity_ = maybe_capacity.value();

  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (!maybe_policy) {
    throw std::runtime_error("Failed to get policy");
  }
  policy_ = maybe_policy.value();

  auto maybe_receiver_address = component->getParameter<std::string>("receiver_address");
  if (!maybe_receiver_address) {
    throw std::runtime_error("Failed to get receiver_address");
  }
  receiver_address_ = maybe_receiver_address.value();

  auto maybe_port = component->getParameter<uint32_t>("port");
  if (!maybe_port) {
    throw std::runtime_error("Failed to get port");
  }
  port_ = maybe_port.value();

  auto maybe_local_address = component->getParameter<std::string>("local_address");
  if (!maybe_local_address) {
    throw std::runtime_error("Failed to get local_address");
  }
  local_address_ = maybe_local_address.value();

  auto maybe_local_port = component->getParameter<uint32_t>("local_port");
  if (!maybe_local_port) {
    throw std::runtime_error("Failed to get local_port");
  }
  local_port_ = maybe_local_port.value();

  auto maybe_max_retry = component->getParameter<uint32_t>("max_retry");
  if (!maybe_max_retry) {
    throw std::runtime_error("Failed to get maximum_connection_retries");
  }
  maximum_connection_retries_ = maybe_max_retry.value();

  // get the serialization buffer object
  auto maybe_buffer =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::UcxSerializationBuffer>>("buffer");
  if (!maybe_buffer) {
    throw std::runtime_error("Failed to get buffer");
  }
  auto buffer_handle = maybe_buffer.value();
  buffer_ = std::make_shared<holoscan::UcxSerializationBuffer>(std::string{buffer_handle->name()},
                                                               buffer_handle.get());
}

void UcxTransmitter::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxTransmitter::setup");
  spec.param(capacity_, "capacity", "Capacity", "", 1UL);
  auto default_policy = holoscan::gxf::get_default_queue_policy();
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", default_policy);
  spec.param(receiver_address_,
             "receiver_address",
             "Receiver address",
             "Address to connect to (IPv4). Default of 0.0.0.0 corresponds to INADDR_ANY.",
             std::string("0.0.0.0"));
  spec.param(local_address_,
             "local_address",
             "Local address",
             "Local Address to use for connection. Default of 0.0.0.0 corresponds to INADDR_ANY.",
             std::string("0.0.0.0"));
  spec.param(maximum_connection_retries_,
             "maximum_connection_retries",
             "Maximum Connection Retries",
             "Maximum Connection Retries",
             static_cast<uint32_t>(10));
  spec.param(port_, "port", "Receiver Port", "Receiver Port", kDefaultUcxPort);
  spec.param(local_port_,
             "local_port",
             "Local port",
             "Local Port to use for connection",
             static_cast<uint32_t>(0));

  spec.param(buffer_, "buffer", "Serialization Buffer", "");

  // TODO(unknown): implement OperatorSpec::resource for managing nvidia::gxf:Resource types
  // spec.resource(gpu_device_, "Optional GPU device resource");
}

nvidia::gxf::UcxTransmitter* UcxTransmitter::get() const {
  return static_cast<nvidia::gxf::UcxTransmitter*>(gxf_cptr_);
}

void UcxTransmitter::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxTransmitter::initialize");
  // Set up prerequisite parameters before calling GXFResource::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'buffer'
  auto has_buffer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "buffer"); });
  // Create an UcxSerializationBuffer if no buffer was provided
  if (has_buffer == args().end()) {
    auto buffer =
        frag->make_resource<holoscan::UcxSerializationBuffer>("ucx_tx_serialization_buffer");
    add_arg(Arg("buffer") = buffer);
    buffer->gxf_cname(buffer->name().c_str());
    if (gxf_eid_ != 0) {
      buffer->gxf_eid(gxf_eid_);
    }
  } else {
    // must set the gxf_eid for the provided buffer or GXF parameter registration will fail
    auto buffer_arg = *has_buffer;
    auto buffer = std::any_cast<std::shared_ptr<Resource>>(buffer_arg.value());
    auto gxf_buffer_resource = std::dynamic_pointer_cast<gxf::GXFResource>(buffer);
    if (gxf_eid_ != 0 && gxf_buffer_resource->gxf_eid() == 0) {
      HOLOSCAN_LOG_TRACE("buffer '{}': setting gxf_eid({}) from UcxTransmitter '{}'",
                         buffer->name(),
                         gxf_eid_,
                         name());
      gxf_buffer_resource->gxf_eid(gxf_eid_);
    }
  }
  GXFResource::initialize();
}

std::string UcxTransmitter::receiver_address() {
  return receiver_address_.get();
}

uint32_t UcxTransmitter::port() {
  return port_.get();
}

std::string UcxTransmitter::local_address() {
  return local_address_.get();
}

uint32_t UcxTransmitter::local_port() {
  return local_port_.get();
}

void UcxTransmitter::track() {
  auto transmitter_ptr = static_cast<holoscan::HoloscanUcxTransmitter*>(gxf_cptr_);
  if (transmitter_ptr)
    transmitter_ptr->track();
  else
    HOLOSCAN_LOG_ERROR("Failed to track UcxTransmitter");
}

}  // namespace holoscan
