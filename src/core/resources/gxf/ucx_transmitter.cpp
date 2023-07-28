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

#include "holoscan/core/resources/gxf/ucx_transmitter.hpp"

#include <memory>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/ucx_receiver.hpp"  // for kDefaultUcxPort
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

namespace holoscan {

UcxTransmitter::UcxTransmitter(const std::string& name, nvidia::gxf::Transmitter* component)
    : Transmitter(name, component) {
  uint64_t capacity = 0;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetUInt64(gxf_context_, gxf_cid_, "capacity", &capacity));
  capacity_ = capacity;
  uint64_t policy = 0;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetUInt64(gxf_context_, gxf_cid_, "policy", &policy));
  policy_ = policy;
  const char* receiver_address;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfParameterGetStr(gxf_context_, gxf_cid_, "receiver_address", &receiver_address));
  receiver_address_ = std::string(receiver_address);
  int32_t port = 0;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetInt32(gxf_context_, gxf_cid_, "port", &port));
  port_ = port;
  int32_t maximum_connection_retries = 0;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetInt32(
      gxf_context_, gxf_cid_, "maximum_connection_retries", &maximum_connection_retries));
  maximum_connection_retries_ = maximum_connection_retries;

  // get the serialization buffer object
  gxf_uid_t buffer_cid;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetHandle(gxf_context_, gxf_cid_, "buffer", &buffer_cid));
  gxf_tid_t ucx_serialization_buffer_tid{};
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
      gxf_context_, "nvidia::gxf::UcxSerializationBuffer", &ucx_serialization_buffer_tid));
  nvidia::gxf::SerializationBuffer* buffer_ptr;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(
      gxf_context_, gxf_cid_, ucx_serialization_buffer_tid, reinterpret_cast<void**>(&buffer_ptr)));
  buffer_ = std::make_shared<UcxSerializationBuffer>(std::string{buffer_ptr->name()}, buffer_ptr);
}

void UcxTransmitter::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxTransmitter::setup");
  spec.param(capacity_, "capacity", "Capacity", "", 1UL);
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", 2UL);
  spec.param(receiver_address_,
             "receiver_address",
             "Receiver address",
             "Address to connect to (IPv4). Default of 0.0.0.0 corresponds to INADDR_ANY.",
             std::string("0.0.0.0"));
  spec.param(maximum_connection_retries_,
             "maximum_connection_retries",
             "Maximum Connection Retries",
             "Maximum Connection Retries",
             10);
  spec.param(port_, "port", "Receiver Port", "Receiver Port", kDefaultUcxPort);
  spec.param(buffer_, "buffer", "Serialization Buffer", "");

  // TODO: implement OperatorSpec::resource for managing nvidia::gxf:Resource types
  // spec.resource(gpu_device_, "Optional GPU device resource");
}

void UcxTransmitter::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxTransmitter::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'buffer'
  auto has_buffer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "buffer"); });
  // Create an UcxSerializationBuffer if no buffer was provided
  if (has_buffer == args().end()) {
    auto buffer =
        frag->make_resource<holoscan::UcxSerializationBuffer>("ucx_tx_serialization_buffer");
    add_arg(Arg("buffer") = buffer);
  }
  GXFResource::initialize();
}

std::string UcxTransmitter::receiver_address() {
  return receiver_address_.get();
}

int32_t UcxTransmitter::port() {
  return port_.get();
}

}  // namespace holoscan
