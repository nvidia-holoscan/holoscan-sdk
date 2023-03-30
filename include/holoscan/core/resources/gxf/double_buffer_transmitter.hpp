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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP

#include <string>
// gxf/std/double_buffer_transmitter.hpp is missing in the GXF SDK
// The following is a copy of the file from the GXF SDK until it is fixed
// TODO: Remove this file once the issue is fixed
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <memory>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/gems/staging_queue/staging_queue.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// A transmitter which uses a double-buffered queue where messages are pushed to a backstage after
// they are published. Outgoing messages are not immediately available and need to be
// moved to the backstage first.
class DoubleBufferTransmitter : public Transmitter {
 public:
  using queue_t = ::gxf::staging_queue::StagingQueue<Entity>;

  gxf_result_t registerInterface(Registrar* registrar) override;

  gxf_result_t initialize() override;

  gxf_result_t deinitialize() override;

  gxf_result_t pop_abi(gxf_uid_t* uid) override;

  gxf_result_t push_abi(gxf_uid_t other) override;

  gxf_result_t peek_abi(gxf_uid_t* uid, int32_t index) override;

  size_t capacity_abi() override;

  size_t size_abi() override;

  gxf_result_t publish_abi(gxf_uid_t uid) override;

  size_t back_size_abi() override;

  gxf_result_t sync_abi() override;

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  std::unique_ptr<queue_t> queue_;
};

}  // namespace gxf
}  // namespace nvidia
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "./transmitter.hpp"

namespace holoscan {

class DoubleBufferTransmitter : public Transmitter {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(DoubleBufferTransmitter, Transmitter)
  DoubleBufferTransmitter() = default;
  DoubleBufferTransmitter(const std::string& name, nvidia::gxf::DoubleBufferTransmitter* component);

  const char* gxf_typename() const override { return "nvidia::gxf::DoubleBufferTransmitter"; }

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_TRANSMITTER_HPP */
