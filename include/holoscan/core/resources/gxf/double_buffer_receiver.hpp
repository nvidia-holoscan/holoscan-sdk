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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_RECEIVER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_RECEIVER_HPP

#include <string>

#include <gxf/std/double_buffer_receiver.hpp>

#include "./receiver.hpp"

namespace holoscan {

// Forward declarations
class AnnotatedDoubleBufferReceiver;

/**
 * @brief Double buffer receiver class.
 *
 * The DoubleBufferReceiver class is used to receive messages from another operator within a
 * fragment. This class uses a double buffer queue where messages first arrive in a "front stage".
 * When an operator is selected for execution, any front stage messages are moved to the main stage
 * just before the `compute` method of the operator is called. During compute, the front stage is
 * now available to receive messages again (and these would then be processed during the next
 * `compute` call).
 *
 * Application authors are not expected to use this class directly. It will be automatically
 * configured for input ports specified via `Operator::setup`.
 *
 * ==Parameters==
 *
 * - **capacity** (uint64_t, optional): The capacity of the double-buffer queue used by the
 * receiver. Defaults to 1.
 * - **policy** (uint64_t, optional): The policy to use when a message arrives, but there is no
 * space in the receiver. The possible values are 0: pop, 1: reject, 2: fault (Default: 2).
 */
class DoubleBufferReceiver : public Receiver {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(DoubleBufferReceiver, Receiver)
  DoubleBufferReceiver() = default;
  DoubleBufferReceiver(const std::string& name, nvidia::gxf::DoubleBufferReceiver* component);

  DoubleBufferReceiver(const std::string& name, AnnotatedDoubleBufferReceiver* component);

  const char* gxf_typename() const override;

  void setup(ComponentSpec& spec) override;

  /**
   * @brief Track the data flow of the receiver and use holoscan::AnnotatedDoubleBufferReceiver as
   * the GXF Component.
   */
  void track();

  nvidia::gxf::DoubleBufferReceiver* get() const;

  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  bool tracking_ = false;  ///< Used to decide whether to use data flow tracking or not.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_DOUBLE_BUFFER_RECEIVER_HPP */
