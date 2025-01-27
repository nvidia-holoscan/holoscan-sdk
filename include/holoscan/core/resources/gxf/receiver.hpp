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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP

#include <cstdint>
#include <string>

#include <gxf/std/receiver.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base receiver class.
 *
 * Receivers are used by input ports to receive messages.
 */
class Receiver : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Receiver, GXFResource)
  Receiver() = default;
  Receiver(const std::string& name, nvidia::gxf::Receiver* component);

  const char* gxf_typename() const override { return "nvidia::gxf::Receiver"; }

  nvidia::gxf::Receiver* get() const;

  /**@brief Get the current capacity of the Receiver queue.
   *
   * For double-buffer queues this is the capacity of the main stage
   * @return The capacity of the Receiver queue.
   */
  size_t capacity() const;

  /**@brief Get the number of elements currently in the main stage of the Receiver queue.
   *
   * @return The number of elements in the main stage.
   */
  size_t size() const;

  /**@brief Get the number of elements currently in the back stage of the Receiver queue.
   *
   * @return The number of elements in the back stage.
   */
  size_t back_size() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP */
