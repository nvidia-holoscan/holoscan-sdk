/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/receiver.hpp"

#include <cstdint>
#include <string>

#include <gxf/std/receiver.hpp>

namespace holoscan {

Receiver::Receiver(const std::string& name, nvidia::gxf::Receiver* component)
    : GXFResource(name, component) {}

nvidia::gxf::Receiver* Receiver::get() const {
  return static_cast<nvidia::gxf::Receiver*>(gxf_cptr_);
}

size_t Receiver::capacity() const {
  auto* receiver = get();
  return receiver ? receiver->capacity() : 0;
}

size_t Receiver::size() const {
  auto* receiver = get();
  return receiver ? receiver->size() : 0;
}

size_t Receiver::back_size() const {
  auto* receiver = get();
  return receiver ? receiver->back_size() : 0;
}

nvidia::gxf::Expected<nvidia::gxf::Entity> Receiver::peek(int32_t index) const {
  auto* receiver = get();
  return receiver ? receiver->peek(index) : nvidia::gxf::Unexpected{GXF_NULL_POINTER};
}

nvidia::gxf::Expected<void> Receiver::sync() {
  auto* receiver = get();
  return receiver ? receiver->sync() : nvidia::gxf::Unexpected{GXF_NULL_POINTER};
}

}  // namespace holoscan
