/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/receiver.hpp>

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
