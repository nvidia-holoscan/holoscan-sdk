/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/transmitter.hpp>

#include <cstdint>
#include <string>

#include <gxf/std/transmitter.hpp>

namespace holoscan {

Transmitter::Transmitter(const std::string& name, nvidia::gxf::Transmitter* component)
    : GXFResource(name, component) {}

nvidia::gxf::Transmitter* Transmitter::get() const {
  return static_cast<nvidia::gxf::Transmitter*>(gxf_cptr_);
}

size_t Transmitter::capacity() const {
  return get()->capacity();
}

size_t Transmitter::size() const {
  return get()->size();
}

size_t Transmitter::back_size() const {
  return get()->back_size();
}

}  // namespace holoscan
