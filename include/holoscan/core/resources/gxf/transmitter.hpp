/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP

#include <cstdint>
#include <string>

#include <gxf/std/transmitter.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base transmitter class.
 *
 * Transmitters are used by output ports to emit messages.
 */
class Transmitter : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Transmitter, GXFResource)
  Transmitter() = default;
  Transmitter(const std::string& name, nvidia::gxf::Transmitter* component);

  const char* gxf_typename() const override { return "nvidia::gxf::Transmitter"; }

  nvidia::gxf::Transmitter* get() const;

  /**@brief Get the current capacity of the Transmitter queue.
   *
   * For double-buffer queues this is the capacity of the main stage.
   * @return The capacity of the Receiver queue.
   */
  size_t capacity() const;

  /**@brief Get the number of elements currently in the main stage of the Transmitter queue.
   *
   * @return The number of elements in the main stage.
   */
  size_t size() const;

  /**@brief Get the number of elements currently in the back stage of the Transmitter queue.
   *
   * @return The number of elements in the back stage.
   */
  size_t back_size() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_TRANSMITTER_HPP */
