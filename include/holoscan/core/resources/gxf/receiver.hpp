/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP

#include <cstdint>
#include <string>

#include <gxf/core/entity.hpp>
#include <gxf/core/expected.hpp>
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

  /**
   * @brief Peek at a message in the queue without consuming it.
   *
   * This method allows inspection of messages in the main stage of the receiver queue
   * without removing them. This is useful for conditions that need to check message
   * contents before the operator executes.
   *
   * @param index The index of the message to peek (0 = oldest message in queue).
   * @return The entity at the given index, or an error if not available.
   */
  nvidia::gxf::Expected<nvidia::gxf::Entity> peek(int32_t index = 0) const;

  /**
   * @brief Sync messages from back stage to main stage.
   *
   * For double-buffer queues, this moves messages from the back stage (where they
   * arrive) to the main stage (where they can be consumed). This is typically called
   * by the framework before operator execution, but may be needed for conditions
   * that check queue state before the operator is scheduled.
   *
   * @return Success if the sync completed, or an error otherwise.
   */
  nvidia::gxf::Expected<void> sync();
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_RECEIVER_HPP */
