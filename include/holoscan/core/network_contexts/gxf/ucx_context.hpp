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

#ifndef HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_UCX_CONTEXT_HPP
#define HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_UCX_CONTEXT_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <gxf/ucx/ucx_context.hpp>

#include "../../gxf/gxf_network_context.hpp"
#include "../../resources/gxf/ucx_entity_serializer.hpp"

namespace holoscan {

/**
 * @brief UCX-based NetworkContext class used by distributed applications.
 *
 * Application authors do not need to use this class directly. It will be initialized by the
 * application at runtime as needed.
 *
 * ==Parameters==
 *
 * - **entity_serializer** (std::shared_ptr<UcxEntitySerializer>): The entity serializer
 * that will be used for any network connections (i.e. `add_flow` connections between fragments).
 * A `UcxEntitySerializer` will be used by default.
 * - **reconnect** (bool, optional): Try to reconnect if a connection is closed during run
 * (default: true).
 * - **cpu_data_only** (bool, optional): This flag should be set to true on a system which does not
 * have any (visible) CUDA capable devices.
 * - **enable_async** (bool, optional): If false, synchronous operation of message transmission
 * will be used (Default: false). The `HOLOSCAN_UCX_ASYNCHRONOUS` environment variable can be used
 * to set the value that Holoscan will use for this parameter when creating its internal
 * `UcxNetworkContext`. This parameter is deprecated in Holoscan v3.7 and will be removed in v4.0.
 * The new behavior will be equivalent to a value of `false` here.
 * - **shutdown_timeout_ms** (uint64_t, optional): Timeout in milliseconds for shutdown operations
 * such as thread joins and pending request cancellation (default: 2000). The
 * `HOLOSCAN_UCX_SHUTDOWN_TIMEOUT_MS` environment variable can be used to override this value.
 */
class UcxContext : public gxf::GXFNetworkContext {
 public:
  HOLOSCAN_NETWORK_CONTEXT_FORWARD_ARGS_SUPER(UcxContext, gxf::GXFNetworkContext)

  UcxContext() = default;
  const char* gxf_typename() const override { return "nvidia::gxf::UcxContext"; }

  // // Finds transmitters and receivers passes the network context to transmitter
  // // and receivers and make connection between them
  // virtual Expected<void> addRoutes(const Entity& entity) = 0;

  // // Closes the connection between transmitters and receivers
  // virtual Expected<void> removeRoutes(const Entity& entity) = 0;

  std::shared_ptr<UcxEntitySerializer> entity_serializer() { return entity_serializer_; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  nvidia::gxf::UcxContext* get() const;

  /**
   * @brief Initiates graceful shutdown of UCX connections.
   *
   * Sets the shutting_down_ flag and signals TX/RX threads to exit.
   * This allows pending operations to complete within the shutdown timeout
   * rather than blocking indefinitely.
   *
   * Call this early in the shutdown sequence (before stop_execution()) to
   * ensure UCX threads exit cleanly and connection errors during shutdown
   * are treated as expected rather than fatal.
   */
  void initiate_shutdown();

  /**
   * @brief Check if shutdown has been initiated.
   * @return true if shutdown is in progress
   */
  bool is_shutting_down() const;

 private:
  Parameter<std::shared_ptr<UcxEntitySerializer>> entity_serializer_;
  Parameter<bool> reconnect_;      ///< Try to reconnect if a connection is closed during run
  Parameter<bool> cpu_data_only_;  ///< Support CPU memory only for UCX communication
  Parameter<bool> enable_async_;   ///< Control whether UCX transmit/receive uses asynchronous mode
  Parameter<uint64_t>
      shutdown_timeout_ms_;  ///< Timeout for shutdown operations (thread joins, etc.)

  // TODO(unknown): support GPUDevice nvidia::gxf::Resource
  // nvidia::gxf::Resource<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>> gpu_device_;

  bool shutting_down_ = false;  ///< Flag to track if shutdown has been initiated
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_UCX_CONTEXT_HPP */
