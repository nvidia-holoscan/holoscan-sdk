/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_NETWORK_CONTEXT_HPP
#define HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_NETWORK_CONTEXT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/pubsub/in_memory_pubsub_context.hpp>  // nvidia::gxf::InMemoryPubSubContext
#include <gxf/pubsub/in_memory_serializer.hpp>

#include <holoscan/core/network_contexts/gxf/pubsub_context.hpp>
#include <holoscan/core/resources/gxf/serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/std_component_serializer.hpp>
#include <holoscan/core/resources/gxf/std_entity_serializer.hpp>
#include <holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp>
#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_session.hpp>

namespace holoscan {

/**
 * @brief Holoscan PubSub NetworkContext backed by in-memory transports.
 *
 * `InMemoryPubSubNetworkContext` wraps the GXF `nvidia::gxf::InMemoryPubSubContext`
 * component, which self-configures with `InMemoryDiscovery`, `InMemoryTransport`,
 * and `InMemorySerializer` backends. No IPC, network sockets, or shared-memory
 * segments are required — messages are delivered by transferring GXF entity UIDs
 * within the same process.
 *
 * ## Private mode (default — no session_id)
 *
 * In the default `kPassthrough` serializer mode, messages are delivered by UID
 * transfer. Both publisher and subscriber must live in the same Holoscan fragment
 * (same GXF context). This is the zero-copy, single-fragment path.
 *
 * ## Session mode (session_id set)
 *
 * When `session_id` is set, multiple fragments (each with their own GXF context)
 * can publish/subscribe on shared topics within the same process. Discovery and
 * transport state are shared via `InMemoryPubSubSession`; serialization always
 * uses `kFullSerialization` to avoid cross-context entity UID issues.
 *
 * ```cpp
 * // fragment A:
 * auto ctx = frag.make_network_context<InMemoryPubSubNetworkContext>(
 *     "pubsub_context", Arg("session_id", "my_test_session"));
 * frag.network_context(ctx);
 *
 * // fragment B (same session_id → shared discovery/transport):
 * auto ctx = frag.make_network_context<InMemoryPubSubNetworkContext>(
 *     "pubsub_context", Arg("session_id", "my_test_session"));
 * frag.network_context(ctx);
 * ```
 *
 * ## Fault injection
 *
 * `drop_pattern` is a cyclic `int32_t` sequence: `0` = deliver, non-zero = drop.
 * An empty vector (the default) disables fault injection.
 * Note: fault injection is only available in private mode.
 *
 * ==Parameters==
 *
 * In addition to the base `PubSubContext` parameters (`node_name`, `clock`):
 *
 * - **session_id** (std::string, optional): Shared session identifier. Fragments with
 *   the same session_id share discovery and transport state within the process.
 *   Empty (default) means private, single-context mode.
 * - **serializer_mode** (int32_t, optional): `0` = `kPassthrough` (default); `1` =
 * `kFullSerialization`. Ignored in session mode (always kFullSerialization).
 * - **drop_pattern** (std::vector<int32_t>, optional): Cyclic drop pattern for fault injection.
 * - **reorder_pattern** (std::vector<int32_t>, optional): Cyclic reorder pattern.
 * - **serialization_buffer_size** (size_t, optional): Size in bytes of the
 *   serialization/deserialization buffers used in session mode. Default: 65536. Ignored in
 *   private mode.
 */
class InMemoryPubSubNetworkContext : public PubSubContext {
 public:
  HOLOSCAN_NETWORK_CONTEXT_FORWARD_ARGS_SUPER(InMemoryPubSubNetworkContext, PubSubContext)

  InMemoryPubSubNetworkContext() = default;
  ~InMemoryPubSubNetworkContext() override;

  /// Points at nvidia::gxf::InMemoryPubSubContext so that the GXF component
  /// self-configures its own backends (InMemoryDiscovery, InMemoryTransport,
  /// InMemorySerializer) via its initialize() method.  The Holoscan layer then
  /// forwards all registered parameters (including drop_pattern, serializer_mode,
  /// reorder_pattern) directly to the GXF component by name.
  const char* gxf_typename() const override { return "nvidia::gxf::InMemoryPubSubContext"; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

 protected:
  /// Called after the GXF InMemoryPubSubContext has been created and its parameters
  /// have been set. In private mode: calls GXF initialize() + init_context().
  /// In session mode: creates shared frontends + serializer chain, calls init_context().
  void setup_backend() override;

 private:
  void setup_private_backend();
  void setup_session_backend(const std::string& sid);

  Parameter<std::string> session_id_;
  /// 0 = kPassthrough (default), 1 = kFullSerialization
  Parameter<int32_t> serializer_mode_;
  Parameter<std::vector<int32_t>> drop_pattern_;
  Parameter<std::vector<int32_t>> reorder_pattern_;
  Parameter<size_t> serialization_buffer_size_;

  // Session mode state (held to keep shared backends alive)
  std::shared_ptr<InMemoryPubSubSession> session_;
  std::shared_ptr<nvidia::gxf::PubSubDiscovery> discovery_frontend_;
  std::shared_ptr<nvidia::gxf::PubSubTransport> transport_frontend_;
  std::shared_ptr<nvidia::gxf::InMemorySerializer> serializer_;
  std::shared_ptr<UcxHoloscanComponentSerializer> holoscan_component_serializer_;
  std::shared_ptr<StdComponentSerializer> std_component_serializer_;
  std::shared_ptr<StdEntitySerializer> std_entity_serializer_;
  std::shared_ptr<SerializationBuffer> serialize_buffer_;
  std::shared_ptr<SerializationBuffer> deserialize_buffer_;
  std::shared_ptr<StdPubSubEntitySerializer> delegate_serializer_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_NETWORK_CONTEXT_HPP
