/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_NETWORK_CONTEXTS_GXF_FASTDDS_PUBSUB_NETWORK_CONTEXT_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_NETWORK_CONTEXTS_GXF_FASTDDS_PUBSUB_NETWORK_CONTEXT_HPP

#include <memory>
#include <string>

#include <holoscan/core/network_contexts/gxf/pubsub_context.hpp>

namespace holoscan {
class NativeBufferProtocolAdapter;

// Forward declarations for DDS backend components
class FastDdsPubSubContext;  // Resource: DDS DomainParticipant + CUDA stream
class FastDdsDiscovery;
class FastDdsTransport;
class FastDdsSerializer;
class FastDdsNativeBufferAdapter;

/**
 * @brief PubSubContext subclass that uses FastDDS as the backend.
 *
 * FastDdsPubSubNetworkContext overrides `setup_backend()` to create and inject
 * FastDDS-based discovery, transport, and serializer components into the
 * GXF PubSubContext.
 *
 * This is the default backend when `HOLOSCAN_ENABLE_FASTDDS` is defined.
 * The auto-creation path in `Fragment::create_pubsub_network_context()`
 * returns an instance of this class.
 *
 * ## Usage
 *
 * Typically auto-created. For explicit use in an Application subclass:
 *
 * ```cpp
 * std::shared_ptr<NetworkContext> create_pubsub_network_context() override {
 *   return make_network_context<FastDdsPubSubNetworkContext>("pubsub_context");
 * }
 * ```
 *
 * @see PubSubContext for the base class and extension points.
 * @see FastDdsPubSubContext for the DDS DomainParticipant resource.
 */
class FastDdsPubSubNetworkContext : public PubSubContext {
 public:
  HOLOSCAN_NETWORK_CONTEXT_FORWARD_ARGS_SUPER(FastDdsPubSubNetworkContext, PubSubContext)

  FastDdsPubSubNetworkContext() = default;
  void setup(ComponentSpec& spec) override;
  std::shared_ptr<NativeBufferProtocolAdapter> native_buffer_adapter() const;

 protected:
  void setup_backend() override;

 private:
  Parameter<std::string> native_buffer_policy_;
  Parameter<int64_t> native_buffer_acquire_timeout_ms_;
  Parameter<int64_t> native_buffer_export_ttl_ms_;
  Parameter<bool> native_buffer_use_eager_acquire_;
  // DDS backend objects — kept alive for the lifetime of this context.
  std::shared_ptr<FastDdsPubSubContext> dds_context_;
  std::shared_ptr<FastDdsDiscovery> dds_discovery_;
  std::shared_ptr<FastDdsTransport> dds_transport_;
  std::shared_ptr<FastDdsSerializer> dds_serializer_;
  std::shared_ptr<FastDdsNativeBufferAdapter> native_adapter_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_FASTDDS_NETWORK_CONTEXTS_GXF_FASTDDS_PUBSUB_NETWORK_CONTEXT_HPP */
