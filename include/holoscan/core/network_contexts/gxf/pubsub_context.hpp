/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_PUBSUB_CONTEXT_HPP
#define HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_PUBSUB_CONTEXT_HPP

#include <memory>
#include <string>
#include <vector>

#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/gid.hpp>
#include <gxf/pubsub/pubsub_context.hpp>

#include "../../gxf/gxf_network_context.hpp"
#include "../../resources/gxf/clock.hpp"

namespace holoscan {

/**
 * @brief Pub/Sub NetworkContext class for topic-based inter-fragment communication.
 *
 * PubSubContext provides topic-based publish/subscribe messaging for distributed
 * Holoscan applications. It wraps the GXF `nvidia::gxf::PubSubContext` component,
 * which is backend-agnostic and supports different discovery and transport implementations.
 *
 * ## Features
 *
 * - **Topic-based routing**: Publishers and subscribers connect via topic names
 * - **Dynamic discovery**: Late-joining fragments can discover existing endpoints
 * - **QoS support**: Configurable reliability, durability, and history policies
 * - **Backend-agnostic**: Works with different PubSubDiscovery, PubSubTransport,
 *   PubSubEntitySerializer implementations (gRPC+UCX, DDS, in-memory for testing, etc.)
 *
 * ## Usage
 *
 * Application authors typically don't use this class directly. It is configured
 * automatically when using pub/sub connectors. For manual configuration:
 *
 * ```cpp
 * auto pubsub_ctx = fragment->make_network_context<PubSubContext>(
 *     "pubsub_context",
 *     Arg("node_name", "my_fragment"));
 * ```
 *
 * ## Backend Injection
 *
 * The underlying GXF PubSubContext requires backend implementations to be injected:
 * - `PubSubDiscovery`: For endpoint discovery (e.g., DDSDiscovery, GrpcDiscovery)
 * - `PubSubTransport`: For message transport (e.g., DDSTransport, UcxTransport)
 * - `PubSubEntitySerializer`: For entity serialization (e.g., DDSSerializer, UcxSerializer)
 *
 * Backend injection is handled by overriding the `setup_backend()` virtual method.
 * The base `PubSubContext` provides an empty default; subclasses (e.g.,
 * `DDSPubSubNetworkContext`) override it to create and inject their backend
 * components.
 *
 * To use a custom backend, subclass `PubSubContext`, override `setup_backend()`,
 * and override `Fragment::create_pubsub_network_context()` in your Application
 * to return an instance of your subclass.
 *
 * ==Parameters==
 *
 * - **node_name** (std::string, optional): Name of this node/fragment for identification
 *   in discovery (default: empty, typically set from fragment name).
 * - **clock** (std::shared_ptr<gxf::Clock>, optional): Clock used for pub/sub timestamps
 *   (creation_timestamp_ns, source_timestamp_ns). If not provided, a RealtimeClock is
 *   auto-created during initialize().
 *
 * @see nvidia::gxf::PubSubContext for the underlying GXF component
 */
class PubSubContext : public gxf::GXFNetworkContext {
 public:
  HOLOSCAN_NETWORK_CONTEXT_FORWARD_ARGS_SUPER(PubSubContext, gxf::GXFNetworkContext)

  PubSubContext() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::PubSubContext"; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;
  std::shared_ptr<Clock> clock() override;

  /**
   * @brief Get the underlying GXF PubSubContext component.
   * @return Pointer to the GXF PubSubContext, or nullptr if not initialized.
   */
  nvidia::gxf::PubSubContext* get() const;

  /**
   * @brief Get the node name for this context.
   * @return The node name string.
   */
  std::string node_name() const;

  /**
   * @brief Get information about all known topics.
   * @return Vector of TopicInfo structures.
   */
  std::vector<nvidia::gxf::TopicInfo> get_topics() const;

  /**
   * @brief Get number of publishers on a topic.
   * @param topic Topic name to query.
   * @return Number of publishers.
   */
  size_t get_publisher_count(const std::string& topic) const;

  /**
   * @brief Get number of subscribers on a topic.
   * @param topic Topic name to query.
   * @return Number of subscribers.
   */
  size_t get_subscriber_count(const std::string& topic) const;

  /**
   * @brief Get the GIDs of all locally registered publishers.
   * @return Vector of publisher GIDs (snapshot under lock).
   */
  std::vector<nvidia::gxf::Gid> registered_publisher_gids() const;

  /**
   * @brief Get the GIDs of all locally registered subscribers.
   * @return Vector of subscriber GIDs (snapshot under lock).
   */
  std::vector<nvidia::gxf::Gid> registered_subscriber_gids() const;

  /**
   * @brief Look up the transmitter Handle for a registered publisher.
   * @param gid Publisher GID.
   * @return Handle to the Transmitter, or Unexpected with GXF_ENTITY_NOT_FOUND.
   */
  nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Transmitter>> get_publisher_transmitter(
      const nvidia::gxf::Gid& gid) const;

  /**
   * @brief Look up the receiver Handle for a registered subscriber.
   * @param gid Subscriber GID.
   * @return Handle to the Receiver, or Unexpected with GXF_ENTITY_NOT_FOUND.
   */
  nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Receiver>> get_subscriber_receiver(
      const nvidia::gxf::Gid& gid) const;

 protected:
  /**
   * @brief Set up the PubSub backend (discovery, transport, serializer).
   *
   * Called automatically from initialize() after the GXF component is created
   * and `get()` returns a valid pointer.
   *
   * The default implementation is a no-op. Subclasses should override this to
   * create backend components and inject them into the GXF PubSubContext:
   *
   * ```cpp
   * void setup_backend() override {
   *   auto* gxf_ctx = get();
   *   gxf_ctx->set_discovery(std::make_shared<MyDiscovery>(...));
   *   gxf_ctx->set_transport(std::make_shared<MyTransport>(...));
   *   gxf_ctx->set_serializer(std::make_shared<MySerializer>(...));
   *   gxf_ctx->initialize();
   *   gxf_ctx->init_context();
   * }
   * ```
   *
   * @see DDSPubSubNetworkContext for the FastDDS backend implementation.
   */
  virtual void setup_backend();

 private:
  Parameter<std::string> node_name_;
  Parameter<std::shared_ptr<gxf::Clock>> clock_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_NETWORK_CONTEXTS_GXF_PUBSUB_CONTEXT_HPP */
