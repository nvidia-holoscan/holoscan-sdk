/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_network_context.hpp>

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

namespace {
constexpr size_t kDefaultSessionBufferSize = 1 << 16;
}  // namespace

InMemoryPubSubNetworkContext::~InMemoryPubSubNetworkContext() {
  if (serializer_) {
    serializer_->set_delegate(nullptr);
  }
  delegate_serializer_.reset();

  if (session_) {
    if (transport_frontend_) {
      transport_frontend_->shutdown();
    }
    if (discovery_frontend_) {
      discovery_frontend_->shutdown();
    }
    auto* gxf_ctx = get();
    if (gxf_ctx) {
      session_->leave(gxf_ctx->context());
    }
  }
}

void InMemoryPubSubNetworkContext::initialize() {
  HOLOSCAN_LOG_DEBUG("InMemoryPubSubNetworkContext::initialize");
  PubSubContext::initialize();
}

void InMemoryPubSubNetworkContext::setup(ComponentSpec& spec) {
  PubSubContext::setup(spec);  // registers node_name and clock

  spec.param(session_id_,
             "session_id",
             "Session ID",
             "Shared session identifier. Processes with the same session_id "
             "share discovery and transport state within the process. "
             "Empty (default) means private, single-context mode.",
             std::string{});

  // These three parameters are forwarded by name to the underlying GXF
  // InMemoryPubSubContext, which registers them in its registerInterface().
  spec.param(serializer_mode_,
             "serializer_mode",
             "Serializer Mode",
             "0 = kPassthrough (default, zero-copy UID transfer); "
             "1 = kFullSerialization (byte-level, requires a delegate serializer). "
             "Ignored in session mode (always kFullSerialization).",
             static_cast<int32_t>(0));
  spec.param(drop_pattern_,
             "drop_pattern",
             "Drop Pattern",
             "Cyclic drop pattern for deterministic fault injection "
             "(0 = deliver, non-zero = drop). Empty disables dropping.",
             std::vector<int32_t>{});
  spec.param(reorder_pattern_,
             "reorder_pattern",
             "Reorder Pattern",
             "Cyclic reorder pattern for deterministic fault injection "
             "(0 = deliver, non-zero = delay). Empty disables reordering.",
             std::vector<int32_t>{});
  spec.param(serialization_buffer_size_,
             "serialization_buffer_size",
             "Serialization Buffer Size",
             "Size in bytes of the serialization/deserialization buffers used in session mode. "
             "Ignored in private mode.",
             kDefaultSessionBufferSize);
}

void InMemoryPubSubNetworkContext::setup_backend() {
  auto* gxf_ctx = get();
  if (!gxf_ctx) {
    HOLOSCAN_LOG_ERROR(
        "InMemoryPubSubNetworkContext::setup_backend: GXF PubSubContext not available");
    return;
  }

  const auto& sid = session_id_.try_get();
  const bool use_session = sid.has_value() && !sid->empty();

  if (use_session) {
    setup_session_backend(*sid);
  } else {
    setup_private_backend();
  }
}

void InMemoryPubSubNetworkContext::setup_private_backend() {
  // Private mode (existing behavior, unchanged):
  // Let the GXF InMemoryPubSubContext create its own private backends via
  // initialize(), then wire up topic routing via init_context().
  auto* gxf_ctx = get();

  const auto init_result = gxf_ctx->initialize();
  if (init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("InMemoryPubSubNetworkContext::setup_backend: GXF initialize() failed ({})",
                       GxfResultStr(init_result));
    return;
  }

  const auto ctx_result = gxf_ctx->init_context();
  if (ctx_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("InMemoryPubSubNetworkContext::setup_backend: init_context() failed ({})",
                       GxfResultStr(ctx_result));
    return;
  }

  HOLOSCAN_LOG_INFO("InMemoryPubSubNetworkContext: private backend ready");
}

void InMemoryPubSubNetworkContext::setup_session_backend(const std::string& sid) {
  // Session mode (cross-context):
  // Call GXF initialize() first so the runtime marks the component as
  // initialized and won't re-initialize it during graph activation (which
  // would create private backends and overwrite our shared ones). The injected
  // discovery/transport frontends must then be initialized by
  // PubSubContext::init_context(); doing that work manually here would diverge
  // from the GXF lifecycle contract the session-mode tests protect.
  // Then override discovery/transport/serializer with session-shared frontends.
  auto* gxf_ctx = get();

  const auto init_result = gxf_ctx->initialize();
  if (init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "InMemoryPubSubNetworkContext::setup_session_backend: GXF initialize() failed ({})",
        GxfResultStr(init_result));
    return;
  }

  session_ = InMemoryPubSubSession::get_or_create(sid);
  session_->join(gxf_ctx->context());

  // Override with per-participant discovery/transport frontends backed by the shared session
  auto frontends = session_->create_frontends();
  discovery_frontend_ = frontends.discovery;
  transport_frontend_ = frontends.transport;
  gxf_ctx->set_discovery(discovery_frontend_);
  gxf_ctx->set_transport(transport_frontend_);

  // Per-context serializer: always kFullSerialization in session mode
  // (entity UIDs are not valid across GXF contexts)
  serializer_ = std::make_shared<nvidia::gxf::InMemorySerializer>(
      nvidia::gxf::SerializerMode::kFullSerialization);

  // Wire the delegate serializer chain:
  //   UcxHoloscanComponentSerializer + StdComponentSerializer
  //     → StdEntitySerializer
  //       → StdPubSubEntitySerializer adapter
  //         → InMemorySerializer delegate
  holoscan_component_serializer_ = fragment()->make_resource<UcxHoloscanComponentSerializer>(
      name() + "__holoscan_component_serializer");
  std_component_serializer_ =
      fragment()->make_resource<StdComponentSerializer>(name() + "__std_component_serializer");
  std_entity_serializer_ = fragment()->make_resource<StdEntitySerializer>(
      name() + "__std_entity_serializer",
      Arg("component_serializers") = std::vector<std::shared_ptr<Resource>>{
          holoscan_component_serializer_, std_component_serializer_});
  const size_t buf_size = serialization_buffer_size_.try_get().value_or(kDefaultSessionBufferSize);
  serialize_buffer_ = fragment()->make_resource<SerializationBuffer>(name() + "__serialize_buffer",
                                                                     Arg("buffer_size") = buf_size);
  deserialize_buffer_ = fragment()->make_resource<SerializationBuffer>(
      name() + "__deserialize_buffer", Arg("buffer_size") = buf_size);

  // Initialize Holoscan resources on this fragment's GXF entity
  auto initialize_resource = [this](const auto& resource) {
    resource->gxf_cname(resource->name().c_str());
    if (gxf_eid_ != 0) {
      resource->gxf_eid(gxf_eid_);
    }
    resource->initialize();
  };
  initialize_resource(holoscan_component_serializer_);
  initialize_resource(std_component_serializer_);
  initialize_resource(std_entity_serializer_);
  initialize_resource(serialize_buffer_);
  initialize_resource(deserialize_buffer_);

  delegate_serializer_ = std::make_shared<StdPubSubEntitySerializer>(
      std_entity_serializer_, serialize_buffer_, deserialize_buffer_, buf_size);
  serializer_->set_delegate(delegate_serializer_.get());
  gxf_ctx->set_serializer(serializer_);

  // Wire topic routing (PubSubContext reads node_name param internally)
  const auto result = gxf_ctx->init_context();
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("InMemoryPubSubNetworkContext::setup_backend: init_context() failed ({})",
                       GxfResultStr(result));
    return;
  }

  if (auto discovery_frontend =
          std::dynamic_pointer_cast<SessionDiscoveryFrontend>(discovery_frontend_);
      discovery_frontend) {
    auto replay_result = discovery_frontend->replay_registered_endpoints();
    if (!replay_result) {
      HOLOSCAN_LOG_ERROR(
          "InMemoryPubSubNetworkContext::setup_backend: replay_registered_endpoints() failed ({})",
          GxfResultStr(replay_result.error()));
      return;
    }
  }

  HOLOSCAN_LOG_INFO("InMemoryPubSubNetworkContext: joined session '{}'", sid);
}

}  // namespace holoscan
