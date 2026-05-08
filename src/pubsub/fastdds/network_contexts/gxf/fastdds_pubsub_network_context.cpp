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

#include <holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp>

#include <chrono>
#include <memory>
#include <string>

#include <gxf/core/handle.hpp>
#include <gxf/std/allocator.hpp>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_discovery.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_native_buffer_adapter.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_serializer.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_transport.hpp>
#include <holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp>

namespace holoscan {

void FastDdsPubSubNetworkContext::setup(ComponentSpec& spec) {
  PubSubContext::setup(spec);
  spec.param(native_buffer_policy_,
             "native_buffer_policy",
             "Native Buffer Policy",
             "Native buffer policy forwarded to the underlying DDS pub/sub resource",
             std::string("preferred"));
  spec.param(native_buffer_acquire_timeout_ms_,
             "native_buffer_acquire_timeout_ms",
             "Native Buffer Acquire Timeout (ms)",
             "Forwarded to FastDdsPubSubContext: max wait for holoscan::ipc acquire_pointer() "
             "(subscriber)",
             static_cast<int64_t>(500));
  spec.param(native_buffer_export_ttl_ms_,
             "native_buffer_export_ttl_ms",
             "Native Buffer Export TTL (ms)",
             "Forwarded to FastDdsPubSubContext: publisher-side stale pending-export eviction age",
             static_cast<int64_t>(5000));
  spec.param(native_buffer_use_eager_acquire_,
             "native_buffer_use_eager_acquire",
             "Use Eager Native Acquire",
             "If true, subscribers use holoipc acquire_pointer_eager() for CUDA IPC native "
             "buffers instead of waiting for ACK. This is opt-in and disabled by default.",
             false);
}

std::shared_ptr<NativeBufferProtocolAdapter> FastDdsPubSubNetworkContext::native_buffer_adapter()
    const {
  return native_adapter_;
}

void FastDdsPubSubNetworkContext::setup_backend() {
  auto* gxf_ctx = get();
  if (!gxf_ctx) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsPubSubNetworkContext::setup_backend: GXF PubSubContext not available");
    return;
  }

  auto frag = fragment();
  if (!frag) {
    HOLOSCAN_LOG_ERROR("FastDdsPubSubNetworkContext::setup_backend: fragment not set");
    return;
  }

  // --- 1. Create FastDdsPubSubContext (DomainParticipant + CUDA stream) ---
  HOLOSCAN_LOG_INFO("PubSubContext: creating DDS backend for fragment '{}'", frag->name());

  // Forward native-buffer parameters to FastDdsPubSubContext. Defaults come from this
  // NetworkContext's parameters; explicit Args on the context override by name.
  std::string native_policy;
  int64_t acquire_timeout_ms = native_buffer_acquire_timeout_ms_.get();
  int64_t export_ttl_ms = native_buffer_export_ttl_ms_.get();
  bool use_eager_acquire = native_buffer_use_eager_acquire_.get();
  for (const auto& a : args()) {
    if (a.name() == "native_buffer_policy") {
      if (auto* p = std::any_cast<std::string>(&a.value())) {
        native_policy = *p;
      } else {
        HOLOSCAN_LOG_WARN("FastDdsPubSubNetworkContext: unexpected type for native_buffer_policy");
      }
    } else if (a.name() == "native_buffer_acquire_timeout_ms") {
      if (auto* p = std::any_cast<int64_t>(&a.value())) {
        acquire_timeout_ms = *p;
      } else {
        HOLOSCAN_LOG_WARN(
            "FastDdsPubSubNetworkContext: unexpected type for native_buffer_acquire_timeout_ms");
      }
    } else if (a.name() == "native_buffer_export_ttl_ms") {
      if (auto* p = std::any_cast<int64_t>(&a.value())) {
        export_ttl_ms = *p;
      } else {
        HOLOSCAN_LOG_WARN(
            "FastDdsPubSubNetworkContext: unexpected type for native_buffer_export_ttl_ms");
      }
    } else if (a.name() == "native_buffer_use_eager_acquire") {
      if (auto* p = std::any_cast<bool>(&a.value())) {
        use_eager_acquire = *p;
      } else {
        HOLOSCAN_LOG_WARN(
            "FastDdsPubSubNetworkContext: unexpected type for native_buffer_use_eager_acquire");
      }
    }
  }

  if (native_policy.empty()) {
    dds_context_ = frag->make_resource<FastDdsPubSubContext>(
        "pubsub__dds_context",
        Arg("domain_id", static_cast<int32_t>(0)),
        Arg("participant_name", std::string("holoscan_") + frag->name()),
        Arg("native_buffer_acquire_timeout_ms", acquire_timeout_ms),
        Arg("native_buffer_export_ttl_ms", export_ttl_ms));
  } else {
    dds_context_ = frag->make_resource<FastDdsPubSubContext>(
        "pubsub__dds_context",
        Arg("domain_id", static_cast<int32_t>(0)),
        Arg("participant_name", std::string("holoscan_") + frag->name()),
        Arg("native_buffer_policy", native_policy),
        Arg("native_buffer_acquire_timeout_ms", acquire_timeout_ms),
        Arg("native_buffer_export_ttl_ms", export_ttl_ms));
  }
  dds_context_->initialize();

  // --- 2. Create DDS backend components ---
  dds_discovery_ = std::make_shared<FastDdsDiscovery>(dds_context_.get());
  dds_transport_ = std::make_shared<FastDdsTransport>(dds_context_.get());
  dds_serializer_ = std::make_shared<FastDdsSerializer>(dds_context_->cuda_stream());

  // --- 2b. Create and wire FastDdsNativeBufferAdapter for CUDA IPC ---
  auto policy = dds_context_->native_buffer_policy();
  if (policy != nvidia::gxf::NativeBufferPolicy::kDisabled) {
    native_adapter_ = std::make_shared<FastDdsNativeBufferAdapter>();
    native_adapter_->set_use_eager_acquire(use_eager_acquire);
    auto adapter_result = native_adapter_->initialize(dds_context_->participant(), policy);
    if (adapter_result) {
      native_adapter_->set_acquire_timeout(
          std::chrono::milliseconds(dds_context_->native_buffer_acquire_timeout_ms()));
      native_adapter_->set_export_ttl(
          std::chrono::milliseconds(dds_context_->native_buffer_export_ttl_ms()));
      dds_serializer_->set_native_buffer_adapter(native_adapter_);
      dds_transport_->set_native_buffers_enabled(true);

      // Set local native capability on discovery for UserData advertisement
      dds_discovery_->set_local_native_capability(dds_context_->native_buffer_capability());

      HOLOSCAN_LOG_INFO(
          "PubSubContext: CUDA IPC native buffer adapter initialized "
          "(gpu_uuid={}, policy={}, eager_acquire={})",
          dds_context_->gpu_device_uuid(),
          policy == nvidia::gxf::NativeBufferPolicy::kPreferred ? "preferred" : "required",
          use_eager_acquire);
    } else if (policy == nvidia::gxf::NativeBufferPolicy::kRequired) {
      HOLOSCAN_LOG_ERROR(
          "PubSubContext: native buffer adapter init failed and policy is 'required'; "
          "aborting backend setup");
      native_adapter_.reset();
      return;
    } else {
      HOLOSCAN_LOG_WARN(
          "PubSubContext: native buffer adapter init failed, "
          "falling back to byte-staging path (policy is 'preferred')");
      native_adapter_.reset();
    }
  } else {
    HOLOSCAN_LOG_INFO("PubSubContext: native buffer (CUDA IPC) disabled by policy");
  }

  // Wire the allocator into GXF PubSubContext so deserialize() gets a non-null handle.
  auto dds_allocator = dds_context_->allocator();
  if (dds_allocator) {
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        gxf_ctx->context(), dds_allocator->gxf_cid());
    if (!allocator_handle) {
      HOLOSCAN_LOG_ERROR(
          "FastDdsPubSubNetworkContext::setup_backend: failed to create allocator handle ({})",
          GxfResultStr(allocator_handle.error()));
      return;
    }
    gxf_ctx->set_allocator(allocator_handle.value());
  } else {
    HOLOSCAN_LOG_WARN("FastDdsPubSubNetworkContext::setup_backend: DDS allocator is null");
  }

  // --- 3. Inject backends into the GXF PubSubContext ---
  gxf_ctx->set_discovery(dds_discovery_);
  gxf_ctx->set_transport(dds_transport_);
  gxf_ctx->set_serializer(dds_serializer_);

  // --- 4. Eagerly call initialize() + init_context() ---
  //
  // GXF entity activation order is: system -> router -> connection -> network -> graph.
  // The network router (router entity) calls init_context() on all registered
  // NetworkContexts during its own activation -- which happens BEFORE the
  // PubSubContext's graph entity is activated (and thus before the GXF component's
  // initialize() lifecycle method is called).
  //
  // To avoid the "init_context() called before initialize()" error, we trigger
  // both methods eagerly here.  Both are safe to call again later:
  //   - initialize() is idempotent (reads params, sets callbacks, sets flag)
  //   - init_context() re-sets callbacks but backends check is_initialized()
  //     before re-initializing, making the second call effectively a no-op.
  auto init_result = gxf_ctx->initialize();
  if (init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("FastDdsPubSubNetworkContext::setup_backend: GXF initialize() failed ({})",
                       GxfResultStr(init_result));
    return;
  }

  auto ctx_result = gxf_ctx->init_context();
  if (ctx_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("FastDdsPubSubNetworkContext::setup_backend: init_context() failed ({})",
                       GxfResultStr(ctx_result));
    return;
  }

  HOLOSCAN_LOG_INFO("PubSubContext: DDS backend ready (domain=0, participant='holoscan_{}')",
                    frag->name());
}

}  // namespace holoscan
