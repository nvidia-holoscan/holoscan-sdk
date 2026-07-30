/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_NETWORK_CONTEXTS_PYDOC_HPP
#define PYHOLOSCAN_NETWORK_CONTEXTS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace UcxContext {

// Constructor
PYDOC(UcxContext, R"doc(
UCX network context class.
)doc")

// PyUcxContext Constructor
PYDOC(UcxContext_python, R"doc(
UCX network context

Parameters
----------
fragment : Fragment
    The fragment the condition will be associated with the UCX network context.
entity_serializer : holoscan.resources.ucx_entity_serializer
    The UCX entity serializer used by the UCX network context.
name : str, optional
    The name of the network context.
)doc")

}  // namespace UcxContext

namespace FastDdsPubSubNetworkContext {

PYDOC(FastDdsPubSubNetworkContext, R"doc(
FastDDS pub/sub network context class.
)doc")

PYDOC(FastDdsPubSubNetworkContext_python, R"doc(
FastDDS pub/sub network context.

Parameters
----------
fragment : Fragment
    The fragment the network context will be associated with.
native_buffer_policy : str, optional
    Native buffer policy forwarded to the FastDDS backend. Typical values are
    ``"disabled"``, ``"preferred"``, and ``"required"``.
native_buffer_acquire_timeout_ms : int, optional
    Maximum wait time in milliseconds for subscriber-side native-buffer acquire.
native_buffer_export_ttl_ms : int, optional
    Publisher-side stale pending-export eviction age in milliseconds.
native_buffer_use_eager_acquire : bool, optional
    If ``True``, use eager acquire for FastDDS CUDA IPC native buffers.
name : str, optional
    The name of the network context.
)doc")

}  // namespace FastDdsPubSubNetworkContext
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_NETWORK_CONTEXTS_PYDOC_HPP
