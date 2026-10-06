/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::doc::graph {

constexpr const char* kGraph = R"doc(Own a mutable Holoscan graph authoring root.)doc";
constexpr const char* kGraphCompose = R"doc(
Author the reusable Graph body.

Override this method in a Graph subclass. Core invokes it once for an embedded
Graph, and ``holoscan.compile`` invokes it once for a top-level Python Graph.
Authoring from the subclass constructor is unsupported; store constructor
arguments there and create Graph objects in this method.
)doc";
constexpr const char* kGraphHandle =
    R"doc(A weak, read-only handle to a reusable Graph instance embedded in a parent Graph.)doc";
constexpr const char* kInputDescriptor =
    R"doc(Declare a typed input on a reusable Graph subclass.)doc";
constexpr const char* kOutputDescriptor =
    R"doc(Declare a typed output on a reusable Graph subclass.)doc";
constexpr const char* kGraphOp = R"doc(
Add a registered C++ operator by registry name.

Keyword arguments are converted and validated from the operator's C++ parameter
schema. Duration fields accept ``datetime.timedelta`` or
``holoscan.time.ns``/``us``/``ms`` values; bare numeric durations are rejected.
)doc";
constexpr const char* kGraphEmbed = R"doc(
Embed a reusable Graph from a Python ``Graph`` subclass or a registered C++ Graph type.

Registered types accept a fully qualified registry name or a proxy from
``holoscan.graphs``/``holoscan.registry.resolve``. Keyword arguments are converted and validated
through the C++ constructor-parameter schema.
)doc";
constexpr const char* kOperatorHandle =
    R"doc(A weak handle to one Graph-owned operator declaration.)doc";
constexpr const char* kPort = R"doc(
A weak, lazily resolved view of one Graph-owned operator port.

Index an OperatorHandle to access arbitrary or Python-keyword port names. Attribute
access is available as sugar for ordinary identifiers.
)doc";
constexpr const char* kPortProject = R"doc(
Select one named Tensor field from a descriptor-backed schema output.

Pass the returned ``TensorProjection`` directly to ``Graph.add_flow``. The target
must be a Tensor input whose setup declaration calls ``accepts_projected_tensor()``.
The initial projection boundary supports one direct, same-process subscriber;
projected fanout, transported projection, and simultaneous delivery of the complete
schema and a projected Tensor are rejected during compilation.
)doc";
constexpr const char* kTensorProjection = R"doc(
A named Tensor field projected from a descriptor-backed schema output.

Create this view with ``output_port.project("field_name")`` and pass it directly
as the source of ``Graph.add_flow``. Projection narrows an existing schema sample;
it does not allocate, copy, or convert the Tensor data.

Projection currently supports one direct, same-process Tensor subscriber. Multiple
projected subscribers, transported projection, and simultaneous delivery of the
complete schema and a projected Tensor are rejected during compilation.
)doc";
constexpr const char* kFlow = R"doc(
A weak handle to one authored Graph flow.

``tensor_projection`` retains the selected field view for a schema-to-Tensor
flow and is ``None`` for an ordinary typed flow. ``source`` remains the original
authored output port in either case.
)doc";
constexpr const char* kPortConfigurator =
    R"doc(Configure one live Graph port through the Core authoring API.)doc";
constexpr const char* kFlowConfigurator =
    R"doc(Configure one live Graph flow through the Core authoring API.)doc";
constexpr const char* kPayloadType = R"doc(A registered payload identity token.)doc";
constexpr const char* kClockType = R"doc(A registered SDK Clock type token.)doc";
constexpr const char* kClockKey = R"doc(A named cold-authoring Clock declaration key.)doc";
constexpr const char* kTopicOptions = R"doc(
Owned, mutable backend-neutral options for one external topic declaration.

Use ``tensor`` for a plain Tensor payload and ``tensor_fields`` for a
descriptor-backed schema payload. Descriptor field keys are generated schema
names and unknown names report the complete known-field list.
)doc";
constexpr const char* kTensorRepresentation =
    R"doc(Required memory, data-type, rank, and stride facts for a Tensor boundary.)doc";
constexpr const char* kTensorBounds = R"doc(
Finite logical and physical bounds for a Tensor boundary.

Shape and stride inputs are copied into the Python value and remain valid through compile.
)doc";
constexpr const char* kTensorTransferPolicy =
    R"doc(External Tensor export and bounded-copy policy.)doc";
constexpr const char* kTensorBoundaryContract = R"doc(
Complete representation, bounds, transfer, and memory-conversion facts for a Tensor topic.
)doc";
constexpr const char* kImportedTopic =
    R"doc(A weak imported-topic handle exposing its Graph-side output port.)doc";
constexpr const char* kExportedTopic =
    R"doc(A weak exported-topic handle exposing its Graph-side input port.)doc";
constexpr const char* kRegionConstraint =
    R"doc(A backend-neutral logical-region constraint value.)doc";
constexpr const char* kRegionHandle = R"doc(A weak fluent handle for one named Graph region.)doc";
constexpr const char* kPartitionHandle =
    R"doc(A weak fluent handle for one named manual execution partition.)doc";
constexpr const char* kGraphLifecycleBuilder =
    R"doc(A weak fluent builder for lifecycle ordering and restart permissions.)doc";
constexpr const char* kLifecycleTransitionPhase =
    R"doc(The lifecycle transition phase associated with supplemental diagnostic evidence.)doc";

}  // namespace holoscan::doc::graph
