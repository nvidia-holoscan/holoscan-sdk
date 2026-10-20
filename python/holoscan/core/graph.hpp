/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan {

class ExecutionPartitionView;
class ExecutionPartitionsView;

namespace config {
class Path;
}  // namespace config

namespace params {
class ParamConfigBinding;
enum class ParamBindingMode : std::uint8_t;
}  // namespace params

namespace nb = nanobind;

void init_graph(nb::module_& module);

// Use the same configured construction route for a caller-owned native Graph.
// Adapted Python constructor callbacks require a GC-visible Python owner and are rejected here.
[[nodiscard]] authoring::OpaqueOperatorRef add_python_operator(Graph& graph, nb::handle type,
                                                               std::string_view name,
                                                               const nb::dict& params);

// The returned string views remain valid while the Python options object is retained.
[[nodiscard]] ConnectionOptions python_connection_options(nb::handle options);

void prepare_python_graph_for_compile(nb::handle value);

class PythonCallbackOwners;

// Adopts the Python callbacks owned by `graph` and by its embedded hosted child Graphs.
void collect_python_callback_owners(nb::handle graph, PythonCallbackOwners& into);

[[nodiscard]] Graph& python_graph(nb::handle value);

[[nodiscard]] nb::object python_external_topic_port(authoring::OpaquePortRef reference,
                                                    PortDirection direction,
                                                    const authoring::RegisteredPayloadType& payload,
                                                    std::string owner_name);

[[nodiscard]] nb::object python_graph_interface_port(
    nb::handle value, const std::string& name, PortDirection direction,
    const authoring::RegisteredPayloadType& payload);

void python_region_add(RegionHandle& region, const nb::args& operators);

void python_partition_add(PartitionHandle& partition, const nb::args& operators);

void python_lifecycle_before(GraphLifecycleBuilder& lifecycle, nb::handle before, nb::handle after,
                             LifecycleStage stage);

void python_lifecycle_allow_self_restart(GraphLifecycleBuilder& lifecycle, nb::handle target);

void python_lifecycle_allow_external_restart(GraphLifecycleBuilder& lifecycle, nb::handle target);

void python_lifecycle_allow_restart_request(GraphLifecycleBuilder& lifecycle, nb::handle requester,
                                            nb::handle target);

[[nodiscard]] std::string python_operator_name(nb::handle value);

[[nodiscard]] params::ParamConfigBinding python_operator_config_binding(
    nb::handle value, config::Path path, params::ParamBindingMode mode);

[[nodiscard]] ExecutionPartitionView python_operator_partition(
    const ExecutionPartitionsView& partitions, nb::handle value);

[[nodiscard]] std::string python_operator_path(
    nb::handle value, const char* type_error_token = "PARAM_CONFIG_OPERATOR_REQUIRED");

}  // namespace holoscan
