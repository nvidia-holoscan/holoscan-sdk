/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include "graph_state.hpp"
#include "operator_spec_compiler_access.hpp"

// Helpers shared by the Graph authoring sources: graph.cpp, graph_composition.cpp,
// graph_flows.cpp, graph_operators.cpp, and graph_snapshot.cpp. Helpers that only one source uses
// stay in that source's anonymous namespace.
namespace holoscan::detail {

template <typename Callable>
void graph_log_noexcept(Callable&& callable) noexcept {
  try {
    std::forward<Callable>(callable)();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is diagnostic-only. Formatting, allocation, and logger backend
    // failures must not replace an authoring result or make committed state
    // appear to have failed.
  }
}

template <typename Enum>
[[nodiscard]] constexpr bool closed_enum(Enum value) noexcept;

// clang-format off
#define HOLOSCAN_GRAPH_CLOSED_ENUM(EnumType, ...)                     \
  template <>                                                         \
  [[nodiscard]] constexpr bool closed_enum(EnumType value) noexcept { \
    switch (value) {                                                  \
      __VA_ARGS__                                                     \
    }                                                                 \
    return false;                                                     \
  }
#define HOLOSCAN_GRAPH_ENUM_CASE(Value) \
  case Value:                           \
    return true;
// clang-format on

// The generated switches deliberately return the same value for every listed enumerator: the
// repetition is the explicit whitelist that rejects fabricated closed-enum values.
// NOLINTBEGIN(bugprone-branch-clone)
HOLOSCAN_GRAPH_CLOSED_ENUM(QueuePolicy, HOLOSCAN_GRAPH_ENUM_CASE(QueuePolicy::kBlocking)
                                            HOLOSCAN_GRAPH_ENUM_CASE(QueuePolicy::kDropOldest))
HOLOSCAN_GRAPH_CLOSED_ENUM(MemoryConversionPolicy,
                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kDefault)
                               HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kDisable)
                                   HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kAllowIfSafe)
                                       HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kRequire))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    InterfaceTensorEmitPolicy,
    HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kRequireExportable)
        HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kCopyIfNeeded)
            HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded))
HOLOSCAN_GRAPH_CLOSED_ENUM(TensorStridePolicy,
                           HOLOSCAN_GRAPH_ENUM_CASE(TensorStridePolicy::kContiguousOnly)
                               HOLOSCAN_GRAPH_ENUM_CASE(TensorStridePolicy::kBoundedStrided))
HOLOSCAN_GRAPH_CLOSED_ENUM(PeerRequirement,
                           HOLOSCAN_GRAPH_ENUM_CASE(PeerRequirement::kOptional)
                               HOLOSCAN_GRAPH_ENUM_CASE(PeerRequirement::kRequired))
HOLOSCAN_GRAPH_CLOSED_ENUM(ClockHealthCapability,
                           HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthCapability::kUnavailable)
                               HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthCapability::kStatusReports))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    ClockHealthRequirement,
    HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthRequirement::kNotRequired)
        HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthRequirement::kFreshHealthyReport))
HOLOSCAN_GRAPH_CLOSED_ENUM(MemoryKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kUnknown)
                               HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kHost)
                                   HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kPinnedHost)
                                       HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kCudaDevice)
                                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kCudaManaged))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    ScalarIdentityKind,
    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kBoolean) HOLOSCAN_GRAPH_ENUM_CASE(
        ScalarIdentityKind::kUInt8) HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt8)
        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt16)
            HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt16)
                HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt32)
                    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt32)
                        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt64)
                            HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt64)
                                HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kFloat32)
                                    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kFloat64)
                                        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kLocalEnum))
HOLOSCAN_GRAPH_CLOSED_ENUM(PayloadKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kInlineTyped)
                               HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kSerializedBytes)
                                   HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kDescriptorBacked)
                                       HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kCompatibilityPayload)
                                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kControlEmpty))
HOLOSCAN_GRAPH_CLOSED_ENUM(PayloadIdentityKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kScalar)
                               HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kTensor)
                                   HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kSchema)
                                       HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kPod))
HOLOSCAN_GRAPH_CLOSED_ENUM(Profile, HOLOSCAN_GRAPH_ENUM_CASE(Profile::kStandard)
                                        HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRobotics)
                                            HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRealTime)
                                                HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRegulated)
                                                    HOLOSCAN_GRAPH_ENUM_CASE(Profile::kResearch))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    RegionConstraintKind,
    HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kStaticTopology)
        HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kBoundedMemory)
            HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kDeviceCompatibleOperators)
                HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kLatencyBudget)
                    HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kCudaGraph))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    LifecycleStage, HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kConfigure)
                        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kDiscover)
                            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kAllocate)
                                HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kWarmUp)
                                    HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kArm)
                                        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kStart)
                                            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kStop)
                                                HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kRelease))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    OperatorSpecCompilerAccess::LifecycleFactKind,
    HOLOSCAN_GRAPH_ENUM_CASE(OperatorSpecCompilerAccess::LifecycleFactKind::kProduces)
        HOLOSCAN_GRAPH_ENUM_CASE(OperatorSpecCompilerAccess::LifecycleFactKind::kRequires))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    LifecycleRestartOrigin,
    HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kSelf)
        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kExternal)
            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kOperatorRequest))
HOLOSCAN_GRAPH_CLOSED_ENUM(ReadinessSourceKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kReadable)
                               HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kNotification)
                                   HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kDoorbell)
                                       HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kSpin))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    PortRateKind, HOLOSCAN_GRAPH_ENUM_CASE(PortRateKind::kUnspecified)
                      HOLOSCAN_GRAPH_ENUM_CASE(PortRateKind::kFixedNominal)
                          HOLOSCAN_GRAPH_ENUM_CASE(PortRateKind::kExpectedInput)
                              HOLOSCAN_GRAPH_ENUM_CASE(PortRateKind::kFollowsInput)
                                  HOLOSCAN_GRAPH_ENUM_CASE(PortRateKind::kLifecycleNominal))
// NOLINTEND(bugprone-branch-clone)

#undef HOLOSCAN_GRAPH_ENUM_CASE
#undef HOLOSCAN_GRAPH_CLOSED_ENUM

// Defined in graph.cpp.

/// Return the percent-encoded instance path of an authored Operator vertex.
[[nodiscard]] std::string authored_operator_path(const GraphState& state,
                                                 const VertexRecord& vertex);
/// Build a Graph authoring error with a stable token.
[[nodiscard]] Error graph_error(ErrorCode code, std::string_view token);
/// Return the display name of a payload identity for diagnostics.
[[nodiscard]] std::string graph_payload_identity_name(const PayloadIdentityRecord& identity);
/// Return whether a name starts or ends with whitespace.
[[nodiscard]] bool edge_whitespace(std::string_view value) noexcept;
/// Validate an authored name; the error carries the given token.
[[nodiscard]] expected<void, Error> validate_name(std::string_view name, std::string_view token,
                                                  bool allow_empty = false);
/// Validate a schema identity record.
[[nodiscard]] expected<void, Error> validate_schema(const SchemaIdentityRecord& schema);
/// Validate a payload identity record.
[[nodiscard]] expected<void, Error> validate_payload_identity(
    const PayloadIdentityRecord& identity);
/// Validate payload options against their payload identity and Tensor layout.
[[nodiscard]] expected<void, Error> validate_payload_options(
    const PayloadOptionsRecord& options, const PayloadIdentityRecord& identity,
    const std::optional<TensorRepresentation>& layout);
/// Check a POD payload identity against the Graph's POD catalog.
[[nodiscard]] expected<void, Error> validate_pod_catalog(const GraphState& state,
                                                         const PayloadIdentityRecord& identity);
/// Add a POD payload identity to the Graph's POD catalog.
void record_pod_identity(GraphState& state, const PayloadIdentityRecord& identity);
/// Find a port by ID; null when the ID is stale or out of range.
[[nodiscard]] PortStorage* find_port(GraphState& state, GraphPortId id) noexcept;
/// Find a port by ID; null when the ID is stale or out of range.
[[nodiscard]] const PortStorage* find_port(const GraphState& state, GraphPortId id) noexcept;

// Defined in graph_operators.cpp.

/// Compare two optional schema identities.
[[nodiscard]] bool schema_equal(const std::optional<SchemaIdentityRecord>& lhs,
                                const std::optional<SchemaIdentityRecord>& rhs);

// Defined in graph_flows.cpp.

/// Reserve adjacency storage for one more edge from a vertex.
void reserve_adjacency(GraphState& state, VertexId source);
/// Record an edge between two vertices in the adjacency lists.
void append_adjacency(GraphState& state, VertexId source, VertexId target);

}  // namespace holoscan::detail
