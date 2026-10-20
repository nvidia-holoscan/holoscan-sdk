/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>

namespace holoscan {

namespace detail {
class GraphState;
class GraphCompilerAccess;
struct GraphSnapshotData;
}  // namespace detail

/// @brief Identify one immutable Graph authoring snapshot generation.
using GraphGeneration = std::uint64_t;

/// @brief Identify one live Graph authoring instance without retaining an address.
///
/// The value is unique within the current process lifetime and remains stable
/// across snapshots of that Graph. It is not a pointer, persistent deployment
/// identity, or cross-process interchange key.
struct GraphInstanceId {
  std::array<std::byte, 16> value{};  ///< Complete process-local instance identity bytes.
  friend constexpr bool operator==(GraphInstanceId, GraphInstanceId) noexcept = default;
};

/// @brief Hold one deterministic 256-bit Graph authoring fingerprint.
///
/// Equal canonical pointer-free authoring records produce equal values
/// independent of Graph object address. The value is an authoring-content
/// fingerprint, not a cryptographic authenticity claim.
struct GraphAuthoringHash {
  std::array<std::byte, 32> value{};  ///< Complete canonical authoring digest bytes.
  friend constexpr bool operator==(GraphAuthoringHash, GraphAuthoringHash) noexcept = default;
};

/// @brief Identify the exact immutable Graph authoring state behind one snapshot.
///
/// The tuple distinguishes the live source Graph, its monotonically increasing
/// authoring version, and its canonical content. Separate Graph instances may
/// have equal \ref hash values while retaining different \ref graph identities.
struct GraphAuthoringIdentity {
  GraphInstanceId graph;      ///< Stable identity of the source Graph object.
  GraphGeneration version{};  ///< Source authoring generation frozen by the snapshot.
  GraphAuthoringHash hash;    ///< Deterministic hash of canonical pointer-free records.
  friend constexpr bool operator==(GraphAuthoringIdentity,
                                   GraphAuthoringIdentity) noexcept = default;
};

/// @brief Own the decoded canonical path of one embedded Graph instance.
///
/// Every value contains at least one embedded-instance segment. Root scope is
/// represented separately by an absent path in \ref GraphRecord; `_root` is an
/// endpoint-name rendering token and never a path segment. Identity comparisons
/// operate on decoded segments rather than a rendered slash-delimited string,
/// so local names may repeat in distinct parent scopes without aliasing.
class HOLOSCAN_CORE_EXPORT GraphInstancePath {
 public:
  GraphInstancePath() = delete;

  /// @brief Return the decoded path segments in root-to-leaf order.
  /// @return Borrowed immutable segments valid for this object's lifetime.
  [[nodiscard]] std::span<const std::string> segments() const noexcept { return segments_; }

  /// @brief Return the number of embedded-instance segments.
  /// @return Segment count.
  [[nodiscard]] std::size_t size() const noexcept { return segments_.size(); }

  friend bool operator==(const GraphInstancePath&, const GraphInstancePath&) noexcept = default;

 private:
  explicit GraphInstancePath(std::vector<std::string> segments) : segments_(std::move(segments)) {}
  std::vector<std::string> segments_;
  friend class detail::GraphState;
  friend class detail::GraphCompilerAccess;
  friend struct detail::GraphSnapshotData;
};

/// @brief Identify one Graph record within a GraphView snapshot.
struct GraphId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(GraphId, GraphId) noexcept = default;
};

/// @brief Identify one vertex record within a GraphView snapshot.
struct VertexId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(VertexId, VertexId) noexcept = default;
};

/// @brief Identify one edge record within a GraphView snapshot.
struct EdgeId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(EdgeId, EdgeId) noexcept = default;
};

/// @brief Identify one region record within a GraphView snapshot.
struct RegionId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(RegionId, RegionId) noexcept = default;
};

/// @brief Identify one manual execution-partition record within a GraphView snapshot.
///
/// @note This developer-preview identity is local to one authoring snapshot.
/// Compiler lowering assigns the corresponding immutable execution-plan
/// partition identity.
struct PartitionId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(PartitionId, PartitionId) noexcept = default;
};

/// @brief Identify one external topic record within a GraphView snapshot.
struct ExternalTopicId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(ExternalTopicId, ExternalTopicId) noexcept = default;
};

/// @brief Identify one port record within a GraphView authoring snapshot.
///
/// This value is never a compiler-assigned \c PortId. Compilation remaps it to a
/// plan-local identity when compilation freezes an authoring snapshot.
struct GraphPortId {
  std::uint64_t value{};  ///< Nonzero snapshot-local authoring identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(GraphPortId, GraphPortId) noexcept = default;
};

/// @brief Classify a logical Graph vertex before compiler-inserted lowering.
enum class VertexKind : std::uint8_t {
  kOperator = 0,        ///< User-authored Operator declaration.
  kGraphInterface = 1,  ///< Public reusable-Graph interface declaration.
  kImportedTopic = 2,   ///< Intrinsic external-to-Graph boundary.
  kExportedTopic = 3,   ///< Intrinsic Graph-to-external boundary.
};

/// @brief Classify which authoring object owns one logical port.
enum class PortOwnerKind : std::uint8_t {
  kOperator = 0,        ///< Operator member declared through `OperatorSpec`.
  kGraphInterface = 1,  ///< Reusable-Graph interface member.
  kImportedTopic = 2,   ///< Imported-topic Graph-side output.
  kExportedTopic = 3,   ///< Exported-topic Graph-side input.
};

}  // namespace holoscan
