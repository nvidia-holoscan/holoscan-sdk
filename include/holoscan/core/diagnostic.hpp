/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan {

class HOLOSCAN_CORE_EXPORT ExecutionGraphView;

/// @brief Identify one stable Core structural diagnostic definition.
///
/// Values are append-only within the Core diagnostic registry. Zero denotes an
/// unspecified identity and never aliases a registered diagnostic.
struct DiagnosticId {
  std::uint32_t value{};  ///< Registry-assigned numeric identity; zero is unspecified.

  /// @brief Return whether this identity is registered and nonzero.
  /// @return \c true when \ref value is nonzero.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }

  friend constexpr bool operator==(DiagnosticId, DiagnosticId) noexcept = default;
};

/// @brief Classify one structural diagnostic's compile-admission or runtime effect.
enum class DiagnosticSeverity : std::uint8_t {
  kWarning = 0,  ///< Preserve evidence without rejecting admission or failing a run.
  kError = 1,    ///< Reject compile admission or fail a run while retaining evidence.
};

/// @brief Identify the bounded plan location associated with one diagnostic.
///
/// Every identity is interpreted only within the enclosing plan or the
/// RunSession that retains that plan. Zero fields are absent. The record
/// contains no authoring address, backend handle, or arbitrary source text.
struct DiagnosticSource {
  GraphGeneration graph_generation{};  ///< Frozen source Graph generation, or zero.
  GraphId graph{};                     ///< Canonical Graph scope, or zero.
  VertexId vertex{};                   ///< Canonical logical vertex, or zero.
  PortId port{};                       ///< Dense plan-local port, or invalid.
};

/// @brief Report one stable bounded compile or runtime result.
///
/// The token is a registry spelling, not free-form log prose. Core producers
/// bound it before publishing immutable evidence. Compile records are
/// plan-owned and serialized deterministically; runtime records are bounded
/// and owned by one RunSession. Typed detail remains private to the producing
/// subsystem.
struct Diagnostic {
  DiagnosticId id{};                                        ///< Stable registry identity.
  DiagnosticSeverity severity{DiagnosticSeverity::kError};  ///< Admission or run-outcome effect.
  std::string token;                                        ///< Owned stable registry token.
  DiagnosticSource source{};                                ///< Optional bounded plan-local source.
  std::string service;       ///< Stable service identity when the diagnostic is catalog-scoped.
  std::string endpoint;      ///< Stable endpoint identity when the diagnostic is endpoint-scoped.
  std::string json_pointer;  ///< RFC 6901 location in the normalized diagnostic document.
  std::string required;      ///< Canonical required-side summary when applicable.
  std::string provided;      ///< Canonical candidate-side summary when applicable.
  std::optional<DiagnosticId> cause_id;  ///< Earlier report record that caused this result.
};

/// @brief Format one structural diagnostic for human-readable output.
/// @param diagnostic Structured diagnostic to render.
/// @param graph Immutable compiled graph used to resolve a nonzero source vertex. A
/// default-constructed view leaves vertex names unresolved.
/// @return One human-readable line without a trailing newline.
/// @note Source vertex IDs are canonical plan IDs, so an authoring GraphView is not accepted.
/// @note Preserve and inspect the structured fields when program logic depends on diagnostics.
/// The rendered text is not a stable machine-readable format.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::string format_diagnostic(const Diagnostic& diagnostic,
                                                                 const ExecutionGraphView& graph);

}  // namespace holoscan
