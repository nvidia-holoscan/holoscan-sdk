/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/temporal_contract_records.hpp>

#include "temporal_contract_internal.hpp"

namespace holoscan::detail {

// These records are the explicit compiler-owned facts that cold Contract normalization cannot
// infer. Requiring them at the seam prevents EA1-005 from manufacturing graph identities, clock
// conversions, retention bounds, or predicate evidence assigned to EA1-009.
struct TemporalResolvedReferenceFacts {
  bool resolved{};
  TemporalClockKeyRecord clock{};
  TemporalRecordConversionKind conversion{TemporalRecordConversionKind::kSameClock};
  std::string conversion_record_id;
};

struct TemporalResolvedEdgeFacts {
  bool resolved{};
  std::string input_name;
  TemporalResolvedReferenceFacts reference{};
  std::uint32_t cache_capacity{};
  TemporalRecordBoundOrigin cache_capacity_origin{TemporalRecordBoundOrigin::kSingleSample};
  std::uint32_t drain_all_max_samples{};
  TemporalRecordProvenance provenance{};
};

struct TemporalResolvedTriggerPinFacts {
  bool resolved{};
  std::string input_name;
  std::uint32_t retained_pin_capacity{};
  TemporalRecordProvenance provenance{};
};

struct TemporalResolvedPredicateFacts {
  bool resolved{};
  std::string record_id;
  std::string implementation_fingerprint;
  std::string build_evidence;
  bool no_dynamic_allocation{};
  bool bounded_call{};
  bool owned_lifetime{};
  bool self_rebind{};
  std::uint64_t state_max_bytes{};
  std::string state_schema;
  std::uint64_t config_max_bytes{};
  std::string config_schema;
  std::string config_digest;
  TemporalPredicateReplayKind replay{TemporalPredicateReplayKind::kDecisionReplay};
  std::string replay_evidence;
};

struct TemporalContractLoweringContext {
  std::string operator_name;
  TemporalRecordProvenance provenance{};
  std::uint32_t connected_input_count{};
  std::vector<TemporalResolvedTriggerPinFacts> trigger_pins;
  std::string source_provenance;
  TemporalResolvedReferenceFacts trigger_reference{};
  std::vector<TemporalResolvedEdgeFacts> edges;
  std::optional<TemporalResolvedPredicateFacts> predicate;
};

struct LoweredTemporalRecords {
  ContractRecord contract;
  std::vector<EdgeSelectionRecord> edges;
};

// Lowering is deliberately private and compiler-consumable. It maps every EA1-005 normalized
// shape but cannot resolve any graph-owned fact; callers must provide those facts explicitly.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<LoweredTemporalRecords, Error>
lower_temporal_contract_records(const TemporalContractSnapshot& snapshot,
                                const TemporalContractLoweringContext& context);

}  // namespace holoscan::detail
