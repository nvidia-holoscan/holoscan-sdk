/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/rate.hpp>

namespace holoscan {

namespace detail {
class RuntimeRateState;

inline constexpr std::uint32_t kRateGenerationRecordLocalBits = 32U;
inline constexpr std::uint64_t kRateGenerationRecordLocalMask = 0x00000000FFFFFFFFULL;
inline constexpr std::uint64_t kRateGenerationRecordGenerationMask = 0xFFFFFFFF00000000ULL;
inline constexpr RateGenerationId kMaximumPackedRateGeneration = kRateGenerationRecordLocalMask;

static_assert((kMaximumPackedRateGeneration << kRateGenerationRecordLocalBits) ==
              kRateGenerationRecordGenerationMask);

[[nodiscard]] constexpr std::uint64_t rate_generation_record_id(RateGenerationId generation,
                                                                std::size_t local_index) noexcept {
  if (generation == 0U || generation > kMaximumPackedRateGeneration ||
      local_index >= kRateGenerationRecordLocalMask) {
    return 0U;
  }
  return (generation << kRateGenerationRecordLocalBits) |
         static_cast<std::uint64_t>(local_index + 1U);
}

[[nodiscard]] constexpr bool rate_generation_record_matches(std::uint64_t record_id,
                                                            RateGenerationId generation) noexcept {
  return record_id != 0U && generation != 0U && generation <= kMaximumPackedRateGeneration &&
         (record_id & kRateGenerationRecordGenerationMask) ==
             (generation << kRateGenerationRecordLocalBits);
}

static_assert(rate_generation_record_id(1U, 0U) == 0x0000000100000001ULL);
static_assert(rate_generation_record_id(kMaximumPackedRateGeneration,
                                        kRateGenerationRecordLocalMask - 1U) ==
              0xFFFFFFFFFFFFFFFFULL);
static_assert(rate_generation_record_id(kMaximumPackedRateGeneration + 1U, 0U) == 0U);
static_assert(rate_generation_record_id(1U, kRateGenerationRecordLocalMask) == 0U);
}  // namespace detail

/// @brief Identify one session-local port-rate result.
struct PortRateGenerationRecordId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(PortRateGenerationRecordId,
                                   PortRateGenerationRecordId) noexcept = default;
};

/// @brief Identify one session-local logical queue-sizing result.
struct QueueSizingGenerationId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(QueueSizingGenerationId,
                                   QueueSizingGenerationId) noexcept = default;
};

/// @brief Identify one evaluated stable-pool dependency.
struct PoolCapacityContributionGenerationId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(PoolCapacityContributionGenerationId,
                                   PoolCapacityContributionGenerationId) noexcept = default;
};

/// @brief Identify one session-local stable-pool capacity result.
struct PoolCapacityGenerationId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(PoolCapacityGenerationId,
                                   PoolCapacityGenerationId) noexcept = default;
};

/// @brief Identify one session-local provider allocation decision.
struct ProviderAllocationGenerationRecordId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProviderAllocationGenerationRecordId,
                                   ProviderAllocationGenerationRecordId) noexcept = default;
};

/// @brief Identify creation or replacement of concrete physical backing.
struct AllocationGenerationId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(AllocationGenerationId,
                                   AllocationGenerationId) noexcept = default;
};

/// @brief Identify one ABA-safe logical binding epoch over physical backing.
struct StorageBindingGenerationId {
  std::uint64_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(StorageBindingGenerationId,
                                   StorageBindingGenerationId) noexcept = default;
};

/// @brief Classify the terminal state of one rate-generation candidate.
enum class RateGenerationDisposition : std::uint8_t {
  kPublished = 0,  ///< The complete generation became active.
  kRejected = 1,   ///< The private candidate did not replace the active generation.
};

/// @brief Classify one provider materialization decision.
enum class AllocationAction : std::uint8_t {
  kFresh = 0,         ///< Materialize first physical backing.
  kReusedExact = 1,   ///< Rebind compatible backing with exact active geometry.
  kReusedLarger = 2,  ///< Rebind larger backing while enforcing a smaller active limit.
  kReplaced = 3,      ///< Materialize replacement backing.
  kRejected = 4,      ///< Reject without publishing a new storage binding.
};

/// @brief Preserve the resolved result and provenance for one port in one generation.
struct PortRateGenerationRecord {
  PortRateGenerationRecordId id{};
  PortId port{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::optional<Rate> selected_binding_rate;
  std::optional<Rate> propagated_rate;
  std::optional<Rate> resolved_rate;
  std::optional<Rate> sizing_rate;
  std::optional<Rate> max_nominal_rate;
  std::optional<PortId> immediate_provenance_port;
  std::optional<PortId> root_provenance_port;
  ResolvedRateOrigin resolved_origin{ResolvedRateOrigin::kUnresolved};
  PortRateViewState state{PortRateViewState::kUnresolved};
  RateComparison expectation_comparison{RateComparison::kNotCompared};
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve the evaluated logical queue result for one generation.
struct QueueSizingGenerationRecord {
  QueueSizingGenerationId id{};
  QueueSizingRecipeId recipe{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::optional<PortRateGenerationRecordId> producer_rate;
  std::optional<HorizonRecordId> accumulation_horizon;
  HardFloorContributorId first_hard_floor_contributor{};
  std::uint32_t hard_floor_contributor_count{};
  std::uint32_t hard_floor{};
  std::optional<std::uint32_t> nominal_arrivals;
  std::optional<std::uint32_t> rate_recommendation;
  std::optional<std::uint32_t> admitted_depth_ceiling;
  StabilityClass stability{StabilityClass::kUnresolvedBoundedEvidence};
  DiagnosticId stability_reason{};
  std::uint32_t effective_logical_depth{};
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve one evaluated immutable pool dependency.
struct PoolCapacityContributionGenerationRecord {
  PoolCapacityContributionGenerationId id{};
  PoolCapacityDependencyId dependency{};
  std::optional<QueueSizingGenerationId> edge_sizing_generation;
  std::uint32_t mapped_entries{};
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve one composed stable-pool capacity result for a generation.
struct PoolCapacityGenerationRecord {
  PoolCapacityGenerationId id{};
  ProviderAllocationPlanRecordId recipe{};
  std::string pool_id;
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  PoolCapacityContributionGenerationId first_contribution{};
  std::uint32_t contribution_count{};
  std::uint32_t required_pool_entries{};
  std::optional<std::uint32_t> admitted_pool_entry_ceiling;
  std::optional<std::uint64_t> max_payload_byte_span_per_slot;
  std::optional<std::uint64_t> logical_payload_capacity_bytes;
  std::optional<std::uint64_t> admitted_logical_payload_capacity_ceiling_bytes;
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve one actual provider allocation, reuse, replacement, or rejection decision.
struct ProviderAllocationGenerationRecord {
  ProviderAllocationGenerationRecordId id{};
  ProviderAllocationPlanRecordId recipe{};
  PoolCapacityGenerationId pool_capacity_generation{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::optional<AllocationGenerationId> allocation_generation;
  std::optional<StorageBindingGenerationId> storage_binding_generation;
  std::optional<ProviderAllocationGenerationRecordId> previous_decision;
  std::optional<AllocationGenerationId> previous_allocation_generation;
  AllocationAction action{AllocationAction::kRejected};
  DiagnosticId compatibility_reason{};
  std::optional<std::uint32_t> provider_materialized_entries;
  std::optional<std::uint64_t> provider_stride_bytes_per_entry;
  std::optional<std::uint64_t> provider_reserved_bytes_total;
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Index one terminal generation in an owning evidence snapshot.
struct RateGenerationEvidenceRecord {
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  RateGenerationDisposition disposition{RateGenerationDisposition::kRejected};
  ErrorCode error_code{ErrorCode::kSuccess};
  Diagnostic diagnostic;
  PortRateGenerationRecordId first_port_rate{};
  std::uint32_t port_rate_count{};
  QueueSizingGenerationId first_queue_sizing{};
  std::uint32_t queue_sizing_count{};
  PoolCapacityContributionGenerationId first_pool_contribution{};
  std::uint32_t pool_contribution_count{};
  PoolCapacityGenerationId first_pool_capacity{};
  std::uint32_t pool_capacity_count{};
  ProviderAllocationGenerationRecordId first_provider_allocation{};
  std::uint32_t provider_allocation_count{};
};

/// @brief Own bounded published and rejected rate-generation evidence for one RunSession.
///
/// The snapshot contains the active published generation and, when present, the most recent
/// rejected candidate. It owns every record and remains valid after the originating session ends.
/// An initial candidate failure can produce a rejected-only snapshot, so published() may return
/// null; use last_rejected() to inspect that startup failure.
/// Generation-specific range overloads accept only the exact record returned by this snapshot;
/// copied, caller-created, and foreign-snapshot records return an empty span.
/// The snapshot is immutable after construction: it may be copy- or move-constructed but cannot
/// be assigned. Copy construction creates independent record storage.
/// Runtime callback contexts cannot capture a snapshot; ordinary compute queries continue to use
/// the allocation-free PortRateView path.
/// @warning Every returned pointer and span is borrowed from this object. It remains valid only
/// while this snapshot remains alive and is not moved from. Do not use a record with another
/// snapshot, including a copied snapshot.
class HOLOSCAN_CORE_EXPORT RateGenerationSnapshot {
 public:
  /// @brief Construct an empty immutable snapshot.
  RateGenerationSnapshot() = default;
  /// @brief Copy records into an independent immutable snapshot.
  RateGenerationSnapshot(const RateGenerationSnapshot&) = default;
  /// @brief Transfer immutable record ownership into a new snapshot.
  RateGenerationSnapshot(RateGenerationSnapshot&&) noexcept = default;
  RateGenerationSnapshot& operator=(const RateGenerationSnapshot&) = delete;
  RateGenerationSnapshot& operator=(RateGenerationSnapshot&&) = delete;

  [[nodiscard]] PlanDigest plan_digest() const noexcept { return plan_digest_; }
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return plan_generation_; }
  [[nodiscard]] RunGenerationId run_generation() const noexcept { return run_generation_; }
  [[nodiscard]] const RateGenerationEvidenceRecord* published() const noexcept {
    for (const auto& generation : generations_) {
      if (generation.disposition == RateGenerationDisposition::kPublished) {
        return &generation;
      }
    }
    return nullptr;
  }
  [[nodiscard]] const RateGenerationEvidenceRecord* last_rejected() const noexcept {
    for (auto generation = generations_.rbegin(); generation != generations_.rend(); ++generation) {
      if (generation->disposition == RateGenerationDisposition::kRejected) {
        return &*generation;
      }
    }
    return nullptr;
  }
  [[nodiscard]] std::span<const RateGenerationEvidenceRecord> generations() const noexcept {
    return generations_;
  }
  [[nodiscard]] std::span<const PortRateGenerationRecord> port_rates() const noexcept {
    return port_rates_;
  }
  [[nodiscard]] std::span<const QueueSizingGenerationRecord> queue_sizings() const noexcept {
    return queue_sizings_;
  }
  [[nodiscard]] std::span<const PoolCapacityContributionGenerationRecord> pool_contributions()
      const noexcept {
    return pool_contributions_;
  }
  [[nodiscard]] std::span<const PoolCapacityGenerationRecord> pool_capacities() const noexcept {
    return pool_capacities_;
  }
  [[nodiscard]] std::span<const ProviderAllocationGenerationRecord> provider_allocations()
      const noexcept {
    return provider_allocations_;
  }

  /// Return the exact port-rate range owned by one generation record.
  [[nodiscard]] std::span<const PortRateGenerationRecord> port_rates(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    return record_range(
        port_rates_, generation, generation.first_port_rate, generation.port_rate_count);
  }

  /// Return the exact logical queue-sizing range owned by one generation record.
  [[nodiscard]] std::span<const QueueSizingGenerationRecord> queue_sizings(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    return record_range(
        queue_sizings_, generation, generation.first_queue_sizing, generation.queue_sizing_count);
  }

  /// Return the exact evaluated pool-contribution range owned by one generation record.
  [[nodiscard]] std::span<const PoolCapacityContributionGenerationRecord> pool_contributions(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    return record_range(pool_contributions_,
                        generation,
                        generation.first_pool_contribution,
                        generation.pool_contribution_count);
  }

  /// Return the exact composed pool-capacity range owned by one generation record.
  [[nodiscard]] std::span<const PoolCapacityGenerationRecord> pool_capacities(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    return record_range(pool_capacities_,
                        generation,
                        generation.first_pool_capacity,
                        generation.pool_capacity_count);
  }

  /// Return the exact provider-decision range owned by one generation record.
  [[nodiscard]] std::span<const ProviderAllocationGenerationRecord> provider_allocations(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    return record_range(provider_allocations_,
                        generation,
                        generation.first_provider_allocation,
                        generation.provider_allocation_count);
  }

 private:
  [[nodiscard]] bool owns_generation(
      const RateGenerationEvidenceRecord& generation) const noexcept {
    for (const auto& owned : generations_) {
      if (&owned == &generation) {
        return generation.run_generation == run_generation_ && generation.rate_generation != 0U;
      }
    }
    return false;
  }

  template <typename Record>
  [[nodiscard]] static bool record_belongs_to_generation(
      const Record& record, const RateGenerationEvidenceRecord& generation) noexcept {
    const bool encoded_generation_matches =
        detail::rate_generation_record_matches(record.id.value, generation.rate_generation);
    if (!encoded_generation_matches) {
      return false;
    }
    if constexpr (requires { record.edge_sizing_generation; }) {
      if (record.edge_sizing_generation &&
          !detail::rate_generation_record_matches(record.edge_sizing_generation->value,
                                                  generation.rate_generation)) {
        return false;
      }
    }
    if constexpr (requires {
                    record.run_generation;
                    record.rate_generation;
                  }) {
      return record.run_generation == generation.run_generation &&
             record.rate_generation == generation.rate_generation;
    }
    return true;
  }

  template <typename Record, typename Id>
  [[nodiscard]] std::span<const Record> record_range(const std::vector<Record>& records,
                                                     const RateGenerationEvidenceRecord& generation,
                                                     Id first, std::uint32_t count) const noexcept {
    if (!owns_generation(generation) || count == 0U) {
      return {};
    }
    const auto belongs = [&](const Record& record) noexcept {
      return record_belongs_to_generation(record, generation);
    };
    std::size_t begin = 0U;
    while (begin < records.size() && !belongs(records[begin])) {
      ++begin;
    }
    if (!first || begin == records.size() || records[begin].id != first) {
      return {};
    }
    std::size_t end = begin;
    while (end < records.size() && belongs(records[end])) {
      ++end;
    }
    if (end - begin != count) {
      return {};
    }
    for (std::size_t index = end; index < records.size(); ++index) {
      if (belongs(records[index])) {
        return {};
      }
    }
    return {records.data() + begin, static_cast<std::size_t>(count)};
  }

  PlanDigest plan_digest_{};
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  std::vector<RateGenerationEvidenceRecord> generations_;
  std::vector<PortRateGenerationRecord> port_rates_;
  std::vector<QueueSizingGenerationRecord> queue_sizings_;
  std::vector<PoolCapacityContributionGenerationRecord> pool_contributions_;
  std::vector<PoolCapacityGenerationRecord> pool_capacities_;
  std::vector<ProviderAllocationGenerationRecord> provider_allocations_;

  friend class detail::RuntimeRateState;
};

/// @brief Select the deterministic representation for a cold rate-aware queue inspection.
enum class RateQueueInspectionFormat : std::uint8_t {
  kHumanText = 0,      ///< Review-oriented line-delimited text.
  kCanonicalJson = 1,  ///< Canonical machine-readable JSON; not an ExecutionPlan artifact.
};

/// @brief Join immutable queue recipes with one session-local rate generation.
///
/// The returned text is a cold, non-mutating explanation. A paste-ready explicit-depth pin is
/// included only when automatic rate-and-contract sizing selected the active logical depth; the
/// function never applies that pin or changes a live queue, pool, or provider.
///
/// @param plan Immutable execution plan that produced @p snapshot.
/// @param snapshot Owning session-local rate-generation evidence.
/// @param rate_generation Exact published or rejected generation to inspect.
/// @param edge Exact logical payload edge to inspect.
/// @param format Deterministic human or canonical-JSON representation.
/// @return The joined explanation, or a typed error when identity or stable joins do not match.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::string, Error> render_rate_queue_inspection(
    const ExecutionPlan& plan, const RateGenerationSnapshot& snapshot,
    RateGenerationId rate_generation, EdgeId edge, RateQueueInspectionFormat format);

}  // namespace holoscan
