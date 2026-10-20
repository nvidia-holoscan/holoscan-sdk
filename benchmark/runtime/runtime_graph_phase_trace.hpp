/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "runtime/runtime_phase_counters.hpp"

namespace holoscan::benchmarking {

struct RuntimePhaseReceiverSummary {
  std::size_t receiver_index{};
  bool source{};
  std::optional<std::uint32_t> downstream_index;
  std::uint64_t activations{};
  std::vector<std::uint64_t> previous_cut_to_candidate;
  std::vector<std::uint64_t> previous_cut_to_readiness_poll_return;
  std::vector<std::uint64_t> readiness_poll_return_to_candidate;
  std::vector<std::uint64_t> candidate_to_selection;
  std::vector<std::uint64_t> selection_to_input_preparation;
  std::vector<std::uint64_t> input_preparation_to_activation;
  std::vector<std::uint64_t> activation_to_dispatch_start;
  std::vector<std::uint64_t> candidate_to_dispatch_start;
  std::vector<std::uint64_t> dispatch_start_to_callback_entry;
  std::vector<std::uint64_t> callback_entry_to_return;
  std::vector<std::uint64_t> callback_return_to_dispatch_return;
  std::vector<std::uint64_t> compute_dispatch;
  std::vector<std::uint64_t> dispatch_return_to_retirement;
  std::vector<std::uint64_t> retirement_to_cut;
  std::vector<std::uint64_t> dispatch_return_to_cut;
  std::vector<std::uint64_t> callback_pre_return_to_dispatch_return;
};

struct RuntimePhaseTraceSummary {
  bool enabled{};
  bool activation_trace_applicable{};
  bool valid{};
  bool capacity_exhausted{};
  bool storage_invariants_valid{};
  std::size_t capacity{};
  std::size_t attempted_records{};
  std::size_t records{};
  std::uint64_t dropped{};
  std::uint64_t readiness_polls{};
  std::uint64_t unmatched_readiness_polls{};
  std::uint64_t complete_activations{};
  std::uint64_t terminal_frozen_candidates{};
  std::uint64_t terminal_run_closing_dispatches{};
  std::uint64_t source_activation_runs{};
  std::uint64_t maximum_consecutive_source_activations{};
  std::uint64_t source_to_source_transitions{};
  std::uint64_t source_to_downstream_transitions{};
  std::uint64_t downstream_to_source_transitions{};
  std::uint64_t downstream_to_downstream_transitions{};
  std::uint64_t complete_tensor_emits{};
  std::vector<std::uint64_t> tensor_emit_total;
  std::vector<std::uint64_t> tensor_emit_validation;
  std::vector<std::uint64_t> tensor_pre_transport;
  std::vector<std::uint64_t> tensor_transport_publication;
  std::vector<std::uint64_t> tensor_post_transport;
  std::uint64_t complete_inline_payload_emits{};
  std::vector<std::uint64_t> inline_payload_emit_total;
  std::vector<std::uint64_t> inline_payload_encoding;
  std::vector<std::uint64_t> inline_payload_pre_transport;
  std::vector<std::uint64_t> inline_payload_transport_publication;
  std::vector<std::uint64_t> inline_payload_post_transport;
  std::vector<RuntimePhaseReceiverSummary> receivers;
};

/// Summarize a stopped run's trace without counting rejected admissions as callbacks.
[[nodiscard]] RuntimePhaseTraceSummary summarize_phase_trace(
    const detail::RuntimePhaseCounters* counters, bool activation_trace_applicable,
    const std::function<std::int64_t(std::uint64_t)>& source_pre_return_timestamp = {});

}  // namespace holoscan::benchmarking
