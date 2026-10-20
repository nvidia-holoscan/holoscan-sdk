/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_graph_phase_trace.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "runtime/runtime_phase_counters.hpp"

namespace holoscan::benchmarking {
namespace {

void append_timestamp_delta(std::vector<std::uint64_t>& values, const std::int64_t start,
                            const std::int64_t end, bool& valid) {
  if (start <= 0 || end < start) {
    valid = false;
    return;
  }
  values.push_back(static_cast<std::uint64_t>(end - start));
}

[[nodiscard]] RuntimePhaseReceiverSummary& phase_receiver(RuntimePhaseTraceSummary& summary,
                                                          const std::size_t receiver_index) {
  const auto existing =
      std::find_if(summary.receivers.begin(), summary.receivers.end(), [&](const auto& receiver) {
        return receiver.receiver_index == receiver_index;
      });
  if (existing != summary.receivers.end()) {
    return *existing;
  }
  const bool source = summary.receivers.empty();
  summary.receivers.push_back(RuntimePhaseReceiverSummary{
      .receiver_index = receiver_index,
      .source = source,
      .downstream_index = source ? std::nullopt
                                 : std::optional<std::uint32_t>{static_cast<std::uint32_t>(
                                       summary.receivers.size() - 1U)},
  });
  return summary.receivers.back();
}

}  // namespace

// The summarizer validates a stateful event grammar and is clearer as one chronological pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RuntimePhaseTraceSummary summarize_phase_trace(
    const detail::RuntimePhaseCounters* counters, const bool activation_trace_applicable,
    const std::function<std::int64_t(std::uint64_t)>& source_pre_return_timestamp) {
  RuntimePhaseTraceSummary summary;
  if (counters == nullptr) {
    return summary;
  }
  summary.enabled = true;
  summary.activation_trace_applicable = activation_trace_applicable;
  summary.capacity_exhausted = counters->trace_capacity_exhausted();
  summary.capacity = counters->trace_capacity();
  summary.attempted_records = counters->trace_attempted();
  summary.records = counters->trace_records().size();
  summary.dropped = counters->trace_dropped();
  summary.storage_invariants_valid = summary.records <= summary.capacity &&
                                     summary.attempted_records >= summary.records &&
                                     summary.attempted_records - summary.records == summary.dropped;
  // Detailed activation attribution is a serial trace. Independent partition owners can append
  // valid records concurrently, so their total order is not an activation order. Keep the bounded
  // storage and phase-counter evidence, but do not publish a misleading serial attribution.
  if (!summary.activation_trace_applicable) {
    summary.valid =
        summary.storage_invariants_valid && !summary.capacity_exhausted && summary.dropped == 0U;
    return summary;
  }
  summary.valid = true;

  struct ActivationTrace {
    std::size_t receiver_index{detail::kNoRuntimePhaseReceiver};
    SourceId source{};
    std::uint64_t source_sequence{};
    std::optional<std::int64_t> readiness_poll_returned;
    std::optional<std::int64_t> readiness_preceding_cut;
    std::optional<std::int64_t> preceding_cut;
    std::int64_t candidate_frozen{};
    std::int64_t candidate_selected{};
    std::int64_t input_selection_prepared{};
    std::int64_t activation_prepared{};
    std::int64_t dispatch_started{};
    std::int64_t callback_entered{};
    std::int64_t callback_returned{};
    std::int64_t dispatch_returned{};
    std::int64_t retirement_completed{};
  } activation;
  struct PendingReadinessPoll {
    SourceId source{};
    std::int64_t timestamp_ns{};
    std::int64_t preceding_cut_ns{};
  };
  struct PendingTensorEmit {
    std::uint64_t output_port_id{};
    std::uint64_t source_sequence{};
    std::int64_t started{};
    std::int64_t validated{};
    std::int64_t transport_started{};
    std::int64_t transport_returned{};
  };
  struct PendingInlinePayloadEmit {
    std::uint64_t output_port_id{};
    std::uint64_t source_sequence{};
    std::int64_t started{};
    std::int64_t encoded{};
    std::int64_t transport_started{};
    std::int64_t transport_returned{};
  };
  std::vector<PendingReadinessPoll> pending_readiness_polls;
  std::vector<PendingTensorEmit> pending_tensor_emits;
  std::vector<PendingInlinePayloadEmit> pending_inline_payload_emits;
  const auto pending_tensor_emit = [&](const detail::RuntimePhaseTraceRecord& record) {
    return std::find_if(pending_tensor_emits.begin(),
                        pending_tensor_emits.end(),
                        [&](const PendingTensorEmit& pending) {
                          return pending.output_port_id == record.output_port_id &&
                                 pending.source_sequence == record.source_sequence;
                        });
  };
  const auto pending_inline_payload_emit = [&](const detail::RuntimePhaseTraceRecord& record) {
    return std::find_if(pending_inline_payload_emits.begin(),
                        pending_inline_payload_emits.end(),
                        [&](const PendingInlinePayloadEmit& pending) {
                          return pending.output_port_id == record.output_port_id &&
                                 pending.source_sequence == record.source_sequence;
                        });
  };
  std::optional<bool> previous_activation_source;
  std::uint64_t consecutive_source_activations{};
  std::int64_t previous_cut{};

  for (const detail::RuntimePhaseTraceRecord& record : counters->trace_records()) {
    switch (record.point) {
      case detail::RuntimePhaseTracePoint::kChannelTransportWaitReturned:
      case detail::RuntimePhaseTracePoint::kChannelIngressRetained:
      case detail::RuntimePhaseTracePoint::kChannelEndOfStreamRetained:
      case detail::RuntimePhaseTracePoint::kChannelReadinessWakeSignaled:
      case detail::RuntimePhaseTracePoint::kControllerWakeSignaled:
      case detail::RuntimePhaseTracePoint::kControllerParkStarted:
      case detail::RuntimePhaseTracePoint::kControllerTimeoutReturned:
      case detail::RuntimePhaseTracePoint::kDirectCompositeStarted:
      case detail::RuntimePhaseTracePoint::kDirectCompositeSuccessorEntered:
      case detail::RuntimePhaseTracePoint::kControllerWakeReturned:
      case detail::RuntimePhaseTracePoint::kDirectCompositeReturned:
        break;
      case detail::RuntimePhaseTracePoint::kReadinessPollReturned:
        ++summary.readiness_polls;
        // Multiple fanout bindings can report readiness for one logical source before any
        // corresponding candidate freezes. Retain each poll and consume them in source-local FIFO
        // order so trace validation remains one-to-one.
        pending_readiness_polls.push_back(PendingReadinessPoll{
            .source = record.source,
            .timestamp_ns = record.timestamp_ns,
            .preceding_cut_ns = previous_cut,
        });
        break;
      case detail::RuntimePhaseTracePoint::kCandidateFrozen:
        if (activation.candidate_frozen != 0) {
          summary.valid = false;
          return summary;
        }
        activation.receiver_index = record.receiver_index;
        activation.source = record.source;
        activation.source_sequence = record.source_sequence;
        if (previous_cut != 0) {
          activation.preceding_cut = previous_cut;
        }
        if (const auto pending =
                std::find_if(pending_readiness_polls.begin(),
                             pending_readiness_polls.end(),
                             [&](const auto& poll) { return poll.source == record.source; });
            pending != pending_readiness_polls.end()) {
          activation.readiness_poll_returned = pending->timestamp_ns;
          if (pending->preceding_cut_ns != 0) {
            activation.readiness_preceding_cut = pending->preceding_cut_ns;
          }
          pending_readiness_polls.erase(pending);
        }
        activation.candidate_frozen = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kCandidateSelected:
        if (activation.candidate_frozen == 0 || activation.candidate_selected != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.candidate_selected = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kInputSelectionPrepared:
        if (activation.candidate_selected == 0 || activation.input_selection_prepared != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.input_selection_prepared = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationPrepared:
        if (activation.input_selection_prepared == 0 || activation.activation_prepared != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.activation_prepared = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchStarted:
        if (activation.candidate_frozen == 0 || activation.dispatch_started != 0 ||
            ((activation.candidate_selected != 0 || activation.input_selection_prepared != 0 ||
              activation.activation_prepared != 0) &&
             (activation.candidate_selected == 0 || activation.input_selection_prepared == 0 ||
              activation.activation_prepared == 0)) ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.dispatch_started = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeCallbackEntered:
        if (activation.dispatch_started == 0 || activation.dispatch_returned != 0 ||
            activation.callback_entered != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.callback_entered = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeCallbackReturned:
        if (activation.callback_entered == 0 || activation.callback_returned != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.callback_returned = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kComputeDispatchReturned:
        if (activation.dispatch_started == 0 ||
            (activation.callback_entered != 0 && activation.callback_returned == 0) ||
            activation.dispatch_returned != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.dispatch_returned = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationRetirementCompleted:
        if (activation.callback_returned == 0 || activation.dispatch_returned == 0 ||
            activation.retirement_completed != 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        activation.retirement_completed = record.timestamp_ns;
        break;
      case detail::RuntimePhaseTracePoint::kActivationCutCompleted: {
        if (activation.retirement_completed == 0 ||
            record.receiver_index != activation.receiver_index ||
            record.source != activation.source ||
            record.source_sequence != activation.source_sequence) {
          summary.valid = false;
          return summary;
        }
        RuntimePhaseReceiverSummary& receiver = phase_receiver(summary, activation.receiver_index);
        ++receiver.activations;
        ++summary.complete_activations;
        if (previous_activation_source.has_value()) {
          if (*previous_activation_source && receiver.source) {
            ++summary.source_to_source_transitions;
          } else if (*previous_activation_source) {
            ++summary.source_to_downstream_transitions;
          } else if (receiver.source) {
            ++summary.downstream_to_source_transitions;
          } else {
            ++summary.downstream_to_downstream_transitions;
          }
        }
        if (receiver.source) {
          if (!previous_activation_source.value_or(false)) {
            ++summary.source_activation_runs;
            consecutive_source_activations = 0U;
          }
          ++consecutive_source_activations;
          summary.maximum_consecutive_source_activations = std::max(
              summary.maximum_consecutive_source_activations, consecutive_source_activations);
        } else {
          consecutive_source_activations = 0U;
        }
        previous_activation_source = receiver.source;
        if (activation.preceding_cut.has_value()) {
          append_timestamp_delta(receiver.previous_cut_to_candidate,
                                 *activation.preceding_cut,
                                 activation.candidate_frozen,
                                 summary.valid);
        }
        if (activation.candidate_selected != 0) {
          append_timestamp_delta(receiver.candidate_to_selection,
                                 activation.candidate_frozen,
                                 activation.candidate_selected,
                                 summary.valid);
          append_timestamp_delta(receiver.selection_to_input_preparation,
                                 activation.candidate_selected,
                                 activation.input_selection_prepared,
                                 summary.valid);
          append_timestamp_delta(receiver.input_preparation_to_activation,
                                 activation.input_selection_prepared,
                                 activation.activation_prepared,
                                 summary.valid);
          append_timestamp_delta(receiver.activation_to_dispatch_start,
                                 activation.activation_prepared,
                                 activation.dispatch_started,
                                 summary.valid);
        }
        append_timestamp_delta(receiver.candidate_to_dispatch_start,
                               activation.candidate_frozen,
                               activation.dispatch_started,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_start_to_callback_entry,
                               activation.dispatch_started,
                               activation.callback_entered,
                               summary.valid);
        append_timestamp_delta(receiver.callback_entry_to_return,
                               activation.callback_entered,
                               activation.callback_returned,
                               summary.valid);
        append_timestamp_delta(receiver.callback_return_to_dispatch_return,
                               activation.callback_returned,
                               activation.dispatch_returned,
                               summary.valid);
        append_timestamp_delta(receiver.compute_dispatch,
                               activation.dispatch_started,
                               activation.dispatch_returned,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_return_to_retirement,
                               activation.dispatch_returned,
                               activation.retirement_completed,
                               summary.valid);
        append_timestamp_delta(receiver.retirement_to_cut,
                               activation.retirement_completed,
                               record.timestamp_ns,
                               summary.valid);
        append_timestamp_delta(receiver.dispatch_return_to_cut,
                               activation.dispatch_returned,
                               record.timestamp_ns,
                               summary.valid);
        if (activation.readiness_poll_returned.has_value()) {
          append_timestamp_delta(receiver.readiness_poll_return_to_candidate,
                                 *activation.readiness_poll_returned,
                                 activation.candidate_frozen,
                                 summary.valid);
          if (activation.readiness_preceding_cut.has_value()) {
            append_timestamp_delta(receiver.previous_cut_to_readiness_poll_return,
                                   *activation.readiness_preceding_cut,
                                   *activation.readiness_poll_returned,
                                   summary.valid);
          }
        }
        if (receiver.source) {
          const std::int64_t pre_return =
              source_pre_return_timestamp ? source_pre_return_timestamp(activation.source_sequence)
                                          : 0;
          // Backpressure retries advance the Runtime activation sequence without advancing the
          // benchmark publication sequence. Keep this optional attribution only while the two
          // sequence domains still identify a timestamp inside the current callback.
          if (pre_return >= activation.callback_entered &&
              pre_return <= activation.callback_returned) {
            append_timestamp_delta(receiver.callback_pre_return_to_dispatch_return,
                                   pre_return,
                                   activation.dispatch_returned,
                                   summary.valid);
          }
        }
        previous_cut = record.timestamp_ns;
        activation = {};
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorEmitStarted:
        if (record.output_port_id == 0U || record.source_sequence == 0U) {
          summary.valid = false;
          return summary;
        }
        if (const auto pending = pending_tensor_emit(record);
            pending != pending_tensor_emits.end()) {
          // A recoverable backpressure return has no terminal emit trace point. Starting the same
          // sequence again therefore closes that incomplete attempt before tracking the retry.
          pending_tensor_emits.erase(pending);
        }
        pending_tensor_emits.push_back(PendingTensorEmit{
            .output_port_id = record.output_port_id,
            .source_sequence = record.source_sequence,
            .started = record.timestamp_ns,
        });
        break;
      case detail::RuntimePhaseTracePoint::kTensorEmitValidated: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated != 0) {
          summary.valid = false;
          return summary;
        }
        pending->validated = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorTransportPublicationStarted: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated == 0 ||
            pending->transport_started != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_started = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorTransportPublicationReturned: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->transport_started == 0 ||
            pending->transport_returned != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_returned = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kTensorEmitCompleted: {
        const auto pending = pending_tensor_emit(record);
        if (pending == pending_tensor_emits.end() || pending->validated == 0 ||
            (pending->transport_started == 0) != (pending->transport_returned == 0)) {
          summary.valid = false;
          return summary;
        }
        append_timestamp_delta(
            summary.tensor_emit_total, pending->started, record.timestamp_ns, summary.valid);
        append_timestamp_delta(
            summary.tensor_emit_validation, pending->started, pending->validated, summary.valid);
        if (pending->transport_started != 0) {
          append_timestamp_delta(summary.tensor_pre_transport,
                                 pending->validated,
                                 pending->transport_started,
                                 summary.valid);
          append_timestamp_delta(summary.tensor_transport_publication,
                                 pending->transport_started,
                                 pending->transport_returned,
                                 summary.valid);
          append_timestamp_delta(summary.tensor_post_transport,
                                 pending->transport_returned,
                                 record.timestamp_ns,
                                 summary.valid);
        }
        ++summary.complete_tensor_emits;
        pending_tensor_emits.erase(pending);
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadEmitStarted:
        if (record.output_port_id == 0U || record.source_sequence == 0U) {
          summary.valid = false;
          return summary;
        }
        if (const auto pending = pending_inline_payload_emit(record);
            pending != pending_inline_payload_emits.end()) {
          // Inline payload emission has the same recoverable-backpressure trace shape as Tensor.
          pending_inline_payload_emits.erase(pending);
        }
        pending_inline_payload_emits.push_back(PendingInlinePayloadEmit{
            .output_port_id = record.output_port_id,
            .source_sequence = record.source_sequence,
            .started = record.timestamp_ns,
        });
        break;
      case detail::RuntimePhaseTracePoint::kInlinePayloadEncoded: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded != 0) {
          summary.valid = false;
          return summary;
        }
        pending->encoded = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadTransportPublicationStarted: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded == 0 ||
            pending->transport_started != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_started = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadTransportPublicationReturned: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->transport_started == 0 ||
            pending->transport_returned != 0) {
          summary.valid = false;
          return summary;
        }
        pending->transport_returned = record.timestamp_ns;
        break;
      }
      case detail::RuntimePhaseTracePoint::kInlinePayloadEmitCompleted: {
        const auto pending = pending_inline_payload_emit(record);
        if (pending == pending_inline_payload_emits.end() || pending->encoded == 0 ||
            (pending->transport_started == 0) != (pending->transport_returned == 0)) {
          summary.valid = false;
          return summary;
        }
        append_timestamp_delta(summary.inline_payload_emit_total,
                               pending->started,
                               record.timestamp_ns,
                               summary.valid);
        append_timestamp_delta(
            summary.inline_payload_encoding, pending->started, pending->encoded, summary.valid);
        if (pending->transport_started != 0) {
          append_timestamp_delta(summary.inline_payload_pre_transport,
                                 pending->encoded,
                                 pending->transport_started,
                                 summary.valid);
          append_timestamp_delta(summary.inline_payload_transport_publication,
                                 pending->transport_started,
                                 pending->transport_returned,
                                 summary.valid);
          append_timestamp_delta(summary.inline_payload_post_transport,
                                 pending->transport_returned,
                                 record.timestamp_ns,
                                 summary.valid);
        }
        ++summary.complete_inline_payload_emits;
        pending_inline_payload_emits.erase(pending);
        break;
      }
      default:
        summary.valid = false;
        return summary;
    }
    if (!summary.valid) {
      return summary;
    }
  }
  summary.unmatched_readiness_polls += pending_readiness_polls.size();
  const bool terminal_frozen_candidate =
      activation.candidate_frozen != 0 && activation.candidate_selected == 0 &&
      activation.input_selection_prepared == 0 && activation.activation_prepared == 0 &&
      activation.dispatch_started == 0 && activation.callback_entered == 0 &&
      activation.callback_returned == 0 && activation.dispatch_returned == 0 &&
      activation.retirement_completed == 0;
  summary.terminal_frozen_candidates = terminal_frozen_candidate ? 1U : 0U;
  // A stop-closed dispatch has no callback, retirement-completed, or cut checkpoints. It must
  // remain the final activation, with a complete ordered preparation/dispatch prefix and exactly
  // matching stop-admission evidence. Closed-island traces omit all three preparation checkpoints.
  const bool preparation_absent = activation.candidate_selected == 0 &&
                                  activation.input_selection_prepared == 0 &&
                                  activation.activation_prepared == 0;
  const bool preparation_ordered =
      activation.candidate_selected >= activation.candidate_frozen &&
      activation.input_selection_prepared >= activation.candidate_selected &&
      activation.activation_prepared >= activation.input_selection_prepared &&
      activation.dispatch_started >= activation.activation_prepared;
  const bool terminal_run_closing_dispatch =
      activation.candidate_frozen > 0 && (preparation_absent || preparation_ordered) &&
      activation.dispatch_started >= activation.candidate_frozen &&
      activation.dispatch_returned >= activation.dispatch_started &&
      activation.callback_entered == 0 && activation.callback_returned == 0 &&
      activation.retirement_completed == 0;
  summary.terminal_run_closing_dispatches = terminal_run_closing_dispatch ? 1U : 0U;
  const auto snapshot = counters->snapshot();
  const auto rejected = snapshot[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kComputeAdmissionsRunClosing)];
  const auto stops =
      snapshot[detail::RuntimePhaseCounters::index(detail::RuntimePhaseCounter::kStopSignals)];
  summary.valid = (activation.candidate_frozen == 0 || terminal_frozen_candidate ||
                   terminal_run_closing_dispatch) &&
                  summary.terminal_run_closing_dispatches == rejected && rejected <= stops &&
                  summary.dropped == 0U && summary.unmatched_readiness_polls == 0U &&
                  pending_tensor_emits.empty() && pending_inline_payload_emits.empty();
  return summary;
}

}  // namespace holoscan::benchmarking
