/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/endpoint_clock_health_lowering.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "compiler/clock_readiness_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/clock_binding_resolution.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool valid_clock_health_contract(const ExternalTopicRecord& topic) noexcept {
  const auto& health = topic.clock_health;
  const bool ordinary = health.capability == ClockHealthCapability::kUnavailable &&
                        health.requirement == ClockHealthRequirement::kNotRequired &&
                        !health.max_report_age.has_value();
  const bool monitored = topic.imported &&
                         health.capability == ClockHealthCapability::kStatusReports &&
                         health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
                         health.max_report_age.has_value() && health.max_report_age->count() > 0;
  return ordinary || monitored;
}

[[nodiscard]] std::string_view clock_health_capability_token(
    ClockHealthCapability capability) noexcept {
  switch (capability) {
    case ClockHealthCapability::kUnavailable:
      return "unavailable";
    case ClockHealthCapability::kStatusReports:
      return "status_reports";
  }
  return "unknown";
}

[[nodiscard]] std::string_view clock_health_requirement_token(
    ClockHealthRequirement requirement) noexcept {
  switch (requirement) {
    case ClockHealthRequirement::kNotRequired:
      return "not_required";
    case ClockHealthRequirement::kFreshHealthyReport:
      return "fresh_healthy_report";
  }
  return "unknown";
}

[[nodiscard]] std::string presented_clock_health_contract(const ExternalTopicRecord& topic) {
  const std::string max_report_age =
      topic.clock_health.max_report_age
          ? fmt::format("{}", topic.clock_health.max_report_age->count())
          : "absent";
  return fmt::format("capability={};requirement={};max_report_age_ns={};role={}",
                     clock_health_capability_token(topic.clock_health.capability),
                     clock_health_requirement_token(topic.clock_health.requirement),
                     max_report_age,
                     topic.imported ? "external_publisher" : "external_subscriber");
}

[[nodiscard]] std::string clock_health_json_pointer(const GraphSnapshotData& snapshot,
                                                    const ExternalTopicRecord& topic) {
  const std::string topic_name = topic.generated_name_requested || !topic.endpoint_name
                                     ? generated_endpoint_name_for_topic(snapshot, topic)
                                     : std::string{*topic.endpoint_name};
  std::size_t index{};
  for (const auto& candidate : snapshot.external_topics) {
    const std::string candidate_name =
        candidate.generated_name_requested || !candidate.endpoint_name
            ? generated_endpoint_name_for_topic(snapshot, candidate)
            : std::string{*candidate.endpoint_name};
    if (candidate_name < topic_name) {
      ++index;
    }
  }
  return fmt::format("/endpoints/{}/clock_health", index);
}

[[nodiscard]] EndpointCatalogDiagnostic invalid_clock_health_diagnostic(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic, DiagnosticSource source) {
  return EndpointCatalogDiagnostic{
      .token = "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID",
      .source = source,
      .json_pointer = clock_health_json_pointer(snapshot, topic),
      .required =
          "unavailable+not_required(age=absent,any_role)|status_reports+fresh_healthy_"
          "report(age=positive,external_publisher)",
      .provided = presented_clock_health_contract(topic),
  };
}

[[nodiscard]] ClockId resolve_topic_capture_clock(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic,
    std::span<const ClockPlanRecord> clocks) noexcept {
  if (!topic.capture_time.has_value()) {
    return {};
  }
  const VertexRecord* owner = find_vertex(snapshot, topic.vertex);
  if (owner == nullptr) {
    return {};
  }
  return resolve_plan_clock_id(snapshot, clocks, owner->graph, topic.capture_time->clock);
}

[[nodiscard]] bool usable_health_age_clock(const ClockPlanRecord& clock,
                                           ClockId capture_clock) noexcept {
  return clock.id != capture_clock && clock.can_read_now && clock.steady &&
         clock.independent_progress && !clock.capability_evidence_id.empty();
}

[[nodiscard]] ClockId resolve_health_age_clock(std::span<const ClockPlanRecord> clocks,
                                               ClockId root_default_clock,
                                               ClockId capture_clock) noexcept {
  const auto root = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.id == root_default_clock && usable_health_age_clock(clock, capture_clock);
  });
  if (root != clocks.end()) {
    return root->id;
  }
  const auto first = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return usable_health_age_clock(clock, capture_clock);
  });
  return first == clocks.end() ? ClockId{} : first->id;
}

[[nodiscard]] bool conversion_matches(std::span<const ClockConversionRecord> conversions,
                                      std::uint64_t id, ClockId from, ClockId to) noexcept {
  return id != 0U && std::any_of(conversions.begin(), conversions.end(), [&](const auto& item) {
           return item.id == id && item.from == from && item.to == to;
         });
}

[[nodiscard]] ClockHealthUnspecifiedDisposition unspecified_disposition(
    std::optional<Profile> profile) noexcept {
  if (!profile.has_value() || *profile == Profile::kRegulated) {
    return ClockHealthUnspecifiedDisposition::kRejected;
  }
  if (*profile == Profile::kRobotics || *profile == Profile::kRealTime) {
    return ClockHealthUnspecifiedDisposition::kQualifiedOnly;
  }
  return ClockHealthUnspecifiedDisposition::kAdmitted;
}

[[nodiscard]] EndpointClockHealthDisposition unsatisfied_disposition(
    std::optional<Profile> profile) noexcept {
  if (profile.has_value() && (*profile == Profile::kRobotics || *profile == Profile::kRealTime ||
                              *profile == Profile::kRegulated)) {
    return EndpointClockHealthDisposition::kFailRun;
  }
  return EndpointClockHealthDisposition::kSkip;
}

[[nodiscard]] constexpr std::uint8_t clock_health_profile_rank(Profile profile) noexcept {
  switch (profile) {
    case Profile::kResearch:
      return 0U;
    case Profile::kStandard:
      return 1U;
    case Profile::kRobotics:
      return 2U;
    case Profile::kRealTime:
      return 3U;
    case Profile::kRegulated:
      return 4U;
  }
  return 4U;
}

[[nodiscard]] std::optional<Profile> stricter_profile(std::optional<Profile> left,
                                                      std::optional<Profile> right) noexcept {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return clock_health_profile_rank(*left) < clock_health_profile_rank(*right) ? right : left;
}

[[nodiscard]] bool imported_topic_feeds_selection(const GraphSnapshotData& snapshot,
                                                  const ExternalTopicRecord& topic,
                                                  PortId selected_input) {
  const PortRecord* input = find_port(snapshot, GraphPortId{selected_input.value()});
  if (input == nullptr || input->direction != PortDirection::kInput) {
    return false;
  }
  const auto target_aliases = interface_alias_component(snapshot, input->id);
  const auto incoming =
      std::find_if(snapshot.edges.begin(), snapshot.edges.end(), [&](const auto& e) {
        return std::find(target_aliases.begin(), target_aliases.end(), e.target) !=
               target_aliases.end();
      });
  if (incoming == snapshot.edges.end()) {
    return false;
  }
  const auto source_aliases = interface_alias_component(snapshot, incoming->source);
  return std::find(source_aliases.begin(), source_aliases.end(), topic.graph_port) !=
         source_aliases.end();
}

[[nodiscard]] bool selector_requires_live_cross_domain_health(
    const InputSelectionPlanRecord& selection) noexcept {
  if (!selection.reference_clock.valid() || selection.conversion_id == 0U) {
    return false;
  }
  switch (selection.selector) {
    case InputSelectorKind::kExact:
    case InputSelectorKind::kLatestWithin:
    case InputSelectorKind::kBracketedWindow:
      return true;
    case InputSelectorKind::kDrainAll:
      return false;
  }
  return false;
}

struct EndpointTemporalHealthNeed {
  bool required{};
  bool conflicting_reference{};
  ClockId reference_clock{};
  std::uint64_t conversion_id{};
  std::optional<Profile> effective_profile;
};

[[nodiscard]] EndpointTemporalHealthNeed temporal_health_need(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic,
    std::span<const InputSelectionPlanRecord> selections,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::optional<Profile> endpoint_profile) {
  EndpointTemporalHealthNeed result{.effective_profile = endpoint_profile};
  if (!topic.imported || !topic.capture_time) {
    return result;
  }
  for (const auto& selection : selections) {
    if (!selector_requires_live_cross_domain_health(selection) ||
        !imported_topic_feeds_selection(snapshot, topic, selection.input)) {
      continue;
    }
    result.required = true;
    if (!result.reference_clock.valid()) {
      result.reference_clock = selection.reference_clock;
      result.conversion_id = selection.conversion_id;
    } else if (result.reference_clock != selection.reference_clock ||
               result.conversion_id != selection.conversion_id) {
      result.conflicting_reference = true;
    }
    const PortRecord* input = find_port(snapshot, GraphPortId{selection.input.value()});
    if (input != nullptr) {
      result.effective_profile =
          stricter_profile(result.effective_profile,
                           effective_profile_for_vertex(effective_vertex_profiles, input->owner));
    }
  }
  return result;
}

struct EndpointQualificationLookup {
  const EndpointClockHealthQualificationRecord* record{};
  bool duplicate{};
};

[[nodiscard]] EndpointQualificationLookup find_qualification(
    std::span<const EndpointClockHealthQualificationRecord> qualifications,
    PortId endpoint) noexcept {
  EndpointQualificationLookup result;
  for (const auto& qualification : qualifications) {
    if (qualification.endpoint != endpoint) {
      continue;
    }
    if (result.record != nullptr) {
      result.duplicate = true;
      return result;
    }
    result.record = &qualification;
  }
  return result;
}

[[nodiscard]] bool qualification_satisfies(
    const EndpointClockHealthQualificationRecord& qualification,
    std::chrono::nanoseconds max_report_age) noexcept {
  const std::array<std::chrono::nanoseconds, 4U> terms{
      qualification.monitor_period,
      qualification.observation_to_call,
      qualification.delivery_delay,
      qualification.jitter_margin,
  };
  if (terms.front().count() <= 0 || max_report_age.count() <= 0 ||
      qualification.evidence_reference.empty() || !qualification.bounded_transition_delivery ||
      !qualification.trace_loss_fail_closed) {
    return false;
  }
  std::int64_t total{};
  for (const auto term : terms) {
    if (term.count() < 0 || total > std::numeric_limits<std::int64_t>::max() - term.count()) {
      return false;
    }
    total += term.count();
  }
  return total < max_report_age.count();
}

[[nodiscard]] ClockHealthUnspecifiedDisposition qualified_unspecified_disposition(
    std::optional<Profile> profile, bool unspecified_reason_qualified) noexcept {
  if (profile == Profile::kRegulated) {
    return unspecified_reason_qualified ? ClockHealthUnspecifiedDisposition::kQualifiedOnly
                                        : ClockHealthUnspecifiedDisposition::kRejected;
  }
  return unspecified_disposition(profile);
}

}  // namespace

// Keep qualification, conversion, and profile admission in one fail-closed pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
EndpointClockHealthLoweringResult lower_endpoint_clock_health(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const EndpointClockHealthQualificationRecord> qualifications) {
  EndpointClockHealthLoweringResult result;
  result.records.reserve(snapshot.external_topics.size());
  const ClockId root_default_clock = [&]() noexcept {
    const GraphRecord* root = find_root_graph(snapshot);
    return root == nullptr ? ClockId{} : find_plan_default_clock_id(clocks, root->id);
  }();

  for (const auto& topic : snapshot.external_topics) {
    const DiagnosticSource source = make_endpoint_diagnostic_source(snapshot, topic);
    const std::optional<Profile> endpoint_profile =
        effective_profile_for_vertex(effective_vertex_profiles, topic.vertex);
    const EndpointTemporalHealthNeed need = temporal_health_need(
        snapshot, topic, input_selections, effective_vertex_profiles, endpoint_profile);
    const std::optional<Profile> profile = need.effective_profile;
    const bool monitored = topic.clock_health.capability == ClockHealthCapability::kStatusReports;
    const ClockId capture_clock = resolve_topic_capture_clock(snapshot, topic, clocks);
    const bool capture_contract_valid = !topic.capture_time.has_value() || capture_clock.valid();
    if (!valid_clock_health_contract(topic) || !capture_contract_valid ||
        (monitored && !capture_clock.valid())) {
      result.errors.push_back(invalid_clock_health_diagnostic(snapshot, topic, source));
    }

    ClockId age_clock{};
    if (monitored && need.required && !need.conflicting_reference) {
      const auto selected = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
        return clock.id == need.reference_clock;
      });
      if (selected != clocks.end() && usable_health_age_clock(*selected, capture_clock)) {
        age_clock = selected->id;
      }
    } else if (monitored) {
      age_clock = resolve_health_age_clock(clocks, root_default_clock, capture_clock);
    }
    if (monitored && !age_clock.valid()) {
      result.errors.push_back(invalid_clock_health_diagnostic(snapshot, topic, source));
    }
    if (need.conflicting_reference) {
      result.errors.push_back(invalid_clock_health_diagnostic(snapshot, topic, source));
    }

    if (need.required && !monitored) {
      // An ordinary endpoint remains ordinary and inspectable. Compilation
      // rejects the insufficient graph/profile capability rather than
      // inventing a monitor or rewriting the temporal Contract.
      result.errors.push_back(EndpointCatalogDiagnostic{"PROFILE_CAPABILITY_UNSATISFIED", source});
    }

    const PortId endpoint{topic.graph_port.value};
    const EndpointQualificationLookup qualification = find_qualification(qualifications, endpoint);
    const bool constrained = profile.has_value() && endpoint_profile_is_constrained(*profile);
    const bool qualified =
        monitored && qualification.record != nullptr && !qualification.duplicate &&
        topic.clock_health.max_report_age.has_value() &&
        qualification_satisfies(*qualification.record, *topic.clock_health.max_report_age);
    if (monitored && constrained && !qualified) {
      result.errors.push_back(EndpointCatalogDiagnostic{"PROFILE_EVIDENCE_UNSATISFIED", source});
    }

    const std::uint64_t conversion_id =
        need.required && !need.conflicting_reference ? need.conversion_id : 0U;
    if (monitored && need.required && capture_clock != age_clock &&
        !conversion_matches(clock_conversions, conversion_id, capture_clock, age_clock)) {
      result.errors.push_back(invalid_clock_health_diagnostic(snapshot, topic, source));
    }
    const EndpointClockHealthDisposition unsatisfied = unsatisfied_disposition(profile);
    result.records.push_back(EndpointClockHealthPlanRecord{
        .endpoint = endpoint,
        .plan_generation = plan_generation,
        .capture_clock = capture_clock,
        .capture_conversion_id = conversion_id,
        .conversion_generation = conversion_id == 0U ? 0U : plan_generation,
        .producer_epoch_source = monitored ? EndpointProducerEpochSource::kAttachmentGeneration
                                           : EndpointProducerEpochSource::kNotApplicable,
        .capability = topic.clock_health.capability,
        .requirement = topic.clock_health.requirement,
        .max_report_age = topic.clock_health.max_report_age,
        .health_age_clock = age_clock,
        .fixed_report_ingress_capacity = monitored ? 1U : 0U,
        .fixed_update_capacity = monitored ? 4U : 0U,
        .ingress_capacity_origin = monitored ? EndpointClockHealthBoundOrigin::kPortableMinimum
                                             : EndpointClockHealthBoundOrigin::kNotApplicable,
        .per_dispatch_report_budget = monitored ? 1U : 0U,
        .publisher_operation_sequence_id =
            monitored ? "holoscan.clock_health.publisher-operation-sequence.v1" : "",
        .report_sequence_id = monitored ? "holoscan.clock_health.report-sequence.v1" : "",
        .health_update_sequence_id = monitored ? "holoscan.clock_health.update-sequence.v1" : "",
        .monitor_path_kind = monitored ? EndpointClockHealthMonitorPathKind::kPublisherSerialized
                                       : EndpointClockHealthMonitorPathKind::kNotApplicable,
        .max_monitor_period =
            qualified ? std::optional{qualification.record->monitor_period} : std::nullopt,
        .max_observation_to_call_delay =
            qualified ? std::optional{qualification.record->observation_to_call} : std::nullopt,
        .max_report_delivery_delay =
            qualified ? std::optional{qualification.record->delivery_delay} : std::nullopt,
        .admitted_jitter_margin =
            qualified ? std::optional{qualification.record->jitter_margin} : std::nullopt,
        .monitor_evidence_reference = qualified ? qualification.record->evidence_reference : "",
        .no_report_disposition =
            monitored ? unsatisfied : EndpointClockHealthDisposition::kNotApplicable,
        .degraded_disposition = monitored ? EndpointClockHealthDisposition::kDegraded
                                          : EndpointClockHealthDisposition::kNotApplicable,
        .unhealthy_disposition =
            monitored ? unsatisfied : EndpointClockHealthDisposition::kNotApplicable,
        .stale_disposition =
            monitored ? unsatisfied : EndpointClockHealthDisposition::kNotApplicable,
        .unspecified_reason_disposition = qualified_unspecified_disposition(
            profile, qualified && qualification.record->unspecified_reason_qualified),
        .clock_health_update_witness_id = monitored ? "holoscan.event.clock-health-update.v1" : "",
        .trace_loss_disposition = monitored
                                      ? EndpointClockHealthTraceLossDisposition::kFailClosed
                                      : EndpointClockHealthTraceLossDisposition::kNotApplicable,
    });
  }
  std::sort(result.records.begin(), result.records.end(), [](const auto& left, const auto& right) {
    return left.endpoint.value() < right.endpoint.value();
  });
  canonicalize_plan_lowering_diagnostics(result.errors);
  return result;
}

}  // namespace holoscan::detail
