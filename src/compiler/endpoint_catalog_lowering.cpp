/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/endpoint_catalog_lowering.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <holoscan/schema/schema_package.hpp>

#include "channel/catalog_diagnostic_detail.hpp"
#include "compiler/clock_plan_resolution.hpp"
#include "compiler/clock_readiness_lowering.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/rate_plan_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/source_id_internal.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kGeneratedPrefix = "__holoscan_auto_endpoint/v1/";
constexpr std::string_view kGeneratedNamespace = "__holoscan_auto_endpoint/";
constexpr std::string_view kGeneratedAlgorithm = "holoscan.auto_endpoint_name";
constexpr std::uint32_t kGeneratedAlgorithmVersion = 1U;
constexpr std::chrono::seconds kOrdinaryAttachWait{2};

[[nodiscard]] std::string encoded_graph_path(const GraphSnapshotData& snapshot,
                                             const ExternalTopicRecord& topic) {
  const VertexRecord* vertex = find_vertex(snapshot, topic.vertex);
  const GraphRecord* graph = vertex == nullptr ? nullptr : find_graph(snapshot, vertex->graph);
  if (graph == nullptr || !graph->path) {
    return std::string{kRootGraphPathComponent};
  }
  std::string result;
  for (const auto& segment : graph->path->segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    append_percent_encoded_uri_component(result, segment);
  }
  return result.empty() ? std::string{kRootGraphPathComponent} : result;
}

[[nodiscard]] std::string generated_endpoint_name(const GraphSnapshotData& snapshot,
                                                  const ExternalTopicRecord& topic) {
  return fmt::format("{}{}{}/{}",
                     kGeneratedPrefix,
                     encoded_graph_path(snapshot, topic),
                     topic.imported ? "/import" : "/export",
                     topic.declaration_ordinal);
}

[[nodiscard]] bool decode_utf8(std::string_view value, std::size_t& index,
                               std::uint32_t& codepoint) noexcept {
  const auto first = static_cast<std::uint8_t>(value[index]);
  if (first < 0x80U) {
    codepoint = first;
    ++index;
    return true;
  }

  std::size_t remaining{};
  if ((first & 0xE0U) == 0xC0U) {
    remaining = 1U;
    codepoint = first & 0x1FU;
  } else if ((first & 0xF0U) == 0xE0U) {
    remaining = 2U;
    codepoint = first & 0x0FU;
  } else if ((first & 0xF8U) == 0xF0U) {
    remaining = 3U;
    codepoint = first & 0x07U;
  } else {
    return false;
  }
  if (index + remaining >= value.size()) {
    return false;
  }
  for (std::size_t offset = 1U; offset <= remaining; ++offset) {
    const auto continuation = static_cast<std::uint8_t>(value[index + offset]);
    if ((continuation & 0xC0U) != 0x80U) {
      return false;
    }
    codepoint = (codepoint << 6U) | (continuation & 0x3FU);
  }
  if ((remaining == 1U && codepoint < 0x80U) || (remaining == 2U && codepoint < 0x800U) ||
      (remaining == 3U && codepoint < 0x10000U) || codepoint > 0x10FFFFU ||
      (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
    return false;
  }
  index += remaining + 1U;
  return true;
}

[[nodiscard]] bool unicode_whitespace(std::uint32_t codepoint) noexcept {
  return (codepoint >= 0x0009U && codepoint <= 0x000DU) || codepoint == 0x0020U ||
         codepoint == 0x0085U || codepoint == 0x00A0U || codepoint == 0x1680U ||
         (codepoint >= 0x2000U && codepoint <= 0x200AU) || codepoint == 0x2028U ||
         codepoint == 0x2029U || codepoint == 0x202FU || codepoint == 0x205FU ||
         codepoint == 0x3000U;
}

[[nodiscard]] bool endpoint_name_shape_valid(std::string_view value) noexcept {
  if (value.empty() || value.size() > 255U) {
    return false;
  }
  std::size_t index = 0U;
  std::uint32_t first_codepoint{};
  std::uint32_t last_codepoint{};
  bool first = true;
  while (index < value.size()) {
    std::uint32_t codepoint{};
    if (!decode_utf8(value, index, codepoint) || codepoint <= 0x001FU ||
        (codepoint >= 0x007FU && codepoint <= 0x009FU)) {
      return false;
    }
    if (first) {
      first_codepoint = codepoint;
      first = false;
    }
    last_codepoint = codepoint;
  }
  return !unicode_whitespace(first_codepoint) && !unicode_whitespace(last_codepoint);
}

[[nodiscard]] bool exact_pin_present(const DeploymentBinding& deployment,
                                     std::string_view final_name) noexcept {
  return std::any_of(deployment.generated_endpoint_names().begin(),
                     deployment.generated_endpoint_names().end(),
                     [&](const GeneratedEndpointNamePinRecord& pin) {
                       return pin.final_name == final_name &&
                              pin.algorithm == kGeneratedAlgorithm &&
                              pin.version == kGeneratedAlgorithmVersion;
                     });
}

[[nodiscard]] bool constrained_endpoint_profile(Profile profile) noexcept {
  return profile == Profile::kRobotics || profile == Profile::kRealTime ||
         profile == Profile::kRegulated;
}

[[nodiscard]] DiagnosticSource root_diagnostic_source(const GraphSnapshotData& snapshot) noexcept {
  const GraphRecord* root = find_root_graph(snapshot);
  return DiagnosticSource{
      .graph_generation = snapshot.generation,
      .graph = root == nullptr ? GraphId{} : root->id,
  };
}

[[nodiscard]] DiagnosticSource endpoint_diagnostic_source(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic) noexcept {
  DiagnosticSource source = root_diagnostic_source(snapshot);
  source.vertex = topic.vertex;
  source.port = PortId{topic.graph_port.value};
  const VertexRecord* vertex = find_vertex(snapshot, topic.vertex);
  if (vertex != nullptr) {
    source.graph = vertex->graph;
  }
  return source;
}

void validate_generated_endpoint_pins(const DeploymentBinding& deployment, DiagnosticSource source,
                                      EndpointCatalogLoweringResult& result) {
  const auto pins = deployment.generated_endpoint_names();
  for (std::size_t index = 0U; index < pins.size(); ++index) {
    const auto& pin = pins[index];
    if (!endpoint_name_shape_valid(pin.final_name)) {
      result.errors.push_back(
          EndpointCatalogDiagnostic{"ENDPOINT_NAME_GENERATED_PIN_INVALID", source});
    }
    if (pin.algorithm != kGeneratedAlgorithm || pin.version != kGeneratedAlgorithmVersion) {
      result.errors.push_back(
          EndpointCatalogDiagnostic{"ENDPOINT_NAME_GENERATED_PIN_MISMATCH", source});
    }
    const auto current = pins.begin() + static_cast<std::ptrdiff_t>(index);
    const auto duplicate = std::find_if(pins.begin(), current, [&](const auto& prior) {
      return prior.final_name == pin.final_name;
    });
    if (duplicate != current) {
      result.errors.push_back(EndpointCatalogDiagnostic{
          duplicate->algorithm == pin.algorithm && duplicate->version == pin.version
              ? "ENDPOINT_NAME_GENERATED_PIN_DUPLICATE"
              : "ENDPOINT_NAME_GENERATED_PIN_MISMATCH",
          source});
    }
  }
}

[[nodiscard]] bool supplied_catalog_pins_generated_endpoint(
    const std::optional<ChannelEndpointCatalog>& supplied_catalog, const ExternalTopicRecord& topic,
    std::string_view final_name) {
  if (!supplied_catalog.has_value()) {
    return false;
  }
  const auto* state = ChannelEndpointCatalogAccess::state(*supplied_catalog);
  if (state == nullptr) {
    return false;
  }
  return std::any_of(state->endpoints.begin(),
                     state->endpoints.end(),
                     [&](const ChannelEndpointRecord& candidate) {
                       return candidate.name == final_name && candidate.uri == topic.uri &&
                              candidate.direction == (topic.imported ? TopicDirection::kImport
                                                                     : TopicDirection::kExport) &&
                              candidate.name_origin == EndpointNameOrigin::kGenerated &&
                              candidate.generated_name.has_value() &&
                              candidate.generated_name->algorithm == kGeneratedAlgorithm &&
                              candidate.generated_name->version == kGeneratedAlgorithmVersion;
                     });
}

// Lower endpoint identity, attachment policy, and bounded queue authority together so diagnostics
// and the returned immutable record always describe the same authoring endpoint.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<ChannelEndpointRecord> lower_endpoint_record(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic,
    const DeploymentBinding& deployment, std::optional<Profile> profile,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings,
    EndpointCatalogLoweringResult& result) {
  const DiagnosticSource source = endpoint_diagnostic_source(snapshot, topic);
  const bool generated = topic.generated_name_requested || !topic.endpoint_name;
  std::string final_name =
      generated ? generated_endpoint_name(snapshot, topic) : std::string{*topic.endpoint_name};
  if (!endpoint_name_shape_valid(final_name)) {
    result.errors.push_back(EndpointCatalogDiagnostic{"ENDPOINT_NAME_INVALID", source});
    return std::nullopt;
  }
  if (!generated && final_name.starts_with(kGeneratedNamespace)) {
    result.errors.push_back(EndpointCatalogDiagnostic{"ENDPOINT_NAME_RESERVED_PREFIX", source});
    return std::nullopt;
  }
  if (generated) {
    const bool pinned =
        supplied_catalog_pins_generated_endpoint(supplied_catalog, topic, final_name) ||
        exact_pin_present(deployment, final_name);
    if (!pinned && (!profile.has_value() || constrained_endpoint_profile(*profile))) {
      result.errors.push_back(
          EndpointCatalogDiagnostic{"ENDPOINT_NAME_GENERATED_PIN_REQUIRED", source});
    } else if (!pinned) {
      result.warnings.push_back(
          EndpointCatalogDiagnostic{"ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED", source});
    }
  }

  TopicAttachPolicy attach = topic.attach;
  if (!attach.max_wait.has_value()) {
    if (!profile.has_value() || constrained_endpoint_profile(*profile)) {
      result.errors.push_back(EndpointCatalogDiagnostic{"ENDPOINT_ATTACH_BOUND_REQUIRED", source});
    } else {
      attach.max_wait = kOrdinaryAttachWait;
    }
  }

  const GraphPortId resolved_boundary = resolve_interface_port(snapshot, topic.graph_port);
  const auto boundary_edge =
      std::find_if(snapshot.edges.begin(), snapshot.edges.end(), [&](const EdgeRecord& edge) {
        const GraphPortId endpoint = topic.imported ? edge.source : edge.target;
        return resolved_boundary && resolve_interface_port(snapshot, endpoint) == resolved_boundary;
      });
  const QueuePolicy queue_policy = boundary_edge == snapshot.edges.end()
                                       ? QueuePolicy::kBlocking
                                       : boundary_edge->graph_override.queue_policy.value_or(
                                             boundary_edge->add_flow_options.queue_policy);
  std::uint32_t graph_queue = 1U;
  std::uint32_t graph_burst = 1U;
  if (boundary_edge != snapshot.edges.end()) {
    const PortRecord* target =
        find_port(snapshot, resolve_interface_port(snapshot, boundary_edge->target));
    const PortRecord* source_port =
        find_port(snapshot, resolve_interface_port(snapshot, boundary_edge->source));
    const std::uint32_t target_queue =
        target == nullptr
            ? 1U
            : target->graph_override.queue_depth.value_or(target->setup.queue_depth.value_or(1U));
    graph_queue = boundary_edge->graph_override.queue_depth.value_or(
        boundary_edge->add_flow_options.queue_depth.value_or(target_queue));
    if (const auto* sizing =
            find_resolved_queue_sizing(resolved_queue_sizings, boundary_edge->id)) {
      graph_queue = sizing->effective_logical_depth;
    }
    graph_burst = source_port == nullptr
                      ? 1U
                      : source_port->graph_override.max_emits_per_compute.value_or(
                            source_port->setup.max_emits_per_compute.value_or(1U));
  }
  const ExternalTopicBoundsResolution bounds =
      resolve_external_topic_bounds(topic, graph_queue, graph_burst);
  const std::uint32_t publisher_loans = bounds.publisher_max_loaned_samples;
  const std::uint32_t subscriber_queue = bounds.subscriber_queue_depth;
  const std::uint32_t subscriber_borrows = bounds.subscriber_max_borrowed_samples;
  EndpointBounds endpoint_bounds = topic.bounds;
  if (topic.imported) {
    endpoint_bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = publisher_loans};
  } else {
    endpoint_bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = subscriber_queue,
        .max_borrowed_samples = subscriber_borrows,
    };
  }

  std::optional<TensorBoundaryContractRecord> endpoint_tensor = topic.tensor;
  std::vector<DescriptorTensorTopicFieldRecord> endpoint_tensor_fields = topic.tensor_fields;
  if (!topic.imported && endpoint_tensor && boundary_edge != snapshot.edges.end()) {
    const GraphPortId source_id = resolve_interface_port(snapshot, boundary_edge->source);
    const auto derivation = std::find_if(tensor_byte_span_derivations.begin(),
                                         tensor_byte_span_derivations.end(),
                                         [&](const auto& candidate) {
                                           return candidate.output.value() == source_id.value &&
                                                  !candidate.field_id && !candidate.tensor_index;
                                         });
    if (derivation != tensor_byte_span_derivations.end()) {
      const std::uint64_t resolved_staging_byte_span =
          derivation->resolved_max_staging_byte_span.value_or(derivation->resolved_max_byte_span);
      const bool copy_capable =
          endpoint_tensor->foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
      const bool byte_span_fits =
          endpoint_tensor->max_byte_span &&
          *endpoint_tensor->max_byte_span >= derivation->resolved_max_byte_span;
      const bool staging_span_fits =
          !copy_capable || (endpoint_tensor->max_staging_byte_span &&
                            *endpoint_tensor->max_staging_byte_span >= resolved_staging_byte_span);
      if (!byte_span_fits || !staging_span_fits) {
        result.errors.push_back(
            EndpointCatalogDiagnostic{"PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT", source});
      } else {
        endpoint_tensor->max_byte_span = derivation->resolved_max_byte_span;
        endpoint_tensor->origins.max_byte_span =
            TensorContractValueOrigin::kCompilerDerivedByteSpan;
        if (copy_capable) {
          const bool preserves_authored_staging =
              endpoint_tensor->origins.max_staging_byte_span ==
                  TensorContractValueOrigin::kAuthored &&
              endpoint_tensor->max_staging_byte_span == resolved_staging_byte_span;
          endpoint_tensor->max_staging_byte_span = resolved_staging_byte_span;
          if (!preserves_authored_staging) {
            endpoint_tensor->origins.max_staging_byte_span =
                TensorContractValueOrigin::kCompilerDerivedByteSpan;
          }
        }
      }
    }
  }
  if (!topic.imported && !endpoint_tensor_fields.empty() && boundary_edge != snapshot.edges.end()) {
    const GraphPortId source_id = resolve_interface_port(snapshot, boundary_edge->source);
    for (auto& field : endpoint_tensor_fields) {
      const auto derivation = std::find_if(tensor_byte_span_derivations.begin(),
                                           tensor_byte_span_derivations.end(),
                                           [&](const auto& candidate) {
                                             return candidate.output.value() == source_id.value &&
                                                    candidate.field_id == field.field_id &&
                                                    candidate.tensor_index == field.tensor_index;
                                           });
      if (derivation == tensor_byte_span_derivations.end()) {
        continue;
      }
      const bool copy_capable =
          field.tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
      const std::uint64_t resolved_staging_byte_span =
          derivation->resolved_max_staging_byte_span.value_or(derivation->resolved_max_byte_span);
      const bool byte_span_fits = field.tensor.max_byte_span &&
                                  *field.tensor.max_byte_span >= derivation->resolved_max_byte_span;
      const bool staging_span_fits =
          !copy_capable || (field.tensor.max_staging_byte_span &&
                            *field.tensor.max_staging_byte_span >= resolved_staging_byte_span);
      if (!byte_span_fits || !staging_span_fits) {
        result.errors.push_back(
            EndpointCatalogDiagnostic{"PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT", source});
        continue;
      }
      field.tensor.max_byte_span = derivation->resolved_max_byte_span;
      field.tensor.origins.max_byte_span = TensorContractValueOrigin::kCompilerDerivedByteSpan;
      if (copy_capable) {
        const bool preserves_authored_staging =
            field.tensor.origins.max_staging_byte_span == TensorContractValueOrigin::kAuthored &&
            field.tensor.max_staging_byte_span == resolved_staging_byte_span;
        field.tensor.max_staging_byte_span = resolved_staging_byte_span;
        if (!preserves_authored_staging) {
          field.tensor.origins.max_staging_byte_span =
              TensorContractValueOrigin::kCompilerDerivedByteSpan;
        }
      }
    }
  }

  const std::string service = fmt::format("{}.service", final_name);
  return ChannelEndpointRecord{
      .name = std::move(final_name),
      .uri = topic.uri,
      .direction = topic.imported ? TopicDirection::kImport : TopicDirection::kExport,
      .name_origin = generated ? EndpointNameOrigin::kGenerated : EndpointNameOrigin::kExplicit,
      .generated_name =
          generated ? std::optional<GeneratedEndpointNameRecord>{GeneratedEndpointNameRecord{
                          .algorithm = std::string{kGeneratedAlgorithm},
                          .version = kGeneratedAlgorithmVersion,
                      }}
                    : std::nullopt,
      // The canonical compiler snapshot has already remapped every GraphPortId
      // to the same dense declaration order used by the baseline plan PortId table.
      .boundary_port = PortId{topic.graph_port.value},
      .service = service,
      .queue_policy = queue_policy,
      .service_max_subscriber_depth = subscriber_queue,
      .service_max_subscriber_borrowed = subscriber_borrows,
      .payload_identity = topic.payload_identity,
      .schema_writer_max_serialized_bytes = topic.serialized_payload_capacity_bytes,
      .schema_identity = topic.schema,
      .memory_conversion = topic.memory_conversion,
      .capture_time_clock =
          topic.capture_time ? std::optional<ClockKey>{topic.capture_time->clock} : std::nullopt,
      .clock_health = topic.clock_health,
      .attach = attach,
      .tensor = std::move(endpoint_tensor),
      .tensor_fields = std::move(endpoint_tensor_fields),
      .bounds = std::move(endpoint_bounds),
  };
}

struct LoweredEndpointRecord {
  ChannelEndpointRecord endpoint;
  DiagnosticSource source;
};

void append_duplicate_endpoint_diagnostics(std::span<const LoweredEndpointRecord> endpoints,
                                           EndpointCatalogLoweringResult& result) {
  for (std::size_t index = 1U; index < endpoints.size(); ++index) {
    if (endpoints[index - 1U].endpoint.name == endpoints[index].endpoint.name) {
      result.errors.push_back(
          EndpointCatalogDiagnostic{"ENDPOINT_NAME_DUPLICATE", endpoints[index].source});
    }
  }
}

[[nodiscard]] bool supplied_catalog_matches(
    const ChannelEndpointCatalog& supplied, std::string_view derived_scope,
    std::span<const ChannelEndpointRecord> derived, std::span<const CatalogClockRecord> clocks,
    std::span<const CatalogClockConversionRecord> clock_conversions) {
  const auto* state = ChannelEndpointCatalogAccess::state(supplied);
  if (state == nullptr) {
    return derived.empty();
  }
  std::vector<ChannelEndpointRecord> normalized_input{derived.begin(), derived.end()};
  auto normalized = ChannelEndpointCatalogAccess::normalize(
      std::string{derived_scope},
      std::move(normalized_input),
      std::vector<CatalogClockRecord>{clocks.begin(), clocks.end()},
      std::vector<CatalogClockConversionRecord>{clock_conversions.begin(),
                                                clock_conversions.end()});
  return normalized && normalized->digest() == supplied.digest();
}

[[nodiscard]] std::string catalog_clock_kind(const ClockPlanRecord& clock) {
  if (clock.capability_evidence_id == "holoscan.clock.realtime.v1") {
    return "realtime";
  }
  if (clock.capability_evidence_id == "holoscan.clock.manual.external-progress-required.v1") {
    return "manual";
  }
  if (clock.capability_evidence_id == "holoscan.clock.synthetic.external-progress-required.v1") {
    return "synthetic";
  }
  return "custom";
}

struct EndpointCatalogClockProjection {
  std::vector<CatalogClockRecord> clocks;
  std::vector<CatalogClockConversionRecord> conversions;
};

/// Derive a portable symmetric source interval whose affine image is always
/// representable. The plan does not currently carry deployment-specific clock
/// coordinate bounds, so the catalog records the largest conservative range
/// derivable from the exact scale and offset instead of claiming the complete
/// int64 domain or inventing an external bound.
[[nodiscard]] std::pair<std::int64_t, std::int64_t> conservative_conversion_range(
    const ClockConversionRecord& conversion) noexcept {
  const std::uint64_t offset_magnitude =
      conversion.offset_ns >= 0 ? static_cast<std::uint64_t>(conversion.offset_ns)
                                : static_cast<std::uint64_t>(-(conversion.offset_ns + 1)) + 1U;
  const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (offset_magnitude > maximum || conversion.scale.numerator <= 0 ||
      conversion.scale.denominator == 0U) {
    return {0, 0};
  }

  const std::uint64_t numerator = static_cast<std::uint64_t>(conversion.scale.numerator);
  const std::uint64_t quotient = (maximum - offset_magnitude) / numerator;
  const std::uint64_t safe_magnitude =
      quotient > maximum / conversion.scale.denominator
          ? maximum
          : std::min(maximum, quotient * conversion.scale.denominator);
  const auto admitted_max = static_cast<std::int64_t>(safe_magnitude);
  return {-admitted_max, admitted_max};
}

/// Project only endpoint-relevant domains from the authoritative plan-local
/// clock catalog. The endpoint catalog must never invent an "external" clock
/// merely because authoring named a capture domain.
// Projection validates all endpoint clock joins before publishing the catalog subset.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] EndpointCatalogClockProjection project_endpoint_catalog_clocks(
    const GraphSnapshotData& snapshot, std::span<const ClockPlanRecord> plan_clocks,
    std::span<const ClockConversionRecord> plan_conversions) {
  EndpointCatalogClockProjection result;
  std::vector<ClockId> included;
  // An exported sample carries the publication coordinate Runtime stamped from the root default
  // clock. Projecting that clock makes the wire identity resolvable even when no authored capture
  // coordinate references it.
  const bool exports_any_endpoint =
      std::any_of(snapshot.external_topics.begin(),
                  snapshot.external_topics.end(),
                  [](const auto& topic) noexcept { return !topic.imported; });
  if (exports_any_endpoint) {
    const GraphRecord* root = find_root_graph(snapshot);
    const ClockId publish_clock =
        root == nullptr ? ClockId{} : find_default_clock_plan_id(plan_clocks, root->id);
    if (publish_clock.valid()) {
      included.push_back(publish_clock);
    }
  }
  for (const auto& topic : snapshot.external_topics) {
    if (!topic.capture_time.has_value()) {
      continue;
    }
    const VertexRecord* owner = find_vertex(snapshot, topic.vertex);
    if (owner == nullptr) {
      continue;
    }
    const ClockId id =
        resolve_clock_plan_id(snapshot, plan_clocks, owner->graph, topic.capture_time->clock);
    if (id.valid() && std::find(included.begin(), included.end(), id) == included.end()) {
      included.push_back(id);
    }
  }
  bool added_clock = true;
  while (added_clock) {
    added_clock = false;
    for (const auto& conversion : plan_conversions) {
      if (std::find(included.begin(), included.end(), conversion.from) == included.end()) {
        continue;
      }
      if (std::find(included.begin(), included.end(), conversion.to) == included.end()) {
        included.push_back(conversion.to);
        added_clock = true;
      }
    }
  }
  std::sort(included.begin(), included.end(), [](ClockId lhs, ClockId rhs) {
    return lhs.value < rhs.value;
  });
  result.clocks.reserve(included.size());
  for (const ClockId id : included) {
    const auto found = std::find_if(
        plan_clocks.begin(), plan_clocks.end(), [id](const auto& clock) { return clock.id == id; });
    if (found == plan_clocks.end()) {
      continue;
    }
    result.clocks.push_back(CatalogClockRecord{
        .id = found->id.value,
        .key = found->key,
        .kind = catalog_clock_kind(*found),
        .unit = "nanoseconds",
        .epoch = "plan_coordinate",
        .monotonicity = found->steady ? "nondecreasing" : "unqualified",
        .discipline_kind =
            found->independent_progress
                ? "independent"
                : (found->external_driver_admitted ? "external_driver_admitted" : "unqualified"),
        .discipline_identity = found->capability_evidence_id,
    });
  }
  for (const auto& conversion : plan_conversions) {
    if (std::find(included.begin(), included.end(), conversion.from) == included.end() ||
        std::find(included.begin(), included.end(), conversion.to) == included.end()) {
      continue;
    }
    const auto from = std::find_if(plan_clocks.begin(), plan_clocks.end(), [&](const auto& clock) {
      return clock.id == conversion.from;
    });
    const auto to = std::find_if(plan_clocks.begin(), plan_clocks.end(), [&](const auto& clock) {
      return clock.id == conversion.to;
    });
    if (from == plan_clocks.end() || to == plan_clocks.end()) {
      continue;
    }
    const auto [admitted_min, admitted_max] = conservative_conversion_range(conversion);
    result.conversions.push_back(CatalogClockConversionRecord{
        .from = from->key,
        .to = to->key,
        .scale_numerator = conversion.scale.numerator,
        .scale_denominator = conversion.scale.denominator,
        .offset_ns = conversion.offset_ns,
        .admitted_min = admitted_min,
        .admitted_max = admitted_max,
        .provenance = fmt::format("holoscan.execution-plan.clock-conversion.v1/{}", conversion.id),
    });
  }
  return result;
}

[[nodiscard]] std::string catalog_scope(const GraphSnapshotData& snapshot) {
  const GraphRecord* root = find_root_graph(snapshot);
  return root != nullptr && !root->name.empty() ? root->name : "_root";
}

}  // namespace

bool endpoint_profile_is_constrained(Profile profile) noexcept {
  return constrained_endpoint_profile(profile);
}

DiagnosticSource make_endpoint_diagnostic_source(const GraphSnapshotData& snapshot,
                                                 const ExternalTopicRecord& topic) noexcept {
  return endpoint_diagnostic_source(snapshot, topic);
}

std::string generated_endpoint_name_for_topic(const GraphSnapshotData& snapshot,
                                              const ExternalTopicRecord& topic) {
  return generated_endpoint_name(snapshot, topic);
}

// Preserve the token-specific diagnostic projections as one linear decoder so every fallback
// retains the same raw detail and source attribution.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
EndpointCatalogDiagnostic endpoint_catalog_normalization_diagnostic(const std::string_view message,
                                                                    const DiagnosticSource source) {
  EndpointCatalogDiagnostic result{.source = source};
  const auto detail_separator = message.find('|');
  result.token = std::string{message.substr(0U, detail_separator)};
  if (result.token.empty()) {
    result.token = "ENDPOINT_CATALOG_NORMALIZATION_FAILED";
  }
  if (detail_separator == std::string_view::npos) {
    return result;
  }

  // Producers escape the structural bytes of every dynamic value, so the first marker and the next
  // '|' delimit exactly one field even when an endpoint name contains '|', '=', or ';'. Decoding
  // restores the authored value for projection.
  const auto detail_value = [&](const std::string_view key) -> std::optional<std::string> {
    const std::string marker = "|" + std::string(key) + "=";
    const auto begin = message.find(marker);
    if (begin == std::string_view::npos) {
      return std::nullopt;
    }
    const auto value_begin = begin + marker.size();
    const auto end = message.find('|', value_begin);
    return channel_internal::decode_detail_value(message.substr(value_begin, end - value_begin));
  };
  const auto append_fact =
      [](std::string& target, const std::string_view key, const std::optional<std::string>& value) {
        if (!value) {
          return;
        }
        if (!target.empty()) {
          target.push_back(';');
        }
        target += key;
        target.push_back('=');
        target += *value;
      };
  const auto use_raw_detail = [&]() {
    result.provided = std::string{message.substr(detail_separator + 1U)};
  };

  if (const auto path = detail_value("path")) {
    result.json_pointer = *path;
  }
  if (result.token == "ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED") {
    append_fact(result.required, "resource", detail_value("resource"));
    append_fact(result.required, "limit", detail_value("limit"));
    append_fact(result.provided, "actual", detail_value("actual"));
    if (result.required.empty() || result.provided.empty()) {
      use_raw_detail();
    }
  } else if (result.token == "ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED") {
    result.required = "required_frame<=maximum_frame";
    append_fact(result.required, "maximum_frame", detail_value("maximum_frame"));
    append_fact(result.provided, "endpoint", detail_value("endpoint"));
    append_fact(result.provided, "application_capacity", detail_value("application_capacity"));
    append_fact(result.provided, "framing_overhead", detail_value("framing_overhead"));
    append_fact(result.provided, "required_frame", detail_value("required_frame"));
    if (result.required == "required_frame<=maximum_frame" || result.provided.empty()) {
      use_raw_detail();
    }
  } else if (result.token == "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID") {
    const std::string_view required_marker = "|required=";
    const std::string_view provided_marker = "|provided=";
    const auto required_begin = message.find(required_marker);
    const auto provided_begin = message.find(provided_marker);
    if (required_begin != std::string_view::npos && provided_begin != std::string_view::npos &&
        provided_begin > required_begin) {
      const auto value_begin = required_begin + required_marker.size();
      result.required = std::string{message.substr(value_begin, provided_begin - value_begin)};
      append_fact(result.provided, "endpoint", detail_value("endpoint"));
      if (!result.provided.empty()) {
        result.provided.push_back(';');
      }
      result.provided += message.substr(provided_begin + provided_marker.size());
    } else {
      use_raw_detail();
    }
  } else {
    use_raw_detail();
  }
  return result;
}

EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings) {
  EndpointCatalogLoweringResult result;
  std::vector<LoweredEndpointRecord> lowered_endpoints;
  lowered_endpoints.reserve(snapshot.external_topics.size());
  const DiagnosticSource root_source = root_diagnostic_source(snapshot);
  validate_generated_endpoint_pins(deployment, root_source, result);

  // REQ: COMPILE-007 - final endpoint identity is frozen before Runtime/backend construction.
  for (const auto& topic : snapshot.external_topics) {
    auto endpoint =
        lower_endpoint_record(snapshot,
                              topic,
                              deployment,
                              effective_profile_for_vertex(effective_vertex_profiles, topic.vertex),
                              supplied_catalog,
                              tensor_byte_span_derivations,
                              resolved_queue_sizings,
                              result);
    if (endpoint.has_value()) {
      lowered_endpoints.push_back(LoweredEndpointRecord{
          .endpoint = std::move(*endpoint),
          .source = endpoint_diagnostic_source(snapshot, topic),
      });
    }
  }

  std::stable_sort(
      lowered_endpoints.begin(), lowered_endpoints.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.endpoint.name < rhs.endpoint.name;
      });
  append_duplicate_endpoint_diagnostics(lowered_endpoints, result);
  std::vector<ChannelEndpointRecord> endpoints;
  endpoints.reserve(lowered_endpoints.size());
  std::transform(lowered_endpoints.begin(),
                 lowered_endpoints.end(),
                 std::back_inserter(endpoints),
                 [](LoweredEndpointRecord& lowered) { return std::move(lowered.endpoint); });
  EndpointCatalogClockProjection catalog_clocks =
      project_endpoint_catalog_clocks(snapshot, clocks, clock_conversions);
  const std::string derived_scope = catalog_scope(snapshot);
  if (supplied_catalog.has_value() && !supplied_catalog_matches(*supplied_catalog,
                                                                derived_scope,
                                                                endpoints,
                                                                catalog_clocks.clocks,
                                                                catalog_clocks.conversions)) {
    result.errors.push_back(EndpointCatalogDiagnostic{"ENDPOINT_CATALOG_MISMATCH", root_source});
  }
  canonicalize_plan_lowering_diagnostics(result.errors);
  canonicalize_plan_lowering_diagnostics(result.warnings);
  auto catalog = ChannelEndpointCatalogAccess::normalize(derived_scope,
                                                         std::move(endpoints),
                                                         std::move(catalog_clocks.clocks),
                                                         std::move(catalog_clocks.conversions));
  if (!catalog) {
    auto normalization_diagnostic =
        endpoint_catalog_normalization_diagnostic(catalog.error().message, root_source);
    const bool already_reported =
        std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& diagnostic) {
          return diagnostic.token == normalization_diagnostic.token;
        });
    if (!already_reported) {
      result.errors.push_back(std::move(normalization_diagnostic));
    }
    canonicalize_plan_lowering_diagnostics(result.errors);
  } else {
    // A rejected plan remains non-authorizing, but its fully normalized attempted
    // catalog is retained as immutable inspection evidence.
    result.catalog = std::move(catalog).value();
  }
  return result;
}

EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment, Profile profile,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings) {
  const std::vector<std::optional<Profile>> uniform_profiles(snapshot.vertices.size(), profile);
  return lower_endpoint_catalog(snapshot,
                                deployment,
                                uniform_profiles,
                                supplied_catalog,
                                clocks,
                                clock_conversions,
                                tensor_byte_span_derivations,
                                resolved_queue_sizings);
}

}  // namespace holoscan::detail
