/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/plan_artifact_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/schema/schema_package.hpp>

#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/detail/snapshot_lookup.hpp"
#include "compiler/inline_partition_execution_plan.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "compiler/partition_runner_plan.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "core/clock_binding_resolution.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/source_id_internal.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kPlanGenerationDomain = "holoscan.plan_generation.v1";
constexpr std::string_view kPlanSchema = "holoscan.execution_plan.private.v1";
constexpr std::string_view kGeneratedPrefix = "__holoscan_auto_endpoint/v1/";
constexpr std::string_view kGeneratedNamespace = "__holoscan_auto_endpoint/";
constexpr std::string_view kGeneratedAlgorithm = "holoscan.auto_endpoint_name";
constexpr std::uint32_t kGeneratedAlgorithmVersion = 1U;
constexpr std::chrono::seconds kOrdinaryAttachWait{2};
// Percent-encoding always escapes '%', so this synthetic component cannot
// collide with any encoded user-authored Graph path segment.
constexpr std::string_view kRootPathComponent = "%root";

[[nodiscard]] Error artifact_error(ErrorCode code, std::string_view token) {
  return Error{.code = code, .message = std::string(token)};
}

template <typename Digest>
[[nodiscard]] Digest digest_bytes(std::string_view bytes) noexcept {
  // REQ: COMPILE-002 - digest canonical bytes, never C++ object representation.
  StableHashBuilder hash;
  for (const char byte : bytes) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  Digest result;
  result.value = hash.finish().value;
  return result;
}

[[nodiscard]] bool uri_unreserved(std::uint8_t byte) noexcept {
  return (byte >= static_cast<std::uint8_t>('A') && byte <= static_cast<std::uint8_t>('Z')) ||
         (byte >= static_cast<std::uint8_t>('a') && byte <= static_cast<std::uint8_t>('z')) ||
         (byte >= static_cast<std::uint8_t>('0') && byte <= static_cast<std::uint8_t>('9')) ||
         byte == static_cast<std::uint8_t>('-') || byte == static_cast<std::uint8_t>('.') ||
         byte == static_cast<std::uint8_t>('_') || byte == static_cast<std::uint8_t>('~');
}

[[nodiscard]] std::string percent_encode(std::string_view value) {
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  std::string encoded;
  encoded.reserve(value.size());
  for (const char character : value) {
    const auto byte = static_cast<std::uint8_t>(character);
    if (uri_unreserved(byte)) {
      encoded.push_back(character);
      continue;
    }
    encoded.push_back('%');
    encoded.push_back(kHex[byte >> 4U]);
    encoded.push_back(kHex[byte & 0x0FU]);
  }
  return encoded;
}

[[nodiscard]] std::string encoded_graph_path(const GraphSnapshotData& snapshot,
                                             const ExternalTopicRecord& topic) {
  const VertexRecord* vertex = find_vertex(snapshot, topic.vertex);
  const GraphRecord* graph = vertex == nullptr ? nullptr : find_graph(snapshot, vertex->graph);
  if (graph == nullptr || !graph->path) {
    return std::string{kRootPathComponent};
  }
  std::string result;
  for (const auto& segment : graph->path->segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    result += percent_encode(segment);
  }
  return result.empty() ? std::string{kRootPathComponent} : result;
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

void append_json_string(std::string& output, std::string_view value) {
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  output.push_back('"');
  for (const char character : value) {
    const auto byte = static_cast<std::uint8_t>(character);
    switch (character) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHex[byte >> 4U]);
          output.push_back(kHex[byte & 0x0FU]);
        } else {
          output.push_back(character);
        }
    }
  }
  output.push_back('"');
}

template <typename T>
void append_number(std::string& output, T value) {
  output += std::to_string(value);
}

void append_bool(std::string& output, bool value) {
  output += value ? "true" : "false";
}

template <typename Bytes>
void append_hex(std::string& output, const Bytes& bytes) {
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  output.push_back('"');
  for (const std::byte value : bytes) {
    const auto byte = std::to_integer<std::uint8_t>(value);
    output.push_back(kHex[byte >> 4U]);
    output.push_back(kHex[byte & 0x0FU]);
  }
  output.push_back('"');
}

template <typename Bytes>
void append_sha256(std::string& output, const Bytes& bytes) {
  output += "\"sha256:";
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  for (const std::byte value : bytes) {
    const auto byte = std::to_integer<std::uint8_t>(value);
    output.push_back(kHex[byte >> 4U]);
    output.push_back(kHex[byte & 0x0FU]);
  }
  output.push_back('"');
}

template <typename Bytes>
[[nodiscard]] std::string sha256_text(const Bytes& bytes) {
  std::string quoted;
  quoted.reserve(73U);
  append_sha256(quoted, bytes);
  return quoted.substr(1U, quoted.size() - 2U);
}

[[nodiscard]] constexpr std::string_view profile_token(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
      return "standard";
    case Profile::kRobotics:
      return "robotics";
    case Profile::kRealTime:
      return "real_time";
    case Profile::kRegulated:
      return "regulated";
    case Profile::kResearch:
      return "research";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view scheduling_policy_token(SchedulingPolicy policy) noexcept {
  switch (policy) {
    case SchedulingPolicy::kOther:
      return "other";
    case SchedulingPolicy::kFIFO:
      return "fifo";
    case SchedulingPolicy::kRoundRobin:
      return "round_robin";
    case SchedulingPolicy::kDeadline:
      return "deadline";
  }
  return "invalid";
}

void append_thread_placement(std::string& json, const ThreadPlacement& placement) {
  json += "{\"affinity\":";
  if (placement.affinity) {
    json.push_back('[');
    for (std::size_t index = 0U; index < placement.affinity->cores.size(); ++index) {
      if (index != 0U) {
        json.push_back(',');
      }
      append_number(json, placement.affinity->cores[index]);
    }
    json.push_back(']');
  } else {
    json += "null";
  }
  json += ",\"irq_isolation\":";
  append_number(json, static_cast<std::uint32_t>(placement.irq_isolation));
  json += ",\"probe_failure_policy\":";
  append_number(json, static_cast<std::uint32_t>(placement.probe_failure_policy));
  json += ",\"scheduling\":{\"deadline\":";
  if (placement.scheduling.deadline) {
    const auto& deadline = *placement.scheduling.deadline;
    json += "{\"deadline_ns\":";
    append_number(json, deadline.deadline_ns);
    json += ",\"period_ns\":";
    append_number(json, deadline.period_ns);
    json += ",\"runtime_ns\":";
    append_number(json, deadline.runtime_ns);
    json.push_back('}');
  } else {
    json += "null";
  }
  json += ",\"policy\":";
  append_json_string(json, scheduling_policy_token(placement.scheduling.policy));
  json += ",\"priority\":";
  if (placement.scheduling.priority) {
    append_number(json, *placement.scheduling.priority);
  } else {
    json += "null";
  }
  json += "},\"wake_mode\":";
  append_number(json, static_cast<std::uint32_t>(placement.wake_mode));
  json.push_back('}');
}

void append_operator_ref(std::string& output, OperatorRef ref) {
  output += "{\"id\":";
  append_number(output, ref.id);
  output += ",\"plan_generation\":";
  append_number(output, ref.plan_generation);
  output += ",\"run_generation\":";
  append_number(output, ref.run_generation);
  output.push_back('}');
}

void append_parameter_storage(std::string& output, const ParamStorageTraitsSummary& storage) {
  output += "{\"fixed_object_extent\":";
  append_bool(output, storage.fixed_object_extent);
  output += ",\"has_portable_codec\":";
  append_bool(output, storage.has_portable_codec);
  output += ",\"inline_bytes\":";
  append_number(output, storage.inline_bytes);
  output += ",\"owns_dynamic_storage\":";
  append_bool(output, storage.owns_dynamic_storage);
  output += ",\"supports_bounded_materialization\":";
  append_bool(output, storage.supports_bounded_materialization);
  output.push_back('}');
}

void append_parameter_type_shape(std::string& output, const ParamFieldTypeShape& shape) {
  output += "{\"bit_width\":";
  append_number(output, shape.bit_width);
  output += ",\"duration_period_denominator\":";
  append_number(output, shape.duration_period_denominator);
  output += ",\"duration_period_numerator\":";
  append_number(output, shape.duration_period_numerator);
  output += ",\"element_or_rep_bit_width\":";
  append_number(output, shape.element_or_rep_bit_width);
  output += ",\"element_or_rep_type\":";
  append_number(output, static_cast<std::uint32_t>(shape.element_or_rep_type));
  output += ",\"has_stable_shape\":";
  append_bool(output, shape.has_stable_shape);
  output += ",\"type\":";
  append_number(output, static_cast<std::uint32_t>(shape.type));
  output.push_back('}');
}

void append_lifecycle_capacity_values(std::string& output,
                                      const LifecycleTransactionCapacityValues& values) {
  output += "{\"max_audit_records\":";
  append_number(output, values.max_audit_records);
  output += ",\"max_candidate_generations\":";
  append_number(output, values.max_candidate_generations);
  output += ",\"max_events\":";
  append_number(output, values.max_events);
  output += ",\"max_fenced_operators\":";
  append_number(output, values.max_fenced_operators);
  output += ",\"max_pending_requests\":";
  append_number(output, values.max_pending_requests);
  output += ",\"max_pending_transactions\":";
  append_number(output, values.max_pending_transactions);
  output += ",\"max_quiesce_records\":";
  append_number(output, values.max_quiesce_records);
  output += ",\"max_receipts\":";
  append_number(output, values.max_receipts);
  output += ",\"max_rollback_records\":";
  append_number(output, values.max_rollback_records);
  output.push_back('}');
}

void append_clock_plan_records(std::string& output, std::span<const ClockPlanRecord> records) {
  output.push_back('[');
  bool first = true;
  for (const ClockPlanRecord& record : records) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    output += "{\"graph\":";
    append_number(output, record.graph.value);
    output += ",\"id\":";
    append_number(output, record.id.value);
    output += ",\"is_default\":";
    append_bool(output, record.is_default);
    output += ",\"can_read_now\":";
    append_bool(output, record.can_read_now);
    output += ",\"can_wait_until\":";
    append_bool(output, record.can_wait_until);
    output += ",\"steady\":";
    append_bool(output, record.steady);
    output += ",\"independent_progress\":";
    append_bool(output, record.independent_progress);
    output += ",\"finite_deadline\":";
    append_bool(output, record.finite_deadline);
    output += ",\"external_driver_admitted\":";
    append_bool(output, record.external_driver_admitted);
    output += ",\"capability_evidence_id\":";
    append_json_string(output, record.capability_evidence_id);
    output += ",\"key\":";
    append_json_string(output, record.key.name());
    output += ",\"shared\":";
    append_bool(output, record.shared);
    output.push_back('}');
  }
  output.push_back(']');
}

template <typename Records, typename AppendRecord>
void append_array(std::string& output, const Records& records, AppendRecord append_record) {
  output.push_back('[');
  bool first = true;
  for (const auto& record : records) {
    if (!first)
      output.push_back(',');
    first = false;
    append_record(output, record);
  }
  output.push_back(']');
}

void append_payload_identity(std::string& output, const PayloadIdentityRecord& identity) {
  output += "{\"kind\":";
  append_number(output, static_cast<std::uint32_t>(identity.kind));
  output += ",\"payload_kind\":";
  append_number(output, static_cast<std::uint32_t>(identity.payload_kind));
  output += ",\"pod\":";
  if (identity.pod) {
    output += "{\"alignment\":";
    append_number(output, identity.pod->object_alignment);
    output += ",\"fields\":";
    append_array(output, identity.pod->fields, [](std::string& json, const auto& field) {
      json += "{\"count\":";
      append_number(json, field.element_count);
      json += ",\"name\":";
      append_json_string(json, field.name);
      json += ",\"offset\":";
      append_number(json, field.byte_offset);
      json += ",\"scalar_kind\":";
      append_number(json, static_cast<std::uint32_t>(field.scalar_kind));
      json.push_back('}');
    });
    output += ",\"name\":";
    append_json_string(output, identity.pod->canonical_name);
    output += ",\"size\":";
    append_number(output, identity.pod->object_size);
    output += ",\"version\":";
    append_number(output, identity.pod->version);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"scalar_kind\":";
  if (identity.scalar) {
    append_number(output, static_cast<std::uint32_t>(identity.scalar->kind));
  } else {
    output += "null";
  }
  output += ",\"schema\":";
  if (identity.schema) {
    output += "{\"digest\":";
    append_hex(output, identity.schema->schema_sha256);
    output += ",\"id\":";
    append_hex(output, identity.schema->id.value);
    output += ",\"name\":";
    append_json_string(output, identity.schema->canonical_name);
    output += ",\"version\":";
    append_number(output, identity.schema->version);
    output.push_back('}');
  } else {
    output += "null";
  }
  output.push_back('}');
}

void append_schema_package_provenance(std::string& output,
                                      std::span<const SchemaPackagePlanRecord> records,
                                      bool include_provenance) {
  if (!include_provenance) {
    return;
  }
  output += ",\"schema_packages\":";
  append_array(output, records, [](std::string& json, const SchemaPackagePlanRecord& record) {
    append_package_plan_record(json, record);
  });
}

void append_optional_tensor_layout(std::string& output,
                                   const std::optional<TensorPortLayout>& layout) {
  if (!layout) {
    output += "null";
    return;
  }
  output += "{\"dtype\":";
  if (layout->dtype) {
    output += "{\"bits\":";
    append_number(output, layout->dtype->bits);
    output += ",\"code\":";
    append_number(output, layout->dtype->code);
    output += ",\"lanes\":";
    append_number(output, layout->dtype->lanes);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"memory_kind\":";
  if (layout->memory_kind) {
    append_number(output, static_cast<std::uint32_t>(*layout->memory_kind));
  } else {
    output += "null";
  }
  output += ",\"rank\":";
  if (layout->rank) {
    append_number(output, *layout->rank);
  } else {
    output += "null";
  }
  output.push_back('}');
}

void append_tensor_allocation_bounds(std::string& output, const TensorAllocationBounds& bounds) {
  output += "{\"max_byte_span\":";
  append_number(output, bounds.max_byte_span);
  output += ",\"max_rank\":";
  append_number(output, bounds.max_rank);
  output.push_back('}');
}

void append_logical_tensor_requirements(std::string& output,
                                        const LogicalTensorRequirementsRecord& logical) {
  output += "{\"axes\":";
  append_array(output, logical.axes, [](std::string& json, const TensorAxisBounds& axis) {
    json += "{\"max_abs_stride_bytes\":";
    append_number(json, axis.max_abs_stride_bytes);
    json += ",\"max_extent\":";
    append_number(json, axis.max_extent);
    json.push_back('}');
  });
  output += ",\"dtype\":";
  if (logical.dtype) {
    output += "{\"bits\":";
    append_number(output, logical.dtype->bits);
    output += ",\"code\":";
    append_number(output, logical.dtype->code);
    output += ",\"lanes\":";
    append_number(output, logical.dtype->lanes);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"foreign_emit_policy\":";
  append_number(output, static_cast<std::uint32_t>(logical.foreign_emit_policy));
  output += ",\"max_byte_span\":";
  if (logical.max_byte_span) {
    append_number(output, *logical.max_byte_span);
  } else {
    output += "null";
  }
  output += ",\"max_inline_bytes\":";
  if (logical.max_inline_bytes) {
    append_number(output, *logical.max_inline_bytes);
  } else {
    output += "null";
  }
  output += ",\"max_rank\":";
  if (logical.max_rank) {
    append_number(output, *logical.max_rank);
  } else {
    output += "null";
  }
  output += ",\"memory_kind\":";
  if (logical.memory_kind) {
    append_number(output, static_cast<std::uint32_t>(*logical.memory_kind));
  } else {
    output += "null";
  }
  output += ",\"rank\":";
  if (logical.rank) {
    append_number(output, *logical.rank);
  } else {
    output += "null";
  }
  output += ",\"strides\":";
  append_number(output, static_cast<std::uint32_t>(logical.strides));
  output.push_back('}');
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

[[nodiscard]] std::optional<Profile> profile_for_vertex(
    std::span<const std::optional<Profile>> effective_vertex_profiles, VertexId vertex) noexcept {
  if (vertex.value == 0U || vertex.value > effective_vertex_profiles.size()) {
    return std::nullopt;
  }
  return effective_vertex_profiles[static_cast<std::size_t>(vertex.value - 1U)];
}

[[nodiscard]] ClockId find_clock_id(std::span<const ClockPlanRecord> clocks, GraphId graph,
                                    const ClockKey& key) noexcept {
  const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.graph == graph && clock.key == key;
  });
  return found == clocks.end() ? ClockId{} : found->id;
}

[[nodiscard]] ClockId find_default_clock_id(std::span<const ClockPlanRecord> clocks,
                                            GraphId graph) noexcept {
  const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.graph == graph && clock.is_default;
  });
  return found == clocks.end() ? ClockId{} : found->id;
}

[[nodiscard]] ClockId resolve_clock_id(const GraphSnapshotData& snapshot,
                                       std::span<const ClockPlanRecord> clocks, GraphId graph,
                                       const ClockKey& key) noexcept {
  const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                  snapshot.clocks,
                                                                  snapshot.clock_imports,
                                                                  snapshot.clock_bindings,
                                                                  graph,
                                                                  key);
  return resolved.resolved()
             ? find_clock_id(clocks, resolved.declaration->graph, resolved.declaration->key)
             : ClockId{};
}

[[nodiscard]] constexpr std::string_view readiness_kind_token(ReadinessSourceKind kind) noexcept {
  switch (kind) {
    case ReadinessSourceKind::kReadable:
      return "readable";
    case ReadinessSourceKind::kNotification:
      return "notification";
    case ReadinessSourceKind::kDoorbell:
      return "doorbell";
    case ReadinessSourceKind::kSpin:
      return "spin";
  }
  return "invalid";
}

struct CanonicalReadinessDeclaration {
  ReadinessSourceDeclarationRecord declaration;
  GraphId owner_graph{};
  std::vector<std::string> graph_path;
  std::string owner_name;
  std::string owner_path;
};

[[nodiscard]] expected<CanonicalReadinessDeclaration, Error> canonical_readiness_declaration(
    const GraphSnapshotData& snapshot, const ReadinessSourceDeclarationRecord& declaration) {
  const VertexRecord* owner = find_vertex(snapshot, declaration.owner);
  if (owner == nullptr) {
    return make_unexpected(artifact_error(ErrorCode::kNotFound, "READINESS_OWNER_UNDECLARED"));
  }
  const GraphRecord* graph = find_graph(snapshot, owner->graph);
  if (graph == nullptr) {
    return make_unexpected(artifact_error(ErrorCode::kNotFound, "READINESS_OWNER_UNDECLARED"));
  }

  CanonicalReadinessDeclaration result{
      .declaration = declaration,
      .owner_graph = owner->graph,
      .owner_name = owner->name,
  };
  result.owner_path = graph->path.has_value() ? std::string{} : std::string{kRootPathComponent};
  if (graph->path.has_value()) {
    result.graph_path.assign(graph->path->segments().begin(), graph->path->segments().end());
    for (const auto& segment : result.graph_path) {
      if (!result.owner_path.empty()) {
        result.owner_path.push_back('/');
      }
      result.owner_path += percent_encode(segment);
    }
  }
  result.owner_path += "/operator/";
  result.owner_path += percent_encode(owner->name);
  return result;
}

[[nodiscard]] DiagnosticSource root_diagnostic_source(const GraphSnapshotData& snapshot) noexcept {
  const auto root = std::find_if(snapshot.graphs.begin(),
                                 snapshot.graphs.end(),
                                 [](const auto& graph) { return !graph.parent.has_value(); });
  return DiagnosticSource{
      .graph_generation = snapshot.generation,
      .graph = root == snapshot.graphs.end() ? GraphId{} : root->id,
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

[[nodiscard]] DiagnosticSource endpoint_diagnostic_source_for_name(
    const GraphSnapshotData& snapshot, std::string_view endpoint_name) {
  const auto found = std::find_if(
      snapshot.external_topics.begin(), snapshot.external_topics.end(), [&](const auto& topic) {
        const bool generated = topic.generated_name_requested || !topic.endpoint_name;
        return generated ? generated_endpoint_name(snapshot, topic) == endpoint_name
                         : *topic.endpoint_name == endpoint_name;
      });
  return found == snapshot.external_topics.end() ? root_diagnostic_source(snapshot)
                                                 : endpoint_diagnostic_source(snapshot, *found);
}

void canonicalize_endpoint_diagnostics(std::vector<EndpointCatalogDiagnostic>& diagnostics) {
  const auto key = [](const EndpointCatalogDiagnostic& diagnostic) {
    return std::tuple{diagnostic.token,
                      diagnostic.source.graph_generation,
                      diagnostic.source.graph.value,
                      diagnostic.source.vertex.value,
                      diagnostic.source.port.value(),
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided};
  };
  std::sort(diagnostics.begin(), diagnostics.end(), [&](const auto& lhs, const auto& rhs) {
    return key(lhs) < key(rhs);
  });
  diagnostics.erase(
      std::unique(diagnostics.begin(),
                  diagnostics.end(),
                  [&](const auto& lhs, const auto& rhs) { return key(lhs) == key(rhs); }),
      diagnostics.end());
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
    graph_burst = source_port == nullptr
                      ? 1U
                      : source_port->graph_override.max_emits_per_compute.value_or(
                            source_port->setup.max_emits_per_compute.value_or(1U));
  }
  const std::uint32_t publisher_loans =
      topic.imported && topic.bounds.publisher && topic.bounds.publisher->max_loaned_samples
          ? *topic.bounds.publisher->max_loaned_samples
          : graph_burst;
  const std::uint32_t subscriber_queue =
      !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->queue_depth
          ? *topic.bounds.subscriber->queue_depth
          : graph_queue;
  const std::uint32_t subscriber_borrows =
      !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->max_borrowed_samples
          ? *topic.bounds.subscriber->max_borrowed_samples
          : subscriber_queue;
  EndpointBounds endpoint_bounds = topic.bounds;
  if (topic.imported) {
    endpoint_bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = publisher_loans};
  } else {
    endpoint_bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = subscriber_queue,
        .max_borrowed_samples = subscriber_borrows,
    };
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
      // to the same dense declaration order used by the EA1 plan PortId table.
      .boundary_port = PortId{topic.graph_port.value},
      .service = service,
      .queue_policy = queue_policy,
      .service_max_subscriber_depth = subscriber_queue,
      .service_max_subscriber_borrowed = subscriber_borrows,
      .payload_identity = topic.payload_identity,
      .serialized_payload_capacity_bytes = topic.serialized_payload_capacity_bytes,
      .schema_identity = topic.schema,
      .memory_conversion = topic.memory_conversion,
      .capture_time_clock =
          topic.capture_time ? std::optional<ClockKey>{topic.capture_time->clock} : std::nullopt,
      .clock_health = topic.clock_health,
      .attach = attach,
      .tensor = topic.tensor,
      .bounds = std::move(endpoint_bounds),
  };
}

void append_duplicate_endpoint_diagnostics(const GraphSnapshotData& snapshot,
                                           std::span<const ChannelEndpointRecord> endpoints,
                                           EndpointCatalogLoweringResult& result) {
  for (std::size_t index = 1U; index < endpoints.size(); ++index) {
    if (endpoints[index - 1U].name == endpoints[index].name) {
      result.errors.push_back(EndpointCatalogDiagnostic{
          "ENDPOINT_NAME_DUPLICATE",
          endpoint_diagnostic_source_for_name(snapshot, endpoints[index].name)});
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
  for (const auto& topic : snapshot.external_topics) {
    if (!topic.capture_time.has_value()) {
      continue;
    }
    const VertexRecord* owner = find_vertex(snapshot, topic.vertex);
    if (owner == nullptr) {
      continue;
    }
    const ClockId id =
        resolve_clock_id(snapshot, plan_clocks, owner->graph, topic.capture_time->clock);
    if (id.valid() && std::find(included.begin(), included.end(), id) == included.end()) {
      included.push_back(id);
    }
  }
  for (const auto& conversion : plan_conversions) {
    if (std::find(included.begin(), included.end(), conversion.from) == included.end()) {
      continue;
    }
    if (std::find(included.begin(), included.end(), conversion.to) == included.end()) {
      included.push_back(conversion.to);
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
  const auto root = std::find_if(snapshot.graphs.begin(),
                                 snapshot.graphs.end(),
                                 [](const auto& graph) { return !graph.parent.has_value(); });
  return root != snapshot.graphs.end() && !root->name.empty() ? root->name : "_root";
}

}  // namespace

PlanGenerationId derive_plan_generation_id(const ExecutionPlanState& canonical_state) noexcept {
  StableHashBuilder hash;
  hash.add(kPlanGenerationDomain);
  for (const std::byte byte : canonical_state.digest.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  hash.add_canonical_size(canonical_state.parameter_generation_zero_operators.size());
  for (const auto& record : canonical_state.parameter_generation_zero_operators) {
    hash.add(record.owner.id);
    hash.add(record.baseline_origin);
    hash.add(record.value_digest.disposition);
    for (const std::byte byte : record.value_digest.digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }
  hash.add_canonical_size(canonical_state.parameter_generation_zero_fields.size());
  for (const auto& record : canonical_state.parameter_generation_zero_fields) {
    hash.add(record.instance);
    hash.add(record.field);
    for (const std::byte byte : record.full_id.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(record.origin);
    hash.add(record.value_digest.disposition);
    for (const std::byte byte : record.value_digest.digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(record.config_source.has_value());
    if (record.config_source) {
      const auto& source = *record.config_source;
      hash.add(source.document.semantic_version);
      for (const std::byte byte : source.document.semantic_digest) {
        hash.add_byte(std::to_integer<std::uint8_t>(byte));
      }
      hash.add(source.document.has_raw_source);
      hash.add(source.document.raw_algorithm_version);
      for (const std::byte byte : source.document.raw_digest) {
        hash.add_byte(std::to_integer<std::uint8_t>(byte));
      }
      hash.add(source.document.raw_byte_size);
      hash.add(source.source_path);
      hash.add(source.source_ordinal);
      hash.add(source.target_operator_path);
      hash.add(source.target_schema_id);
      hash.add(source.target_schema_version);
      hash.add(source.target_field_id);
      for (const std::byte byte : source.target_full_id.value) {
        hash.add_byte(std::to_integer<std::uint8_t>(byte));
      }
      hash.add(source.origin);
    }
  }
  const auto digest = hash.finish();
  std::uint64_t generation{};
  for (std::size_t index = 0U; index < sizeof(generation); ++index) {
    generation = (generation << 8U) | std::to_integer<std::uint8_t>(digest.value[index]);
  }
  return generation == 0U ? 1U : generation;
}

// Rebind every plan-local identity family together to preserve generation consistency.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void rebind_plan_generation(ExecutionPlanState& state, PlanGenerationId plan_generation) {
  const auto rebind_operator = [plan_generation](OperatorRef& reference) {
    if (reference.id != 0U) {
      reference.plan_generation = plan_generation;
    }
  };
  const auto rebind_owner_records = [&](auto& records) {
    for (auto& record : records) {
      rebind_operator(record.owner);
    }
  };
  const auto plan_local_id = [plan_generation](std::string_view prefix, std::uint64_t local) {
    return fmt::format("{}.{}.{}", prefix, plan_generation, local);
  };
  const auto allocator_id = [plan_generation](ProcessGroupId process_group, DeviceId device) {
    return device.valid() ? fmt::format("tensor-allocator.{}.process.{}.device.{}",
                                        plan_generation,
                                        process_group.value,
                                        device.value)
                          : fmt::format("tensor-allocator.{}.process.{}.device.host",
                                        plan_generation,
                                        process_group.value);
  };
  std::vector<std::pair<std::string, std::string>> rebound_logical_ids;
  const auto rebind_logical_id = [&](std::string& identity, std::string replacement) {
    rebound_logical_ids.emplace_back(identity, replacement);
    identity = std::move(replacement);
  };

  state.plan_generation = plan_generation;
  for (auto& record : state.clock_conversions) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.endpoint_clock_health) {
    record.plan_generation = plan_generation;
    if (record.conversion_generation != 0U) {
      record.conversion_generation = plan_generation;
    }
  }
  for (auto& record : state.readiness) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.readable_triggers) {
    rebind_operator(record.owner);
    record.plan_generation = plan_generation;
  }
  std::vector<std::pair<std::string, std::string>> rebound_process_local_services;
  rebound_process_local_services.reserve(state.process_local_channel_services.size());
  for (auto& record : state.process_local_channel_services) {
    std::string replacement = fmt::format("process-local-channel.{}.source.{}.policy.{}",
                                          plan_generation,
                                          record.source.value(),
                                          static_cast<std::uint32_t>(record.queue_policy));
    rebound_process_local_services.emplace_back(record.service_id, replacement);
    record.service_id = std::move(replacement);
  }
  const auto rebind_process_local_service = [&](std::string& service_id) {
    const auto found =
        std::find_if(rebound_process_local_services.begin(),
                     rebound_process_local_services.end(),
                     [&](const auto& identity) { return identity.first == service_id; });
    if (found != rebound_process_local_services.end()) {
      service_id = found->second;
    }
  };
  for (auto& record : state.process_local_channel_publishers) {
    rebind_process_local_service(record.service_id);
  }
  for (auto& record : state.process_local_channel_subscribers) {
    rebind_process_local_service(record.service_id);
  }
  std::vector<std::pair<std::string, std::string>> rebound_cross_process_services;
  rebound_cross_process_services.reserve(state.cross_process_channel_services.size());
  for (auto& record : state.cross_process_channel_services) {
    std::string replacement = fmt::format("cross-process-channel.{}.source.{}.policy.{}.anchor.{}",
                                          plan_generation,
                                          record.source.value(),
                                          static_cast<std::uint32_t>(record.queue_policy),
                                          record.edge.value);
    rebound_cross_process_services.emplace_back(record.service_id, replacement);
    record.service_id = std::move(replacement);
  }
  const auto rebind_cross_process_service = [&](std::string& service_id) {
    const auto found =
        std::find_if(rebound_cross_process_services.begin(),
                     rebound_cross_process_services.end(),
                     [&](const auto& identity) { return identity.first == service_id; });
    if (found != rebound_cross_process_services.end()) {
      service_id = found->second;
    }
  };
  for (auto& record : state.cross_process_channel_publishers) {
    rebind_cross_process_service(record.service_id);
  }
  for (auto& record : state.cross_process_channel_subscribers) {
    rebind_cross_process_service(record.service_id);
  }
  for (auto& record : state.generic_output_loans) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id, plan_local_id("generic-output-pool", record.output.value()));
  }
  for (auto& record : state.tensor_allocations) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id, plan_local_id("tensor-output-pool", record.output.value()));
    record.allocator_root_id = allocator_id(record.allocator_process_group, record.device);
  }
  for (auto& record : state.tensor_field_allocations) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id,
                      fmt::format("tensor-field-output-pool.{}.{}.field.{}",
                                  plan_generation,
                                  record.output.value(),
                                  record.field_id));
    record.allocator_root_id = allocator_id(record.allocator_process_group, record.device);
  }
  for (auto& record : state.tensor_interface_pools) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id, plan_local_id("tensor-interface-pool", record.topic.value));
    record.allocator_root_id = allocator_id(record.allocator_process_group, record.device);
    record.exporter_subpool_id = fmt::format("{}.exporter-generation", record.pool_id);
  }
  for (auto& record : state.tensor_device_resources) {
    record.plan_generation = plan_generation;
    record.allocator_root_id = allocator_id(record.process_group, record.device);
    rebind_logical_id(record.stream_id,
                      fmt::format("tensor-stream.{}.process.{}.device.{}",
                                  plan_generation,
                                  record.process_group.value,
                                  record.device.value));
  }
  for (auto& record : state.process_local_channel_services) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == record.tensor_pool_id;
        });
    if (found != rebound_logical_ids.end()) {
      record.tensor_pool_id = found->second;
    }
    for (auto& field : record.tensor_fields) {
      const auto field_pool = std::find_if(
          rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
            return identity.first == field.pool_id;
          });
      if (field_pool != rebound_logical_ids.end()) {
        field.pool_id = field_pool->second;
      }
    }
  }
  for (auto& record : state.cross_process_channel_services) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == record.tensor_pool_id;
        });
    if (found != rebound_logical_ids.end()) {
      record.tensor_pool_id = found->second;
    }
    for (auto& field : record.tensor_fields) {
      const auto field_pool = std::find_if(
          rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
            return identity.first == field.pool_id;
          });
      if (field_pool != rebound_logical_ids.end()) {
        field.pool_id = field_pool->second;
      }
    }
  }
  for (auto& record : state.tensor_inline_records) {
    rebind_logical_id(record.fixed_record_bucket_id,
                      plan_local_id("tensor-inline-record-bucket", record.topic.value));
  }
  for (auto& record : state.imported_generation_capacities) {
    rebind_logical_id(record.runtime_scope_id, plan_local_id("runtime-scope", 1U));
  }
  for (auto& contributor : state.pool_capacity_contributors) {
    for (const auto& [prior, rebound] : rebound_logical_ids) {
      if (contributor.pool_id != prior) {
        continue;
      }
      if (contributor.contributor_id.starts_with(prior)) {
        contributor.contributor_id = rebound + contributor.contributor_id.substr(prior.size());
      }
      contributor.pool_id = rebound;
      break;
    }
  }
  const auto rebind_manifest_object = [&](std::string& object_id) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == object_id;
        });
    if (found != rebound_logical_ids.end()) {
      object_id = found->second;
    }
  };
  for (auto& capacity : state.profile_manifest.derived_capacities) {
    rebind_manifest_object(capacity.object_id);
  }
  for (auto& obligation : state.profile_manifest.materialization_obligations) {
    rebind_manifest_object(obligation.object_id);
  }

  rebind_owner_records(state.temporal_triggers);
  rebind_owner_records(state.aligned_triggers);
  rebind_owner_records(state.clock_triggers);
  rebind_owner_records(state.deadlines);
  rebind_owner_records(state.missing_aggregates);
  rebind_owner_records(state.window_triggers);
  rebind_owner_records(state.bracketed_windows);
  rebind_owner_records(state.parameter_derived_terms);
  for (auto& record : state.predicate_triggers) {
    rebind_operator(record.owner);
    record.graph_generation = plan_generation;
    record.state_generation = plan_generation;
  }
  for (auto& record : state.predicate_callables) {
    record.evidence_validity_epoch = plan_generation;
  }

  for (auto& record : state.lifecycle_checkpoints) {
    rebind_operator(record.operator_instance);
  }
  for (auto& record : state.lifecycle_dependencies) {
    rebind_operator(record.before);
    rebind_operator(record.after);
  }
  for (auto& record : state.lifecycle_restart_permissions) {
    rebind_operator(record.requester);
    rebind_operator(record.target);
  }
  for (auto& record : state.lifecycle_restart_targets) {
    rebind_operator(record.target);
  }
  for (auto& record : state.lifecycle_restart_closures) {
    rebind_operator(record.target);
    for (auto& member : record.closure) {
      rebind_operator(member);
    }
  }
  rebind_owner_records(state.lifecycle_mutable_fields);
  rebind_owner_records(state.operator_parameters);
  rebind_owner_records(state.parameter_fields);
  rebind_owner_records(state.parameter_generation_zero_operators);
  state.parameter_catalog =
      rebind_parameter_catalog_plan_generation(state.parameter_catalog, plan_generation);
}

namespace {

[[nodiscard]] expected<void, Error> validate_clock_binding_sources(
    const GraphSnapshotData& snapshot) {
  for (const auto& binding : snapshot.clock_bindings) {
    const auto imported = std::find_if(
        snapshot.clock_imports.begin(), snapshot.clock_imports.end(), [&](const auto& row) {
          return row.graph == binding.child_graph && row.key == binding.child_clock;
        });
    if (imported == snapshot.clock_imports.end()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_default_clock_inventory(
    const GraphSnapshotData& snapshot) {
  std::vector<GraphId> defaults;
  defaults.reserve(snapshot.clocks.size() + snapshot.clock_imports.size());
  for (const auto& clock : snapshot.clocks) {
    if (clock.is_default) {
      defaults.push_back(clock.graph);
    }
  }
  for (const auto& clock_import : snapshot.clock_imports) {
    if (clock_import.is_default) {
      defaults.push_back(clock_import.graph);
    }
  }
  std::ranges::sort(defaults, {}, &GraphId::value);
  if (std::ranges::adjacent_find(defaults) != defaults.end()) {
    return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_DEFAULT_DUPLICATE"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_import_declarations(
    const GraphSnapshotData& snapshot) {
  for (std::size_t index = 0U; index < snapshot.clock_imports.size(); ++index) {
    const auto& clock_import = snapshot.clock_imports[index];
    if (!clock_import.graph || clock_import.key.is_default() || clock_import.key.name().empty()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_DECLARATION_INVALID"));
    }
    const auto duplicate = std::find_if(
        std::next(snapshot.clock_imports.begin(), static_cast<std::ptrdiff_t>(index + 1U)),
        snapshot.clock_imports.end(),
        [&](const auto& candidate) {
          return candidate.graph == clock_import.graph && candidate.key == clock_import.key;
        });
    const bool concrete_collision =
        std::any_of(snapshot.clocks.begin(), snapshot.clocks.end(), [&](const auto& candidate) {
          return candidate.graph == clock_import.graph && candidate.key == clock_import.key;
        });
    if (duplicate != snapshot.clock_imports.end() || concrete_collision) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_IMPORT_DUPLICATE"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_import_bindings(
    const GraphSnapshotData& snapshot) {
  // Import identity is more specific than default selection. Validate it first so a corrupted
  // record that violates both contracts retains the exact import/collision diagnostic.
  if (auto declarations = validate_clock_import_declarations(snapshot); !declarations) {
    return declarations;
  }
  if (auto defaults = validate_default_clock_inventory(snapshot); !defaults) {
    return defaults;
  }
  for (const auto& clock_import : snapshot.clock_imports) {
    const auto binding_count = static_cast<std::size_t>(std::count_if(
        snapshot.clock_bindings.begin(), snapshot.clock_bindings.end(), [&](const auto& candidate) {
          return candidate.child_graph == clock_import.graph &&
                 candidate.child_clock == clock_import.key;
        }));
    if (binding_count != 1U) {
      return make_unexpected(artifact_error(
          binding_count == 0U ? ErrorCode::kNotFound : ErrorCode::kInvalidArgument,
          binding_count == 0U ? "CLOCK_IMPORT_UNBOUND" : "CLOCK_IMPORT_BINDING_INVALID"));
    }
    const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                    snapshot.clocks,
                                                                    snapshot.clock_imports,
                                                                    snapshot.clock_bindings,
                                                                    clock_import.graph,
                                                                    clock_import.key);
    if (resolved.status == AuthoredClockResolutionStatus::kImportUnbound) {
      return make_unexpected(artifact_error(ErrorCode::kNotFound, "CLOCK_IMPORT_UNBOUND"));
    }
    if (!resolved.resolved()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
  }
  return validate_clock_binding_sources(snapshot);
}

[[nodiscard]] expected<void, Error> append_clock_declarations(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clocks.reserve(snapshot.clocks.size());
  for (std::size_t index = 0U; index < snapshot.clocks.size(); ++index) {
    const auto& declaration = snapshot.clocks[index];
    const auto capability = std::find_if(snapshot.clock_capabilities.begin(),
                                         snapshot.clock_capabilities.end(),
                                         [&](const auto& candidate) {
                                           return candidate.graph == declaration.graph &&
                                                  candidate.key == declaration.key;
                                         });
    if (declaration.key.is_default() || declaration.key.name().empty()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_DECLARATION_INVALID"));
    }
    if (find_clock_id(result.clocks, declaration.graph, declaration.key).valid()) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_CATALOG_DUPLICATE"));
    }
    if (declaration.is_default && find_default_clock_id(result.clocks, declaration.graph).valid()) {
      return make_unexpected(artifact_error(ErrorCode::kDuplicateName, "CLOCK_DEFAULT_DUPLICATE"));
    }
    result.clocks.push_back(ClockPlanRecord{
        .id = ClockId{static_cast<std::uint64_t>(index + 1U)},
        .graph = declaration.graph,
        .key = declaration.key,
        .shared = declaration.shared,
        .is_default = declaration.is_default,
        .can_read_now = capability != snapshot.clock_capabilities.end() && capability->can_read_now,
        .can_wait_until =
            capability != snapshot.clock_capabilities.end() && capability->can_wait_until,
        .steady = capability != snapshot.clock_capabilities.end() && capability->steady,
        .independent_progress =
            capability != snapshot.clock_capabilities.end() && capability->independent_progress,
        .finite_deadline =
            capability != snapshot.clock_capabilities.end() && capability->finite_deadline,
        .external_driver_admitted =
            capability != snapshot.clock_capabilities.end() && capability->external_driver_admitted,
        .capability_evidence_id = capability == snapshot.clock_capabilities.end()
                                      ? "holoscan.clock.capability.missing.v1"
                                      : capability->evidence_id,
    });
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_clock_import_resolutions(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clock_import_resolutions.reserve(snapshot.clock_imports.size());
  for (const auto& clock_import : snapshot.clock_imports) {
    const auto binding = std::find_if(
        snapshot.clock_bindings.begin(), snapshot.clock_bindings.end(), [&](const auto& candidate) {
          return candidate.child_graph == clock_import.graph &&
                 candidate.child_clock == clock_import.key;
        });
    const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                    snapshot.clocks,
                                                                    snapshot.clock_imports,
                                                                    snapshot.clock_bindings,
                                                                    clock_import.graph,
                                                                    clock_import.key);
    if (binding == snapshot.clock_bindings.end() || !resolved.resolved()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
    const ClockId resolved_clock =
        find_clock_id(result.clocks, resolved.declaration->graph, resolved.declaration->key);
    if (!resolved_clock.valid()) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_IMPORT_BINDING_INVALID"));
    }
    result.clock_import_resolutions.push_back(ClockImportResolutionPlanRecord{
        .child_graph = clock_import.graph,
        .child_clock = clock_import.key,
        .is_default = clock_import.is_default,
        .parent_graph = binding->parent_graph,
        .parent_clock = binding->parent_clock,
        .resolved_clock = resolved_clock,
    });
  }
  return {};
}

[[nodiscard]] expected<ClockAndReadinessLoweringResult, Error> lower_clock_catalog(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation) {
  if (plan_generation == 0U) {
    return make_unexpected(artifact_error(ErrorCode::kInvalidArgument, "PLAN_GENERATION_INVALID"));
  }
  if (auto bindings = validate_clock_import_bindings(snapshot); !bindings) {
    return make_unexpected(std::move(bindings).error());
  }

  ClockAndReadinessLoweringResult result;
  result.plan_generation = plan_generation;
  if (auto declarations = append_clock_declarations(snapshot, result); !declarations) {
    return make_unexpected(std::move(declarations).error());
  }
  if (auto imports = append_clock_import_resolutions(snapshot, result); !imports) {
    return make_unexpected(std::move(imports).error());
  }

  const auto root = std::find_if(snapshot.graphs.begin(),
                                 snapshot.graphs.end(),
                                 [](const auto& graph) { return !graph.parent.has_value(); });
  if (root != snapshot.graphs.end()) {
    result.root_default_clock = find_default_clock_id(result.clocks, root->id);
  }
  return result;
}

[[nodiscard]] expected<void, Error> append_clock_conversions(
    const GraphSnapshotData& snapshot, ClockAndReadinessLoweringResult& result) {
  result.clock_conversions.reserve(snapshot.clock_conversions.size());
  for (std::size_t index = 0U; index < snapshot.clock_conversions.size(); ++index) {
    const auto& declaration = snapshot.clock_conversions[index];
    const ClockId from =
        resolve_clock_id(snapshot, result.clocks, declaration.graph, declaration.from);
    const ClockId to = resolve_clock_id(snapshot, result.clocks, declaration.graph, declaration.to);
    if (!from.valid() || !to.valid()) {
      return make_unexpected(artifact_error(ErrorCode::kNotFound, "CLOCK_CONVERSION_UNDECLARED"));
    }
    if (from == to || declaration.conversion.scale.numerator <= 0 ||
        declaration.conversion.scale.denominator == 0U) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "CLOCK_CONVERSION_INVALID"));
    }
    result.clock_conversions.push_back(ClockConversionRecord{
        .id = static_cast<std::uint64_t>(index + 1U),
        .plan_generation = result.plan_generation,
        .graph = declaration.graph,
        .from = from,
        .to = to,
        .scale = declaration.conversion.scale,
        .offset_ns = declaration.conversion.offset_ns,
        .rounding = ClockRoundingMode::kNearestTiesToEven,
        .overflow = ClockOverflowPolicy::kError,
    });
  }
  return {};
}

[[nodiscard]] expected<std::vector<CanonicalReadinessDeclaration>, Error>
canonical_readiness_declarations(const GraphSnapshotData& snapshot) {
  std::vector<CanonicalReadinessDeclaration> declarations;
  declarations.reserve(snapshot.readiness_sources.size());
  for (const auto& declaration : snapshot.readiness_sources) {
    auto canonical = canonical_readiness_declaration(snapshot, declaration);
    if (!canonical) {
      return make_unexpected(std::move(canonical).error());
    }
    declarations.push_back(std::move(canonical).value());
  }
  std::sort(declarations.begin(), declarations.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.graph_path,
                      lhs.owner_name,
                      readiness_kind_token(lhs.declaration.kind),
                      lhs.declaration.name,
                      lhs.declaration.declaration_ordinal} <
           std::tuple{rhs.graph_path,
                      rhs.owner_name,
                      readiness_kind_token(rhs.declaration.kind),
                      rhs.declaration.name,
                      rhs.declaration.declaration_ordinal};
  });
  if (declarations.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        artifact_error(ErrorCode::kResourceExhausted, "READINESS_SOURCE_ID_EXHAUSTED"));
  }
  return declarations;
}

[[nodiscard]] bool readiness_capacity_valid(
    const ReadinessSourceDeclarationRecord& source) noexcept {
  const bool capacity_one =
      source.kind == ReadinessSourceKind::kReadable || source.kind == ReadinessSourceKind::kSpin;
  if (capacity_one) {
    return source.requested_capacity == 1U;
  }
  if (source.requested_capacity == 0U) {
    return false;
  }
  return source.kind != ReadinessSourceKind::kNotification ||
         source.requested_sender_reference_capacity != 0U;
}

[[nodiscard]] expected<ClockId, Error> resolve_readiness_clock(
    const GraphSnapshotData& snapshot, std::span<const ClockPlanRecord> clocks,
    const CanonicalReadinessDeclaration& candidate) {
  const auto& source = candidate.declaration;
  const ClockId resolved =
      resolve_clock_id(snapshot, clocks, candidate.owner_graph, source.requested_clock);
  if (resolved.valid()) {
    return resolved;
  }
  return make_unexpected(artifact_error(ErrorCode::kNotFound,
                                        source.requested_clock.is_default()
                                            ? "READINESS_DEFAULT_CLOCK_UNDECLARED"
                                            : "READINESS_CLOCK_UNDECLARED"));
}

[[nodiscard]] expected<std::vector<ClockId>, Error> validate_and_resolve_readiness(
    const GraphSnapshotData& snapshot, std::span<const CanonicalReadinessDeclaration> declarations,
    std::span<const ClockPlanRecord> clocks) {
  std::vector<ClockId> resolved_clocks;
  resolved_clocks.reserve(declarations.size());
  for (const auto& candidate : declarations) {
    const auto& source = candidate.declaration;
    if (!readiness_capacity_valid(source)) {
      return make_unexpected(
          artifact_error(ErrorCode::kInvalidArgument, "READINESS_CAPACITY_INVALID"));
    }
    auto resolved = resolve_readiness_clock(snapshot, clocks, candidate);
    if (!resolved) {
      return make_unexpected(std::move(resolved).error());
    }
    resolved_clocks.push_back(std::move(resolved).value());
  }
  return resolved_clocks;
}

void append_readiness_records(std::span<const CanonicalReadinessDeclaration> declarations,
                              std::span<const ClockId> resolved_clocks,
                              std::span<const std::optional<Profile>> effective_vertex_profiles,
                              ClockAndReadinessLoweringResult& result) {
  result.readiness.reserve(declarations.size());
  for (std::size_t index = 0U; index < declarations.size(); ++index) {
    const auto& candidate = declarations[index];
    const auto& source = candidate.declaration;
    result.readiness.push_back(ReadinessPlanRecord{
        .source_id = make_source_id(static_cast<std::uint32_t>(index + 1U)),
        .plan_generation = result.plan_generation,
        .owner = source.owner,
        .owner_path = candidate.owner_path,
        .declaration_ordinal = source.declaration_ordinal,
        .kind = source.kind,
        .name = source.name,
        .effective_profile = profile_for_vertex(effective_vertex_profiles, source.owner),
        .requested_clock = source.requested_clock,
        .resolved_clock = resolved_clocks[index],
        .requested_capacity = source.requested_capacity,
        .requested_sender_reference_capacity = source.requested_sender_reference_capacity,
        .wake_requirement = source.wake_requirement,
        .accepted_capacity = source.requested_capacity,
        .accepted_sender_reference_capacity = source.requested_sender_reference_capacity,
        .referenced = source.referenced,
        .materialized = source.referenced,
        .initial_source_generation = 1U,
        .initial_source_sequence = 0U,
        .disposition_reason =
            source.referenced ? DiagnosticId{} : diagnostic_id("READINESS_SOURCE_UNUSED"),
    });
    if (!source.referenced) {
      result.unused_source_warnings.push_back(DiagnosticSource{
          .graph = candidate.owner_graph,
          .vertex = source.owner,
      });
    }
  }
}

}  // namespace

expected<ClockAndReadinessLoweringResult, Error> lower_clocks_and_readiness(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_vertex_profiles) {
  // REQ: COMPILE-007 - validate the whole clock/conversion/source table before
  // publishing any row. Numeric ClockId values never escape without the same
  // nonzero plan-generation provenance.
  auto result = lower_clock_catalog(snapshot, plan_generation);
  if (!result) {
    return make_unexpected(std::move(result).error());
  }
  auto conversions = append_clock_conversions(snapshot, result.value());
  if (!conversions) {
    return make_unexpected(std::move(conversions).error());
  }
  auto declarations = canonical_readiness_declarations(snapshot);
  if (!declarations) {
    return make_unexpected(std::move(declarations).error());
  }
  auto resolved_clocks =
      validate_and_resolve_readiness(snapshot, declarations.value(), result.value().clocks);
  if (!resolved_clocks) {
    return make_unexpected(std::move(resolved_clocks).error());
  }
  append_readiness_records(
      declarations.value(), resolved_clocks.value(), effective_vertex_profiles, result.value());
  return result;
}

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
                                     ? generated_endpoint_name(snapshot, topic)
                                     : std::string{*topic.endpoint_name};
  std::size_t index{};
  for (const auto& candidate : snapshot.external_topics) {
    const std::string candidate_name =
        candidate.generated_name_requested || !candidate.endpoint_name
            ? generated_endpoint_name(snapshot, candidate)
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
  return resolve_clock_id(snapshot, clocks, owner->graph, topic.capture_time->clock);
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

[[nodiscard]] std::vector<GraphPortId> endpoint_alias_component(const GraphSnapshotData& snapshot,
                                                                GraphPortId initial) {
  std::vector<GraphPortId> result{initial};
  for (std::size_t cursor = 0U; cursor < result.size(); ++cursor) {
    for (const auto& binding : snapshot.interface_bindings) {
      GraphPortId adjacent{};
      if (binding.public_port == result[cursor]) {
        adjacent = binding.internal_port;
      } else if (binding.internal_port == result[cursor]) {
        adjacent = binding.public_port;
      }
      if (adjacent && std::find(result.begin(), result.end(), adjacent) == result.end()) {
        result.push_back(adjacent);
      }
    }
  }
  return result;
}

[[nodiscard]] bool imported_topic_feeds_selection(const GraphSnapshotData& snapshot,
                                                  const ExternalTopicRecord& topic,
                                                  PortId selected_input) {
  const PortRecord* input = find_port(snapshot, GraphPortId{selected_input.value()});
  if (input == nullptr || input->direction != PortDirection::kInput) {
    return false;
  }
  const auto target_aliases = endpoint_alias_component(snapshot, input->id);
  const auto incoming =
      std::find_if(snapshot.edges.begin(), snapshot.edges.end(), [&](const auto& e) {
        return std::find(target_aliases.begin(), target_aliases.end(), e.target) !=
               target_aliases.end();
      });
  if (incoming == snapshot.edges.end()) {
    return false;
  }
  const auto source_aliases = endpoint_alias_component(snapshot, incoming->source);
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
      result.effective_profile = stricter_profile(
          result.effective_profile, profile_for_vertex(effective_vertex_profiles, input->owner));
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
    const auto root = std::find_if(snapshot.graphs.begin(),
                                   snapshot.graphs.end(),
                                   [](const auto& g) { return !g.parent.has_value(); });
    return root == snapshot.graphs.end() ? ClockId{} : find_default_clock_id(clocks, root->id);
  }();

  for (const auto& topic : snapshot.external_topics) {
    const DiagnosticSource source = endpoint_diagnostic_source(snapshot, topic);
    const std::optional<Profile> endpoint_profile =
        profile_for_vertex(effective_vertex_profiles, topic.vertex);
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
    const bool constrained = profile.has_value() && constrained_endpoint_profile(*profile);
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
  canonicalize_endpoint_diagnostics(result.errors);
  return result;
}

EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions) {
  EndpointCatalogLoweringResult result;
  std::vector<ChannelEndpointRecord> endpoints;
  endpoints.reserve(snapshot.external_topics.size());
  const DiagnosticSource root_source = root_diagnostic_source(snapshot);
  validate_generated_endpoint_pins(deployment, root_source, result);

  // REQ: COMPILE-007 - final endpoint identity is frozen before Runtime/backend construction.
  for (const auto& topic : snapshot.external_topics) {
    auto endpoint =
        lower_endpoint_record(snapshot,
                              topic,
                              deployment,
                              profile_for_vertex(effective_vertex_profiles, topic.vertex),
                              supplied_catalog,
                              result);
    if (endpoint.has_value()) {
      endpoints.push_back(std::move(*endpoint));
    }
  }

  std::sort(endpoints.begin(), endpoints.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.name < rhs.name;
  });
  append_duplicate_endpoint_diagnostics(snapshot, endpoints, result);
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
  canonicalize_endpoint_diagnostics(result.errors);
  canonicalize_endpoint_diagnostics(result.warnings);
  auto catalog = ChannelEndpointCatalogAccess::normalize(derived_scope,
                                                         std::move(endpoints),
                                                         std::move(catalog_clocks.clocks),
                                                         std::move(catalog_clocks.conversions));
  if (!catalog) {
    result.errors.push_back(EndpointCatalogDiagnostic{catalog.error().message.empty()
                                                          ? "ENDPOINT_CATALOG_NORMALIZATION_FAILED"
                                                          : catalog.error().message,
                                                      root_source});
    canonicalize_endpoint_diagnostics(result.errors);
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
    std::span<const ClockConversionRecord> clock_conversions) {
  const std::vector<std::optional<Profile>> uniform_profiles(snapshot.vertices.size(), profile);
  return lower_endpoint_catalog(
      snapshot, deployment, uniform_profiles, supplied_catalog, clocks, clock_conversions);
}

namespace {

void append_optional_algorithm(std::string& output, const std::optional<AlgorithmKey>& algorithm) {
  if (!algorithm.has_value()) {
    output += "null";
    return;
  }
  output += "{\"id\":";
  append_json_string(output, algorithm->id);
  output += ",\"version\":";
  append_number(output, algorithm->version);
  output.push_back('}');
}

void append_optional_maturity(std::string& output, std::optional<AlgorithmMaturity> maturity) {
  if (!maturity.has_value()) {
    output += "null";
    return;
  }
  append_number(output, static_cast<std::uint32_t>(*maturity));
}

void append_optional_profile(std::string& output, std::optional<Profile> profile) {
  if (!profile.has_value()) {
    output += "null";
    return;
  }
  append_json_string(output, profile_token(*profile));
}

void append_optional_string(std::string& output, const std::optional<std::string>& value) {
  if (!value) {
    output += "null";
    return;
  }
  append_json_string(output, *value);
}

void append_optional_time_point(std::string& output, const std::optional<TimePoint>& value) {
  if (!value) {
    output += "null";
    return;
  }
  output += "{\"clock\":";
  append_number(output, value->clock_id.value);
  output += ",\"timestamp_ns\":";
  append_number(output, value->timestamp_ns);
  output.push_back('}');
}

void append_profile_provenance(std::string& output, const ProfileManifest& manifest,
                               bool include_provenance) {
  if (!include_provenance) {
    return;
  }
  output += ",\"producer_build_id\":";
  append_json_string(output, manifest.producer_build_id);
}

}  // namespace

// Canonical serialization emits the complete public semantic plan schema in one order. Private
// Runtime construction facts are validated above but remain derived, process-local state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::string, Error> plan_artifact_json(const ExecutionPlanState& state,
                                                bool include_provenance) {
  // REQ: COMPILE-002 - canonical semantics, not diagnostic build provenance,
  // form the digest preimage. The complete artifact retains both.
  // REQ: COMPILE-009 - serialization reads immutable records and performs no runtime/OS mutation.
  if (!valid_process_local_channel_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID"));
  }
  if (!valid_closed_island_activation_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "CLOSED_ISLAND_ACTIVATION_PLAN_INVALID"));
  }
  if (!valid_inline_partition_execution_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "INLINE_PARTITION_EXECUTION_PLAN_INVALID"));
  }
  if (!valid_partition_runner_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "PARTITION_RESOURCE_REALIZATION_FAILED"));
  }
  if (state.admitted && !valid_external_channel_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "EXTERNAL_CHANNEL_PLAN_INVALID"));
  }
  if (state.admitted && !valid_cross_process_channel_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "CROSS_PROCESS_CHANNEL_PLAN_INVALID"));
  }
  std::string output{"{\"admitted\":"};
  append_bool(output, state.admitted);
  // REQ: COMPILE-015, COMPILE-016, COMPILE-017 - temporal record families are
  // separate canonical arrays whose pointer-free joins contribute to identity.
  output += ",\"aligned_triggers\":";
  append_array(output, state.aligned_triggers, [](std::string& json, const auto& record) {
    json += "{\"accepted_once_consumption\":";
    append_bool(json, record.accepted_once_consumption);
    json += ",\"algorithm\":";
    append_json_string(json, record.algorithm);
    json += ",\"checked_direct_conversion\":";
    append_bool(json, record.checked_direct_conversion);
    json += ",\"checked_spread_subtraction\":";
    append_bool(json, record.checked_spread_subtraction);
    json += ",\"closure_proofs\":";
    append_array(json, record.closure_proofs, [](std::string& values, auto proof) {
      append_number(values, static_cast<std::uint32_t>(proof));
    });
    json += ",\"comparison_clock\":";
    append_number(json, record.comparison_clock.value);
    json += ",\"exact_window_matches\":";
    append_bool(json, record.exact_window_matches);
    json += ",\"fifo_heads_only\":";
    append_bool(json, record.fifo_heads_only);
    json += ",\"formation_failure_identity\":";
    append_json_string(json, record.formation_failure_identity);
    json += ",\"inclusive_window_ns\":";
    append_number(json, record.inclusive_window_ns);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& members, const auto& member) {
      members += "{\"cache_capacity\":";
      append_number(members, member.cache_capacity);
      members += ",\"capture_time_required\":";
      append_bool(members, member.capture_time_required);
      members += ",\"conversion_id\":";
      append_number(members, member.conversion_id);
      members += ",\"event_clock\":";
      append_number(members, member.event_clock.value);
      members += ",\"input\":";
      append_number(members, member.input.value());
      members += ",\"member_ordinal\":";
      append_number(members, member.member_ordinal);
      members += ",\"nondecreasing_normalized_time\":";
      append_bool(members, member.nondecreasing_normalized_time);
      members += ",\"queue_capacity\":";
      append_number(members, member.queue_capacity);
      members.push_back('}');
    });
    json += ",\"miss_witness_identity\":";
    append_json_string(json, record.miss_witness_identity);
    json += ",\"missing_requires_finite_proof\":";
    append_bool(json, record.missing_requires_finite_proof);
    json += ",\"no_scan_ahead\":";
    append_bool(json, record.no_scan_ahead);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"public_watermark_protocol\":";
    append_bool(json, record.public_watermark_protocol);
    json += ",\"reference_is_maximum_head\":";
    append_bool(json, record.reference_is_maximum_head);
    json += ",\"retire_all_minimum_heads\":";
    append_bool(json, record.retire_all_minimum_heads);
    json += ",\"retirement_order\":";
    append_json_string(json, record.retirement_order);
    json += ",\"silence_is_not_closure\":";
    append_bool(json, record.silence_is_not_closure);
    json.push_back('}');
  });
  output += ",\"bracketed_windows\":";
  append_array(output, state.bracketed_windows, [](std::string& json, const auto& record) {
    json += "{\"anchor\":";
    append_number(json, record.anchor.value());
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"close_condition\":";
    append_json_string(json, record.close_condition);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"event_clock\":";
    append_number(json, record.event_clock.value);
    json += ",\"inclusive_interior\":";
    append_bool(json, record.inclusive_interior);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"insufficient_history_reason\":";
    append_json_string(json, record.insufficient_history_reason);
    json += ",\"insufficient_history_witness_identity\":";
    append_json_string(json, record.insufficient_history_witness_identity);
    json += ",\"lease_lifetime_evidence\":";
    append_json_string(json, record.lease_lifetime_evidence);
    json += ",\"lookahead\":";
    append_number(json, record.lookahead);
    json += ",\"lookbehind\":";
    append_number(json, record.lookbehind);
    json += ",\"max_samples\":";
    append_number(json, record.max_samples);
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"ordering\":";
    append_json_string(json, record.ordering);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"producer_epoch_identity_schema\":";
    append_json_string(json, record.producer_epoch_identity_schema);
    json += ",\"producer_max_emits_per_compute\":";
    append_number(json, record.producer_max_emits_per_compute);
    json += ",\"queue_capacity\":";
    append_number(json, record.queue_capacity);
    json += ",\"retain_overlap\":";
    append_bool(json, record.retain_overlap);
    json += ",\"retained_overlap_capacity\":";
    append_number(json, record.retained_overlap_capacity);
    json += ",\"sample_limit_reason\":";
    append_json_string(json, record.sample_limit_reason);
    json += ",\"sample_limit_witness_identity\":";
    append_json_string(json, record.sample_limit_witness_identity);
    json += ",\"semantic_reference\":";
    append_json_string(json, record.semantic_reference);
    json += ",\"source_sequence_identity_schema\":";
    append_json_string(json, record.source_sequence_identity_schema);
    json += ",\"strict_outer_samples\":";
    append_bool(json, record.strict_outer_samples);
    json += ",\"timeout_reason\":";
    append_json_string(json, record.timeout_reason);
    json += ",\"timeout_witness_identity\":";
    append_json_string(json, record.timeout_witness_identity);
    json += ",\"activation_time_policy\":";
    append_json_string(json, record.activation_time_policy);
    json.push_back('}');
  });
  output += ",\"channel_locality\":";
  append_array(output, state.channel_locality, [](std::string& json, const auto& record) {
    json += "{\"accepted\":";
    append_number(json, static_cast<std::uint32_t>(record.accepted));
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"requested\":";
    append_number(json, static_cast<std::uint32_t>(record.requested));
    json += ",\"requires_runtime_attach\":";
    append_bool(json, record.requires_runtime_attach);
    json.push_back('}');
  });
  output += ",\"channel_publishers\":";
  append_array(output, state.channel_publishers, [](std::string& json, const auto& record) {
    json += "{\"max_loaned_samples\":";
    append_number(json, record.max_loaned_samples);
    json += ",\"replacement_attribution_required\":";
    append_bool(json, record.replacement_attribution_required);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"channel_services\":";
  append_array(output, state.channel_services, [](std::string& json, const auto& record) {
    json += "{\"event_wake\":";
    append_bool(json, record.event_wake);
    json += ",\"max_subscriber_borrowed_samples\":";
    append_number(json, record.max_subscriber_borrowed_samples);
    json += ",\"max_subscriber_queue_depth\":";
    append_number(json, record.max_subscriber_queue_depth);
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"publisher_history_depth\":";
    append_number(json, record.publisher_history_depth);
    json += ",\"queue_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.queue_policy));
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"tensor_emit_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.tensor_emit_policy));
    json += ",\"tensor_payload\":";
    append_bool(json, record.tensor_payload);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"channel_subscribers\":";
  append_array(output, state.channel_subscribers, [](std::string& json, const auto& record) {
    json += "{\"event_wake\":";
    append_bool(json, record.event_wake);
    json += ",\"max_borrowed_samples\":";
    append_number(json, record.max_borrowed_samples);
    json += ",\"requested_queue_depth\":";
    append_number(json, record.requested_queue_depth);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"clock_catalog\":{\"clocks\":";
  append_clock_plan_records(output, state.clocks);
  output += ",\"plan_generation\":";
  append_number(output, state.plan_generation);
  output += ",\"root_default_clock\":";
  append_number(output, state.root_default_clock.value);
  output += "},\"clock_conversions\":";
  append_array(
      output, state.clock_conversions, [](std::string& json, const ClockConversionRecord& record) {
        json += "{\"denominator\":";
        append_number(json, record.scale.denominator);
        json += ",\"from\":";
        append_number(json, record.from.value);
        json += ",\"graph\":";
        append_number(json, record.graph.value);
        json += ",\"id\":";
        append_number(json, record.id);
        json += ",\"numerator\":";
        append_number(json, record.scale.numerator);
        json += ",\"offset_ns\":";
        append_number(json, record.offset_ns);
        json += ",\"overflow\":";
        append_json_string(json,
                           record.overflow == ClockOverflowPolicy::kError ? "error" : "unknown");
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"rounding\":";
        append_json_string(
            json,
            record.rounding == ClockRoundingMode::kNearestTiesToEven ? "ties_to_even" : "unknown");
        json += ",\"to\":";
        append_number(json, record.to.value);
        json.push_back('}');
      });
  output += ",\"clock_import_resolutions\":";
  append_array(output, state.clock_import_resolutions, [](std::string& json, const auto& record) {
    json += "{\"child_clock\":";
    append_json_string(json, record.child_clock.name());
    json += ",\"child_graph\":";
    append_number(json, record.child_graph.value);
    json += ",\"is_default\":";
    append_bool(json, record.is_default);
    json += ",\"parent_clock\":";
    append_json_string(json, record.parent_clock.name());
    json += ",\"parent_graph\":";
    append_number(json, record.parent_graph.value);
    json += ",\"resolved_clock\":";
    append_number(json, record.resolved_clock.value);
    json.push_back('}');
  });
  output += ",\"clock_triggers\":";
  append_array(output, state.clock_triggers, [](std::string& json, const auto& record) {
    json += "{\"arithmetic_failure_identity\":";
    append_json_string(json, record.arithmetic_failure_identity);
    json += ",\"checked_terminal_no_wrap\":";
    append_bool(json, record.checked_terminal_no_wrap);
    json += ",\"clock\":";
    append_number(json, record.clock.value);
    json += ",\"coalesced_range_capacity\":";
    append_number(json, record.coalesced_range_capacity);
    json += ",\"coalesced_range_identity\":";
    append_json_string(json, record.coalesced_range_identity);
    json += ",\"epoch_discontinuity\":";
    append_number(json, static_cast<std::uint32_t>(record.epoch_discontinuity));
    json += ",\"exact_now_eligible\":";
    append_bool(json, record.exact_now_eligible);
    json += ",\"explicit_phase\":";
    append_optional_time_point(json, record.explicit_phase);
    json += ",\"explicit_phase_preserved_on_restart\":";
    append_bool(json, record.explicit_phase_preserved_on_restart);
    json += ",\"failed_rollback_restores_watermark\":";
    append_bool(json, record.failed_rollback_restores_watermark);
    json += ",\"global_start_barrier\":";
    append_bool(json, record.global_start_barrier);
    json += ",\"greatest_due_constant_time\":";
    append_bool(json, record.greatest_due_constant_time);
    json += ",\"initial_handled_index\":";
    if (record.initial_handled_index) {
      append_number(json, *record.initial_handled_index);
    } else {
      json += "null";
    }
    json += ",\"initial_tick_index\":";
    append_number(json, record.initial_tick_index);
    json += ",\"omitted_phase_index_zero_immediate\":";
    append_bool(json, record.omitted_phase_index_zero_immediate);
    json += ",\"omitted_phase_recaptured_on_restart\":";
    append_bool(json, record.omitted_phase_recaptured_on_restart);
    json += ",\"one_candidate_per_observation\":";
    append_bool(json, record.one_candidate_per_observation);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase_conversion_id\":";
    append_number(json, record.phase_conversion_id);
    json += ",\"phase_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.phase_origin));
    json += ",\"scheduled_tick_is_reference\":";
    append_bool(json, record.scheduled_tick_is_reference);
    json += ",\"skipped_indices_have_no_candidate\":";
    append_bool(json, record.skipped_indices_have_no_candidate);
    json += ",\"stale_anchor_identity\":";
    append_json_string(json, record.stale_anchor_identity);
    json += ",\"stop_blocks_future_freeze\":";
    append_bool(json, record.stop_blocks_future_freeze);
    json += ",\"tick_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.tick_policy));
    json.push_back('}');
  });
  output += ",\"deadlines\":";
  append_array(output, state.deadlines, [](std::string& json, const auto& record) {
    json += "{\"action\":";
    append_number(json, static_cast<std::uint32_t>(record.action));
    json += ",\"automatic_output\":";
    append_bool(json, record.automatic_output);
    json += ",\"budget_ns\":";
    append_number(json, record.budget_ns);
    json += ",\"callback_count_if_late\":";
    append_number(json, record.callback_count_if_late);
    json += ",\"callback_state_rollback\":";
    append_bool(json, record.callback_state_rollback);
    json += ",\"checked_addition\":";
    append_bool(json, record.checked_addition);
    json += ",\"consumes_candidate_tag\":";
    append_bool(json, record.consumes_candidate_tag);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"duration_measurement_separate\":";
    append_bool(json, record.duration_measurement_separate);
    json += ",\"exact_boundary_is_on_time\":";
    append_bool(json, record.exact_boundary_is_on_time);
    json += ",\"final_invocation_gate\":";
    append_bool(json, record.final_invocation_gate);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"source_clock\":";
    append_number(json, record.source_clock.value);
    json += ",\"source_coordinate\":";
    append_number(json, static_cast<std::uint32_t>(record.source_coordinate));
    json += ",\"source_coordinate_identity\":";
    append_json_string(json, record.source_coordinate_identity);
    json += ",\"violation_witness_identity\":";
    append_json_string(json, record.violation_witness_identity);
    json.push_back('}');
  });
  output += ",\"diagnostics\":";
  append_array(output, state.diagnostics, [](std::string& json, const Diagnostic& diagnostic) {
    json += "{\"cause_id\":";
    if (diagnostic.cause_id) {
      append_number(json, diagnostic.cause_id->value);
    } else {
      json += "null";
    }
    json += ",\"endpoint\":";
    append_json_string(json, diagnostic.endpoint);
    json += ",\"id\":";
    append_number(json, diagnostic.id.value);
    json += ",\"json_pointer\":";
    append_json_string(json, diagnostic.json_pointer);
    json += ",\"provided\":";
    append_json_string(json, diagnostic.provided);
    json += ",\"required\":";
    append_json_string(json, diagnostic.required);
    json += ",\"service\":";
    append_json_string(json, diagnostic.service);
    json += ",\"severity\":";
    append_number(json, static_cast<std::uint32_t>(diagnostic.severity));
    json += ",\"source\":{\"graph\":";
    append_number(json, diagnostic.source.graph.value);
    json += ",\"graph_generation\":";
    append_number(json, diagnostic.source.graph_generation);
    json += ",\"port\":";
    append_number(json, diagnostic.source.port.value());
    json += ",\"vertex\":";
    append_number(json, diagnostic.source.vertex.value);
    json.push_back('}');
    json += ",\"token\":";
    append_json_string(json, diagnostic.token);
    json.push_back('}');
  });
  output += ",\"endpoint_catalog\":";
  output += state.endpoint_catalog.json().empty() ? "{}" : state.endpoint_catalog.json();
  output += ",\"endpoint_clock_health\":";
  append_array(output, state.endpoint_clock_health, [](std::string& json, const auto& record) {
    const auto append_duration = [](std::string& target, const auto& duration) {
      if (duration.has_value()) {
        append_number(target, duration->count());
      } else {
        target += "null";
      }
    };
    json += "{\"admitted_jitter_margin_ns\":";
    append_duration(json, record.admitted_jitter_margin);
    json += ",\"capability\":";
    append_number(json, static_cast<std::uint32_t>(record.capability));
    json += ",\"capture_clock\":";
    append_number(json, record.capture_clock.value);
    json += ",\"capture_conversion_id\":";
    append_number(json, record.capture_conversion_id);
    json += ",\"clock_health_update_witness_id\":";
    append_json_string(json, record.clock_health_update_witness_id);
    json += ",\"conversion_generation\":";
    append_number(json, record.conversion_generation);
    json += ",\"degraded_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.degraded_disposition));
    json += ",\"endpoint\":";
    append_number(json, record.endpoint.value());
    json += ",\"fixed_report_ingress_capacity\":";
    append_number(json, record.fixed_report_ingress_capacity);
    json += ",\"fixed_update_capacity\":";
    append_number(json, record.fixed_update_capacity);
    json += ",\"health_age_clock\":";
    append_number(json, record.health_age_clock.value);
    json += ",\"health_update_sequence_id\":";
    append_json_string(json, record.health_update_sequence_id);
    json += ",\"ingress_capacity_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.ingress_capacity_origin));
    json += ",\"max_monitor_period_ns\":";
    append_duration(json, record.max_monitor_period);
    json += ",\"max_observation_to_call_delay_ns\":";
    append_duration(json, record.max_observation_to_call_delay);
    json += ",\"max_report_age_ns\":";
    append_duration(json, record.max_report_age);
    json += ",\"max_report_delivery_delay_ns\":";
    append_duration(json, record.max_report_delivery_delay);
    json += ",\"monitor_evidence_reference\":";
    append_json_string(json, record.monitor_evidence_reference);
    json += ",\"monitor_path_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.monitor_path_kind));
    json += ",\"no_report_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.no_report_disposition));
    json += ",\"per_dispatch_report_budget\":";
    append_number(json, record.per_dispatch_report_budget);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"producer_epoch_source\":";
    append_number(json, static_cast<std::uint32_t>(record.producer_epoch_source));
    json += ",\"publisher_operation_sequence_id\":";
    append_json_string(json, record.publisher_operation_sequence_id);
    json += ",\"report_sequence_id\":";
    append_json_string(json, record.report_sequence_id);
    json += ",\"requirement\":";
    append_number(json, static_cast<std::uint32_t>(record.requirement));
    json += ",\"stale_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.stale_disposition));
    json += ",\"trace_loss_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.trace_loss_disposition));
    json += ",\"unhealthy_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.unhealthy_disposition));
    json += ",\"unspecified_reason_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.unspecified_reason_disposition));
    json.push_back('}');
  });
  output += ",\"clock_health_reason_schema\":[";
  constexpr std::array<std::string_view, 10> kClockHealthReasonTokens{
      "none",
      "synchronization_lock_lost",
      "phase_error_limit_exceeded",
      "rate_error_limit_exceeded",
      "calibration_expired",
      "clock_stalled",
      "timestamp_regression",
      "timestamp_source_fault",
      "monitor_fault",
      "unspecified",
  };
  for (std::size_t index = 0U; index < kClockHealthReasonTokens.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    output += "{\"id\":";
    append_number(output, index);
    output += ",\"token\":";
    append_json_string(output, kClockHealthReasonTokens[index]);
    output.push_back('}');
  }
  output.push_back(']');
  output += ",\"graph_generation\":";
  append_number(output, state.graph_identity.version);
  output += ",\"graph_hash\":";
  append_hex(output, state.graph_identity.hash.value);
  output += ",\"generic_output_loans\":";
  append_array(output, state.generic_output_loans, [](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"max_live_loans\":";
    append_number(json, record.max_live_loans);
    json += ",\"output\":";
    append_number(json, record.output.value());
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"pre_submit_preserves_same_activation_loan\":";
    append_bool(json, record.pre_submit_preserves_same_activation_loan);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"runtime_retry_enabled\":";
    append_bool(json, record.runtime_retry_enabled);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json += ",\"slot_size_bytes\":";
    append_number(json, record.slot_size_bytes);
    json += ",\"trait_evidence\":";
    append_number(json, static_cast<std::uint32_t>(record.trait_evidence));
    json.push_back('}');
  });
  output += ",\"imported_generation_capacities\":";
  append_array(
      output, state.imported_generation_capacities, [](std::string& json, const auto& record) {
        json += "{\"attach_max_wait_ns\":";
        if (record.attach_max_wait_ns) {
          append_number(json, *record.attach_max_wait_ns);
        } else {
          json += "null";
        }
        json += ",\"candidate_reservation_capacity\":";
        append_number(json, record.candidate_reservation_capacity);
        json += ",\"effective_profile\":";
        append_optional_profile(json, record.effective_profile);
        json += ",\"endpoint_max_generations\":";
        append_number(json, record.endpoint_max_generations);
        json += ",\"endpoint_or_service_id\":";
        append_json_string(json, record.endpoint_or_service_id);
        json += ",\"fallback_rejected\":";
        append_bool(json, record.fallback_rejected);
        json += ",\"no_wrap_aba_generation\":";
        append_bool(json, record.no_wrap_aba_generation);
        json += ",\"peer_requirement\":";
        append_number(json, static_cast<std::uint32_t>(record.peer_requirement));
        json += ",\"quarantine_record_capacity\":";
        append_number(json, record.quarantine_record_capacity);
        json += ",\"release_record_capacity\":";
        append_number(json, record.release_record_capacity);
        json += ",\"restart_capable_minimum_two\":";
        append_bool(json, record.restart_capable_minimum_two);
        json += ",\"runtime_max_generations\":";
        append_number(json, record.runtime_max_generations);
        json += ",\"runtime_scope_id\":";
        append_json_string(json, record.runtime_scope_id);
        json += ",\"service_max_generations\":";
        append_number(json, record.service_max_generations);
        json += ",\"topic\":";
        append_number(json, record.topic.value);
        json.push_back('}');
      });
  output += ",\"input_selections\":";
  append_array(output, state.input_selections, [](std::string& json, const auto& record) {
    json += "{\"batch_selection_mode\":";
    append_number(json, static_cast<std::uint32_t>(record.batch_selection_mode));
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"cardinality\":";
    append_number(json, static_cast<std::uint32_t>(record.cardinality));
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"lookahead\":";
    append_number(json, record.lookahead);
    json += ",\"lookbehind\":";
    append_number(json, record.lookbehind);
    json += ",\"max_age_ns\":";
    append_number(json, record.max_age_ns);
    json += ",\"max_selected_samples\":";
    append_number(json, record.max_selected_samples);
    json += ",\"max_selected_samples_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.max_selected_samples_origin));
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"reuse_max_age_ns\":";
    append_number(json, record.reuse_max_age_ns);
    json += ",\"selector\":";
    append_number(json, static_cast<std::uint32_t>(record.selector));
    json.push_back('}');
  });
  output += ",\"interface_port_bindings\":";
  append_array(output, state.interface_port_bindings, [](std::string& json, const auto& record) {
    json += "{\"direction\":";
    append_number(json, static_cast<std::uint32_t>(record.direction));
    json += ",\"internal_port\":";
    append_number(json, record.internal_port.value());
    json += ",\"public_port\":";
    append_number(json, record.public_port.value());
    json.push_back('}');
  });
  output += ",\"lifecycle_checkpoints\":";
  append_array(output, state.lifecycle_checkpoints, [](std::string& json, const auto& record) {
    json += "{\"callback\":";
    append_bool(json, record.callback_present);
    json += ",\"level\":";
    append_number(json, record.stable_topological_level);
    json += ",\"operator\":";
    append_operator_ref(json, record.operator_instance);
    json += ",\"required\":";
    append_bool(json, record.required);
    json += ",\"stage\":";
    append_number(json, static_cast<std::uint32_t>(record.stage));
    json.push_back('}');
  });
  output += ",\"lifecycle_dependencies\":";
  append_array(output, state.lifecycle_dependencies, [](std::string& json, const auto& record) {
    json += "{\"after\":";
    append_operator_ref(json, record.after);
    json += ",\"before\":";
    append_operator_ref(json, record.before);
    json += ",\"key\":";
    append_json_string(json, record.dependency_key);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"stage\":";
    append_number(json, static_cast<std::uint32_t>(record.stage));
    json.push_back('}');
  });
  output += ",\"lifecycle_mutable_fields\":";
  append_array(output, state.lifecycle_mutable_fields, [](std::string& json, const auto& record) {
    json += "{\"field\":";
    append_number(json, record.field);
    json += ",\"from_stage\":";
    append_number(json, static_cast<std::uint32_t>(record.from_stage));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json.push_back('}');
  });
  output += ",\"lifecycle_restart_closures\":";
  append_array(output, state.lifecycle_restart_closures, [](std::string& json, const auto& record) {
    json += "{\"closure\":";
    append_array(json, record.closure, [](std::string& values, const OperatorRef& member) {
      append_operator_ref(values, member);
    });
    json += ",\"required_fenced_operators\":";
    append_number(json, record.required_fenced_operators);
    json += ",\"required_quiesce_records\":";
    append_number(json, record.required_quiesce_records);
    json += ",\"required_rollback_records\":";
    append_number(json, record.required_rollback_records);
    json += ",\"stage_resolutions\":";
    append_array(json, record.stage_resolutions, [](std::string& stages, const auto& stage) {
      stages += "{\"effective_stage\":";
      append_number(stages, static_cast<std::uint32_t>(stage.effective_stage));
      stages += ",\"reason\":";
      append_number(stages, static_cast<std::uint32_t>(stage.reason));
      stages += ",\"required_quiesce_records\":";
      append_number(stages, stage.required_quiesce_records);
      stages += ",\"required_rollback_records\":";
      append_number(stages, stage.required_rollback_records);
      stages += ",\"required_structural_event_records\":";
      append_number(stages, stage.required_structural_event_records);
      stages += ",\"requested_stage\":";
      append_number(stages, static_cast<std::uint32_t>(stage.requested_stage));
      stages.push_back('}');
    });
    json += ",\"target\":";
    append_operator_ref(json, record.target);
    json += ",\"valid_requested_stage_mask\":";
    append_number(json, record.valid_requested_stage_mask);
    json.push_back('}');
  });
  output += ",\"lifecycle_restart_permissions\":";
  append_array(
      output, state.lifecycle_restart_permissions, [](std::string& json, const auto& record) {
        json += "{\"requester\":";
        append_operator_ref(json, record.requester);
        json += ",\"target\":";
        append_operator_ref(json, record.target);
        json.push_back('}');
      });
  output += ",\"lifecycle_restart_targets\":";
  append_array(output, state.lifecycle_restart_targets, [](std::string& json, const auto& record) {
    json += "{\"allowed_from_stage_mask\":";
    append_number(json, record.allowed_from_stage_mask);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"external_control_allowed\":";
    append_bool(json, record.external_control_allowed);
    json += ",\"parameter_origin_allowed\":";
    append_bool(json, record.parameter_origin_allowed);
    json += ",\"profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"reason\":";
    append_number(json, record.disposition_reason);
    json += ",\"scope\":";
    append_json_string(json, record.effective_scope_id);
    json += ",\"self_allowed\":";
    append_bool(json, record.self_allowed);
    json += ",\"target\":";
    append_operator_ref(json, record.target);
    json.push_back('}');
  });
  output += ",\"lifecycle_transaction_capacity\":";
  if (state.lifecycle_transaction_capacity) {
    const auto& capacity = *state.lifecycle_transaction_capacity;
    output += "{\"configured\":";
    append_lifecycle_capacity_values(output, capacity.configured);
    output += ",\"derivation_reason\":";
    append_number(output, capacity.derivation_reason);
    output += ",\"parameter_origin_enabled\":";
    append_bool(output, capacity.parameter_origin_enabled);
    output += ",\"required_generation_slots_per_operator\":";
    append_number(output, capacity.required_generation_slots_per_operator);
    output += ",\"required_minimum\":";
    append_lifecycle_capacity_values(output, capacity.required_minimum);
    output += ",\"required_parameter_arena_bytes\":";
    append_number(output, capacity.required_parameter_arena_bytes);
    output += ",\"required_parameter_audit_records\":";
    append_number(output, capacity.required_parameter_audit_records);
    output += ",\"resolved_parameter_update_policy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.resolved_parameter_update_policy));
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"missing_aggregates\":";
  append_array(output, state.missing_aggregates, [](std::string& json, const auto& record) {
    json += "{\"absent_callback_count\":";
    append_number(json, record.absent_callback_count);
    json += ",\"absent_without_safe_default_no_fires\":";
    append_bool(json, record.absent_without_safe_default_no_fires);
    json += ",\"aggregate_identity\":";
    append_json_string(json, record.aggregate_identity);
    json += ",\"bracketed_window_all_or_nothing\":";
    append_bool(json, record.bracketed_window_all_or_nothing);
    json += ",\"degraded_callback_count\":";
    append_number(json, record.degraded_callback_count);
    json += ",\"degraded_commits_destructive_selection\":";
    append_bool(json, record.degraded_commits_destructive_selection);
    json += ",\"evaluate_complete_set\":";
    append_bool(json, record.evaluate_complete_set);
    json += ",\"input_state_identity\":";
    append_json_string(json, record.input_state_identity);
    json += ",\"inputs\":";
    append_array(json, record.inputs, [](std::string& inputs, const auto& input) {
      inputs += "{\"input\":";
      append_number(inputs, input.input.value());
      inputs += ",\"missing\":";
      append_number(inputs, static_cast<std::uint32_t>(input.missing));
      inputs += ",\"ordinal\":";
      append_number(inputs, input.ordinal);
      inputs += ",\"selector\":";
      append_number(inputs, static_cast<std::uint32_t>(input.selector));
      inputs.push_back('}');
    });
    json += ",\"no_fire_preserves_provisional_selection\":";
    append_bool(json, record.no_fire_preserves_provisional_selection);
    json += ",\"nominal_callback_count\":";
    append_number(json, record.nominal_callback_count);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"phase\":";
    append_number(json, static_cast<std::uint32_t>(record.phase));
    json += ",\"safe_default_dominates\":";
    append_bool(json, record.safe_default_dominates);
    json += ",\"selection_reason_identity\":";
    append_json_string(json, record.selection_reason_identity);
    json.push_back('}');
  });
  output += ",\"operator_parameters\":";
  append_array(output, state.operator_parameters, [](std::string& json, const auto& record) {
    json += "{\"contract_hash\":";
    append_hex(json, record.contract_hash.value);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fixed_scalar_generation_strategy\":";
    append_bool(json, record.fixed_scalar_generation_strategy);
    json += ",\"initial_generation\":";
    append_number(json, record.initial_generation);
    json += ",\"lifecycle_updates_enabled\":";
    append_bool(json, record.lifecycle_updates_enabled);
    json += ",\"metadata_hash\":";
    append_hex(json, record.metadata_hash.value);
    json += ",\"no_params\":";
    append_bool(json, record.no_params);
    json += ",\"object_alignment\":";
    append_number(json, record.object_alignment);
    json += ",\"object_size\":";
    append_number(json, record.object_size);
    json += ",\"operator_path\":";
    append_json_string(json, record.operator_path);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"runtime_updates_enabled\":";
    append_bool(json, record.runtime_updates_enabled);
    json += ",\"schema_availability\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_availability));
    json += ",\"schema_coverage\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_coverage));
    json += ",\"schema_id\":";
    append_json_string(json, record.schema_id);
    json += ",\"schema_proof\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_proof));
    json += ",\"schema_version\":";
    append_number(json, record.schema_version);
    json += ",\"storage\":";
    append_parameter_storage(json, record.storage);
    json += ",\"structural_hash\":";
    append_hex(json, record.structural_hash.value);
    json.push_back('}');
  });
  output += ",\"parameter_derived_terms\":";
  append_array(output, state.parameter_derived_terms, [](std::string& json, const auto& record) {
    json += "{\"field\":";
    append_number(json, record.field);
    json += ",\"field_full_id\":";
    append_hex(json, record.field_full_id.value);
    json += ",\"field_path\":";
    append_json_string(json, record.field_path);
    json += ",\"initial_value_ns\":";
    append_number(json, record.initial_value_ns);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"maximum_value_ns\":";
    append_number(json, record.maximum_value_ns);
    json += ",\"minimum_value_ns\":";
    append_number(json, record.minimum_value_ns);
    json += ",\"mutability\":";
    append_number(json, static_cast<std::uint32_t>(record.mutability));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"schema_id\":";
    append_json_string(json, record.schema_id);
    json += ",\"schema_version\":";
    append_number(json, record.schema_version);
    json += ",\"term_id\":";
    append_json_string(json, record.term_id);
    json += ",\"update_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.update_disposition));
    json.push_back('}');
  });
  output += ",\"parameter_fields\":";
  append_array(output, state.parameter_fields, [](std::string& json, const auto& record) {
    json += "{\"canonical_name\":";
    append_json_string(json, record.canonical_name);
    json += ",\"earliest_invalidated_stage\":";
    append_number(json, static_cast<std::uint32_t>(record.earliest_invalidated_stage));
    json += ",\"external_access\":";
    append_number(json, static_cast<std::uint32_t>(record.external_access));
    json += ",\"field\":";
    append_number(json, record.field);
    json += ",\"full_id\":";
    append_hex(json, record.full_id.value);
    json += ",\"logical_type\":";
    append_number(json, static_cast<std::uint32_t>(record.logical_type));
    json += ",\"mutability\":";
    append_number(json, static_cast<std::uint32_t>(record.mutability));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"persistence\":";
    append_number(json, static_cast<std::uint32_t>(record.persistence));
    json += ",\"sensitivity\":";
    append_number(json, static_cast<std::uint32_t>(record.sensitivity));
    json += ",\"storage\":";
    append_parameter_storage(json, record.storage);
    json += ",\"type_shape\":";
    append_parameter_type_shape(json, record.type_shape);
    json += ",\"update_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.update_disposition));
    json += ",\"value_route\":";
    append_number(json, static_cast<std::uint32_t>(record.value_route));
    json += ",\"visibility\":";
    append_number(json, static_cast<std::uint32_t>(record.visibility));
    json.push_back('}');
  });
  output += ",\"parameter_runtime_capacity\":";
  if (state.parameter_runtime_capacity) {
    const auto& capacity = *state.parameter_runtime_capacity;
    output += "{\"arena_strategy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.arena_strategy));
    output += ",\"audit_capacity\":";
    append_number(output, capacity.audit_capacity);
    output += ",\"compute_ingress_capacity\":";
    append_number(output, capacity.compute_ingress_capacity);
    output += ",\"enabled_operator_count\":";
    append_number(output, capacity.enabled_operator_count);
    output += ",\"event_capacity\":";
    append_number(output, capacity.event_capacity);
    output += ",\"generation_slots_per_operator\":";
    append_number(output, capacity.generation_slots_per_operator);
    output += ",\"lifecycle_updates_enabled\":";
    append_bool(output, capacity.lifecycle_updates_enabled);
    output += ",\"max_arena_alignment\":";
    append_number(output, capacity.max_arena_alignment);
    output += ",\"max_compute_patches_per_request\":";
    append_number(output, capacity.max_compute_patches_per_request);
    output += ",\"max_event_cursors\":";
    append_number(output, capacity.max_event_cursors);
    output += ",\"max_pending_updates_per_operator\":";
    append_number(output, capacity.max_pending_updates_per_operator);
    output += ",\"parameter_arena_bytes\":";
    append_number(output, capacity.parameter_arena_bytes);
    output += ",\"required_lifecycle_capacity\":";
    append_lifecycle_capacity_values(output, capacity.required_lifecycle_capacity);
    output += ",\"resolved_update_policy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.resolved_update_policy));
    output += ",\"runtime_updates_enabled\":";
    append_bool(output, capacity.runtime_updates_enabled);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"payload_edges\":";
  append_array(output, state.payload_edges, [](std::string& json, const auto& record) {
    json += "{\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"fanout\":";
    append_number(json, record.fanout);
    json += ",\"locality\":";
    append_number(json, static_cast<std::uint32_t>(record.locality));
    json += ",\"max_inflight_samples\":";
    append_number(json, record.max_inflight_samples);
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"producer_max_emits_per_compute\":";
    append_number(json, record.producer_max_emits_per_compute);
    json += ",\"queue_depth\":";
    append_number(json, record.queue_depth);
    json += ",\"queue_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.queue_policy));
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"source\":";
    append_number(json, record.source.value());
    json += ",\"target\":";
    append_number(json, record.target.value());
    json.push_back('}');
  });
  output += ",\"pool_capacity_contributors\":";
  append_array(output, state.pool_capacity_contributors, [](std::string& json, const auto& record) {
    json += "{\"contributor_id\":";
    append_json_string(json, record.contributor_id);
    json += ",\"maximum_distinct_slots\":";
    append_number(json, record.maximum_distinct_slots);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"reasons\":";
    append_array(json, record.reasons, [](std::string& values, TensorSlotClaimReason reason) {
      append_json_string(values, tensor_slot_claim_reason_name(reason));
    });
    json.push_back('}');
  });
  output += ",\"ports\":";
  append_array(output, state.ports, [](std::string& json, const auto& record) {
    json += "{\"direction\":";
    append_number(json, static_cast<std::uint32_t>(record.direction));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"generic_loan_slot_size_bytes\":";
    append_number(json, record.generic_loan_slot_size_bytes);
    json += ",\"generic_loan_supported\":";
    append_bool(json, record.generic_loan_supported);
    json += ",\"id\":";
    append_number(json, record.id.value());
    json += ",\"late_joiner_history_depth\":";
    append_number(json, record.late_joiner_history_depth);
    json += ",\"local_name\":";
    append_json_string(json, record.local_name);
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"owner\":";
    append_number(json, record.owner.value);
    json += ",\"owner_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.owner_kind));
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"queue_depth\":";
    append_number(json, record.queue_depth);
    json += ",\"serialized_payload_capacity_bytes\":";
    append_number(json, record.serialized_payload_capacity_bytes);
    json += ",\"tensor_layout\":";
    append_optional_tensor_layout(json, record.tensor_layout);
    json.push_back('}');
  });
  output += ",\"publication_outcomes\":";
  append_array(output, state.publication_outcomes, [](std::string& json, const auto& record) {
    json += "{\"output\":";
    append_number(json, record.output.value());
    json += ",\"post_submit_assigns_sequence\":";
    append_bool(json, record.post_submit_assigns_sequence);
    json += ",\"post_submit_consumes_authority\":";
    append_bool(json, record.post_submit_consumes_authority);
    json += ",\"post_submit_error\":";
    append_number(json, static_cast<std::uint32_t>(record.post_submit_error));
    json += ",\"post_submit_partial_error\":";
    append_number(json, static_cast<std::uint32_t>(record.post_submit_partial_error));
    json += ",\"pre_submit_assigns_sequence\":";
    append_bool(json, record.pre_submit_assigns_sequence);
    json += ",\"pre_submit_authority\":";
    append_number(json, static_cast<std::uint32_t>(record.pre_submit_authority));
    json += ",\"pre_submit_error\":";
    append_number(json, static_cast<std::uint32_t>(record.pre_submit_error));
    json += ",\"retain_claims_until_retirement\":";
    append_bool(json, record.retain_claims_until_retirement);
    json += ",\"runtime_retry_enabled\":";
    append_bool(json, record.runtime_retry_enabled);
    json += ",\"tensor\":";
    append_bool(json, record.tensor);
    json.push_back('}');
  });
  output += ",\"partition_decisions\":";
  append_array(output, state.partition_decisions, [](std::string& json, const auto& record) {
    json += "{\"decision\":";
    append_number(json, static_cast<std::uint32_t>(record.decision));
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"reasons\":";
    append_array(json, record.reasons, [](std::string& values, auto reason) {
      append_number(values, static_cast<std::uint32_t>(reason));
    });
    json += ",\"vertex\":";
    append_number(json, record.vertex.value);
    json.push_back('}');
  });
  output += ",\"partitioning\":{";
  output += "\"accepted_algorithm\":";
  append_optional_algorithm(output, state.partitioning_plan.accepted_algorithm);
  output += ",\"config_digest\":";
  append_hex(output, state.partitioning_plan.config_digest.value);
  output += ",\"disposition\":";
  append_number(output, static_cast<std::uint32_t>(state.partitioning_plan.disposition));
  output += ",\"interface_version\":";
  append_number(output, state.partitioning_plan.interface_version);
  output += ",\"maturity\":";
  append_optional_maturity(output, state.partitioning_plan.maturity);
  output += ",\"reason\":";
  append_number(output, state.partitioning_plan.reason.value);
  output += ",\"requested_algorithm\":";
  append_optional_algorithm(output, state.partitioning_plan.requested_algorithm);
  output += ",\"selection_source\":";
  append_number(output, static_cast<std::uint32_t>(state.partitioning_plan.selection_source));
  output += "},\"partitions\":";
  append_array(output, state.execution_partitions, [](std::string& json, const auto& record) {
    json += "{\"authored_name\":";
    if (record.authored_name) {
      append_json_string(json, *record.authored_name);
    } else {
      json += "null";
    }
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"lane\":";
    append_number(json, record.execution_lane);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json.push_back('}');
  });
  output += ",\"predicate_inputs\":";
  append_array(output, state.predicate_inputs, [](std::string& json, const auto& record) {
    json += "{\"argument_ordinal\":";
    append_number(json, record.argument_ordinal);
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"capture_time_required\":";
    append_bool(json, record.capture_time_required);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"implicit_driver\":";
    append_bool(json, record.implicit_driver);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"max_age_ns\":";
    append_number(json, record.max_age_ns);
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"predicate_id\":";
    append_json_string(json, record.predicate_id);
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"reuse_max_age_ns\":";
    append_number(json, record.reuse_max_age_ns);
    json += ",\"selection_reason_schema\":";
    append_json_string(json, record.selection_reason_schema);
    json += ",\"selection_state_schema\":";
    append_json_string(json, record.selection_state_schema);
    json += ",\"selector\":";
    append_number(json, static_cast<std::uint32_t>(record.selector));
    json.push_back('}');
  });
  output += ",\"predicate_callables\":";
  append_array(output, state.predicate_callables, [](std::string& json, const auto& record) {
    json += "{\"alignment_bytes\":";
    append_number(json, record.alignment_bytes);
    json += ",\"bounded_call\":";
    append_bool(json, record.bounded_call);
    json += ",\"build_evidence\":";
    append_json_string(json, record.build_evidence);
    json += ",\"capability_evidence_references\":";
    append_array(
        json,
        record.capability_evidence_references,
        [](std::string& values, const std::string& value) { append_json_string(values, value); });
    json += ",\"capture_provenance\":";
    append_json_string(json, record.capture_provenance);
    json += ",\"criticality_constraints\":";
    append_array(
        json, record.criticality_constraints, [](std::string& values, const std::string& value) {
          append_json_string(values, value);
        });
    json += ",\"diagnostic_ids\":";
    append_array(json, record.diagnostic_ids, [](std::string& values, const std::string& value) {
      append_json_string(values, value);
    });
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"evidence_revocation_epoch\":";
    append_number(json, record.evidence_revocation_epoch);
    json += ",\"evidence_validity_epoch\":";
    append_number(json, record.evidence_validity_epoch);
    json += ",\"implementation_fingerprint\":";
    append_json_string(json, record.implementation_fingerprint);
    json += ",\"is_noexcept\":";
    append_bool(json, record.is_noexcept);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"argument_contract\":";
    append_json_string(json, record.argument_contract);
    json += ",\"binding_provenance\":";
    append_json_string(json, record.binding_provenance);
    json += ",\"dispatch_model\":";
    append_json_string(json, record.dispatch_model);
    json += ",\"no_dynamic_allocation\":";
    append_bool(json, record.no_dynamic_allocation);
    json += ",\"owned_lifetime\":";
    append_bool(json, record.owned_lifetime);
    json += ",\"possible_dynamic_allocation\":";
    append_bool(json, record.possible_dynamic_allocation);
    json += ",\"predicate_id\":";
    append_json_string(json, record.predicate_id);
    json += ",\"predicate_key\":";
    append_optional_string(json, record.predicate_key);
    json += ",\"predicate_key_version\":";
    if (record.predicate_key_version) {
      json += "{\"major\":";
      append_number(json, record.predicate_key_version->major);
      json += ",\"minor\":";
      append_number(json, record.predicate_key_version->minor);
      json += ",\"patch\":";
      append_number(json, record.predicate_key_version->patch);
      json.push_back('}');
    } else {
      json += "null";
    }
    json += ",\"receiver_role\":";
    append_json_string(json, record.receiver_role);
    json += ",\"requires_python_gil\":";
    append_bool(json, record.requires_python_gil);
    json += ",\"result_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.result_kind));
    json += ",\"self_rebind\":";
    append_bool(json, record.self_rebind);
    json += ",\"size_bytes\":";
    append_number(json, record.size_bytes);
    json += ",\"toolchain_evidence\":";
    append_json_string(json, record.toolchain_evidence);
    json += ",\"type_erased\":";
    append_bool(json, record.type_erased);
    json.push_back('}');
  });
  output += ",\"predicate_triggers\":";
  append_array(output, state.predicate_triggers, [](std::string& json, const auto& record) {
    json += "{\"candidate_state_is_transactional\":";
    append_bool(json, record.candidate_state_is_transactional);
    json += ",\"clock\":";
    append_number(json, record.clock.value);
    json += ",\"compute_adopts_evaluator_snapshot\":";
    append_bool(json, record.compute_adopts_evaluator_snapshot);
    json += ",\"config_alignment_bytes\":";
    append_number(json, record.config_alignment_bytes);
    json += ",\"config_digest\":";
    append_optional_string(json, record.config_digest);
    json += ",\"config_schema\":";
    append_optional_string(json, record.config_schema);
    json += ",\"config_size_bytes\":";
    append_number(json, record.config_size_bytes);
    json += ",\"config_type_digest\":";
    append_optional_string(json, record.config_type_digest);
    json += ",\"consumes_reserved_candidate_tag\":";
    append_bool(json, record.consumes_reserved_candidate_tag);
    json += ",\"driver\":";
    append_number(json, record.driver.value());
    json += ",\"evaluation_witness_identity\":";
    append_json_string(json, record.evaluation_witness_identity);
    json += ",\"error_witness_identity\":";
    append_json_string(json, record.error_witness_identity);
    json += ",\"evaluator_error_policy\":";
    append_json_string(json, record.evaluator_error_policy);
    json += ",\"firing\":";
    append_number(json, static_cast<std::uint32_t>(record.firing));
    json += ",\"has_config\":";
    append_bool(json, record.has_config);
    json += ",\"graph_generation\":";
    append_number(json, record.graph_generation);
    json += ",\"id\":";
    append_json_string(json, record.id);
    json += ",\"identity\":";
    append_json_string(json, record.identity);
    json += ",\"identity_scope\":";
    append_number(json, static_cast<std::uint32_t>(record.identity_scope));
    json += ",\"initial_truth\":";
    append_bool(json, record.initial_truth);
    json += ",\"missing_or_error_preserves_truth\":";
    append_bool(json, record.missing_or_error_preserves_truth);
    json += ",\"missing_precedence\":";
    append_json_string(json, record.missing_precedence);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase\":";
    append_optional_time_point(json, record.phase);
    json += ",\"replay\":";
    append_number(json, static_cast<std::uint32_t>(record.replay));
    json += ",\"replay_evidence\":";
    append_json_string(json, record.replay_evidence);
    json += ",\"reset_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.reset_policy));
    json += ",\"reset_reasons\":";
    append_array(json, record.reset_reasons, [](std::string& values, const std::string& value) {
      append_json_string(values, value);
    });
    json += ",\"reset_witness_identity\":";
    append_json_string(json, record.reset_witness_identity);
    json += ",\"semantic_version\":";
    if (record.semantic_version) {
      json += "{\"major\":";
      append_number(json, record.semantic_version->major);
      json += ",\"minor\":";
      append_number(json, record.semantic_version->minor);
      json += ",\"patch\":";
      append_number(json, record.semantic_version->patch);
      json.push_back('}');
    } else {
      json += "null";
    }
    json += ",\"source\":";
    append_number(json, static_cast<std::uint32_t>(record.source));
    json += ",\"state_alignment_bytes\":";
    append_number(json, record.state_alignment_bytes);
    json += ",\"state_generation\":";
    append_number(json, record.state_generation);
    json += ",\"state_initialization\":";
    append_json_string(json, record.state_initialization);
    json += ",\"state_schema\":";
    append_json_string(json, record.state_schema);
    json += ",\"state_size_bytes\":";
    append_number(json, record.state_size_bytes);
    json += ",\"state_type_digest\":";
    append_json_string(json, record.state_type_digest);
    json += ",\"safe_default_witness_identity\":";
    append_json_string(json, record.safe_default_witness_identity);
    json += ",\"suppression_witness_identity\":";
    append_json_string(json, record.suppression_witness_identity);
    json += ",\"successful_evaluation_commits_truth\":";
    append_bool(json, record.successful_evaluation_commits_truth);
    json += ",\"transition_witness_identity\":";
    append_json_string(json, record.transition_witness_identity);
    json += ",\"reaction_fire_witness_identity\":";
    append_json_string(json, record.reaction_fire_witness_identity);
    json.push_back('}');
  });
  output += ",\"process_roles\":";
  append_array(output, state.process_roles, [](std::string& json, const auto& record) {
    json += "{\"accepted_isolation\":";
    if (record.accepted_isolation) {
      append_number(json, static_cast<std::uint32_t>(*record.accepted_isolation));
    } else {
      json += "null";
    }
    json += ",\"accepted_launch_group\":";
    append_json_string(json, record.accepted_launch_group);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"require_locality_probe_success\":";
    append_bool(json, record.require_locality_probe_success);
    json += ",\"requested_isolation\":";
    append_number(json, static_cast<std::uint32_t>(record.requested_isolation));
    json += ",\"requested_launch_group\":";
    append_json_string(json, record.requested_launch_group);
    json += ",\"role\":";
    append_json_string(json, record.role);
    json += ",\"runtime_seal_required\":";
    append_bool(json, record.runtime_seal_required);
    json.push_back('}');
  });
  output += ",\"process_selectors\":";
  append_array(output, state.process_selectors, [](std::string& json, const auto& record) {
    json += "{\"authored_value\":";
    append_json_string(json, record.authored_value);
    json += ",\"canonical_value\":";
    append_json_string(json, record.canonical_value);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"external_topic\":";
    if (record.external_topic) {
      append_number(json, record.external_topic->value);
    } else {
      json += "null";
    }
    json += ",\"graph\":";
    if (record.graph) {
      append_number(json, record.graph->value);
    } else {
      json += "null";
    }
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json += ",\"ordinal\":";
    append_number(json, record.ordinal);
    json += ",\"process_role\":";
    append_number(json, record.process_role.value);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"region\":";
    if (record.region) {
      append_number(json, record.region->value);
    } else {
      json += "null";
    }
    json += ",\"vertex\":";
    if (record.vertex) {
      append_number(json, record.vertex->value);
    } else {
      json += "null";
    }
    json.push_back('}');
  });
  output += ",\"vertex_process_placements\":";
  append_array(output, state.vertex_process_placements, [](std::string& json, const auto& record) {
    json += "{\"isolation\":";
    append_number(json, static_cast<std::uint32_t>(record.isolation));
    json += ",\"launch_group\":";
    append_json_string(json, record.launch_group);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"process_role\":";
    if (record.process_role) {
      append_number(json, record.process_role->value);
    } else {
      json += "null";
    }
    json += ",\"runtime_seal_required\":";
    append_bool(json, record.runtime_seal_required);
    json += ",\"vertex\":";
    append_number(json, record.vertex.value);
    json.push_back('}');
  });
  output += ",\"profile\":{";
  output += "\"disposition\":";
  append_number(output, static_cast<std::uint32_t>(state.profile_manifest.disposition));
  output += ",\"derived_capacities\":";
  append_array(
      output, state.profile_manifest.derived_capacities, [](std::string& json, const auto& record) {
        json += "{\"capacity_id\":";
        append_json_string(json, record.capacity_id);
        json += ",\"limit\":";
        append_number(json, record.limit);
        json += ",\"object_id\":";
        append_json_string(json, record.object_id);
        json += ",\"unit\":";
        append_json_string(json, record.unit);
        json.push_back('}');
      });
  output += ",\"effective\":";
  append_optional_profile(output, state.profile_manifest.effective_graph_profile);
  output += ",\"evidence_references\":";
  append_array(output,
               state.profile_manifest.evidence_references,
               [](std::string& json, const auto& record) {
                 json += "{\"id\":";
                 append_json_string(json, record.id);
                 json += ",\"kind\":";
                 append_json_string(json, record.kind);
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"explicit_fallbacks\":";
  append_array(
      output, state.profile_manifest.explicit_fallbacks, [](std::string& json, const auto& record) {
        json += "{\"evidence_id\":";
        append_json_string(json, record.evidence_id);
        json += ",\"fallback_id\":";
        append_json_string(json, record.fallback_id);
        json += ",\"feature_id\":";
        append_json_string(json, record.feature_id);
        json += ",\"reason_id\":";
        append_json_string(json, record.reason_id);
        json.push_back('}');
      });
  output += ",\"feature_dispositions\":";
  append_array(output,
               state.profile_manifest.feature_dispositions,
               [](std::string& json, const auto& record) {
                 json += "{\"disposition\":";
                 append_number(json, static_cast<std::uint32_t>(record.disposition));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"feature_id\":";
                 append_json_string(json, record.feature_id);
                 json += ",\"rule_id\":";
                 append_json_string(json, record.rule_id);
                 json.push_back('}');
               });
  output += ",\"generated_endpoint_names\":";
  append_array(output,
               state.profile_manifest.generated_endpoint_names,
               [](std::string& json, const auto& record) {
                 json += "{\"algorithm\":";
                 append_json_string(json, record.algorithm);
                 json += ",\"disposition\":";
                 append_number(json, static_cast<std::uint32_t>(record.disposition));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"final_name\":";
                 append_json_string(json, record.final_name);
                 json += ",\"source\":";
                 append_number(json, static_cast<std::uint32_t>(record.source));
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"materialization_obligations\":";
  append_array(output,
               state.profile_manifest.materialization_obligations,
               [](std::string& json, const auto& record) {
                 json += "{\"cleanup_stage_mask\":";
                 append_number(json, record.cleanup_stage_mask);
                 json += ",\"complete_before_stage\":";
                 append_number(json, static_cast<std::uint32_t>(record.complete_before_stage));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"max_cleanup_actions\":";
                 append_number(json, record.max_cleanup_actions);
                 json += ",\"object_id\":";
                 append_json_string(json, record.object_id);
                 json += ",\"strategy_id\":";
                 append_json_string(json, record.strategy_id);
                 json.push_back('}');
               });
  output += ",\"policy_digest\":";
  append_sha256(output, state.profile_manifest.profile_policy_id.digest.value);
  output += ",\"policy_major\":";
  append_number(output, state.profile_manifest.profile_policy_id.major);
  output += ",\"policy_minor\":";
  append_number(output, state.profile_manifest.profile_policy_id.minor);
  output += ",\"plan_schema_version\":";
  append_json_string(output, state.profile_manifest.plan_schema_version);
  append_profile_provenance(output, state.profile_manifest, include_provenance);
  output += ",\"regions\":";
  append_array(
      output, state.profile_manifest.effective_regions, [](std::string& json, const auto& record) {
        json += "{\"effective\":";
        if (record.effective_profile) {
          append_json_string(json, profile_token(*record.effective_profile));
        } else {
          json += "null";
        }
        json += ",\"id\":";
        append_number(json, record.region_id.value);
        json += ",\"origin\":";
        append_number(json, static_cast<std::uint32_t>(record.origin));
        json += ",\"requested\":";
        if (record.requested_profile) {
          append_json_string(json, profile_token(*record.requested_profile));
        } else {
          json += "null";
        }
        json.push_back('}');
      });
  output += ",\"requested\":";
  append_optional_profile(output, state.profile_manifest.requested_profile);
  output += ",\"required_capabilities\":";
  append_array(output,
               state.profile_manifest.required_capabilities,
               [](std::string& json, const auto& record) {
                 json += "{\"capability_id\":";
                 append_json_string(json, record.capability_id);
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"runtime_abi_version\":";
  append_json_string(output, state.profile_manifest.runtime_abi_version);
  output += ",\"scope_resolutions\":";
  append_array(
      output, state.profile_manifest.scope_resolutions, [](std::string& json, const auto& record) {
        json += "{\"contributing_regions\":";
        append_array(json, record.contributing_regions, [](std::string& values, RegionId region) {
          append_number(values, region.value);
        });
        json += ",\"effective_profile\":";
        append_json_string(json, profile_token(record.effective_profile));
        json += ",\"object_id\":";
        append_json_string(json, record.object_id);
        json += ",\"object_kind\":";
        append_json_string(json, record.object_kind);
        json.push_back('}');
      });
  output += ",\"target_triple\":";
  append_json_string(output, state.profile_manifest.target_triple);
  output += "},\"readable_triggers\":";
  append_array(output, state.readable_triggers, [](std::string& json, const auto& record) {
    json += "{\"automatic_retry\":";
    append_bool(json, record.automatic_retry);
    json += ",\"bind_at_arm\":";
    append_bool(json, record.bind_at_arm);
    json += ",\"disable_before_stop\":";
    append_bool(json, record.disable_before_stop);
    json += ",\"effective_capacity\":";
    append_number(json, record.effective_capacity);
    json += ",\"enable_after_global_start\":";
    append_bool(json, record.enable_after_global_start);
    json += ",\"error_creates_candidate\":";
    append_bool(json, record.error_creates_candidate);
    json += ",\"fatal_error\":";
    append_number(json, static_cast<std::uint32_t>(record.fatal_error));
    json += ",\"initial_source_generation\":";
    append_number(json, record.initial_source_generation);
    json += ",\"initial_source_sequence\":";
    append_number(json, record.initial_source_sequence);
    json += ",\"level_readable\":";
    append_bool(json, record.level_readable);
    json += ",\"max_outstanding_candidates\":";
    append_number(json, record.max_outstanding_candidates);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"readiness_witness_identity\":";
    append_json_string(json, record.readiness_witness_identity);
    json += ",\"rearm_only_if_still_readable\":";
    append_bool(json, record.rearm_only_if_still_readable);
    json += ",\"restart_advances_generation\":";
    append_bool(json, record.restart_advances_generation);
    json += ",\"source\":";
    append_number(json, source_id_value(record.source));
    json += ",\"source_error_identity\":";
    append_json_string(json, record.source_error_identity);
    json += ",\"stale_generation_identity\":";
    append_json_string(json, record.stale_generation_identity);
    json += ",\"stale_generation_rejected\":";
    append_bool(json, record.stale_generation_rejected);
    json += ",\"unbind_at_release\":";
    append_bool(json, record.unbind_at_release);
    json += ",\"witness_clock\":";
    append_number(json, record.witness_clock.value);
    json.push_back('}');
  });
  output += ",\"readiness\":";
  append_array(output, state.readiness, [](std::string& json, const ReadinessPlanRecord& record) {
    json += "{\"capacity\":";
    append_number(json, record.accepted_capacity);
    json += ",\"declaration_ordinal\":";
    append_number(json, record.declaration_ordinal);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"materialized\":";
    append_bool(json, record.materialized);
    json += ",\"name\":";
    append_json_string(json, record.name);
    json += ",\"owner\":";
    append_number(json, record.owner.value);
    json += ",\"owner_path\":";
    append_json_string(json, record.owner_path);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"referenced\":";
    append_bool(json, record.referenced);
    json += ",\"requested_capacity\":";
    append_number(json, record.requested_capacity);
    json += ",\"requested_clock\":{\"default\":";
    append_bool(json, record.requested_clock.is_default());
    json += ",\"name\":";
    append_json_string(json, record.requested_clock.name());
    json += "},\"requested_sender_capacity\":";
    append_number(json, record.requested_sender_reference_capacity);
    json += ",\"resolved_clock\":";
    append_number(json, record.resolved_clock.value);
    json += ",\"sender_capacity\":";
    append_number(json, record.accepted_sender_reference_capacity);
    json += ",\"source_id\":";
    append_number(json, source_id_value(record.source_id));
    json += ",\"initial_source_generation\":";
    append_number(json, record.initial_source_generation);
    json += ",\"initial_source_sequence\":";
    append_number(json, record.initial_source_sequence);
    json += ",\"disposition_reason\":";
    append_number(json, record.disposition_reason.value);
    json += ",\"wake_requirement\":";
    append_number(json, static_cast<std::uint32_t>(record.wake_requirement));
    json.push_back('}');
  });
  output += ",\"schedule_edges\":";
  append_array(output, state.schedule_edges, [](std::string& json, const auto& record) {
    json += "{\"destination\":";
    append_number(json, record.destination.value);
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"source\":";
    append_number(json, record.source.value);
    json.push_back('}');
  });
  output += ",\"scheduler\":{";
  output += "\"activation_overlap\":";
  append_bool(output, state.scheduler_plan.activation_overlap);
  output += ",\"max_concurrent_activations\":";
  append_number(output, state.scheduler_plan.max_concurrent_activations);
  output += ",\"execution_lane_count\":";
  append_number(output, state.scheduler_plan.execution_lane_count);
  output += ",\"intra_path_parallelism\":";
  append_number(output, state.scheduler_plan.intra_path_parallelism);
  output += ",\"ready_policy\":";
  append_json_string(output, state.scheduler_plan.ready_policy.id);
  output += ",\"ready_policy_version\":";
  append_number(output, state.scheduler_plan.ready_policy.version);
  output += ",\"reason\":";
  append_number(output, state.scheduler_plan.reason.value);
  output += ",\"worker_count\":";
  append_number(output, state.scheduler_plan.worker_count);
  output.push_back('}');
  append_schema_package_provenance(output, state.schema_packages, include_provenance);
  output += ",\"temporal_triggers\":";
  append_array(output, state.temporal_triggers, [](std::string& json, const auto& record) {
    json += "{\"bounded_state_capacity\":";
    append_number(json, record.bounded_state_capacity);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"deadline_ns\":";
    if (record.deadline_ns) {
      append_number(json, *record.deadline_ns);
    } else {
      json += "null";
    }
    json += ",\"declared_inputs\":";
    append_array(json, record.declared_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase\":";
    append_optional_time_point(json, record.phase);
    json += ",\"presence_guaranteed_inputs\":";
    append_array(json, record.presence_guaranteed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"readiness_source\":";
    append_number(json, source_id_value(record.readiness_source));
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"replay_identity\":";
    append_json_string(json, record.replay_identity);
    json += ",\"trigger_clock\":";
    append_number(json, record.trigger_clock.value);
    json += ",\"trigger_input\":";
    append_number(json, record.trigger_input.value());
    json += ",\"violation\":";
    if (record.violation) {
      append_number(json, static_cast<std::uint32_t>(*record.violation));
    } else {
      json += "null";
    }
    json += ",\"witness_identity\":";
    append_json_string(json, record.witness_identity);
    json.push_back('}');
  });
  if (!state.tensor_field_allocations.empty()) {
    output += ",\"tensor_field_allocations\":";
    append_array(output, state.tensor_field_allocations, [](std::string& json, const auto& record) {
      json += "{\"aggregate_byte_capacity\":";
      append_number(json, record.aggregate_byte_capacity);
      json += ",\"allocator_process_group\":";
      append_number(json, record.allocator_process_group.value);
      json += ",\"allocator_root_id\":";
      append_json_string(json, record.allocator_root_id);
      json += ",\"bounds\":";
      append_tensor_allocation_bounds(json, record.bounds);
      json += ",\"control_capacity\":";
      append_number(json, record.control_capacity);
      json += ",\"device\":";
      append_number(json, record.device.value);
      json += ",\"disposition\":";
      append_number(json, static_cast<std::uint32_t>(record.disposition));
      json += ",\"effective_profile\":";
      append_optional_profile(json, record.effective_profile);
      json += ",\"fallback_rejected\":";
      append_bool(json, record.fallback_rejected);
      json += ",\"field_id\":";
      append_number(json, record.field_id);
      json += ",\"layout\":";
      append_optional_tensor_layout(json, record.layout);
      json += ",\"materialize_before\":";
      append_number(json, static_cast<std::uint32_t>(record.materialize_before));
      json += ",\"max_emits_per_compute\":";
      append_number(json, record.max_emits_per_compute);
      json += ",\"max_outstanding_loans\":";
      append_number(json, record.max_outstanding_loans);
      json += ",\"memory_kind\":";
      append_number(json, static_cast<std::uint32_t>(record.memory_kind));
      json += ",\"output\":";
      append_number(json, record.output.value());
      json += ",\"plan_generation\":";
      append_number(json, record.plan_generation);
      json += ",\"pool_id\":";
      append_json_string(json, record.pool_id);
      json += ",\"quarantine_capacity\":";
      append_number(json, record.quarantine_capacity);
      json += ",\"readiness_capacity\":";
      append_number(json, record.readiness_capacity);
      json += ",\"release_capacity\":";
      append_number(json, record.release_capacity);
      json += ",\"retirement_capacity\":";
      append_number(json, record.retirement_capacity);
      json += ",\"run_generation\":";
      append_number(json, record.run_generation);
      json += ",\"slot_count\":";
      append_number(json, record.slot_count);
      json += ",\"tensor_index\":";
      append_number(json, record.tensor_index);
      json.push_back('}');
    });
  }
  output += ",\"tensor_allocations\":";
  append_array(output, state.tensor_allocations, [](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"allocator_process_group\":";
    append_number(json, record.allocator_process_group.value);
    json += ",\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"bounds\":";
    append_tensor_allocation_bounds(json, record.bounds);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"layout\":";
    append_optional_tensor_layout(json, record.layout);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"max_outstanding_loans\":";
    append_number(json, record.max_outstanding_loans);
    json += ",\"memory_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.memory_kind));
    json += ",\"output\":";
    append_number(json, record.output.value());
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json.push_back('}');
  });
  output += ",\"tensor_device_resources\":";
  append_array(output, state.tensor_device_resources, [](std::string& json, const auto& record) {
    json += "{\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"fused_linear_edges\":";
    append_array(json, record.fused_linear_edges, [](std::string& values, EdgeId edge) {
      append_number(values, edge.value);
    });
    json += ",\"gpu_resident_backend_selected\":";
    append_bool(json, record.gpu_resident_backend_selected);
    json += ",\"loan_credit_capacity\":";
    append_number(json, record.loan_credit_capacity);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"per_partition_event\":";
    append_bool(json, record.per_partition_event);
    json += ",\"per_partition_green_context\":";
    append_bool(json, record.per_partition_green_context);
    json += ",\"per_partition_stream\":";
    append_bool(json, record.per_partition_stream);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_count\":";
    append_number(json, record.pool_count);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_capacity\":";
    append_number(json, record.slot_capacity);
    json += ",\"stream_id\":";
    append_json_string(json, record.stream_id);
    json.push_back('}');
  });
  output += ",\"tensor_inline_records\":";
  append_array(output, state.tensor_inline_records, [](std::string& json, const auto& record) {
    json += "{\"accepted_max_inline_logical_bytes\":";
    append_number(json, record.accepted_max_inline_logical_bytes);
    json += ",\"capability_evidence_id\":";
    append_json_string(json, record.capability_evidence_id);
    json += ",\"capability_id\":";
    append_json_string(json, record.capability_id);
    json += ",\"capability_version\":";
    append_number(json, record.capability_version);
    json += ",\"descriptor_arm_materialized\":";
    append_bool(json, record.descriptor_arm_materialized);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_max_inline_logical_bytes\":";
    append_number(json, record.effective_max_inline_logical_bytes);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"fixed_metadata_overhead_bytes\":";
    append_number(json, record.fixed_metadata_overhead_bytes);
    json += ",\"fixed_record_bucket_bytes\":";
    append_number(json, record.fixed_record_bucket_bytes);
    json += ",\"fixed_record_bucket_id\":";
    append_json_string(json, record.fixed_record_bucket_id);
    json += ",\"inline_arm_materialized\":";
    append_bool(json, record.inline_arm_materialized);
    json += ",\"inline_arm_schema_id\":";
    append_json_string(json, record.inline_arm_schema_id);
    json += ",\"inline_discriminator\":";
    append_json_string(json, record.inline_discriminator);
    json += ",\"opaque_descriptor_arm_schema_id\":";
    append_json_string(json, record.opaque_descriptor_arm_schema_id);
    json += ",\"opaque_descriptor_discriminator\":";
    append_json_string(json, record.opaque_descriptor_discriminator);
    json += ",\"profile_max_inline_logical_bytes\":";
    append_number(json, record.profile_max_inline_logical_bytes);
    json += ",\"provider_id\":";
    append_json_string(json, record.provider_id);
    json += ",\"provider_max_fixed_record_bytes\":";
    append_number(json, record.provider_max_fixed_record_bytes);
    json += ",\"provider_max_inline_logical_bytes\":";
    append_number(json, record.provider_max_inline_logical_bytes);
    json += ",\"provider_max_tensor_rank\":";
    append_number(json, record.provider_max_tensor_rank);
    json += ",\"record_format_id\":";
    append_json_string(json, record.record_format_id);
    json += ",\"record_format_version\":";
    append_number(json, record.record_format_version);
    json += ",\"service_family_id\":";
    append_json_string(json, record.service_family_id);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"tensor_interface_pools\":";
  append_array(output, state.tensor_interface_pools, [](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"allocator_process_group\":";
    append_number(json, record.allocator_process_group.value);
    json += ",\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"emit_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.emit_policy));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"exporter_subpool_id\":";
    append_json_string(json, record.exporter_subpool_id);
    json += ",\"graph_port\":";
    append_number(json, record.graph_port.value());
    json += ",\"imported\":";
    append_bool(json, record.imported);
    json += ",\"logical\":";
    append_logical_tensor_requirements(json, record.logical);
    json += ",\"memory_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.memory_kind));
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"publisher_loan_capacity\":";
    append_number(json, record.publisher_loan_capacity);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"tensor_transfer_plans\":";
  append_array(output, state.tensor_transfer_plans, [](std::string& json, const auto& record) {
    json += "{\"delivered_read_only\":";
    append_bool(json, record.delivered_read_only);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"evidence_id\":";
    append_json_string(json, record.evidence_id);
    json += ",\"max_logical_bytes\":";
    append_number(json, record.max_logical_bytes);
    json += ",\"min_logical_bytes\":";
    append_number(json, record.min_logical_bytes);
    json += ",\"realization\":";
    if (record.realization) {
      append_number(json, static_cast<std::uint32_t>(*record.realization));
    } else {
      json += "null";
    }
    json += ",\"requires_exportable_copy_pool\":";
    append_bool(json, record.requires_exportable_copy_pool);
    json += ",\"requires_inline_record\":";
    append_bool(json, record.requires_inline_record);
    json += ",\"source_class\":";
    append_number(json, static_cast<std::uint32_t>(record.source_class));
    json += ",\"source_must_be_exportable\":";
    append_bool(json, record.source_must_be_exportable);
    json += ",\"source_must_be_host_ready\":";
    append_bool(json, record.source_must_be_host_ready);
    json += ",\"source_retained_until_materialization\":";
    append_bool(json, record.source_retained_until_materialization);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"partition_placements\":";
  append_array(output, state.partition_placements, [](std::string& json, const auto& record) {
    json += "{\"accepted\":";
    if (record.accepted) {
      append_thread_placement(json, *record.accepted);
    } else {
      json += "null";
    }
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"execution_lane\":";
    append_number(json, record.execution_lane);
    json += ",\"authored_name\":";
    append_json_string(json, record.authored_name);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"requested\":";
    append_thread_placement(json, record.requested);
    json.push_back('}');
  });
  output += ",\"placement_probes\":";
  append_array(output, state.placement_probes, [](std::string& json, const auto& record) {
    json += "{\"evidence_id\":";
    append_json_string(json, record.evidence_id);
    json += ",\"probe_id\":";
    append_json_string(json, record.probe_id);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"status\":";
    append_number(json, static_cast<std::uint32_t>(record.status));
    json.push_back('}');
  });
  output += ",\"execution_lanes\":";
  append_array(output, state.execution_lanes, [](std::string& json, const auto& record) {
    json += "{\"accepted_placement\":";
    append_thread_placement(json, record.accepted_placement);
    json += ",\"execution_lane\":";
    append_number(json, record.execution_lane);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json.push_back('}');
  });
  output += ",\"window_triggers\":";
  append_array(output, state.window_triggers, [](std::string& json, const auto& record) {
    json += "{\"advance_on_every_anchor\":";
    append_bool(json, record.advance_on_every_anchor);
    json += ",\"anchor\":";
    append_number(json, record.anchor.value());
    json += ",\"bracketed_inputs\":";
    append_array(json, record.bracketed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"close_witness_identity\":";
    append_json_string(json, record.close_witness_identity);
    json += ",\"cancelled_witness_identity\":";
    append_json_string(json, record.cancelled_witness_identity);
    json += ",\"capability_evidence_id\":";
    append_json_string(json, record.capability_evidence_id);
    json += ",\"external_driver_used\":";
    append_bool(json, record.external_driver_used);
    json += ",\"first_anchor_prime_only\":";
    append_bool(json, record.first_anchor_prime_only);
    json += ",\"end_of_stream_witness_identity\":";
    append_json_string(json, record.end_of_stream_witness_identity);
    json += ",\"max_pending_windows\":";
    append_number(json, record.max_pending_windows);
    json += ",\"max_wait_ns\":";
    append_number(json, record.max_wait_ns);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"open_witness_identity\":";
    append_json_string(json, record.open_witness_identity);
    json += ",\"pending_limit_witness_identity\":";
    append_json_string(json, record.pending_limit_witness_identity);
    json += ",\"presence_guaranteed_inputs\":";
    append_array(json, record.presence_guaranteed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"primed_witness_identity\":";
    append_json_string(json, record.primed_witness_identity);
    json += ",\"producer_epoch_identity_schema\":";
    append_json_string(json, record.producer_epoch_identity_schema);
    json += ",\"reset_on_producer_epoch\":";
    append_bool(json, record.reset_on_producer_epoch);
    json += ",\"reset_on_timestamp_regression\":";
    append_bool(json, record.reset_on_timestamp_regression);
    json += ",\"timeout_witness_identity\":";
    append_json_string(json, record.timeout_witness_identity);
    json += ",\"timestamp_regression_witness_identity\":";
    append_json_string(json, record.timestamp_regression_witness_identity);
    json += ",\"wait_clock\":";
    append_number(json, record.wait_clock.value);
    json.push_back('}');
  });
  output += ",\"schema\":";
  append_json_string(output, kPlanSchema);
  output.push_back('}');

  return output;
}

expected<void, Error> finalize_plan_artifact(ExecutionPlanState& state) {
  auto semantic = plan_artifact_json(state, false);
  if (!semantic) {
    return make_unexpected(std::move(semantic).error());
  }
  state.digest = digest_bytes<PlanDigest>(semantic.value());

  auto complete = plan_artifact_json(state, true);
  if (!complete) {
    return make_unexpected(std::move(complete).error());
  }
  state.json = std::move(complete).value();
  return {};
}

}  // namespace holoscan::detail
