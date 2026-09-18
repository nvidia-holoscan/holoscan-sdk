/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/reflection_generated.h>
#include <flatbuffers/verifier.h>

#include "../../common/schema_id_test_utils.hpp"
#include "collision_a_generated.h"
#include "collision_b_generated.h"
#include "flatbuffer_schema_evolution_v2_traits.hpp"
#include "holoscan/core/schema_identity.hpp"
#include "holoscan/schema/schema_package.hpp"

namespace {

// The generated roots need distinct C++ namespaces to coexist in this test. Their independent
// Holoscan metadata intentionally claims the same external schema family.
template <typename Root>
struct CollisionFixtureFamilyClaim;

template <>
struct CollisionFixtureFamilyClaim<schema_collision_a::Frame> {
  static constexpr std::string_view canonical_name = "com.acme.Frame";
  static constexpr std::uint32_t compatibility_epoch = 1U;
};

template <>
struct CollisionFixtureFamilyClaim<schema_collision_b::Frame> {
  static constexpr std::string_view canonical_name = "com.acme.Frame";
  static constexpr std::uint32_t compatibility_epoch = 1U;
};

template <typename Root>
constexpr holoscan::SchemaId collision_fixture_family_id() {
  using Claim = CollisionFixtureFamilyClaim<Root>;
  return holoscan::derive_schema_family_id(Claim::canonical_name, Claim::compatibility_epoch);
}

constexpr auto kV2FamilyIdentity = holoscan::schema_family_identity<schema_evolution::FrameT>();
constexpr auto kV2Identity = holoscan::schema_identity<schema_evolution::FrameT>();
constexpr holoscan::SchemaId kV2ExpectedFamilyId =
    holoscan::test::schema_id_from_hex("fdee7efad74013f8dc3a79926278eec6");
static_assert(kV2FamilyIdentity.canonical_name == schema_evolution::Frame::GetFullyQualifiedName());
static_assert(kV2FamilyIdentity.compatibility_epoch == 3U);
static_assert(kV2FamilyIdentity.id == kV2ExpectedFamilyId);

std::vector<std::byte> load_bytes(const char* path) {
  std::ifstream input(path, std::ios::binary);
  const std::vector<char> chars{std::istreambuf_iterator<char>{input},
                                std::istreambuf_iterator<char>{}};
  std::vector<std::byte> bytes;
  bytes.reserve(chars.size());
  for (const char value : chars) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(value)));
  }
  return bytes;
}

bool require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

const reflection::Schema* reflection_schema(std::span<const std::byte> bytes) {
  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  if (!reflection::VerifySchemaBuffer(verifier)) {
    return nullptr;
  }
  return reflection::GetSchema(bytes.data());
}

std::optional<holoscan::FlatBufferFileIdentifier> file_identifier(
    const reflection::Schema* schema) {
  if (schema == nullptr || schema->file_ident() == nullptr || schema->file_ident()->size() != 4U) {
    return std::nullopt;
  }
  const auto* identifier = schema->file_ident();
  return holoscan::FlatBufferFileIdentifier{
      .value = {identifier->Get(0U), identifier->Get(1U), identifier->Get(2U), identifier->Get(3U)},
  };
}

const reflection::Field* field_by_id(const reflection::Object* object, std::uint16_t id) {
  if (object == nullptr || object->fields() == nullptr) {
    return nullptr;
  }
  for (const reflection::Field* field : *object->fields()) {
    if (field->id() == id) {
      return field;
    }
  }
  return nullptr;
}

bool required_field_policy_accepts(const reflection::Schema* baseline,
                                   const reflection::Schema* candidate) {
  if (baseline == nullptr || candidate == nullptr) {
    return false;
  }
  const reflection::Object* baseline_frame =
      baseline->objects()->LookupByKey("schema_evolution.Frame");
  const reflection::Object* candidate_frame =
      candidate->objects()->LookupByKey("schema_evolution.Frame");
  const reflection::Field* baseline_label = field_by_id(baseline_frame, 1U);
  const reflection::Field* candidate_label = field_by_id(candidate_frame, 1U);
  return baseline_label != nullptr && candidate_label != nullptr &&
         baseline_label->required() == candidate_label->required();
}

std::vector<std::uint8_t> make_collision_a_buffer() {
  flatbuffers::FlatBufferBuilder builder;
  const auto label = builder.CreateString("same-shape");
  const auto frame = schema_collision_a::CreateFrame(builder, 17U, label);
  schema_collision_a::FinishFrameBuffer(builder, frame);
  return std::vector<std::uint8_t>(builder.GetBufferPointer(),
                                   builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace

// REQ: SCHEMA-001
// REQ: SCHEMA-002
// REQ: SCHEMA-003
int main() {
  const std::vector<std::byte> v1_bfbs = load_bytes(EVOLUTION_V1_BFBS);
  const std::vector<std::byte> renamed_bfbs = load_bytes(EVOLUTION_RENAMED_BFBS);
  const std::vector<std::byte> required_bfbs = load_bytes(EVOLUTION_REQUIRED_BFBS);
  if (!require(!v1_bfbs.empty() && !renamed_bfbs.empty() && !required_bfbs.empty(),
               "failed to load schema characterization artifacts")) {
    return 1;
  }

  const auto& v2_package = holoscan::schema_package<schema_evolution::FrameT>();
  const reflection::Schema* v1_schema = reflection_schema(v1_bfbs);
  const auto v1_file_identifier = file_identifier(v1_schema);
  constexpr auto v1_family = holoscan::make_schema_family_identity("schema_evolution.Frame", 3U);
  const holoscan::SchemaIdentity v1_identity{
      .canonical_name = v1_family.canonical_name,
      .compatibility_epoch = v1_family.compatibility_epoch,
      .family_id = v1_family.id,
      .file_identifier = v1_file_identifier.value_or(holoscan::FlatBufferFileIdentifier{}),
  };
  const auto attachment = holoscan::validate_schema_package_for_attach(v1_identity, v2_package);
  if (!require(v1_file_identifier.has_value(),
               "V1 BFBS does not carry a four-byte file identifier") ||
      !require(*v1_file_identifier == kV2Identity.file_identifier,
               "compatible V1/V2 schemas do not carry the same file identifier") ||
      !require(v1_family == kV2FamilyIdentity,
               "compatible V1/V2 evolution changed the v2 family identity") ||
      !require(v1_identity == v2_package.identity(),
               "legal V1/V2 evolution changed stable attachment identity") ||
      !require(attachment.has_value(), "stable family attachment rejected legal V1/V2 evolution")) {
    return 1;
  }

  const reflection::Schema* renamed_schema = reflection_schema(renamed_bfbs);
  const reflection::Object* v1_frame =
      v1_schema == nullptr ? nullptr : v1_schema->objects()->LookupByKey("schema_evolution.Frame");
  const reflection::Object* renamed_frame =
      renamed_schema == nullptr ? nullptr
                                : renamed_schema->objects()->LookupByKey("schema_evolution.Frame");
  const reflection::Field* v1_label = field_by_id(v1_frame, 1U);
  const reflection::Field* renamed_title = field_by_id(renamed_frame, 1U);
  if (!require(v1_label != nullptr && renamed_title != nullptr &&
                   v1_label->name()->string_view() == "label" &&
                   renamed_title->name()->string_view() == "title" &&
                   v1_label->id() == renamed_title->id() &&
                   v1_label->offset() == renamed_title->offset(),
               "field rename did not retain binary field identity")) {
    return 1;
  }

  const reflection::Schema* required_schema = reflection_schema(required_bfbs);
  if (!require(!required_field_policy_accepts(v1_schema, required_schema),
               "required-field policy unexpectedly accepted an optional-to-required change")) {
    return 1;
  }

  const std::vector<std::uint8_t> collision_a = make_collision_a_buffer();
  flatbuffers::Verifier own_identifier_verifier(collision_a.data(), collision_a.size());
  flatbuffers::Verifier structural_verifier(collision_a.data(), collision_a.size());
  flatbuffers::Verifier identifier_verifier(collision_a.data(), collision_a.size());
  constexpr holoscan::SchemaId claimed_a = collision_fixture_family_id<schema_collision_a::Frame>();
  constexpr holoscan::SchemaId claimed_b = collision_fixture_family_id<schema_collision_b::Frame>();
  if (!require(claimed_a == claimed_b, "test family-name collision was not reproduced") ||
      !require(schema_collision_a::VerifyFrameBuffer(own_identifier_verifier),
               "valid collision fixture did not carry its own file identifier") ||
      !require(structural_verifier.VerifyBuffer<schema_collision_b::Frame>(nullptr),
               "structurally compatible wrong root did not pass identifier-free verification") ||
      !require(!identifier_verifier.VerifyBuffer<schema_collision_b::Frame>("CLB1"),
               "wrong file identifier did not reject before typed exposure")) {
    return 1;
  }

  return 0;
}
