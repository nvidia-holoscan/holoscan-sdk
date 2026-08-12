/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/params/persistence.hpp>

#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {
namespace {

// RAII helper that creates a uniquely-named temp file on construction and removes it (plus any
// .holoscan.tmp sidecar) on destruction. Keeps IO tests hermetic and filesystem-clean.
class TemporaryArchive {
 public:
  TemporaryArchive() {
    path_ = std::filesystem::temp_directory_path() /
            ("holoscan-param-archive-" + std::to_string(next_++) + ".json");
  }
  ~TemporaryArchive() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_.string() + ".holoscan.tmp", ignored);
  }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  inline static std::uint64_t next_{};
  std::filesystem::path path_;
};

// Builds a snapshot with one operator record ("scalar-source") covering all 11 supported scalar
// types: bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64. Used by
// ScalarTypeIdentityRestoresEveryExactSupportedWidth to verify exact bit-width round-trips.
[[nodiscard]] ParamSnapshot scalar_snapshot() {
  struct ScalarCase {
    std::string_view name;
    std::string_view type;
    ParamLogicalType logical_type;
    ParamValue value;
  };
  const std::array values{
      ScalarCase{"bool", "holoscan.bool", ParamLogicalType::kBool, ParamValue::from(true)},
      ScalarCase{
          "i8", "holoscan.i8", ParamLogicalType::kSignedInteger, ParamValue::from(std::int8_t{-7})},
      ScalarCase{"i16",
                 "holoscan.i16",
                 ParamLogicalType::kSignedInteger,
                 ParamValue::from(std::int16_t{-513})},
      ScalarCase{"i32",
                 "holoscan.i32",
                 ParamLogicalType::kSignedInteger,
                 ParamValue::from(std::int32_t{-65'537})},
      ScalarCase{"i64",
                 "holoscan.i64",
                 ParamLogicalType::kSignedInteger,
                 ParamValue::from(std::int64_t{-4'294'967'297})},
      ScalarCase{"u8",
                 "holoscan.u8",
                 ParamLogicalType::kUnsignedInteger,
                 ParamValue::from(std::uint8_t{251})},
      ScalarCase{"u16",
                 "holoscan.u16",
                 ParamLogicalType::kUnsignedInteger,
                 ParamValue::from(std::uint16_t{65'000})},
      ScalarCase{"u32",
                 "holoscan.u32",
                 ParamLogicalType::kUnsignedInteger,
                 ParamValue::from(std::uint32_t{4'000'000'001U})},
      ScalarCase{"u64",
                 "holoscan.u64",
                 ParamLogicalType::kUnsignedInteger,
                 ParamValue::from(std::uint64_t{9'000'000'000'000'000'001ULL})},
      ScalarCase{"f32", "holoscan.f32", ParamLogicalType::kFloatingPoint, ParamValue::from(1.25F)},
      ScalarCase{"f64", "holoscan.f64", ParamLogicalType::kFloatingPoint, ParamValue::from(-9.5)},
  };

  std::vector<ParamSnapshotFieldRecord> fields;
  fields.reserve(values.size());
  ParamFieldId id = 1U;
  for (const auto& item : values) {
    auto path = ParamPath::parse(item.name);
    if (!path) {
      return {};
    }
    ParamFieldTypeShape shape{};
    shape.type = item.logical_type;
    shape.has_stable_shape = true;
    shape.bit_width = item.value.bit_width();
    fields.push_back(ParamSnapshotFieldRecord{
        .address = ParamAddress{.field = id++},
        .path = std::move(*path),
        .logical_type = item.logical_type,
        .type =
            ParamSnapshotTypeIdentity{.id = std::string{item.type}, .version = 1U, .shape = shape},
        .codec =
            ParamSnapshotCodecIdentity{.id = "holoscan.config." + std::string{item.type.substr(9U)},
                                       .version = 1U},
        .projection = ParamProjectionStatus::kIncluded,
        .value = item.value,
    });
  }
  auto path = OperatorPath::parse("scalar-source");
  if (!path) {
    return {};
  }
  ParamSnapshotOperatorRecord record{
      .path = std::move(*path),
      .effective_profile = Profile::kStandard,
      .schema_availability = ParamSchemaAvailability::kDeclared,
      .schema_id = "holoscan.test.scalar-archive",
      .schema_version = 1U,
      .schema_coverage = ParamSchemaCoverage::kComplete,
      .schema_proof = ParamSchemaProof::kProven,
      .generation = 1U,
      .accepted_update_sequence = 1U,
      .published_update_sequence = 1U,
      .state = ParamUpdateState::kPublished,
      .filter = ParamSnapshotFilter::kPolicyDefault,
      .value_coverage = ParamSnapshotValueCoverage::kAllDeclaredFields,
      .fields = std::move(fields),
  };
  std::vector<ParamSnapshotOperatorRecord> operators;
  operators.push_back(std::move(record));
  return holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      ParamSnapshotScope::root(),
      ParamUpdateSequence{1U},
      ParamUpdateSequence{1U},
      std::move(operators));
}

// Builds a snapshot with 5 operators each carrying ~3.5 MB of schema_id metadata (total >17 MB).
// Exceeds the default 16 MB output ceiling; used by SaveHonorsConfiguredOutputBeyondPublicInput-
// Ceiling to verify that raising the limit to 32 MB is honoured.
[[nodiscard]] ParamSnapshot large_metadata_snapshot() {
  constexpr std::size_t kMetadataBytesPerOperator = 3U * 1024U * 1024U + 512U * 1024U;
  std::vector<ParamSnapshotOperatorRecord> operators;
  operators.reserve(5U);
  for (std::uint32_t index = 0; index < 5U; ++index) {
    auto path = OperatorPath::parse("large-" + std::to_string(index));
    if (!path) {
      return {};
    }
    operators.push_back(ParamSnapshotOperatorRecord{
        .path = std::move(*path),
        .effective_profile = Profile::kStandard,
        .schema_availability = ParamSchemaAvailability::kDeclared,
        .schema_id = std::string(kMetadataBytesPerOperator, static_cast<char>('a' + index)),
        .schema_version = 1U,
        .schema_coverage = ParamSchemaCoverage::kComplete,
        .schema_proof = ParamSchemaProof::kProven,
        .generation = 1U,
        .accepted_update_sequence = 1U,
        .published_update_sequence = 1U,
        .state = ParamUpdateState::kPublished,
        .filter = ParamSnapshotFilter::kPolicyDefault,
        .value_coverage = ParamSnapshotValueCoverage::kAllDeclaredFields,
    });
  }
  return holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      ParamSnapshotScope::root(),
      ParamUpdateSequence{1U},
      ParamUpdateSequence{1U},
      std::move(operators));
}

// REQ: PARAM-025
// REQ: PARAM-026
TEST(ParameterArchive, EmptySnapshotRoundTripsThroughCanonicalEnvelope) {
  // Key setup: save an empty snapshot with require_complete=true, then verify raw file content
  // matches the canonical single-line JSON envelope structure.
  const TemporaryArchive archive;
  const ParamSnapshot original;
  auto saved = save(original, archive.path(), {.require_complete = true});
  ASSERT_TRUE(saved.has_value());
  EXPECT_GT(saved->written_source_digest.byte_size, 0U);
  EXPECT_TRUE(saved->complete_for_normal_persistence);

  std::ifstream stream{archive.path(), std::ios::binary};
  const std::string text{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
  EXPECT_TRUE(text.starts_with(
      "{\"format\":\"holoscan.param-snapshot\",\"version\":1,\"profile\":\"standard\","));
  EXPECT_NE(text.find("\"source\":"), std::string::npos);
  EXPECT_NE(text.find("\"scope\":"), std::string::npos);
  EXPECT_NE(text.find("\"capture\":"), std::string::npos);
  EXPECT_NE(text.find("\"frontier\":"), std::string::npos);
  EXPECT_NE(text.find("\"operators\":"), std::string::npos);
  EXPECT_NE(text.find("\"extensions\":"), std::string::npos);
  EXPECT_NE(text.find("\"digest\":\"sha256:"), std::string::npos);
  EXPECT_EQ(text.find('\n'), std::string::npos);

  auto loaded = load(archive.path());
  ASSERT_TRUE(loaded.has_value());
  EXPECT_TRUE(loaded->operators().empty());
  EXPECT_EQ(loaded->scope().kind(), ParamSnapshotScopeKind::kRoot);
  EXPECT_EQ(loaded->plan_digest(), original.plan_digest());
  const auto identity = archive_identity(*loaded);
  ASSERT_TRUE(identity.has_value());
  if (identity.has_value()) {
    EXPECT_EQ(identity->raw_source_digest, saved->written_source_digest);
  }

  // Key assertion: slicing strips archive identity (only the top-level loaded snapshot has it).
  auto sliced = loaded->slice(ParamSnapshotScope::root());
  ASSERT_TRUE(sliced.has_value());
  EXPECT_FALSE(archive_identity(*sliced).has_value());
}

// REQ: PARAM-025
TEST(ParameterArchive, SaveHonorsConfiguredOutputBeyondPublicInputCeiling) {
  const TemporaryArchive archive;
  ParamSaveOptions options;
  options.io.output_limits.max_output_bytes = 32U * 1024U * 1024U;

  auto saved = save(large_metadata_snapshot(), archive.path(), options);

  ASSERT_TRUE(saved.has_value()) << saved.error().detail().token();
  EXPECT_GT(saved->written_source_digest.byte_size, 16U * 1024U * 1024U);
  EXPECT_EQ(saved->written_source_digest.byte_size, std::filesystem::file_size(archive.path()));
}

// REQ: PARAM-025
TEST(ParameterArchive, ScalarTypeIdentityRestoresEveryExactSupportedWidth) {
  const TemporaryArchive archive;
  const ParamSnapshot original = scalar_snapshot();
  ASSERT_EQ(original.operators().size(), 1U);
  auto saved = save(original, archive.path(), {.require_complete = true});
  ASSERT_TRUE(saved.has_value());

  auto loaded = load(archive.path());
  ASSERT_TRUE(loaded.has_value());
  ASSERT_EQ(loaded->operators().size(), 1U);
  const auto original_fields = original.operators().front().fields;
  const auto loaded_fields = loaded->operators().front().fields;
  ASSERT_EQ(loaded_fields.size(), original_fields.size());
  for (std::size_t index = 0; index < original_fields.size(); ++index) {
    const auto& original_value = original_fields[index].value;
    const auto& loaded_value = loaded_fields[index].value;
    ASSERT_TRUE(original_value.has_value());
    ASSERT_TRUE(loaded_value.has_value());
    EXPECT_EQ(loaded_fields[index].type.id, original_fields[index].type.id);
    EXPECT_EQ(loaded_fields[index].type.shape.bit_width,
              original_fields[index].type.shape.bit_width);
    if (loaded_value.has_value() && original_value.has_value()) {
      EXPECT_EQ(*loaded_value, *original_value);
    }
  }
}

// REQ: PARAM-025
TEST(ParameterArchive, DetectsSemanticDigestCorruption) {
  // Key setup: locate the first hex byte of the sha256 digest field in the saved JSON and
  // flip it in-place, leaving every other byte intact.
  const TemporaryArchive archive;
  ASSERT_TRUE(save(ParamSnapshot{}, archive.path()).has_value());
  std::fstream stream{archive.path(), std::ios::in | std::ios::out | std::ios::binary};
  ASSERT_TRUE(stream.good());
  std::string text{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
  const auto digest = text.find("\"digest\":\"sha256:");
  ASSERT_NE(digest, std::string::npos);
  const auto byte = digest + std::string_view{"\"digest\":\"sha256:"}.size();
  text[byte] = text[byte] == '0' ? '1' : '0';
  stream.clear();
  stream.seekp(0);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  stream.close();

  // Key assertion: load must fail with kDigestMismatch even for a single-byte change.
  auto loaded = load(archive.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code(), ParamArchiveErrorCode::kDigestMismatch);
}

// REQ: PARAM-025
// Whitespace is injected after the opening '{' and after the first ',' of the canonical JSON.
// The semantic digest (content-addressed) must not change, but the raw-source digest must differ
// because the byte sequence changed.
TEST(ParameterArchive, AcceptsWhitespaceWithoutChangingSemanticIdentity) {
  const TemporaryArchive canonical;
  auto saved = save(ParamSnapshot{}, canonical.path());
  ASSERT_TRUE(saved.has_value());
  auto loaded = load(canonical.path());
  ASSERT_TRUE(loaded.has_value());
  const auto canonical_identity = archive_identity(*loaded);
  ASSERT_TRUE(canonical_identity.has_value());

  std::ifstream input{canonical.path(), std::ios::binary};
  std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  text.insert(1U, "\n  ");
  const auto comma = text.find(',');
  ASSERT_NE(comma, std::string::npos);
  text.insert(comma + 1U, "\n  ");
  std::ofstream output{canonical.path(), std::ios::binary | std::ios::trunc};
  output << text;
  output.close();

  auto whitespace = load(canonical.path());
  ASSERT_TRUE(whitespace.has_value());
  const auto whitespace_identity = archive_identity(*whitespace);
  ASSERT_TRUE(whitespace_identity.has_value());
  // Key assertion: archive_digest (semantic) is identical; raw_source_digest differs.
  if (whitespace_identity.has_value() && canonical_identity.has_value()) {
    EXPECT_EQ(whitespace_identity->archive_digest, canonical_identity->archive_digest);
    EXPECT_NE(whitespace_identity->raw_source_digest, canonical_identity->raw_source_digest);
  }
}

// REQ: PARAM-025
TEST(ParameterArchive, RejectsUnknownTopLevelMember) {
  const TemporaryArchive archive;
  ASSERT_TRUE(save(ParamSnapshot{}, archive.path()).has_value());
  std::ifstream input{archive.path(), std::ios::binary};
  std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  text.insert(1U, "\"unknown\":null,");
  std::ofstream output{archive.path(), std::ios::binary | std::ios::trunc};
  output << text;
  output.close();

  auto loaded = load(archive.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code(), ParamArchiveErrorCode::kInvalidArchive);
}

// REQ: PARAM-025
TEST(ParameterArchive, RejectsNonArchiveInput) {
  const TemporaryArchive archive;
  std::ofstream{archive.path()} << "not an archive";
  auto loaded = load(archive.path());
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code(), ParamArchiveErrorCode::kInvalidArchive);
}

// REQ: PARAM-025
TEST(ParameterArchive, MissingSourceReportsPersistenceFailure) {
  const TemporaryArchive archive;

  auto loaded = load(archive.path());

  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().code(), ParamArchiveErrorCode::kPersistenceFailure);
  EXPECT_EQ(loaded.error().detail().token(), "PARAM_ARCHIVE_LOAD_FAILED");
}

}  // namespace
}  // namespace holoscan::params
