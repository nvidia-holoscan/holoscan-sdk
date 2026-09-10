/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
/// @brief Pin the wire identity of the sensor payload schemas this library took over from Core.
///
/// image.fbs and imu.fbs moved out of public/schemas and into src/sensor_io/schemas. That was
/// meant to be a source-tree change and nothing more: the same bytes on the wire, under the same
/// names, with the same digests, readable by anything that recorded a payload before the move.
///
/// Nothing about the build enforces that on its own. flatc writes the filename of every
/// declaration file into the binary schema, spelled relative to the root schema's own directory,
/// and the digest is taken over exactly those bytes. Compiling image.fbs from its new home while
/// reaching header.fbs through an include path pointing back at Core records
/// "//../../../schemas/header.fbs" where it used to record "//header.fbs" -- different bytes,
/// different digest, different schema id, and a recorded Image that no longer matches the schema
/// claiming to describe it. Every compile-time check still passes. The first sign of trouble is a
/// subscriber rejecting a message.
///
/// src/sensor_io/schemas/CMakeLists.txt avoids that by staging the included Core schemas flat
/// beside these two before codegen. The tests below are what makes that staging load-bearing
/// rather than incidental: remove it and the digests move, and these fail with the old and new
/// values side by side instead of shipping quietly.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/schema_identity.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_traits.hpp>

namespace holoscan {
namespace {

// SchemaIdentity has no operator==; the runtime compares it field by field through an internal
// helper, so do the same here rather than reach into detail from a test.
void expect_same_identity(const SchemaIdentity& actual, const SchemaIdentity& expected) {
  EXPECT_EQ(actual.canonical_name, expected.canonical_name);
  EXPECT_EQ(actual.version, expected.version);
  EXPECT_EQ(actual.schema_sha256, expected.schema_sha256);
  EXPECT_EQ(actual.id, expected.id);
}

/// @brief Render a digest as lowercase hex so a mismatch prints the old and new values.
///
/// Comparing the raw arrays would report only that two 32-byte blobs differ, which for a value
/// whose whole job is to be transcribed into a header is the wrong failure: the person reading it
/// needs the new digest, not the news that it moved.
[[nodiscard]] std::string to_hex(std::span<const std::byte> bytes) {
  static constexpr std::string_view kDigits{"0123456789abcdef"};
  std::string text;
  text.reserve(bytes.size() * 2U);
  for (const std::byte value : bytes) {
    const auto byte = static_cast<std::uint8_t>(value);
    text.push_back(kDigits[byte >> 4U]);
    text.push_back(kDigits[byte & 0x0FU]);
  }
  return text;
}

/// @brief One expected include, spelled as a literal rather than read back from the header.
struct ExpectedInclude {
  std::string_view logical_path{};    ///< Exact logical path the evidence must record.
  std::string_view content_sha256{};  ///< Exact source digest, lowercase hex.
};

// The two closures below are transcribed literals on purpose. Reading them from
// schema::detail::kImageIncludes would assert that an array equals itself, which is what the
// preceding formatting-only loop amounted to. Spelled out, a change to the arrays or to any of the
// four hand-maintained source digests in flatbuffer_schema_package.hpp fails here and has to be
// acknowledged, which is the whole point: those digests are recorded evidence rather than a
// checkable fact, because the .fbs sources do not exist at run time.
//
// The order is the closure order, not the include-statement order. image.fbs includes tensor.fbs
// and header.fbs, and header.fbs pulls in time.fbs; imu.fbs includes geometry.fbs and header.fbs.
constexpr std::array<ExpectedInclude, 3U> kExpectedImageIncludes{
    ExpectedInclude{
        .logical_path = "tensor.fbs",
        .content_sha256 = "1f6a3fb98deab8fe9d3d2c47f4e8c44e3b1b4b2c011f5472cecaf88df607bf76"},
    ExpectedInclude{
        .logical_path = "header.fbs",
        .content_sha256 = "85054752451442caef375d7507000ed84ff0ca78c65ec736d12411205380496e"},
    ExpectedInclude{
        .logical_path = "time.fbs",
        .content_sha256 = "51d9cc42ca6d3ca4589eac955767c9ea772512128c72a2ec299254bc87f5855a"},
};
constexpr std::array<ExpectedInclude, 3U> kExpectedImuIncludes{
    ExpectedInclude{
        .logical_path = "geometry.fbs",
        .content_sha256 = "7b316a005dbd64a03c4e8c42c2e6392e3364fe68cba57603133f89f486fafe3e"},
    ExpectedInclude{
        .logical_path = "header.fbs",
        .content_sha256 = "85054752451442caef375d7507000ed84ff0ca78c65ec736d12411205380496e"},
    ExpectedInclude{
        .logical_path = "time.fbs",
        .content_sha256 = "51d9cc42ca6d3ca4589eac955767c9ea772512128c72a2ec299254bc87f5855a"},
};

// Domain-separated digest over the ordered closure, defined at
// public/include/holoscan/schema/schema_package.hpp. This is the single value a plan record carries
// to stand for the whole closure, so pinning it catches a reordering that the per-entry assertions
// above would also catch and, more usefully, a change to the digest preimage itself, which they
// would not.
constexpr std::string_view kExpectedImageClosureSha256{
    "e24240ba5813fca984e11436ce8fa84710572f4445808f707e53c107065076c0"};
constexpr std::string_view kExpectedImuClosureSha256{
    "851f7e281c9e9cd49998273f0037b7c226cac8fc02a91e28dd2f905cf29fa056"};

/// @brief Assert a package's include closure exactly: count, order, paths, digests, and closure.
/// @param evidence Package evidence under test.
/// @param expected Ordered expected closure.
/// @param expected_closure_sha256 Expected ordered-closure digest, lowercase hex.
void expect_exact_include_closure(const SchemaPackageEvidence& evidence,
                                  std::span<const ExpectedInclude> expected,
                                  std::string_view expected_closure_sha256) {
  // Proof that no host path survived codegen, which is the precondition for the paths below being
  // comparable at all.
  EXPECT_TRUE(evidence.host_paths_stripped);

  // Count first and as an assertion: a closure that gained or lost an entry makes every index
  // comparison below misleading, and a short read would be undefined rather than a failure.
  ASSERT_EQ(evidence.includes.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    // Exact paths subsume the weaker check that no path carries a directory component: a closure
    // resolved outside the staging directory would record "../../../schemas/header.fbs" here.
    EXPECT_EQ(evidence.includes[index].logical_path, expected[index].logical_path);
    EXPECT_EQ(to_hex(evidence.includes[index].content_sha256), expected[index].content_sha256);
  }

  EXPECT_EQ(to_hex(schema_include_closure_sha256(evidence.includes)), expected_closure_sha256);
}

// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, ImagePackageStillMatchesTheDigestItShippedWith) {
  const auto& package = schema_package<schema::Image>();
  const auto identity = schema_identity<schema::Image>();

  auto valid = validate_schema_package(package);
  ASSERT_TRUE(valid) << runtime_error_message(valid.error());

  // The name is unchanged because a FlatBuffers namespace is part of the wire identity. Ownership
  // moved to holoscan::sensor_io; the format did not, so this is still holoscan.schema.Image.
  EXPECT_EQ(package.identity().canonical_name, "holoscan.schema.Image");
  EXPECT_EQ(package.evidence().root_logical_path, "image.fbs");

  // The load-bearing assertion. schema_artifact_sha256 hashes the embedded binary schema the
  // codegen actually produced; identity.schema_sha256 is the constant recorded in
  // sensor_schema_traits.hpp before the move. They agree only if codegen saw the include layout it
  // used to see.
  EXPECT_EQ(package.identity().schema_sha256, schema_artifact_sha256(package.binary_schema()));
  EXPECT_EQ(package.identity().id, derive_schema_id(identity));
}

// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, ImuPackageStillMatchesTheDigestItShippedWith) {
  const auto& package = schema_package<schema::Imu>();
  const auto identity = schema_identity<schema::Imu>();

  auto valid = validate_schema_package(package);
  ASSERT_TRUE(valid) << runtime_error_message(valid.error());

  EXPECT_EQ(package.identity().canonical_name, "holoscan.schema.Imu");
  EXPECT_EQ(package.evidence().root_logical_path, "imu.fbs");
  EXPECT_EQ(package.identity().schema_sha256, schema_artifact_sha256(package.binary_schema()));
  EXPECT_EQ(package.identity().id, derive_schema_id(identity));
}

// The owning native types are what an operator names on a port, and each carries its own identity
// specialization. Those have to resolve to the same schema as the generated root, or a producer
// and a consumer of the same message can disagree about what they are exchanging.
// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, OwningNativeTypesShareTheirRootIdentity) {
  expect_same_identity(schema_identity<schema::ImageT>(), schema_identity<schema::Image>());
  expect_same_identity(schema_identity<schema::ImuT>(), schema_identity<schema::Imu>());

  const auto& native_package = schema_package<schema::ImuT>();
  auto valid = validate_schema_package(native_package);
  ASSERT_TRUE(valid) << runtime_error_message(valid.error());
  expect_same_identity(native_package.identity(), schema_identity<schema::Imu>());
}

// The include evidence names Core-owned schemas by logical path and content hash, neither of which
// depends on where the including schema lives. That independence is the reason the move did not
// disturb them, and it is worth pinning: a future change that started spelling these relative to
// src/sensor_io/schemas would move the include-closure digest without touching the schema text.
//
// The arrays this reads are maintained by hand, in two senses. The membership is written out in
// sensor_schema_package.hpp, and kImuIncludes was in fact missing geometry.fbs until recently; the
// four source digests are written out in flatbuffer_schema_package.hpp, whose own comment records
// that kHeaderSource has drifted four times. Neither is derived from anything at run time, because
// the .fbs sources are not present then. Asserting the closure exactly is the only thing that turns
// either into a checkable fact.
// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, ImageIncludeClosureIsExactlyWhatItShippedWith) {
  expect_exact_include_closure(schema_package<schema::Image>().evidence(),
                               std::span{kExpectedImageIncludes},
                               kExpectedImageClosureSha256);
}

// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, ImuIncludeClosureIsExactlyWhatItShippedWith) {
  expect_exact_include_closure(schema_package<schema::Imu>().evidence(),
                               std::span{kExpectedImuIncludes},
                               kExpectedImuClosureSha256);
}

// The owning native package is built from the same include array as the generated root, so a change
// that updated one call site and not the other would leave a producer naming ImuT and a consumer
// naming Imu recording different closures for one wire format.
// REQ: SENSOR-IO-006
TEST(SensorIoSchemaIdentity, OwningImuNativePackageRecordsTheSameClosure) {
  expect_exact_include_closure(schema_package<schema::ImuT>().evidence(),
                               std::span{kExpectedImuIncludes},
                               kExpectedImuClosureSha256);
}

}  // namespace
}  // namespace holoscan
