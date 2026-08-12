/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/schema/schema_package.hpp>

namespace holoscan::schema_test {

// Minimal empty root type used as the schema anchor for SchemaPackageDescriptor tests.
// No FlatBuffer table fields are needed — only the identity and artifact bytes matter.
struct PackageValidationRoot {};

// Four-byte stub FlatBuffer artifact. Content is arbitrary; only identity/digest
// derivation and non-empty presence are exercised by these tests.
inline constexpr std::array<std::byte, 4> kArtifact{
    std::byte{0x01}, std::byte{0x23}, std::byte{0x45}, std::byte{0x67}};

}  // namespace holoscan::schema_test

namespace holoscan {

template <>
consteval SchemaIdentity schema_identity<schema_test::PackageValidationRoot>() {
  constexpr std::string_view kName = "nvidia.holoscan.test.PackageValidationRoot";
  constexpr std::uint32_t kVersion = 1U;
  constexpr auto kDigest = schema_artifact_sha256(schema_test::kArtifact);
  return SchemaIdentity{
      .canonical_name = kName,
      .version = kVersion,
      .schema_sha256 = kDigest,
      .id = derive_schema_id(kName, kVersion, kDigest),
  };
}

}  // namespace holoscan

namespace {

using Root = holoscan::schema_test::PackageValidationRoot;

// Reference ordered flatc flags containing a logical placeholder (<schema-root-0>)
// and a canonical relative path. All make_evidence() calls default to these flags.
inline constexpr std::array<std::string_view, 4> kCanonicalFlags{
    "--schema", "-I", "<schema-root-0>", "schemas/root.fbs"};
inline constexpr std::array<char, 3> kInvalidUtf8Bytes{'x', static_cast<char>(0xC3), '('};
inline constexpr std::string_view kInvalidUtf8{kInvalidUtf8Bytes.data(), kInvalidUtf8Bytes.size()};

// Stub verifier: accepts any non-empty byte span as a structurally valid FlatBuffer.
// Used wherever a test package needs a non-null verifier but correctness is irrelevant.
[[nodiscard]] bool verify_payload(std::span<const std::byte> bytes) noexcept {
  return !bytes.empty();
}

// Stub borrowed parser returning a pointer to a static Root sentinel.
// Used for packages that need a non-null parse_borrowed callback.
[[nodiscard]] const Root* parse_payload(std::span<const std::byte>) noexcept {
  static constexpr Root kRoot;
  return &kRoot;
}

// Stub encoder writing a single 0x01 byte. Returns kResourceExhausted when the
// destination buffer is empty. Used by BorrowedParseRejectsOwningOnlyPackage*.
[[nodiscard]] holoscan::expected<std::size_t, holoscan::Error> encode_payload(
    const Root&, std::span<std::byte> destination) noexcept {
  if (destination.empty()) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kResourceExhausted});
  }
  destination.front() = std::byte{0x01};
  return 1U;
}

// Stub materializer returning a default-constructed Root. Used by
// BorrowedParseRejectsOwningOnlyPackage* as the owning package materializer.
[[nodiscard]] holoscan::expected<Root, holoscan::Error> materialize_payload(
    std::span<const std::byte>) noexcept {
  return Root{};
}

// Convenience factory for SchemaPackageEvidence with canonical defaults. Pass
// non-default arguments to vary the evidence field under test.
[[nodiscard]] holoscan::SchemaPackageEvidence make_evidence(
    std::string_view root_logical_path, std::span<const std::string_view> flags = kCanonicalFlags,
    std::span<const holoscan::SchemaPackageSource> includes = {},
    std::string_view include_root_mapping = "schema-root-0=.") {
  return holoscan::SchemaPackageEvidence{
      .flatc_build = "flatc-test-build",
      .ordered_flags = flags,
      .root_logical_path = root_logical_path,
      .includes = includes,
      .include_root_mapping = include_root_mapping,
      .host_paths_stripped = true,
      .generator_format_version = 1U,
      .package_format_version = 1U,
      .runtime_abi = holoscan::FlatBufferRuntimeABI{.major = 1U},
  };
}

// Convenience factory for SchemaPackageDescriptor<Root> using stub callbacks.
// Omits encoder and materializer — the package is import-side (borrowed-parse) only.
[[nodiscard]] holoscan::SchemaPackageDescriptor<Root> make_package(
    holoscan::SchemaPackageEvidence evidence) {
  return holoscan::SchemaPackageDescriptor<Root>{
      holoscan::schema_identity<Root>(),
      holoscan::schema_test::kArtifact,
      evidence,
      &verify_payload,
      &parse_payload,
  };
}

// Logical paths retain exact package-relative spelling. A repeated dot inside a filename is not a
// traversal segment, while POSIX, drive-relative, drive-absolute, and UNC host spellings all fail.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(SchemaPackageValidation, AcceptsOnlyCanonicalPackageRelativeLogicalPaths) {
  // Key setup: kAccepted paths have no traversal segments, no leading slash, no URI
  //   scheme. kRejected covers every non-canonical form the validator must block.
  constexpr std::array<std::string_view, 4> kAccepted{
      "root.fbs", "schemas/root.fbs", "schemas/schema..v1.fbs", ".generated/root.fbs"};
  for (const std::string_view path : kAccepted) {
    EXPECT_TRUE(holoscan::detail::canonical_package_path(path)) << path;
  }

  constexpr std::array<std::string_view, 14> kRejected{
      "/host/root.fbs",
      "//server/share/root.fbs",
      R"(\\server\share\root.fbs)",
      R"(schemas\root.fbs)",
      "C:/work/root.fbs",
      "C:root.fbs",
      "file:///work/root.fbs",
      "./root.fbs",
      "../root.fbs",
      "schemas/./root.fbs",
      "schemas/../root.fbs",
      "schemas//root.fbs",
      "schemas/",
      "",
  };
  // Key assertion: canonical_package_path accepts all kAccepted forms and rejects
  //   all 14 kRejected forms without exception.
  for (const std::string_view path : kRejected) {
    EXPECT_FALSE(holoscan::detail::canonical_package_path(path)) << path;
  }
}

// Ordered flags may retain normalized logical placeholders and relative logical paths, but never
// smuggle a platform host path or a path requiring normalization into package evidence.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(SchemaPackageValidation, RejectsHostAndNoncanonicalPathsInsideOrderedFlags) {
  // Key setup: kAccepted includes logical placeholders and relative paths that are
  //   safe to embed; kRejected includes host paths inside --include= values.
  constexpr std::array<std::string_view, 7> kAccepted{
      "--bfbs-gen-embed",
      "--cpp-ptr-type",
      "std::shared_ptr",
      "-I",
      "<schema-root-0>",
      "schemas/root.fbs",
      "--include=schemas/root.fbs",
  };
  for (const std::string_view flag : kAccepted) {
    EXPECT_TRUE(holoscan::detail::canonical_generator_flag(flag)) << flag;
  }

  constexpr std::array<std::string_view, 15> kRejected{
      "/host/root.fbs",
      "//server/share/root.fbs",
      R"(\\server\share\root.fbs)",
      R"(schemas\root.fbs)",
      "C:/work/root.fbs",
      "C:root.fbs",
      "--include=C:/work/root.fbs",
      "--include=/work/root.fbs",
      "--include=../root.fbs",
      "--include=schemas/./root.fbs",
      "--include=schemas//root.fbs",
      "-I../schemas",
      "-I/host/schemas",
      "schemas/",
      "file:///work/root.fbs",
  };
  // Key assertion: canonical_generator_flag rejects every host-path form embedded in
  //   flags, including paths inside --include= and -I prefixes.
  for (const std::string_view flag : kRejected) {
    EXPECT_FALSE(holoscan::detail::canonical_generator_flag(flag)) << flag;
  }
}

// Generated evidence uses one indexed grammar for singleton and multi-root invocations.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(SchemaPackageValidation, AcceptsCanonicalOrderedIncludeRootMappings) {
  constexpr std::array<std::string_view, 3> kAccepted{
      "schema-root-0=.",
      "schema-root-0=.;schema-root-1=.",
      "schema-root-0=.;schema-root-1=.;schema-root-2=.",
  };
  for (const std::string_view mapping : kAccepted) {
    EXPECT_TRUE(holoscan::detail::canonical_include_root_mapping(mapping)) << mapping;
  }

  constexpr std::array<std::string_view, 11> kRejected{
      "",
      "schema-root=.",
      "application=schemas;holoscan=sdk/schemas",
      ";schema-root-0=.",
      "schema-root-0=.;",
      "schema-root-0=.;;schema-root-1=.",
      "schema-root-1=.",
      "schema-root-0=.;schema-root-2=.",
      "schema-root-00=.",
      "schema-root-0=schemas",
      "schema-root-0=/host/schemas",
  };
  for (const std::string_view mapping : kRejected) {
    EXPECT_FALSE(holoscan::detail::canonical_include_root_mapping(mapping)) << mapping;
  }
}

// Package validation applies the same fail-closed rules to the root, every include, the ordered
// flag vector, and the include-root mapping rather than trusting host_paths_stripped by itself.
// REQ: CHAN-120
TEST(SchemaPackageValidation, PackageRejectsEveryResidualHostPathEvidence) {
  // Key setup: each of the four evidence fields (root path, ordered flags, include
  //   sources, include-root mapping) is exercised with a distinct bad input.

  auto valid = holoscan::validate_schema_package(make_package(make_evidence("schemas/root.fbs")));
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());

  constexpr std::array<std::string_view, 3> kBadRoots{
      R"(C:\work\root.fbs)", "schemas/../root.fbs", "schemas//root.fbs"};
  for (const std::string_view root : kBadRoots) {
    auto rejected = holoscan::validate_schema_package(make_package(make_evidence(root)));
    ASSERT_FALSE(rejected) << root;
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument) << root;
  }

  for (const std::string_view flag : std::array<std::string_view, 3>{
           "--include=C:/work/root.fbs", R"(-I\\server\share)", "--include=schemas/../root.fbs"}) {
    const std::array<std::string_view, 2> flags{"--schema", flag};
    auto rejected =
        holoscan::validate_schema_package(make_package(make_evidence("schemas/root.fbs", flags)));
    ASSERT_FALSE(rejected) << flag;
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument) << flag;
  }

  for (const std::string_view include_path : std::array<std::string_view, 3>{
           "/host/include.fbs", R"(C:\work\include.fbs)", "schemas/./include.fbs"}) {
    const std::array<holoscan::SchemaPackageSource, 1> includes{
        holoscan::SchemaPackageSource{.logical_path = include_path, .content_sha256 = {}}};
    auto rejected = holoscan::validate_schema_package(
        make_package(make_evidence("schemas/root.fbs", kCanonicalFlags, includes)));
    ASSERT_FALSE(rejected) << include_path;
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument) << include_path;
  }

  constexpr std::array<std::string_view, 4> kBadMappings{R"(schema-root-0=C:\work)",
                                                         "schema-root-0=/work",
                                                         "schema-root-0=schemas/../work",
                                                         "=schemas"};
  for (const std::string_view mapping : kBadMappings) {
    auto rejected = holoscan::validate_schema_package(
        make_package(make_evidence("schemas/root.fbs", kCanonicalFlags, {}, mapping)));
    // Key assertion: validate_schema_package returns kInvalidArgument for every
    //   residual host-path form regardless of which evidence field carries it.
    ASSERT_FALSE(rejected) << mapping;
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument) << mapping;
  }
}

// Package evidence enters plan and attachment diagnostic JSON, so every public evidence string
// must be valid UTF-8 before the package is admitted.
// REQ: CHAN-120
// REQ: CHAN-121
TEST(SchemaPackageValidation, PackageRejectsInvalidUtf8DiagnosticEvidence) {
  const auto expect_rejected = [](holoscan::SchemaPackageEvidence evidence) {
    auto result = holoscan::validate_schema_package(make_package(evidence));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, holoscan::ErrorCode::kInvalidArgument);
  };

  // Key setup: place one malformed two-byte UTF-8 sequence in every evidence string family that
  // can be copied into plan provenance or used to derive it.
  auto flatc_build = make_evidence("schemas/root.fbs");
  flatc_build.flatc_build = kInvalidUtf8;
  expect_rejected(flatc_build);

  const std::array<std::string_view, 2> flags{"--schema", kInvalidUtf8};
  expect_rejected(make_evidence("schemas/root.fbs", flags));

  const std::string root_path = "schemas/" + std::string(kInvalidUtf8) + ".fbs";
  expect_rejected(make_evidence(root_path));

  const std::string mapping = "schema-root-0=" + std::string(kInvalidUtf8);
  expect_rejected(make_evidence("schemas/root.fbs", kCanonicalFlags, {}, mapping));

  const std::string include_path = "schemas/" + std::string(kInvalidUtf8) + ".fbs";
  const std::array<holoscan::SchemaPackageSource, 1> includes{
      holoscan::SchemaPackageSource{.logical_path = include_path, .content_sha256 = {}}};
  // Key assertion: every malformed field fails package validation before any diagnostic renderer
  // can copy the invalid bytes into JSON.
  expect_rejected(make_evidence("schemas/root.fbs", kCanonicalFlags, includes));
}

// A valid owning package deliberately has no borrowed parser. The borrowed-view API must report
// that missing capability instead of calling a null parser and misclassifying it as bad wire data.
// REQ: CHAN-075
TEST(SchemaPackageValidation, BorrowedParseRejectsOwningOnlyPackageAsUnsupported) {
  // Key setup: owning_only package has no parse_borrowed callback (null pointer);
  //   the encoder and materializer are present so the package itself is valid.
  const holoscan::SchemaPackageDescriptor<Root> owning_only{
      holoscan::schema_identity<Root>(),
      holoscan::schema_test::kArtifact,
      make_evidence("schemas/root.fbs"),
      &verify_payload,
      4U,
      &encode_payload,
      &materialize_payload,
  };
  const holoscan::SchemaEnvelopeHeader header{
      .schema_id = holoscan::schema_identity<Root>().id,
      .schema_version = holoscan::schema_identity<Root>().version,
      .payload_size = 1U,
      .endpoint_generation = 1U,
  };
  const std::array payload{std::byte{0x01}};

  auto parsed = holoscan::verify_and_parse_received_schema(
      owning_only, header, holoscan::schema_identity<Root>(), 1U, payload);
  ASSERT_FALSE(parsed);
  // Key assertion: error code is kNotSupported, not a wire-data codec error.
  EXPECT_EQ(parsed.error().code, holoscan::ErrorCode::kNotSupported);
}

}  // namespace
