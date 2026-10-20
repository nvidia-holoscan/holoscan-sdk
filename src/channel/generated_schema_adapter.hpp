/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <holoscan/schema/schema_package.hpp>

namespace holoscan::channel_internal {

/// Bind one exact generated package and endpoint generation after catalog attachment.
///
/// This process-local adapter is the integration seam for catalog-backed Channel
/// subscribers. It owns no queue or payload storage. Callers must retain the
/// returned serialized bytes through typed Sample construction and must never
/// expose the parser result after this adapter returns an error.
template <typename T>
class GeneratedSchemaAdapter {
 public:
  /// Validate and bind an exact installed package for one endpoint generation.
  [[nodiscard]] static expected<GeneratedSchemaAdapter, Error> create(
      const SchemaIdentity& expected_identity, std::uint64_t endpoint_generation,
      const SchemaPackageDescriptor<T>& package) {
    if (endpoint_generation == 0U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "schema endpoint generation is zero"});
    }
    auto status = validate_schema_package_for_attach(expected_identity, package);
    if (!status) {
      return make_unexpected(std::move(status).error());
    }
    // Attachment proved exact tuple equality. Retain the package-owned identity
    // view so an authoring/catalog string view cannot outlive its owner while
    // the endpoint remains active.
    return GeneratedSchemaAdapter(package.identity(), endpoint_generation, package);
  }

  /// Verify one received envelope and payload before returning a typed root view.
  [[nodiscard]] expected<const T*, Error> decode(const SchemaEnvelopeHeader& header,
                                                 std::span<const std::byte> payload) const {
    return verify_and_parse_received_schema(
        *package_, header, identity_, endpoint_generation_, payload);
  }

  /// Serialize one owning generated value into caller-owned bounded storage.
  ///
  /// This operation is available when the bound descriptor carries an owning
  /// codec. Borrowed table-root packages return \ref ErrorCode::kNotSupported.
  [[nodiscard]] expected<std::size_t, Error> encode(
      const T& value, std::span<std::byte> destination) const noexcept {
    return package_->encode(value, destination);
  }

  /// Verify one received envelope and materialize an owning semantic value.
  ///
  /// The adapter validates envelope ABI, identity, and generation before the
  /// exact package verifier and owning decoder. No borrowed root escapes an
  /// unsuccessful call.
  [[nodiscard]] expected<T, Error> materialize(const SchemaEnvelopeHeader& header,
                                               std::span<const std::byte> payload) const {
    return verify_and_materialize_received_schema(
        *package_, header, identity_, endpoint_generation_, payload);
  }

  /// Return the exact endpoint generation bound by this adapter.
  [[nodiscard]] std::uint64_t endpoint_generation() const noexcept { return endpoint_generation_; }

 private:
  GeneratedSchemaAdapter(SchemaIdentity identity, std::uint64_t endpoint_generation,
                         const SchemaPackageDescriptor<T>& package) noexcept
      : identity_(identity), endpoint_generation_(endpoint_generation), package_(&package) {}

  SchemaIdentity identity_{};
  std::uint64_t endpoint_generation_{};
  const SchemaPackageDescriptor<T>* package_{};
};

}  // namespace holoscan::channel_internal
