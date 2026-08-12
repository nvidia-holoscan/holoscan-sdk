/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_snapshot.hpp>

namespace holoscan::detail {

/// Core-owned private construction seam consumed by optional snapshot archive tooling.
class ParamSnapshotArchiveAccess {
 public:
  struct ArchiveIdentity {
    std::uint32_t semantic_version{};
    std::array<std::byte, 32> semantic_sha256{};
    std::uint32_t raw_source_version{};
    std::array<std::byte, 32> raw_source_sha256{};
    std::uint64_t raw_source_bytes{};
  };

  [[nodiscard]] static constexpr ParamValue value(ParamScalarValue scalar) noexcept {
    return ParamValue{scalar};
  }

  [[nodiscard]] static ParamSnapshotScope scope(ParamSnapshotScopeKind kind, OperatorPath path) {
    return ParamSnapshotScope{kind, std::move(path)};
  }

  [[nodiscard]] static ParamSnapshot snapshot(
      PlanDigest plan_digest, PlanGenerationId plan_generation, RunGenerationId run_generation,
      ParamCatalogDigest catalog_digest, ParamReadView view, ParamCaptureConsistency consistency,
      ParamSnapshotScope scope, std::optional<ParamUpdateSequence> accepted_through,
      std::optional<ParamUpdateSequence> published_through,
      std::vector<ParamSnapshotOperatorRecord> operators) {
    ParamSnapshot result;
    result.plan_digest_ = plan_digest;
    result.plan_generation_ = plan_generation;
    result.run_generation_ = run_generation;
    result.catalog_digest_ = catalog_digest;
    result.view_ = view;
    result.consistency_ = consistency;
    result.scope_ = std::move(scope);
    result.accepted_through_ = accepted_through;
    result.published_through_ = published_through;
    result.operators_ = std::move(operators);
    return result;
  }

  static void set_archive_identity(ParamSnapshot& snapshot, const ArchiveIdentity& identity) {
    snapshot.has_archive_identity_ = true;
    snapshot.archive_semantic_version_ = identity.semantic_version;
    snapshot.archive_semantic_sha256_ = identity.semantic_sha256;
    snapshot.archive_raw_source_version_ = identity.raw_source_version;
    snapshot.archive_raw_source_sha256_ = identity.raw_source_sha256;
    snapshot.archive_raw_source_bytes_ = identity.raw_source_bytes;
  }

  [[nodiscard]] static std::optional<ArchiveIdentity> archive_identity(
      const ParamSnapshot& snapshot) noexcept {
    if (!snapshot.has_archive_identity_) {
      return std::nullopt;
    }
    return ArchiveIdentity{
        .semantic_version = snapshot.archive_semantic_version_,
        .semantic_sha256 = snapshot.archive_semantic_sha256_,
        .raw_source_version = snapshot.archive_raw_source_version_,
        .raw_source_sha256 = snapshot.archive_raw_source_sha256_,
        .raw_source_bytes = snapshot.archive_raw_source_bytes_,
    };
  }

  static void set_archive_provenance(ParamSnapshot& snapshot, Profile profile,
                                     std::optional<std::string> source_run_identity) {
    snapshot.archive_profile_ = profile;
    snapshot.archive_source_run_identity_ = std::move(source_run_identity);
  }

  [[nodiscard]] static std::optional<Profile> archive_profile(
      const ParamSnapshot& snapshot) noexcept {
    return snapshot.archive_profile_;
  }

  [[nodiscard]] static const std::optional<std::string>& archive_source_run_identity(
      const ParamSnapshot& snapshot) noexcept {
    return snapshot.archive_source_run_identity_;
  }
};

}  // namespace holoscan::detail
