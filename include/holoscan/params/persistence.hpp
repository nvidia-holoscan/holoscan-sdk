/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include <holoscan/holoscan_params_io_export.h>
#include <holoscan/config/io.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/params/error.hpp>
#include <holoscan/params/rebind.hpp>

namespace holoscan::params {

/// @brief Classify one durable snapshot archive failure.
enum class ParamArchiveErrorCode : std::uint8_t {
  kInvalidArchive = 0,             ///< Source is malformed or violates the closed schema.
  kUnsupportedArchiveVersion = 1,  ///< Format token is known but its version is unsupported.
  kDigestMismatch = 2,             ///< Semantic content does not match the stored digest.
  kProjectionFailure = 3,          ///< Safe projection cannot classify a snapshot record.
  kIncomplete = 4,                 ///< Complete persistence was requested but cannot be proven.
  kPersistenceFailure = 5,         ///< Filesystem read, write, flush, or replacement failed.
};

/// @brief Own one typed archive or persistence failure.
///
/// The value owns structured bounded tooling detail. Concurrent const access is
/// safe while no thread mutates or destroys the value.
class HOLOSCAN_PARAMS_IO_EXPORT ParamArchiveError {
 public:
  /// @brief Construct one archive failure.
  /// @param code Stable archive category.
  /// @param detail Owning tooling detail and optional nested cause.
  ParamArchiveError(ParamArchiveErrorCode code, ParamToolError detail);
  /// @brief Return the stable archive category.
  [[nodiscard]] ParamArchiveErrorCode code() const noexcept { return code_; }
  /// @brief Return structured detail retained by this failure.
  /// @note The reference remains valid until this error is assigned or destroyed.
  [[nodiscard]] const ParamToolError& detail() const noexcept { return detail_; }

 private:
  ParamArchiveErrorCode code_{ParamArchiveErrorCode::kInvalidArchive};
  ParamToolError detail_;
};

/// @brief Result type used by snapshot archive operations.
/// @tparam T Successful value type.
template <typename T>
using ParamArchiveResult = expected<T, ParamArchiveError>;

/// @brief Select the single ordinary safe snapshot projection.
enum class ParamSafeProjection : std::uint8_t {
  kPersistedPublicOnly = 0,  ///< Include only Core records admitted for ordinary persistence.
};

/// @brief Configure strict JSON snapshot persistence.
struct ParamSaveOptions {
  /// Finite source-size and atomic replacement policy.
  config_io::SaveOptions io{};
  /// Sole ordinary safe projection.
  ParamSafeProjection safe_projection{ParamSafeProjection::kPersistedPublicOnly};
  /// Reject before filesystem mutation unless every restorable record is included.
  bool require_complete{};
};

/// @brief Configure strict JSON snapshot loading.
struct ParamLoadOptions {
  /// Bounded Config document parse policy.
  config_io::ParseOptions parser{};
};

/// @brief Identify one verified archive's semantic content and exact source bytes.
struct ParamArchiveIdentity {
  /// Canonical semantic archive identity.
  config::SemanticDigest archive_digest{};
  /// Exact source-byte identity and byte count.
  config::RawSourceDigest raw_source_digest{};
};

/// @brief Report the exact ordinary projection written to one archive.
struct ParamSaveResult {
  /// Exact committed destination bytes.
  config::RawSourceDigest written_source_digest{};
  /// Value-bearing records included in the archive.
  std::uint64_t included_count{};
  /// Records retained without a restorable value.
  std::uint64_t redacted_count{};
  /// Records omitted by the safe projection.
  std::uint64_t excluded_count{};
  /// Records whose projection could not be persisted.
  std::uint64_t error_count{};
  /// Whether the archive is complete for ordinary non-privileged restoration.
  bool complete_for_normal_persistence{};
};

/// @brief Return verified archive identity carried by a loaded snapshot.
/// @param snapshot Snapshot returned by \ref load or derived from another snapshot.
/// @return Semantic and raw-source identity for a directly loaded snapshot; empty for captures and
/// slices.
[[nodiscard]] HOLOSCAN_PARAMS_IO_EXPORT std::optional<ParamArchiveIdentity> archive_identity(
    const ParamSnapshot& snapshot) noexcept;

/// @brief Atomically persist one policy-filtered snapshot as JSON v1.
/// @param snapshot Owning snapshot to project; the function retains no reference.
/// @param path Destination path. The parent directory must already exist.
/// @param options Safe projection, finite I/O, and completeness policy.
/// @return Exact committed-byte identity and projection counts, or a typed failure.
/// @note This cold operation allocates and blocks on filesystem I/O. It writes a
/// same-directory temporary file and replaces the destination only after complete encoding.
[[nodiscard]] HOLOSCAN_PARAMS_IO_EXPORT ParamArchiveResult<ParamSaveResult> save(
    const ParamSnapshot& snapshot, const std::filesystem::path& path,
    ParamSaveOptions options = {});

/// @brief Load and transactionally validate one JSON v1 snapshot archive.
/// @param path Existing archive path.
/// @param options Finite parser/source-size policy.
/// @return New owning snapshot with archive identity, or a typed failure with no partial result.
/// @note This cold operation allocates and blocks on filesystem I/O. It validates the
/// closed schema and semantic digest before publishing a snapshot.
[[nodiscard]] HOLOSCAN_PARAMS_IO_EXPORT ParamArchiveResult<ParamSnapshot> load(
    const std::filesystem::path& path, ParamLoadOptions options = {});

/// @brief Load and explicitly rebind one archive to a target Graph.
/// @param path Existing archive path.
/// @param graph Draft destination Graph captured synchronously under the standard context.
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Immutable target-bound override set or a typed load/rebind failure.
/// @note This cold convenience allocates and blocks on filesystem I/O. It does not mutate graph;
/// do not mutate graph concurrently.
[[nodiscard]] HOLOSCAN_PARAMS_IO_EXPORT ParamRebindResult<ParamOverrideSet> load_overrides(
    const std::filesystem::path& path, Graph& graph, ParamRebindOptions options = {});

/// @brief Load and explicitly rebind one archive to an immutable target.
/// @param path Existing archive path.
/// @param target Immutable destination captured by make_param_rebind_target().
/// @param options Coverage mode and borrowed directed conversion tables.
/// @return Immutable target-bound override set or a typed load/rebind failure.
/// @note This cold convenience allocates and blocks on filesystem I/O.
[[nodiscard]] HOLOSCAN_PARAMS_IO_EXPORT ParamRebindResult<ParamOverrideSet> load_overrides(
    const std::filesystem::path& path, const ParamRebindTarget& target,
    ParamRebindOptions options = {});

}  // namespace holoscan::params
