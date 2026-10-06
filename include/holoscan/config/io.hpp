/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_config_io_export.h>
#include <holoscan/config/document.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace holoscan::config_io {

/// @brief Select an explicit configuration source or destination syntax.
enum class Format : std::uint8_t {
  kJson,  ///< Strict JSON with the Holoscan version-one typed scalar envelope.
  kYaml   ///< The admitted fail-closed YAML semantic subset.
};

/// @brief Select semantic rewriting or required exact source-style retention.
enum class Preservation : std::uint8_t {
  kSemanticOnly,    ///< Permit deterministic semantic rendering after edits.
  kRequireLossless  ///< Require same-format exact replay or admitted same-kind scalar rewriting.
};

/// @brief Select the persistence flush boundary required after atomic replacement.
enum class Durability : std::uint8_t {
  kData,            ///< Flush the sibling temporary file before atomic replacement.
  kDataAndMetadata  ///< Also flush the parent directory after replacement.
};

/// @brief Bound one parser invocation below fixed package maxima.
struct ParseOptions {
  Preservation preservation{Preservation::kSemanticOnly};  ///< Source-retention requirement.
  std::size_t max_source_bytes{16U * 1024U * 1024U};       ///< Maximum input bytes.
  std::uint32_t max_depth{64};                             ///< Maximum tree depth; root is one.
  std::uint32_t max_nodes{1'000'000};                      ///< Maximum values including root.
  std::size_t max_string_bytes{4U * 1024U * 1024U};        ///< Maximum decoded key/string bytes.
  std::uint32_t max_container_entries{1'000'000};          ///< Maximum children per container.
};

/// @brief Bound serialization and temporary output below fixed package maxima.
struct OutputLimits {
  std::size_t max_output_bytes{64U * 1024U * 1024U};  ///< Maximum rendered bytes.
  std::uint32_t max_depth{64};                        ///< Maximum tree depth; root is one.
  std::uint32_t max_nodes{1'000'000};                 ///< Maximum values including root.
  std::size_t max_string_bytes{4U * 1024U * 1024U};   ///< Maximum key/string bytes.
  std::uint32_t max_container_entries{1'000'000};     ///< Maximum children per container.
};

/// @brief Configure one explicit-format serialization attempt.
struct SerializeOptions {
  OutputLimits output_limits{};  ///< Bounds validated before rendering output.
};

/// @brief Configure guarded sibling atomic replacement.
struct SaveOptions {
  /// Optional exact identity required of the current destination bytes; absence disables the check.
  /// When saving an edited Document, pass `document.digest().base_source` to guard replacement of
  /// the parsed file from which the Document was derived.
  std::optional<config::RawSourceDigest> expected_source_digest;
  /// Flush boundary required for success.
  Durability durability{Durability::kDataAndMetadata};
  /// Maximum wait for the deterministic sibling guard, from zero through 30 seconds.
  std::chrono::milliseconds guard_timeout{5000};
  /// Bounds validated before creating a temporary file.
  OutputLimits output_limits{};
};

/// @brief Report the exact bytes and guarantees installed by a successful save.
struct SaveResult {
  config::RawSourceDigest written_source_digest;  ///< Identity of the exact installed bytes.
  Durability achieved_durability{Durability::kDataAndMetadata};     ///< Completed flush boundary.
  Preservation achieved_preservation{Preservation::kSemanticOnly};  ///< Rendering mode used.
};

/// @brief Parse an explicitly selected JSON or admitted YAML source into an owned document.
///
/// Duplicate keys, invalid closed-enum values, unsupported YAML features, and all bound violations
/// fail before a Document is published. Lossless mode retains immutable same-format source evidence
/// and scalar token spans; parser-provider objects never cross this API.
///
/// @param source Complete source bytes, borrowed only for this call.
/// @param format Syntax selected by the caller; no filename or content inference occurs.
/// @param options Preservation and resource-limit policy.
/// @return The immutable Document with semantic and raw-source identity.
/// @throws config::ConfigException on validation or parse failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::Document parse(std::string_view source,
                                                               Format format,
                                                               ParseOptions options = {});

/// @brief Try to parse an explicitly selected JSON or admitted YAML source.
/// @param source Complete source bytes, borrowed only for this call.
/// @param format Syntax selected by the caller; no filename or content inference occurs.
/// @param options Preservation and resource-limit policy.
/// @return The immutable Document or a structured failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::ConfigResult<config::Document> try_parse(
    std::string_view source, Format format, ParseOptions options = {});

/// @brief Load and parse an explicitly selected configuration file.
///
/// The file size is checked before allocating the input buffer and the same parsing contract as
/// `parse()` applies. This call may block on filesystem I/O and is not a real-time operation.
///
/// @param path Existing regular-file path to read.
/// @param format Explicit source syntax.
/// @param options Preservation and resource-limit policy.
/// @return The immutable Document.
/// @throws config::ConfigException on validation, I/O, or parse failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::Document load(const std::filesystem::path& path,
                                                              Format format,
                                                              ParseOptions options = {});

/// @brief Try to load and parse an explicitly selected configuration file.
/// @param path Existing regular-file path to read.
/// @param format Explicit source syntax.
/// @param options Preservation and resource-limit policy.
/// @return The immutable Document or a structured validation, I/O, or parse failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::ConfigResult<config::Document> try_load(
    const std::filesystem::path& path, Format format, ParseOptions options = {});

/// @brief Serialize one immutable document in an explicitly selected format.
///
/// Semantic-only JSON output is compact and deterministic. Semantic-only YAML output uses the
/// canonical JSON representation, which is valid in the admitted YAML 1.2 subset. A lossless
/// document requires the original format and either exact unchanged replay or previously admitted
/// same-kind scalar edits; the function never silently falls back to canonical rendering.
///
/// @param document Immutable source document.
/// @param format Explicit destination syntax.
/// @param options Output resource bounds.
/// @return Owning output bytes.
/// @throws config::ConfigException on preservation, limit, or codec failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT std::string serialize(const config::Document& document,
                                                              Format format,
                                                              SerializeOptions options = {});

/// @brief Try to serialize one immutable document in an explicitly selected format.
/// @param document Immutable source document.
/// @param format Explicit destination syntax.
/// @param options Output resource bounds.
/// @return Owning output bytes or a structured preservation, limit, or codec failure.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::ConfigResult<std::string> try_serialize(
    const config::Document& document, Format format, SerializeOptions options = {});

/// @brief Persist one document with guarded same-directory atomic replacement.
///
/// Serialization completes before filesystem mutation. The baseline Linux implementation rejects
/// symlinked parent components and destinations, acquires a deterministic sibling guard, optionally
/// compares exact destination bytes, writes and data-flushes an exclusive sibling temporary,
/// revalidates destination identity, atomically renames, and optionally flushes parent metadata.
/// The call may block until `SaveOptions::guard_timeout`. Native Windows returns
/// `CONFIG_ATOMIC_REPLACE_UNSUPPORTED` in baseline; mount-specific crash guarantees remain a
/// platform qualification rather than a portable promise.
///
/// @param document Immutable source document.
/// @param path Destination path. Its parent must already exist.
/// @param format Explicit destination syntax.
/// @param options Compare-and-swap, durability, wait, and output-limit policy.
/// @return Installed-byte evidence.
/// @throws config::ConfigException on failure. A `kReplaced` failure means replacement completed
/// but metadata flush failed.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT SaveResult save(const config::Document& document,
                                                        const std::filesystem::path& path,
                                                        Format format, SaveOptions options = {});

/// @brief Try to persist one document with guarded same-directory atomic replacement.
/// @param document Immutable source document.
/// @param path Destination path. Its parent must already exist.
/// @param format Explicit destination syntax.
/// @param options Compare-and-swap, durability, wait, and output-limit policy.
/// @return Installed-byte evidence, or a structured failure carrying operation stage and
/// destination state. A `kReplaced` failure means replacement completed but metadata flush failed.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::ConfigResult<SaveResult> try_save(
    const config::Document& document, const std::filesystem::path& path, Format format,
    SaveOptions options = {});

}  // namespace holoscan::config_io
