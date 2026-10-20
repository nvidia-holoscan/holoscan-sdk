/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tl/expected.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/holoscan_config_export.h>

namespace holoscan::config {

/// @brief Enumerate the closed semantic alternatives accepted by a configuration document.
enum class ConfigValueKind : std::uint8_t {
  kNull,             ///< The null alternative.
  kBool,             ///< A Boolean value.
  kSignedInteger,    ///< A signed 64-bit integer.
  kUnsignedInteger,  ///< An unsigned 64-bit integer.
  kFloating,         ///< A finite binary64 value with negative zero normalized.
  kString,           ///< A valid UTF-8 byte string.
  kArray,            ///< An ordered sequence of values.
  kObject,           ///< An insertion-ordered sequence of unique UTF-8 keys and values.
};

/// @brief Enumerate stable failures from configuration value, codec, and persistence operations.
enum class ConfigErrorCode : std::uint16_t {
  kInvalidArgument,    ///< An argument or option is outside its closed domain.
  kParseError,         ///< Input bytes do not conform to the selected syntax.
  kUnsupportedFormat,  ///< The requested source or destination format is unavailable.
  kUnsupportedValue,   ///< A syntactically valid value is outside the admitted subset.
  kInvalidUtf8,        ///< A required UTF-8 string or key is malformed.
  kDuplicateKey,       ///< An object contains the same key more than once.
  kPathSyntax,         ///< An RFC 6901 path is malformed.
  kPathNotFound,       ///< A requested path or child does not exist.
  kPathConflict,       ///< Transactional edit paths overlap or otherwise conflict.
  kTypeMismatch,       ///< A value has a different semantic kind than required.
  kOutOfRange,         ///< A numeric value cannot be represented by its destination.
  kLimitExceeded,      ///< A configured or package resource bound would be exceeded.
  kLosslessPreservationUnavailable,  ///< Exact source-style preservation cannot be satisfied.
  kSourceChanged,                    ///< Optimistic source identity no longer matches.
  kGuardBusy,                 ///< The persistence guard was not acquired before its deadline.
  kCodecMissing,              ///< No explicit typed codec is available.
  kCodecFailure,              ///< A codec violated its contract or failed during conversion.
  kIoError,                   ///< A filesystem or byte-stream operation failed.
  kPermissionDenied,          ///< The platform denied a required filesystem operation.
  kAtomicReplaceUnsupported,  ///< The requested atomic guarantee is unavailable.
  kDigestMismatch,            ///< Content does not match an expected digest.
  kUnsupportedVersion,        ///< A persisted grammar or envelope version is unsupported.
};

/// @brief Identify the stage at which a configuration operation failed.
enum class ConfigOperationStage : std::uint8_t {
  kNone,                      ///< No persistence or codec stage applies.
  kParse,                     ///< Source bytes were being parsed or loaded.
  kValidate,                  ///< Arguments, limits, or semantic constraints were being checked.
  kSerialize,                 ///< A document was being rendered to destination bytes.
  kWriteTemporary,            ///< A sibling temporary file was being written or data-flushed.
  kAcquireGuard,              ///< The save guard or stable destination observation was pending.
  kReplaceDestination,        ///< Atomic replacement of the destination was attempted.
  kFlushDestinationMetadata,  ///< Parent-directory metadata was being flushed after replacement.
};

/// @brief Describe whether a failed persistence operation changed its destination.
enum class ConfigDestinationState : std::uint8_t {
  kNotApplicable,  ///< The operation did not address a persistent destination.
  kUnchanged,      ///< The destination was not replaced.
  kReplaced,       ///< Replacement succeeded, although a later metadata flush may have failed.
};

/// @brief Return the stable token associated with a configuration error code.
///
/// @param code One of the 22 closed configuration error categories.
/// @return The process-lifetime `CONFIG_*` token. An unknown numeric value maps to
/// `CONFIG_INVALID_ARGUMENT`.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT std::string_view config_error_token(
    ConfigErrorCode code) noexcept;

/// @brief Borrow one bounded entry from a configuration error's nested-cause chain.
///
/// Instances are stored inside ConfigError. Returned text/excerpt views remain valid only while the
/// owning ConfigError is alive and unmodified. Inspection is allocation-free and thread-safe for
/// concurrent readers of the same immutable error.
class HOLOSCAN_CONFIG_EXPORT ConfigErrorCause {
 public:
  /// @brief Return the stable machine-readable failure category.
  [[nodiscard]] ConfigErrorCode code() const noexcept { return code_; }
  /// @brief Return the stable token corresponding to \ref code.
  [[nodiscard]] std::string_view token() const noexcept { return config_error_token(code_); }
  /// @brief Return the copied message bytes.
  [[nodiscard]] std::string_view message() const noexcept;
  /// @brief Return the copied rendered path bytes.
  [[nodiscard]] std::string_view path() const noexcept;
  /// @brief Return the copied source-name bytes.
  [[nodiscard]] std::string_view source() const noexcept;
  /// @brief Return the copied raw excerpt bytes.
  [[nodiscard]] std::span<const std::byte> excerpt() const noexcept;
  /// @brief Return whether message text was truncated.
  [[nodiscard]] bool message_truncated() const noexcept { return message_truncated_; }
  /// @brief Return whether rendered path text was truncated.
  [[nodiscard]] bool path_truncated() const noexcept { return path_truncated_; }
  /// @brief Return whether source-name text was truncated.
  [[nodiscard]] bool source_truncated() const noexcept { return source_truncated_; }
  /// @brief Return whether excerpt bytes were truncated.
  [[nodiscard]] bool excerpt_truncated() const noexcept { return excerpt_truncated_; }
  /// @brief Return whether deeper causes were discarded after this entry.
  [[nodiscard]] bool cause_overflow() const noexcept { return cause_overflow_; }
  /// @brief Return the operation stage associated with this cause.
  [[nodiscard]] ConfigOperationStage operation_stage() const noexcept { return stage_; }
  /// @brief Return the destination state associated with this cause.
  [[nodiscard]] ConfigDestinationState destination_state() const noexcept { return destination_; }

 private:
  static constexpr std::size_t kMessageStorageBytes = 257;
  static constexpr std::size_t kPathStorageBytes = 513;
  static constexpr std::size_t kSourceStorageBytes = 129;
  static constexpr std::size_t kExcerptStorageBytes = 256;

  ConfigErrorCode code_{ConfigErrorCode::kInvalidArgument};
  ConfigOperationStage stage_{ConfigOperationStage::kNone};
  ConfigDestinationState destination_{ConfigDestinationState::kNotApplicable};
  std::array<char, kMessageStorageBytes> message_{};
  std::array<char, kPathStorageBytes> path_{};
  std::array<char, kSourceStorageBytes> source_{};
  std::array<std::byte, kExcerptStorageBytes> excerpt_{};
  std::uint16_t message_size_{0};
  std::uint16_t path_size_{0};
  std::uint16_t source_size_{0};
  std::uint16_t excerpt_size_{0};
  bool message_truncated_{false};
  bool path_truncated_{false};
  bool source_truncated_{false};
  bool excerpt_truncated_{false};
  bool cause_overflow_{false};

  friend class ConfigError;
};

/// @brief Own one bounded, allocation-free configuration diagnostic.
///
/// Construction copies and deterministically truncates all caller-provided evidence. The type is
/// copyable without allocation; concurrent inspection is safe when no thread assigns to it.
class HOLOSCAN_CONFIG_EXPORT ConfigError {
 public:
  /// Maximum copied message bytes, excluding the internal terminator.
  static constexpr std::size_t kMaxMessageBytes = 256;
  /// Maximum copied rendered-path bytes, excluding the internal terminator.
  static constexpr std::size_t kMaxPathBytes = 512;
  /// Maximum copied source-name bytes, excluding the internal terminator.
  static constexpr std::size_t kMaxSourceBytes = 128;
  /// Maximum copied raw excerpt bytes.
  static constexpr std::size_t kMaxExcerptBytes = 256;
  /// Maximum number of copied nested causes.
  static constexpr std::size_t kMaxNestedCauseDepth = 4;

  /// @brief Construct a bounded error while copying all supplied diagnostic text.
  ///
  /// The returned value owns every supplied byte. Text is truncated to the largest complete UTF-8
  /// prefix that fits its field; the raw excerpt retains its first bytes without interpretation.
  /// Construction performs no dynamic allocation and never throws.
  ///
  /// @param code Stable machine-readable failure category.
  /// @param message Human-readable context that is not semantic control authority.
  /// @param path Rendered diagnostic path; truncation makes it unsuitable for addressing.
  /// @param source Source name associated with the failure.
  /// @param excerpt Raw source bytes near the failure.
  /// @param stage Operation stage at which the failure occurred.
  /// @param destination Destination state after a persistence failure.
  /// @return An owning bounded diagnostic.
  static ConfigError make(
      ConfigErrorCode code, std::string_view message = {}, std::string_view path = {},
      std::string_view source = {}, std::span<const std::byte> excerpt = {},
      ConfigOperationStage stage = ConfigOperationStage::kNone,
      ConfigDestinationState destination = ConfigDestinationState::kNotApplicable) noexcept;

  /// @brief Append a copied cause chain to an existing bounded diagnostic.
  ///
  /// The direct cause is retained first, followed by its nearest causes. At most
  /// \ref kMaxNestedCauseDepth entries are retained. Overflow marks both the root and deepest
  /// retained cause. The input values are not modified and the operation performs no allocation.
  ///
  /// @param root Root diagnostic to return with nested-cause evidence.
  /// @param cause Direct cause whose bounded chain is copied.
  /// @return The augmented owning diagnostic.
  [[nodiscard]] static ConfigError nest(ConfigError root, const ConfigError& cause) noexcept;

  /// @brief Return the stable machine-readable failure category.
  [[nodiscard]] ConfigErrorCode code() const noexcept { return code_; }
  /// @brief Return the stable token for \ref code.
  [[nodiscard]] std::string_view token() const noexcept { return config_error_token(code_); }
  /// @brief Return the copied message bytes.
  [[nodiscard]] std::string_view message() const noexcept;
  /// @brief Return the copied rendered path bytes.
  [[nodiscard]] std::string_view path() const noexcept;
  /// @brief Return the copied source-name bytes.
  [[nodiscard]] std::string_view source() const noexcept;
  /// @brief Return the copied raw excerpt bytes.
  [[nodiscard]] std::span<const std::byte> excerpt() const noexcept;
  /// @brief Return whether message text was truncated.
  [[nodiscard]] bool message_truncated() const noexcept { return message_truncated_; }
  /// @brief Return whether rendered path text was truncated.
  [[nodiscard]] bool path_truncated() const noexcept { return path_truncated_; }
  /// @brief Return whether source-name text was truncated.
  [[nodiscard]] bool source_truncated() const noexcept { return source_truncated_; }
  /// @brief Return whether raw excerpt bytes were truncated.
  [[nodiscard]] bool excerpt_truncated() const noexcept { return excerpt_truncated_; }
  /// @brief Return whether a nested-cause chain exceeded its fixed capacity.
  [[nodiscard]] bool cause_overflow() const noexcept { return cause_overflow_; }
  /// @brief Return the number of retained nested causes.
  [[nodiscard]] std::size_t cause_count() const noexcept { return cause_count_; }
  /// @brief Borrow one retained cause, or return null when \p index is out of range.
  ///
  /// The returned pointer remains valid only while this error is alive and unmodified.
  [[nodiscard]] const ConfigErrorCause* cause(std::size_t index) const noexcept {
    return index < cause_count_ ? &causes_[index] : nullptr;
  }
  /// @brief Return the operation stage associated with the failure.
  [[nodiscard]] ConfigOperationStage operation_stage() const noexcept { return stage_; }
  /// @brief Return the destination state associated with the failure.
  [[nodiscard]] ConfigDestinationState destination_state() const noexcept { return destination_; }

 private:
  ConfigErrorCode code_{ConfigErrorCode::kInvalidArgument};
  ConfigOperationStage stage_{ConfigOperationStage::kNone};
  ConfigDestinationState destination_{ConfigDestinationState::kNotApplicable};
  std::array<char, kMaxMessageBytes + 1> message_{};
  std::array<char, kMaxPathBytes + 1> path_{};
  std::array<char, kMaxSourceBytes + 1> source_{};
  std::array<std::byte, kMaxExcerptBytes> excerpt_{};
  std::uint16_t message_size_{0};
  std::uint16_t path_size_{0};
  std::uint16_t source_size_{0};
  std::uint16_t excerpt_size_{0};
  bool message_truncated_{false};
  bool path_truncated_{false};
  bool source_truncated_{false};
  bool excerpt_truncated_{false};
  bool cause_overflow_{false};
  std::uint8_t cause_count_{0};
  std::array<ConfigErrorCause, kMaxNestedCauseDepth> causes_{};
};

/// @brief Carry one complete structured configuration failure through an exception boundary.
///
/// `what()` is the result of `format_error(error())`. Constructing the exception can allocate;
/// callers that must preserve allocation-failure handling use the corresponding `try_*` API.
class HOLOSCAN_CONFIG_EXPORT ConfigException : public std::runtime_error {
 public:
  /// @brief Construct an exception while preserving the complete structured error.
  /// @param error Owning configuration failure to retain.
  explicit ConfigException(ConfigError error);

  /// @brief Destroy the exception through its exported shared-library implementation.
  ~ConfigException() override;

  /// @brief Copy an exception without changing its structured failure.
  ConfigException(const ConfigException&) = default;
  /// @brief Replace an exception with a copy of another structured failure.
  ConfigException& operator=(const ConfigException&) = default;

  /// @brief Return the retained structured configuration failure.
  [[nodiscard]] const ConfigError& error() const noexcept { return error_; }

 private:
  ConfigError error_;
};

/// @brief Render a configuration error as a single line for logs and diagnostics.
///
/// The result begins with the stable error token and includes available message, source, path,
/// and retained cause context. Truncated fields are marked. Raw excerpt bytes are never included
/// because they may contain sensitive configuration data. The wording is intended for people and
/// is not a stable format; use ConfigError accessors for programmatic decisions.
///
/// @param error Configuration error to render.
/// @return Human-readable text without a trailing newline.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT std::string format_error(const ConfigError& error);

/// @brief Represent either a configuration result or a bounded structured error.
template <class T>
using ConfigResult = tl::expected<T, ConfigError>;

namespace detail {

/// @brief Throw a structured configuration failure from a shared-library boundary.
/// @param error Owning configuration failure retained by ConfigException.
/// @throws ConfigException Always.
[[noreturn]] HOLOSCAN_CONFIG_EXPORT void throw_config_error(ConfigError error);

/// @brief Return a successful configuration result or throw its complete structured error.
template <class T>
[[nodiscard]] T value_or_throw(ConfigResult<T> result) {
  if (!result) {
    throw_config_error(std::move(result.error()));
  }
  return T(std::move(*result));
}

}  // namespace detail

/// @brief Bound construction of one owned semantic value tree.
struct ConfigValueLimits {
  /// Maximum tree depth, where the root has depth one.
  std::uint32_t max_depth{64};
  /// Maximum number of semantic values, including the root.
  std::uint32_t max_nodes{1'000'000};
  /// Maximum decoded UTF-8 bytes in each key or string value.
  std::size_t max_string_bytes{4U * 1024U * 1024U};
  /// Maximum immediate entries in each array or object.
  std::uint32_t max_container_entries{1'000'000};
};

namespace detail {
struct ConfigNode;
struct DocumentState;
}  // namespace detail

class ConfigView;
struct ConfigObjectEntry;
struct SemanticEncodeOptions;
/// @brief Select behavior for missing fields while decoding a typed value.
enum class MissingValuePolicy : std::uint8_t {
  kReject,          ///< Reject a missing value required by the codec.
  kUseTypeDefaults  ///< Permit a codec to retain its type-defined default.
};

/// @brief Select behavior for unconsumed object fields.
enum class UnknownFieldPolicy : std::uint8_t {
  kReject,  ///< Reject any object member not consumed by the codec.
  kIgnore   ///< Permit unconsumed object members.
};

/// @brief Select strict or explicitly checked lossy numeric conversion.
enum class NumericConversionPolicy : std::uint8_t {
  kLossless,          ///< Reject semantic-kind changes and inexact conversions.
  kAllowCheckedLossy  ///< Permit only finite, range-checked conversions admitted by the codec.
};

/// @brief Own immutable typed-decoding policy for one codec call.
class DecodeOptions {
 public:
  /// @brief Create immutable typed-decoding policy.
  ///
  /// @param missing Policy applied by codecs to absent values.
  /// @param unknown Policy applied by object decoders to unconsumed members.
  /// @param numeric Policy applied to numeric kind and width conversion.
  explicit DecodeOptions(
      MissingValuePolicy missing = MissingValuePolicy::kReject,
      UnknownFieldPolicy unknown = UnknownFieldPolicy::kReject,
      NumericConversionPolicy numeric = NumericConversionPolicy::kLossless) noexcept
      : missing_(missing), unknown_(unknown), numeric_(numeric) {}
  /// @brief Return the missing-value policy.
  [[nodiscard]] MissingValuePolicy missing_fields() const noexcept { return missing_; }
  /// @brief Return the unknown-field policy.
  [[nodiscard]] UnknownFieldPolicy unknown_fields() const noexcept { return unknown_; }
  /// @brief Return the numeric-conversion policy.
  [[nodiscard]] NumericConversionPolicy numeric_conversion() const noexcept { return numeric_; }

 private:
  MissingValuePolicy missing_;
  UnknownFieldPolicy unknown_;
  NumericConversionPolicy numeric_;
};

/// @brief Own immutable typed-encoding policy for one codec call.
class EncodeOptions {
 public:
  /// @brief Create the version-one typed-encoding policy.
  EncodeOptions() noexcept = default;
};

/// @brief Own one value from the closed configuration semantic algebra.
///
/// Values are immutable after construction. Copies share immutable storage and may be read
/// concurrently. Except for `null()`, factories can allocate. Short forms throw on failure;
/// `try_*` forms return a structured error, including allocation exhaustion. Use `try_*` when
/// allocation exhaustion must remain reportable because constructing an exception can allocate.
class HOLOSCAN_CONFIG_EXPORT ConfigValue {
 public:
  /// @brief Create the null value without allocating.
  ConfigValue() noexcept;

  /// @brief Create the null value without allocating.
  /// @return An owning null value.
  static ConfigValue null() noexcept;
  /// @brief Create a Boolean value.
  /// @param value Boolean to copy.
  /// @return The owning value.
  /// @throws ConfigException when an allocation failure can be reported.
  [[nodiscard]] static ConfigValue boolean(bool value);
  /// @brief Try to create a Boolean value.
  /// @param value Boolean to copy.
  /// @return The owning value or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] static ConfigResult<ConfigValue> try_boolean(bool value) noexcept;
  /// @brief Create a signed 64-bit integer value.
  /// @param value Integer to copy.
  /// @return The owning value.
  /// @throws ConfigException when an allocation failure can be reported.
  [[nodiscard]] static ConfigValue signed_integer(std::int64_t value);
  /// @brief Try to create a signed 64-bit integer value.
  /// @param value Integer to copy.
  /// @return The owning value or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] static ConfigResult<ConfigValue> try_signed_integer(std::int64_t value) noexcept;
  /// @brief Create an unsigned 64-bit integer value.
  /// @param value Integer to copy.
  /// @return The owning value.
  /// @throws ConfigException when an allocation failure can be reported.
  [[nodiscard]] static ConfigValue unsigned_integer(std::uint64_t value);
  /// @brief Try to create an unsigned 64-bit integer value.
  /// @param value Integer to copy.
  /// @return The owning value or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] static ConfigResult<ConfigValue> try_unsigned_integer(std::uint64_t value) noexcept;
  /// @brief Create a finite binary64 value.
  /// @param value Number to copy. Negative zero is normalized to positive zero.
  /// @return The owning value.
  /// @throws ConfigException on an unsupported value or a reportable allocation failure.
  [[nodiscard]] static ConfigValue floating(double value);
  /// @brief Try to create a finite binary64 value.
  /// @param value Number to copy. Negative zero is normalized to positive zero.
  /// @return The owning value, `CONFIG_UNSUPPORTED_VALUE`, or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] static ConfigResult<ConfigValue> try_floating(double value) noexcept;
  /// @brief Create an owning UTF-8 string value.
  /// @param value UTF-8 bytes to copy.
  /// @param limits Construction limits applied before publication.
  /// @return The owning value.
  /// @throws ConfigException on validation or a reportable allocation failure.
  [[nodiscard]] static ConfigValue string(std::string_view value, ConfigValueLimits limits = {});
  /// @brief Try to create an owning UTF-8 string value.
  /// @param value UTF-8 bytes to copy.
  /// @param limits Construction limits applied before publication.
  /// @return The owning value or a structured validation/allocation failure.
  [[nodiscard]] static ConfigResult<ConfigValue> try_string(std::string_view value,
                                                            ConfigValueLimits limits = {});
  /// @brief Create an immutable array by sharing each child value.
  /// @param values Child values in preserved array order.
  /// @param limits Bounds applied to the complete reachable tree.
  /// @return The owning aggregate.
  /// @throws ConfigException on a limit violation or reportable allocation failure.
  [[nodiscard]] static ConfigValue array(std::span<const ConfigValue> values,
                                         ConfigValueLimits limits = {});
  /// @brief Try to create an immutable array by sharing each child value.
  /// @param values Child values in preserved array order.
  /// @param limits Bounds applied to the complete reachable tree.
  /// @return The owning aggregate or a structured limit/allocation failure.
  [[nodiscard]] static ConfigResult<ConfigValue> try_array(std::span<const ConfigValue> values,
                                                           ConfigValueLimits limits = {});
  /// @brief Create an immutable unique-key object.
  /// @param entries UTF-8 keys and child values in insertion order. Keys are copied.
  /// @param limits Bounds applied to the complete reachable tree.
  /// @return The owning aggregate.
  /// @throws ConfigException on UTF-8, duplicate-key, limit, or reportable allocation failure.
  [[nodiscard]] static ConfigValue object(std::span<const ConfigObjectEntry> entries,
                                          ConfigValueLimits limits = {});
  /// @brief Try to create an immutable unique-key object.
  /// @param entries UTF-8 keys and child values in insertion order. Keys are copied.
  /// @param limits Bounds applied to the complete reachable tree.
  /// @return The owning aggregate or a structured UTF-8, duplicate, limit, or allocation failure.
  [[nodiscard]] static ConfigResult<ConfigValue> try_object(
      std::span<const ConfigObjectEntry> entries, ConfigValueLimits limits = {});

  /// @brief Return the stored semantic alternative.
  [[nodiscard]] ConfigValueKind kind() const noexcept;
  /// @brief Return a shareable immutable view of this value.
  /// @return A view that remains valid independently of this object's lifetime.
  [[nodiscard]] ConfigView view() const noexcept;

 private:
  explicit ConfigValue(std::shared_ptr<const detail::ConfigNode> node) noexcept;
  std::shared_ptr<const detail::ConfigNode> node_;

  friend class ConfigView;
  friend class Document;
  friend class Edit;
  friend class ConfigIoAccess;
  friend ConfigResult<ConfigValue> try_merge_config_values(const ConfigValue&, const ConfigValue&);
};

/// @brief Describe one key/value input to checked object construction.
struct ConfigObjectEntry {
  std::string_view key;  ///< UTF-8 key borrowed until `ConfigValue::object()` returns.
  ConfigValue value;     ///< Immutable child value shared by the constructed object.
};

/// @brief Borrow an immutable semantic value for inspection and codec traversal.
///
/// A view shares ownership of its immutable node; values returned from a Document or ConfigValue
/// remain usable after the originating object is destroyed. A default-constructed view is invalid.
/// Concurrent reads of valid views are safe.
class HOLOSCAN_CONFIG_EXPORT ConfigView {
 public:
  /// @brief Create an invalid sentinel view.
  ConfigView() noexcept = default;

  /// @brief Return whether this view addresses a semantic value.
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  /// @brief Return the semantic alternative, or `kNull` for an invalid view.
  [[nodiscard]] ConfigValueKind kind() const noexcept;
  /// @brief Return the immediate child count for an array or object.
  /// @return Zero for a scalar or invalid view.
  [[nodiscard]] std::size_t size() const noexcept;
  /// @brief Return the Boolean value.
  /// @return The stored Boolean.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` when this is not Boolean.
  [[nodiscard]] bool as_bool() const;
  /// @brief Try to return the Boolean value.
  /// @return The stored Boolean or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<bool> try_as_bool() const noexcept;
  /// @brief Return the signed 64-bit integer value.
  /// @return The stored integer.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` when the kind differs.
  [[nodiscard]] std::int64_t as_signed_integer() const;
  /// @brief Try to return the signed 64-bit integer value.
  /// @return The stored integer or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<std::int64_t> try_as_signed_integer() const noexcept;
  /// @brief Return the unsigned 64-bit integer value.
  /// @return The stored integer.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` when the kind differs.
  [[nodiscard]] std::uint64_t as_unsigned_integer() const;
  /// @brief Try to return the unsigned 64-bit integer value.
  /// @return The stored integer or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<std::uint64_t> try_as_unsigned_integer() const noexcept;
  /// @brief Return the finite binary64 value.
  /// @return The stored number.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` when the kind differs.
  [[nodiscard]] double as_floating() const;
  /// @brief Try to return the finite binary64 value.
  /// @return The stored number or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<double> try_as_floating() const noexcept;
  /// @brief Borrow the UTF-8 string bytes.
  /// @return A view valid while this ConfigView remains alive.
  /// @warning Do not retain the returned view from a temporary unless another object owns the node.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` when the kind differs.
  [[nodiscard]] std::string_view as_string() const;
  /// @brief Try to borrow the UTF-8 string bytes.
  /// @return A borrowed view or `CONFIG_TYPE_MISMATCH`.
  /// @warning Do not retain the returned view from a temporary unless another object owns the node.
  [[nodiscard]] ConfigResult<std::string_view> try_as_string() const noexcept;
  /// @brief Return one array element or object value by insertion index.
  /// @param index Zero-based child index.
  /// @return A shareable child view.
  /// @throws ConfigException with `CONFIG_PATH_NOT_FOUND` or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigView element(std::size_t index) const;
  /// @brief Try to return one child by insertion index.
  /// @param index Zero-based child index.
  /// @return A shareable child view or a structured kind/index failure.
  [[nodiscard]] ConfigResult<ConfigView> try_element(std::size_t index) const noexcept;
  /// @brief Borrow one object key by insertion index.
  /// @param index Zero-based object entry index.
  /// @return UTF-8 key bytes valid while this view remains alive.
  /// @warning Do not retain the returned view from a temporary unless another object owns the node.
  /// @throws ConfigException on an invalid kind or index.
  [[nodiscard]] std::string_view key(std::size_t index) const;
  /// @brief Try to borrow one object key by insertion index.
  /// @param index Zero-based object entry index.
  /// @return A borrowed key or a structured kind/index failure.
  /// @warning Do not retain the returned view from a temporary unless another object owns the node.
  [[nodiscard]] ConfigResult<std::string_view> try_key(std::size_t index) const noexcept;
  /// @brief Return a required object member.
  /// @param key Exact UTF-8 key bytes.
  /// @return A shareable child view.
  /// @throws ConfigException with `CONFIG_PATH_NOT_FOUND` or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigView member(std::string_view key) const;
  /// @brief Try to return a required object member.
  /// @param key Exact UTF-8 key bytes.
  /// @return A shareable child view or a structured kind/missing-member failure.
  [[nodiscard]] ConfigResult<ConfigView> try_member(std::string_view key) const noexcept;
  /// @brief Find an object member without treating absence as failure.
  /// @param key Exact UTF-8 key bytes.
  /// @return An optional child view; absence is not an error.
  /// @throws ConfigException with `CONFIG_TYPE_MISMATCH` for a non-object or invalid view.
  [[nodiscard]] std::optional<ConfigView> find_member(std::string_view key) const;
  /// @brief Try to find an object member without treating absence as failure.
  /// @param key Exact UTF-8 key bytes.
  /// @return An optional child view or `CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<std::optional<ConfigView>> try_find_member(
      std::string_view key) const noexcept;
  /// @brief Convert this view into a value sharing the same immutable node.
  /// @return The shareable owning value. An invalid view produces a null value.
  [[nodiscard]] ConfigValue to_value() const noexcept;

 private:
  explicit ConfigView(std::shared_ptr<const detail::ConfigNode> node) noexcept
      : node_(std::move(node)), valid_(true) {}
  std::shared_ptr<const detail::ConfigNode> node_;
  bool valid_{false};

  friend class ConfigValue;
  friend class Document;
  friend ConfigResult<std::vector<std::byte>> try_encode_semantic_value_v1(ConfigView,
                                                                           SemanticEncodeOptions);
};

/// @brief Own an RFC 6901 JSON Pointer path.
///
/// Paths copy and validate every token. They are independent of the source text and safe for
/// concurrent reads after construction.
class HOLOSCAN_CONFIG_EXPORT Path {
 public:
  /// @brief Parse and own an RFC 6901 JSON Pointer.
  /// @param text Empty root text or slash-prefixed UTF-8 pointer text.
  /// @return The owned path.
  /// @throws ConfigException with `CONFIG_PATH_SYNTAX`, `CONFIG_INVALID_UTF8`, or
  /// `CONFIG_LIMIT_EXCEEDED` on failure.
  /// @see try_parse
  [[nodiscard]] static Path parse(std::string_view text);
  /// @brief Try to parse and own an RFC 6901 JSON Pointer.
  /// @param text Empty root text or slash-prefixed UTF-8 pointer text.
  /// @return The owned path or `CONFIG_PATH_SYNTAX`/`CONFIG_INVALID_UTF8`/
  /// `CONFIG_LIMIT_EXCEEDED`.
  /// @see parse
  [[nodiscard]] static ConfigResult<Path> try_parse(std::string_view text);
  /// @brief Create the empty root path without allocating.
  static Path root() noexcept;

  /// @brief Return the number of decoded tokens.
  [[nodiscard]] std::size_t token_count() const noexcept { return tokens_.size(); }
  /// @brief Borrow one decoded token.
  /// @param index Zero-based token index.
  /// @return The token, or an empty view when \p index is out of range.
  [[nodiscard]] std::string_view token(std::size_t index) const noexcept;
  /// @brief Render the path with RFC 6901 escaping.
  /// @return Newly allocated pointer text; the root renders as an empty string.
  [[nodiscard]] std::string to_string() const;
  /// @brief Return whether this path addresses the document root.
  [[nodiscard]] bool is_root() const noexcept { return tokens_.empty(); }

 private:
  std::vector<std::string> tokens_;
};

namespace detail {

/// @brief Build a path diagnostic without allowing path rendering allocation to escape.
/// @param code Failure category used when rendering succeeds.
/// @param message Failure context used when rendering succeeds.
/// @param path Path rendered into the diagnostic.
/// @return The requested diagnostic, or an allocation-limit diagnostic.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigError make_path_error(ConfigErrorCode code,
                                                                 std::string_view message,
                                                                 const Path& path) noexcept;

}  // namespace detail

/// @brief Store a SHA-256 digest as 32 raw bytes in digest order.
using Sha256Digest = std::array<std::byte, 32>;

/// @brief Identify the exact source bytes from which a document was parsed or loaded.
struct RawSourceDigest {
  std::uint32_t algorithm_version{1};  ///< Raw-byte digest grammar version.
  Sha256Digest sha256{};               ///< SHA-256 of the exact source bytes.
  std::uint64_t byte_size{0};          ///< Exact number of source bytes covered.
  /// @brief Compare every identity field.
  friend bool operator==(const RawSourceDigest&, const RawSourceDigest&) = default;
};

/// @brief Identify a semantic configuration tree independent of its renderer.
struct SemanticDigest {
  std::uint32_t canonical_version{1};  ///< Typed semantic grammar version.
  Sha256Digest sha256{};               ///< Domain-separated SHA-256 of canonical semantic bytes.
  /// @brief Compare every identity field.
  friend bool operator==(const SemanticDigest&, const SemanticDigest&) = default;
};

/// @brief Pair semantic identity with optional exact-current and parsed-base source evidence.
struct ConfigDigest {
  SemanticDigest semantic{};  ///< Identity of the immutable semantic tree.
  std::optional<RawSourceDigest>
      source;  ///< Exact retained bytes for the current tree; absent after semantic-only edits.
  std::optional<RawSourceDigest>
      base_source;  ///< First parsed bytes behind a derived tree; use for persistence CAS.
};

/// @brief Bound the backend-neutral semantic value encoding output.
struct SemanticEncodeOptions {
  std::size_t max_output_bytes{64U * 1024U * 1024U};  ///< Maximum emitted semantic bytes.
};

/// @brief Encode one value with the version-one typed semantic grammar.
///
/// Object entries are sorted by unsigned UTF-8 bytes; array order and numeric semantic kinds are
/// preserved. The output is backend-neutral and is not JSON or YAML text.
///
/// @param value Valid semantic value to encode.
/// @param options Output byte bound checked before each append.
/// @return Canonical typed bytes.
/// @throws ConfigException on invalid input, limit, or allocation failure.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT std::vector<std::byte> encode_semantic_value_v1(
    ConfigView value, SemanticEncodeOptions options = {});

/// @brief Try to encode one value with the version-one typed semantic grammar.
/// @param value Valid semantic value to encode.
/// @param options Output byte bound checked before each append.
/// @return Canonical typed bytes or a structured invalid-input/limit/allocation failure.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigResult<std::vector<std::byte>>
try_encode_semantic_value_v1(ConfigView value, SemanticEncodeOptions options = {});

/// @brief Enumerate immutable document edit operations.
enum class EditKind : std::uint8_t {
  kAdd,         ///< Add a previously absent object member or array element.
  kReplace,     ///< Replace an existing value.
  kRemove,      ///< Remove an existing object member or array element.
  kMergeObject  ///< Recursively merge an object at an existing path.
};

/// @brief Own one transactional edit request.
class HOLOSCAN_CONFIG_EXPORT Edit {
 public:
  /// @brief Create an add operation.
  /// @param path Destination path owned by the edit.
  /// @param value Immutable value to add.
  static Edit add(Path path, ConfigValue value);
  /// @brief Parse a destination path and create an add operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Immutable value to add.
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit add(std::string_view path, ConfigValue value);
  /// @brief Parse a destination path and create an add operation from a shareable value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Shareable value view converted to an owning value.
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit add(std::string_view path, ConfigView value);
  /// @brief Try to parse a destination path and create an add operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Immutable value to add.
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_add(std::string_view path, ConfigValue value);
  /// @brief Try to parse a destination path and create an add operation from a value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Shareable value view converted through ConfigView::to_value().
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_add(std::string_view path, ConfigView value);
  /// @brief Create a replacement operation.
  /// @param path Existing destination path owned by the edit.
  /// @param value Immutable replacement value.
  static Edit replace(Path path, ConfigValue value);
  /// @brief Parse a destination path and create a replacement operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Immutable replacement value.
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit replace(std::string_view path, ConfigValue value);
  /// @brief Parse a destination path and create a replacement operation from a value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Shareable value view converted through ConfigView::to_value().
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit replace(std::string_view path, ConfigView value);
  /// @brief Try to parse a destination path and create a replacement operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Immutable replacement value.
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_replace(std::string_view path, ConfigValue value);
  /// @brief Try to parse a destination path and create a replacement operation from a value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param value Shareable value view converted through ConfigView::to_value().
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_replace(std::string_view path, ConfigView value);
  /// @brief Create a removal operation.
  /// @param path Existing destination path owned by the edit.
  static Edit remove(Path path);
  /// @brief Parse a destination path and create a removal operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit remove(std::string_view path);
  /// @brief Try to parse a destination path and create a removal operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_remove(std::string_view path);
  /// @brief Create a recursive object-merge operation.
  /// @param path Existing destination path owned by the edit.
  /// @param object Object-valued patch; a non-object is rejected when applied.
  static Edit merge_object(Path path, ConfigValue object);
  /// @brief Parse a destination path and create a recursive object-merge operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param object Immutable object-valued patch.
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit merge_object(std::string_view path, ConfigValue object);
  /// @brief Parse a destination path and create an object-merge operation from a value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param object Shareable value view converted through ConfigView::to_value().
  /// @return The owning edit.
  /// @throws ConfigException when \p path is not valid RFC 6901 text.
  static Edit merge_object(std::string_view path, ConfigView object);
  /// @brief Try to parse a destination path and create a recursive object-merge operation.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param object Immutable object-valued patch.
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_merge_object(std::string_view path,
                                                           ConfigValue object);
  /// @brief Try to parse a destination path and create an object-merge operation from a value view.
  /// @param path RFC 6901 destination path copied by the edit.
  /// @param object Shareable value view converted through ConfigView::to_value().
  /// @return The owning edit or a structured path error.
  [[nodiscard]] static ConfigResult<Edit> try_merge_object(std::string_view path,
                                                           ConfigView object);

  /// @brief Return the requested operation.
  [[nodiscard]] EditKind kind() const noexcept { return kind_; }
  /// @brief Return the owned destination path.
  [[nodiscard]] const Path& path() const noexcept { return path_; }
  /// @brief Return the immutable operation value.
  ///
  /// Remove operations return their internal null placeholder.
  [[nodiscard]] const ConfigValue& value() const noexcept { return value_; }

 private:
  Edit(EditKind kind, Path path, ConfigValue value) noexcept;
  EditKind kind_;
  Path path_;
  ConfigValue value_;
};

/// @brief Own an immutable backend-neutral configuration document.
///
/// Copies share immutable state and are safe for concurrent reads. Every successful edit publishes
/// a distinct Document; no operation mutates the baseline. Documents created by config_io may also
/// retain original-source identity and an internal lossless scalar-span sidecar.
class HOLOSCAN_CONFIG_EXPORT Document {
 public:
  /// @brief Create a document and compute its semantic identity.
  /// @param root Root value to share with the document.
  /// @return The immutable document.
  /// @throws ConfigException on allocation or semantic encoding failure.
  [[nodiscard]] static Document from_value(ConfigValue root);
  /// @brief Try to create a document and compute its semantic identity.
  /// @param root Root value to share with the document.
  /// @return The immutable document or a structured allocation/encoding failure.
  [[nodiscard]] static ConfigResult<Document> try_from_value(ConfigValue root) noexcept;

  /// @brief Return a shareable view of the root value.
  [[nodiscard]] ConfigView root() const noexcept;
  /// @brief Return the value at an owned path.
  /// @param path Root-relative RFC 6901 path.
  /// @return A shareable view.
  /// @throws ConfigException with `CONFIG_PATH_NOT_FOUND` or `CONFIG_TYPE_MISMATCH` on lookup
  /// failure.
  [[nodiscard]] ConfigView at(const Path& path) const;
  /// @brief Try to return the value at an owned path.
  /// @param path Root-relative RFC 6901 path.
  /// @return A shareable view or `CONFIG_PATH_NOT_FOUND`/`CONFIG_TYPE_MISMATCH`.
  [[nodiscard]] ConfigResult<ConfigView> try_at(const Path& path) const;
  /// @brief Parse a path and return the addressed value.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @return A shareable view.
  /// @throws ConfigException on path syntax, missing value, or scalar traversal.
  [[nodiscard]] ConfigView at(std::string_view path) const;
  /// @brief Try to parse a path and return the addressed value.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @return A shareable view or a structured path/lookup/type failure.
  [[nodiscard]] ConfigResult<ConfigView> try_at(std::string_view path) const;
  /// @brief Find a value without treating absence as failure.
  /// @param path Root-relative RFC 6901 path.
  /// @return The optional shareable view; absence is not an error.
  /// @throws ConfigException on a traversal/type failure.
  [[nodiscard]] std::optional<ConfigView> find(const Path& path) const;
  /// @brief Try to find a value without treating absence as failure.
  /// @param path Root-relative RFC 6901 path.
  /// @return The optional shareable view, or a traversal/type failure.
  [[nodiscard]] ConfigResult<std::optional<ConfigView>> try_find(const Path& path) const;
  /// @brief Parse a path and find a value without treating absence as failure.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @return The optional shareable view; absence is not an error.
  /// @throws ConfigException on a path syntax or traversal failure.
  [[nodiscard]] std::optional<ConfigView> find(std::string_view path) const;
  /// @brief Try to parse a path and find a value without treating absence as failure.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @return The optional shareable view, or a path/traversal failure.
  [[nodiscard]] ConfigResult<std::optional<ConfigView>> try_find(std::string_view path) const;
  /// @brief Decode one addressed value through the explicit ConfigCodec specialization.
  /// @tparam T Owning destination type satisfying ConfigCodable.
  /// @param path Root-relative path.
  /// @param options Immutable typed-decoding policy.
  /// @return The decoded value.
  /// @throws ConfigException on a path or codec failure.
  template <class T>
  [[nodiscard]] T get(const Path& path, DecodeOptions options = DecodeOptions{}) const;
  /// @brief Try to decode one addressed value through the explicit ConfigCodec specialization.
  /// @tparam T Owning destination type satisfying ConfigCodable.
  /// @param path Root-relative path.
  /// @param options Immutable typed-decoding policy.
  /// @return The decoded value or a structured path/codec failure.
  template <class T>
  [[nodiscard]] ConfigResult<T> try_get(const Path& path,
                                        DecodeOptions options = DecodeOptions{}) const;
  /// @brief Parse a path and decode its value through the explicit ConfigCodec specialization.
  /// @tparam T Owning destination type satisfying ConfigCodable.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @param options Immutable typed-decoding policy.
  /// @return The decoded value.
  /// @throws ConfigException on a path or codec failure.
  template <class T>
  [[nodiscard]] T get(std::string_view path, DecodeOptions options = DecodeOptions{}) const;
  /// @brief Try to parse a path and decode it through the explicit ConfigCodec specialization.
  /// @tparam T Owning destination type satisfying ConfigCodable.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @param options Immutable typed-decoding policy.
  /// @return The decoded value or a structured path/codec failure.
  template <class T>
  [[nodiscard]] ConfigResult<T> try_get(std::string_view path,
                                        DecodeOptions options = DecodeOptions{}) const;
  /// @brief Apply an object-valued configuration patch to a typed baseline transactionally.
  ///
  /// The baseline is encoded, recursively merged with the addressed object, and decoded into one
  /// successor. Neither \p baseline nor this document is modified on success or failure.
  ///
  /// @tparam T Owning type satisfying ConfigCodable.
  /// @param path Path to an object-valued patch.
  /// @param baseline Baseline value copied into the transaction.
  /// @param options Immutable decoding policy for the successor.
  /// @return The typed successor.
  /// @throws ConfigException on a path, type, or codec failure.
  template <class T>
  [[nodiscard]] T patch(const Path& path, T baseline,
                        DecodeOptions options = DecodeOptions{}) const;
  /// @brief Try to apply an object-valued configuration patch to a typed baseline.
  /// @tparam T Owning type satisfying ConfigCodable.
  /// @param path Path to an object-valued patch.
  /// @param baseline Baseline value copied into the transaction.
  /// @param options Immutable decoding policy for the successor.
  /// @return The typed successor or a structured path/type/codec failure.
  template <class T>
  [[nodiscard]] ConfigResult<T> try_patch(const Path& path, T baseline,
                                          DecodeOptions options = DecodeOptions{}) const;
  /// @brief Encode a typed value and transactionally replace an existing path.
  /// @tparam T Type satisfying ConfigCodable after cv/ref removal.
  /// @param path Existing destination path.
  /// @param value Typed replacement value; the caller retains ownership.
  /// @return A successor document.
  /// @throws ConfigException on a codec or edit failure.
  template <class T>
  [[nodiscard]] Document replace(const Path& path, T&& value) const;
  /// @brief Try to encode a typed value and replace an existing path transactionally.
  /// @tparam T Type satisfying ConfigCodable after cv/ref removal.
  /// @param path Existing destination path.
  /// @param value Typed replacement value; the caller retains ownership.
  /// @return A successor document or a structured codec/edit failure.
  template <class T>
  [[nodiscard]] ConfigResult<Document> try_replace(const Path& path, T&& value) const;
  /// @brief Parse a path, encode a typed value, and transactionally replace it.
  /// @tparam T Type satisfying ConfigCodable after cv/ref removal.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @param value Typed replacement value; the caller retains ownership.
  /// @return A successor document.
  /// @throws ConfigException on a path, codec, or edit failure.
  template <class T>
  [[nodiscard]] Document replace(std::string_view path, T&& value) const;
  /// @brief Try to parse a path, encode a typed value, and replace it transactionally.
  /// @tparam T Type satisfying ConfigCodable after cv/ref removal.
  /// @param path Empty root text or slash-prefixed RFC 6901 text.
  /// @param value Typed replacement value; the caller retains ownership.
  /// @return A successor document or a structured path/codec/edit failure.
  template <class T>
  [[nodiscard]] ConfigResult<Document> try_replace(std::string_view path, T&& value) const;
  /// @brief Apply a non-overlapping batch of immutable edits transactionally.
  ///
  /// In lossless mode, only existing non-root scalar replacements that retain semantic kind and
  /// source style are admitted; every byte outside the replaced scalar tokens is preserved.
  /// A lossless document parsed from alias-bearing YAML rejects every edit; parse it with
  /// `Preservation::kSemanticOnly` when edits are required.
  /// Structural edits clear source-byte preservation authority in semantic-only documents.
  ///
  /// @param edits Complete transaction. Duplicate or ancestor/descendant paths conflict.
  /// @return A successor document.
  /// @throws ConfigException on failure, with this document unchanged.
  [[nodiscard]] Document patched(std::span<const Edit> edits) const;
  /// @brief Try to apply a non-overlapping batch of immutable edits transactionally.
  /// @param edits Complete transaction. Duplicate or ancestor/descendant paths conflict.
  /// @return A successor document, or a structured failure with this document unchanged.
  [[nodiscard]] ConfigResult<Document> try_patched(std::span<const Edit> edits) const;
  /// @brief Apply one immutable edit transactionally.
  /// @param edit Complete single-edit transaction.
  /// @return A successor document.
  /// @throws ConfigException on failure, with this document unchanged.
  /// @see patched(std::span<const Edit>)
  [[nodiscard]] Document patched(const Edit& edit) const {
    return patched(std::span<const Edit>{&edit, 1});
  }
  /// @brief Try to apply one immutable edit transactionally.
  /// @param edit Complete single-edit transaction.
  /// @return A successor document, or a structured failure with this document unchanged.
  /// @see try_patched(std::span<const Edit>)
  [[nodiscard]] ConfigResult<Document> try_patched(const Edit& edit) const {
    return try_patched(std::span<const Edit>{&edit, 1});
  }
  /// @brief Return immutable semantic identity and optional original-source provenance.
  [[nodiscard]] ConfigDigest digest() const noexcept;

 private:
  explicit Document(std::shared_ptr<const detail::DocumentState> state) noexcept;
  std::shared_ptr<const detail::DocumentState> state_;

  friend class DocumentAccess;
  friend class ConfigIoAccess;
};

/// @brief Recursively merge object values while replacing arrays and scalars as whole values.
///
/// @param baseline Immutable baseline value.
/// @param patch Immutable patch value. Object/object pairs recurse; every other pair replaces the
/// baseline with \p patch.
/// @return The merged immutable value.
/// @throws ConfigException on limit or allocation failure.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigValue merge_config_values(const ConfigValue& baseline,
                                                                     const ConfigValue& patch);

/// @brief Try to recursively merge object values while replacing other values as wholes.
/// @param baseline Immutable baseline value.
/// @param patch Immutable patch value.
/// @return The merged immutable value or a structured limit/allocation failure.
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigResult<ConfigValue> try_merge_config_values(
    const ConfigValue& baseline, const ConfigValue& patch);

}  // namespace holoscan::config
