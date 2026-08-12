/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/io.hpp>

#include "config/config_internal.hpp"
#include "config/io_internal.hpp"
#include "config/json_yyjson.hpp"
#include "config/reserved_envelope.hpp"
#include "config/utf8.hpp"
#include "config/yaml_scalar_policy.hpp"

#if HOLOSCAN_CONFIG_IO_USE_YAML_CPP
#include <yaml-cpp/exceptions.h>
#include <yaml-cpp/yaml.h>
#endif

#if !defined(__cpp_exceptions) && !defined(_CPPUNWIND)
#error "config_io requires exceptions; config remains exception-optional"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace holoscan::config_io {
namespace {

constexpr std::size_t kMaxSourceBytes = 16U * 1024U * 1024U;
constexpr std::size_t kMaxOutputBytes = 64U * 1024U * 1024U;
constexpr std::uint32_t kMaxDepth = 64;
constexpr std::uint32_t kMaxNodes = 1'000'000;
constexpr std::size_t kMaxStringBytes = 4U * 1024U * 1024U;
constexpr std::uint32_t kMaxContainerEntries = 1'000'000;

[[nodiscard]] config::ConfigError failure(
    config::ConfigErrorCode code, std::string_view message,
    config::ConfigOperationStage stage = config::ConfigOperationStage::kNone,
    config::ConfigDestinationState state = config::ConfigDestinationState::kNotApplicable,
    std::string_view source = {}) noexcept {
  return config::ConfigError::make(code, message, {}, source, {}, stage, state);
}

[[nodiscard]] bool valid_parse_options(const ParseOptions& options) noexcept {
  return (options.preservation == Preservation::kSemanticOnly ||
          options.preservation == Preservation::kRequireLossless) &&
         options.max_source_bytes > 0 && options.max_source_bytes <= kMaxSourceBytes &&
         options.max_depth > 0 && options.max_depth <= kMaxDepth && options.max_nodes > 0 &&
         options.max_nodes <= kMaxNodes && options.max_string_bytes > 0 &&
         options.max_string_bytes <= kMaxStringBytes && options.max_container_entries > 0 &&
         options.max_container_entries <= kMaxContainerEntries;
}

[[nodiscard]] bool valid_format(Format format) noexcept {
  return format == Format::kJson || format == Format::kYaml;
}

[[nodiscard]] bool valid_durability(Durability durability) noexcept {
  return durability == Durability::kData || durability == Durability::kDataAndMetadata;
}

[[nodiscard]] bool valid_output_limits(const OutputLimits& limits) noexcept {
  return limits.max_output_bytes > 0 && limits.max_output_bytes <= kMaxOutputBytes &&
         limits.max_depth > 0 && limits.max_depth <= kMaxDepth && limits.max_nodes > 0 &&
         limits.max_nodes <= kMaxNodes && limits.max_string_bytes > 0 &&
         limits.max_string_bytes <= kMaxStringBytes && limits.max_container_entries > 0 &&
         limits.max_container_entries <= kMaxContainerEntries;
}

[[nodiscard]] config::ConfigValueLimits construction_limits(const ParseOptions& options) noexcept {
  return {.max_depth = options.max_depth,
          .max_nodes = options.max_nodes,
          .max_string_bytes = options.max_string_bytes,
          .max_container_entries = options.max_container_entries};
}

[[nodiscard]] std::string append_pointer_token(std::string_view path, std::string_view token) {
  std::string result{path};
  result.push_back('/');
  for (const char byte : token) {
    if (byte == '~') {
      result += "~0";
    } else if (byte == '/') {
      result += "~1";
    } else {
      result.push_back(byte);
    }
  }
  return result;
}

#if HOLOSCAN_CONFIG_IO_USE_YAML_CPP
enum class YamlScalarStyle : std::uint8_t {
  kPlain,
  kSingleQuoted,
  kDoubleQuoted,
};

struct ExactScalarSource {
  ExactScalarSource(YamlScalarStyle source_style, std::string_view source_bytes,
                    std::size_t source_begin) noexcept
      : style(source_style), bytes(source_bytes), begin(source_begin) {}

  [[nodiscard]] std::size_t end() const noexcept { return begin + bytes.size(); }

  YamlScalarStyle style;
  std::string_view bytes;
  std::size_t begin;
};

struct AbsentScalarSource {};

enum class ScalarSourceFailure : std::uint8_t {
  kUnstableMark,
  kMissingToken,
  kDecodedValueMismatch,
  kMalformedQuotedToken,
};

struct UntrustedScalarSource {
  ScalarSourceFailure reason;
};

using RecoveredScalarSource =
    std::variant<ExactScalarSource, AbsentScalarSource, UntrustedScalarSource>;

[[nodiscard]] std::string_view scalar_source_failure_message(ScalarSourceFailure reason,
                                                             bool mapping_key = false) noexcept {
  switch (reason) {
    case ScalarSourceFailure::kUnstableMark:
      return "YAML node has no stable source token";
    case ScalarSourceFailure::kMissingToken:
      return mapping_key ? "YAML mapping key has no attributable source token"
                         : "YAML scalar has no attributable source token";
    case ScalarSourceFailure::kDecodedValueMismatch:
      return mapping_key ? "folded YAML mapping keys are unsupported"
                         : "folded YAML plain scalars are unsupported";
    case ScalarSourceFailure::kMalformedQuotedToken:
      return mapping_key ? "YAML quoted mapping key source is unterminated"
                         : "YAML quoted scalar source is unterminated";
  }
  return "YAML scalar source token is untrusted";
}

[[nodiscard]] bool yaml_cpp_null_spelling(std::string_view token) noexcept {
  return token == "~" || token == "null" || token == "Null" || token == "NULL";
}

struct DecodedYamlKey {
  std::string value;
  bool alias_occurrence{false};
  bool plain_merge_key{false};
};

struct YamlSourcePolicy {
  std::optional<std::size_t> directive_offset;
  std::optional<std::size_t> multi_document_offset;
};

[[nodiscard]] bool has_utf8_bom(std::string_view source) noexcept {
  return source.size() >= 3U && static_cast<unsigned char>(source[0]) == 0xEFU &&
         static_cast<unsigned char>(source[1]) == 0xBBU &&
         static_cast<unsigned char>(source[2]) == 0xBFU;
}

// yaml-cpp admits a comment immediately after some closed tokens. Callers prove that lexical
// boundary explicitly because the same punctuation can also be ordinary plain-scalar content.
[[nodiscard]] bool yaml_comment_start(std::string_view source, std::size_t position,
                                      bool after_token_boundary = false) noexcept {
  if (position >= source.size() || source[position] != '#') {
    return false;
  }
  if (position == 0U) {
    return true;
  }
  const char previous = source[position - 1U];
  return previous == ' ' || previous == '\t' || previous == '\r' || previous == '\n' ||
         after_token_boundary;
}

[[nodiscard]] bool yaml_document_marker(std::string_view source, std::size_t position,
                                        std::string_view marker) noexcept {
  if (position > source.size() || source.size() - position < marker.size() ||
      source.substr(position, marker.size()) != marker) {
    return false;
  }
  const auto after = position + marker.size();
  return after == source.size() || source[after] == ' ' || source[after] == '\t' ||
         source[after] == '\r' || source[after] == '\n';
}

// This scan records preservation metadata and fail-closed stream policy only. Collection topology
// always comes from yaml-cpp's node graph; the scan never discovers or reconstructs YAML parents
// or children.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] YamlSourcePolicy inspect_yaml_source(std::string_view source) noexcept {
  YamlSourcePolicy result;
  bool in_single_quote = false;
  bool in_double_quote = false;
  bool escaped = false;
  bool in_comment = false;
  bool at_line_start = true;
  bool line_indented = false;
  bool line_has_content = false;
  bool scalar_can_start = true;
  bool in_plain_scalar = false;
  bool plain_scalar_continuation = false;
  bool in_property = false;
  std::uint32_t flow_depth = 0;
  bool comment_may_start = false;
  bool saw_document_start = false;
  bool saw_document_content = false;
  bool saw_document_end = false;
  for (std::size_t index = 0; index < source.size(); ++index) {
    if (index == 0U && has_utf8_bom(source)) {
      index = 2U;
      continue;
    }
    const char value = source[index];
    if (value == '\r' || value == '\n') {
      const bool second_byte_of_crlf = value == '\n' && index > 0U && source[index - 1U] == '\r';
      if (!in_single_quote && !in_double_quote && !second_byte_of_crlf) {
        if (line_has_content) {
          plain_scalar_continuation = in_plain_scalar;
        }
        scalar_can_start = true;
        in_plain_scalar = false;
        in_property = false;
        line_has_content = false;
      }
      in_comment = false;
      at_line_start = true;
      line_indented = false;
      comment_may_start = false;
      escaped = false;
      continue;
    }
    if (in_comment) {
      continue;
    }
    if (in_double_quote) {
      if (escaped) {
        escaped = false;
      } else if (value == '\\') {
        escaped = true;
      } else if (value == '"') {
        in_double_quote = false;
        comment_may_start = true;
        scalar_can_start = false;
      }
      continue;
    }
    if (in_single_quote) {
      if (value == '\'' && index + 1U < source.size() && source[index + 1U] == '\'') {
        ++index;
      } else if (value == '\'') {
        in_single_quote = false;
        comment_may_start = true;
        scalar_can_start = false;
      }
      continue;
    }
    if (at_line_start && (value == ' ' || value == '\t')) {
      line_indented = true;
      continue;
    }
    const bool comment_start = yaml_comment_start(source, index, comment_may_start);
    comment_may_start = false;
    if (comment_start) {
      in_comment = true;
      continue;
    }
    if (at_line_start) {
      at_line_start = false;
      if (!line_indented && value == '%') {
        result.directive_offset = index;
      }
      if (!line_indented && yaml_document_marker(source, index, "---")) {
        if (saw_document_start || saw_document_content || saw_document_end) {
          result.multi_document_offset = index;
        }
        saw_document_start = true;
        plain_scalar_continuation = false;
        scalar_can_start = true;
        in_plain_scalar = false;
        line_has_content = true;
        index += 2U;
        continue;
      }
      if (!line_indented && yaml_document_marker(source, index, "...")) {
        if (saw_document_end) {
          result.multi_document_offset = index;
        }
        saw_document_end = true;
        plain_scalar_continuation = false;
        scalar_can_start = false;
        in_plain_scalar = false;
        line_has_content = true;
        index += 2U;
        continue;
      }
      if (line_indented && plain_scalar_continuation) {
        scalar_can_start = false;
        in_plain_scalar = true;
      } else if (!line_indented) {
        plain_scalar_continuation = false;
      }
    }
    if (value == ' ' || value == '\t') {
      if (in_property) {
        in_property = false;
        scalar_can_start = true;
      }
      continue;
    }
    if (saw_document_end && !result.multi_document_offset) {
      result.multi_document_offset = index;
    }
    saw_document_content = true;
    line_has_content = true;
    if (in_property) {
      if (value != ',' && value != '[' && value != ']' && value != '{' && value != '}') {
        continue;
      }
      in_property = false;
      scalar_can_start = true;
    }
    if (scalar_can_start && value == '\'') {
      in_single_quote = true;
      scalar_can_start = false;
      in_plain_scalar = false;
      continue;
    }
    if (scalar_can_start && value == '"') {
      in_double_quote = true;
      scalar_can_start = false;
      in_plain_scalar = false;
      continue;
    }
    if (scalar_can_start && (value == '&' || value == '!')) {
      in_property = true;
      scalar_can_start = false;
      in_plain_scalar = false;
      continue;
    }
    const auto next_is_separation = [&]() {
      if (index + 1U == source.size()) {
        return true;
      }
      const char next = source[index + 1U];
      return next == ' ' || next == '\t' || next == '\r' || next == '\n' || next == ',' ||
             next == '[' || next == ']' || next == '{' || next == '}';
    };
    const bool scalar_separator =
        (value == ':' && (next_is_separation() || (flow_depth > 0U && !in_plain_scalar))) ||
        (flow_depth > 0U && value == ',');
    if (scalar_separator) {
      comment_may_start = true;
      scalar_can_start = true;
      in_plain_scalar = false;
    } else if (scalar_can_start && (value == '[' || value == '{')) {
      ++flow_depth;
      comment_may_start = true;
      scalar_can_start = true;
      in_plain_scalar = false;
    } else if (flow_depth > 0U && (value == ']' || value == '}')) {
      --flow_depth;
      comment_may_start = true;
      scalar_can_start = false;
      in_plain_scalar = false;
    } else if (scalar_can_start && (value == '-' || value == '?') && next_is_separation()) {
      in_plain_scalar = false;
    } else if (scalar_can_start) {
      scalar_can_start = false;
      in_plain_scalar = true;
    }
  }
  return result;
}

// Keep the closed numeric grammar explicit so each accepted transition is auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool exact_json_number_grammar(std::string_view token) noexcept {
  std::size_t position = 0;
  if (position < token.size() && token[position] == '-') {
    ++position;
  }
  if (position == token.size()) {
    return false;
  }
  if (token[position] == '0') {
    ++position;
  } else if (token[position] >= '1' && token[position] <= '9') {
    do {
      ++position;
    } while (position < token.size() && token[position] >= '0' && token[position] <= '9');
  } else {
    return false;
  }
  if (position < token.size() && token[position] == '.') {
    ++position;
    const std::size_t fractional_begin = position;
    while (position < token.size() && token[position] >= '0' && token[position] <= '9') {
      ++position;
    }
    if (position == fractional_begin) {
      return false;
    }
  }
  if (position < token.size() && (token[position] == 'e' || token[position] == 'E')) {
    ++position;
    if (position < token.size() && (token[position] == '+' || token[position] == '-')) {
      ++position;
    }
    const std::size_t exponent_begin = position;
    while (position < token.size() && token[position] >= '0' && token[position] <= '9') {
      ++position;
    }
    if (position == exponent_begin) {
      return false;
    }
  }
  return position == token.size();
}

class YamlNodeConverter {
 public:
  YamlNodeConverter(std::string_view source, std::string_view source_name, ParseOptions options,
                    std::vector<config::detail::LosslessScalarSpan>* spans, bool* contains_alias)
      : source_(source),
        source_name_(source_name),
        options_(options),
        limits_(construction_limits(options)),
        spans_(spans),
        contains_alias_output_(contains_alias) {}

  [[nodiscard]] config::ConfigResult<config::ConfigValue> run() {
    const auto policy = inspect_yaml_source(source_);
    const auto invalid_utf8 = config::detail::invalid_utf8_offset(source_);
    try {
      const auto document = YAML::Load(std::string{source_});
      if (invalid_utf8) {
        if (!document.Mark().is_null()) {
          auto converted = convert(document, 1U, {});
          if (!converted && converted.error().code() == config::ConfigErrorCode::kInvalidUtf8) {
            return converted;
          }
        }
        return tl::make_unexpected(make_error(
            config::ConfigErrorCode::kInvalidUtf8, "YAML source is not UTF-8", {}, *invalid_utf8));
      }
      if (policy.directive_offset) {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                              "YAML directives are unsupported",
                                              {},
                                              *policy.directive_offset));
      }
      if (policy.multi_document_offset) {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                              "multiple YAML documents are unsupported",
                                              {},
                                              *policy.multi_document_offset));
      }
      config::ConfigResult<config::ConfigValue> converted =
          document.Mark().is_null() ? convert_empty_document() : convert(document, 1U, {});
      if (contains_alias_output_ != nullptr) {
        *contains_alias_output_ = contains_alias_;
      }
      if (converted && contains_alias_ && spans_ != nullptr) {
        // yaml-cpp retains an anchor-definition mark for aliases, not the alias-use mark. Keeping
        // those duplicated spans would make an edited tree appear losslessly representable.
        spans_->clear();
      }
      return converted;
    } catch (const YAML::Exception& error) {
      if (invalid_utf8) {
        return tl::make_unexpected(make_error(
            config::ConfigErrorCode::kInvalidUtf8, "YAML source is not UTF-8", {}, *invalid_utf8));
      }
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kParseError,
                                            "input does not conform to YAML syntax",
                                            {},
                                            error.mark));
    }
  }

 private:
  struct NodeMarkHash {
    [[nodiscard]] std::size_t operator()(const YAML::Node& node) const {
      return static_cast<std::size_t>(node.Mark().pos);
    }
  };

  struct NodeIdentityEqual {
    [[nodiscard]] bool operator()(const YAML::Node& lhs, const YAML::Node& rhs) const {
      return lhs.Mark().pos == rhs.Mark().pos && lhs.is(rhs);
    }
  };

  [[nodiscard]] std::size_t source_position(YAML::Mark mark) const noexcept {
    const std::size_t parser_position = static_cast<std::size_t>(mark.pos);
    const std::size_t bom_size = has_utf8_bom(source_) ? 3U : 0U;
    return parser_position > source_.size() - bom_size ? source_.size()
                                                       : parser_position + bom_size;
  }

  [[nodiscard]] config::ConfigError make_error(config::ConfigErrorCode code,
                                               std::string_view message, std::string_view path,
                                               YAML::Mark mark) const {
    if (mark.is_null() || mark.pos < 0) {
      return config::ConfigError::make(
          code, message, path, source_name_, {}, config::ConfigOperationStage::kParse);
    }
    return make_error(code, message, path, source_position(mark), mark.line, mark.column);
  }

  [[nodiscard]] config::ConfigError make_error(config::ConfigErrorCode code,
                                               std::string_view message, std::string_view path,
                                               std::size_t position, int line = -1,
                                               int column = -1) const {
    const std::size_t offset = std::min(position, source_.size());
    const std::size_t excerpt_offset =
        offset == source_.size() && !source_.empty()
            ? source_.size() - std::min<std::size_t>(source_.size(), 256U)
            : offset;
    const auto* excerpt = source_.empty()
                              ? nullptr
                              : reinterpret_cast<const std::byte*>(source_.data() + excerpt_offset);
    if (line < 0 || column < 0) {
      line = 0;
      column = 0;
      for (std::size_t index = has_utf8_bom(source_) ? 3U : 0U; index < offset; ++index) {
        if (source_[index] == '\r') {
          ++line;
          column = 0;
          if (index + 1U < offset && source_[index + 1U] == '\n') {
            ++index;
          }
        } else if (source_[index] == '\n') {
          ++line;
          column = 0;
        } else {
          ++column;
        }
      }
    }
    std::string located{message};
    if (line >= 0 && column >= 0) {
      located += " at line ";
      located += std::to_string(line + 1);
      located += ", column ";
      located += std::to_string(column + 1);
    }
    return config::ConfigError::make(
        code,
        located,
        path,
        source_name_,
        std::span<const std::byte>{excerpt,
                                   std::min<std::size_t>(source_.size() - excerpt_offset, 256U)},
        config::ConfigOperationStage::kParse);
  }

  [[nodiscard]] config::ConfigError contextualize(const config::ConfigError& error,
                                                  std::string_view path, YAML::Mark mark) const {
    return make_error(error.code(), error.message(), path, mark);
  }

  [[nodiscard]] config::ConfigError make_error(config::ConfigErrorCode code,
                                               std::string_view message, std::string_view path,
                                               const ExactScalarSource& source,
                                               YAML::Mark fallback_mark) const {
    return fallback_mark.is_null() ? make_error(code, message, path, fallback_mark)
                                   : make_error(code, message, path, source.begin);
  }

  [[nodiscard]] config::ConfigError contextualize(const config::ConfigError& error,
                                                  std::string_view path,
                                                  const ExactScalarSource& source,
                                                  YAML::Mark fallback_mark) const {
    return make_error(error.code(), error.message(), path, source, fallback_mark);
  }

  [[nodiscard]] config::ConfigResult<void> charge(std::uint32_t depth, std::string_view path,
                                                  YAML::Mark mark) {
    if (depth > options_.max_depth || nodes_ >= options_.max_nodes) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kLimitExceeded,
                                            "YAML semantic tree limit exceeded",
                                            path,
                                            mark));
    }
    ++nodes_;
    return {};
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_empty_document() {
    auto charged = charge(1U, {}, YAML::Mark::null_mark());
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return config::ConfigValue::null();
  }

  [[nodiscard]] std::size_t skip_anchor_separation(std::size_t position) const noexcept {
    while (position < source_.size()) {
      while (position < source_.size() && (source_[position] == ' ' || source_[position] == '\t')) {
        ++position;
      }
      if (position < source_.size() && source_[position] == '#') {
        while (position < source_.size() && source_[position] != '\r' &&
               source_[position] != '\n') {
          ++position;
        }
        continue;
      }
      if (position < source_.size() && source_[position] == '\r') {
        ++position;
        if (position < source_.size() && source_[position] == '\n') {
          ++position;
        }
        continue;
      }
      if (position < source_.size() && source_[position] == '\n') {
        ++position;
        continue;
      }
      break;
    }
    return position;
  }

  [[nodiscard]] config::ConfigResult<std::size_t> token_start(const YAML::Node& node,
                                                              std::string_view path) const {
    const auto mark = node.Mark();
    if (mark.is_null() || mark.pos < 0) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "YAML node has no stable source token",
                                            path,
                                            YAML::Mark::null_mark()));
    }
    std::size_t position = source_position(mark);
    while (position < source_.size()) {
      while (position < source_.size() && (source_[position] == ' ' || source_[position] == '\t')) {
        ++position;
      }
      if (position == source_.size()) {
        break;
      }
      if (source_[position] == '!') {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                              "YAML tags are unsupported",
                                              path,
                                              position));
      }
      if (source_[position] != '&') {
        break;
      }
      ++position;
      while (position < source_.size()) {
        const char value = source_[position];
        if (value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == ',' ||
            value == '[' || value == ']' || value == '{' || value == '}') {
          break;
        }
        ++position;
      }
      position = skip_anchor_separation(position);
    }
    return position;
  }

  [[nodiscard]] config::ConfigResult<void> validate_tag(const YAML::Node& node,
                                                        std::string_view path,
                                                        YAML::Mark diagnostic_mark) const {
    const auto& tag = node.Tag();
    if (tag.empty() || tag == "?") {
      return {};
    }
    // yaml-cpp reports "!" for both ordinary non-plain scalars and an explicit nonspecific tag.
    // Inspect the bounded source token only for that ambiguous case and for actual custom tags.
    auto start = token_start(node, path);
    if (!start) {
      return tl::make_unexpected(start.error());
    }
    if (tag != "!") {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "YAML tags are unsupported",
                                            path,
                                            diagnostic_mark));
    }
    return {};
  }

  [[nodiscard]] bool observe_alias(const YAML::Node& node) {
    const auto mark = node.Mark();
    if (mark.is_null() || mark.pos < 0) {
      return false;
    }
    if (!observed_nodes_.insert(node).second) {
      contains_alias_ = true;
      return true;
    }
    return false;
  }

  // This bounded scan recovers scalar-local metadata only; yaml-cpp remains the sole collection
  // topology authority. Only ExactScalarSource can publish a lossless byte span.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] config::ConfigResult<RecoveredScalarSource> recover_scalar_source(
      const YAML::Node& node, std::string_view path) const {
    const auto mark = node.Mark();
    if (mark.is_null() || mark.pos < 0) {
      return RecoveredScalarSource{std::in_place_type<UntrustedScalarSource>,
                                   ScalarSourceFailure::kUnstableMark};
    }
    auto start = token_start(node, path);
    if (!start) {
      return tl::make_unexpected(start.error());
    }
    std::size_t position = *start;
    if (position == source_.size() || source_[position] == '\r' || source_[position] == '\n' ||
        source_[position] == ',' || source_[position] == ']' || source_[position] == '}' ||
        source_[position] == '#') {
      if (node.IsNull()) {
        return RecoveredScalarSource{std::in_place_type<AbsentScalarSource>};
      }
      return RecoveredScalarSource{std::in_place_type<UntrustedScalarSource>,
                                   ScalarSourceFailure::kMissingToken};
    }
    if (source_[position] == '|' || source_[position] == '>') {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "YAML block scalar styles are unsupported",
                                            path,
                                            position));
    }
    const std::size_t begin = position;
    if (source_[position] == '"') {
      bool escaped = false;
      for (++position; position < source_.size(); ++position) {
        const char value = source_[position];
        if (escaped) {
          escaped = false;
        } else if (value == '\\') {
          escaped = true;
        } else if (value == '"') {
          ++position;
          if (node.IsNull()) {
            return RecoveredScalarSource{std::in_place_type<AbsentScalarSource>};
          }
          return RecoveredScalarSource{std::in_place_type<ExactScalarSource>,
                                       YamlScalarStyle::kDoubleQuoted,
                                       source_.substr(begin, position - begin),
                                       begin};
        }
      }
    } else if (source_[position] == '\'') {
      for (++position; position < source_.size(); ++position) {
        if (source_[position] != '\'') {
          continue;
        }
        if (position + 1U < source_.size() && source_[position + 1U] == '\'') {
          ++position;
          continue;
        }
        ++position;
        if (node.IsNull()) {
          return RecoveredScalarSource{std::in_place_type<AbsentScalarSource>};
        }
        return RecoveredScalarSource{std::in_place_type<ExactScalarSource>,
                                     YamlScalarStyle::kSingleQuoted,
                                     source_.substr(begin, position - begin),
                                     begin};
      }
    } else if (node.IsScalar()) {
      if (node.Scalar().size() > source_.size() - begin ||
          source_.substr(begin, node.Scalar().size()) != node.Scalar()) {
        return RecoveredScalarSource{std::in_place_type<UntrustedScalarSource>,
                                     ScalarSourceFailure::kDecodedValueMismatch};
      }
      position = begin + node.Scalar().size();
      return RecoveredScalarSource{std::in_place_type<ExactScalarSource>,
                                   YamlScalarStyle::kPlain,
                                   source_.substr(begin, position - begin),
                                   begin};
    } else {
      for (; position < source_.size(); ++position) {
        const char value = source_[position];
        if (value == '\r' || value == '\n' || value == ',' || value == ']' || value == '}' ||
            (value == '#' && (position == begin || source_[position - 1U] == ' ' ||
                              source_[position - 1U] == '\t'))) {
          break;
        }
      }
      while (position > begin &&
             (source_[position - 1U] == ' ' || source_[position - 1U] == '\t')) {
        --position;
      }
      const auto candidate = source_.substr(begin, position - begin);
      if (node.IsNull() && yaml_cpp_null_spelling(candidate)) {
        return RecoveredScalarSource{
            std::in_place_type<ExactScalarSource>, YamlScalarStyle::kPlain, candidate, begin};
      }
      return RecoveredScalarSource{std::in_place_type<AbsentScalarSource>};
    }
    return RecoveredScalarSource{std::in_place_type<UntrustedScalarSource>,
                                 ScalarSourceFailure::kMalformedQuotedToken};
  }

  // Quote, escape, comment, and nesting states are all needed to retain one exact flow token.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> flow_collection_span(
      const YAML::Node& node, std::string_view path) const {
    auto start = token_start(node, path);
    if (!start || *start == source_.size() || (source_[*start] != '{' && source_[*start] != '[')) {
      return std::nullopt;
    }
    std::uint32_t depth = 0;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool escaped = false;
    bool in_comment = false;
    bool comment_may_start = false;
    bool colon_may_follow = false;
    for (std::size_t position = *start; position < source_.size(); ++position) {
      const char value = source_[position];
      if (value == '\r' || value == '\n') {
        in_comment = false;
        comment_may_start = false;
        escaped = false;
        continue;
      }
      if (in_comment) {
        continue;
      }
      if (in_double_quote) {
        if (escaped) {
          escaped = false;
        } else if (value == '\\') {
          escaped = true;
        } else if (value == '"') {
          in_double_quote = false;
          comment_may_start = true;
          colon_may_follow = true;
        }
        continue;
      }
      if (in_single_quote) {
        if (value == '\'' && position + 1U < source_.size() && source_[position + 1U] == '\'') {
          ++position;
        } else if (value == '\'') {
          in_single_quote = false;
          comment_may_start = true;
          colon_may_follow = true;
        }
        continue;
      }
      const bool after_token_boundary = comment_may_start;
      comment_may_start = false;
      if (yaml_comment_start(source_, position, after_token_boundary)) {
        in_comment = true;
      } else if (value == ' ' || value == '\t') {
        continue;
      } else if (value == '"') {
        in_double_quote = true;
        colon_may_follow = false;
      } else if (value == '\'') {
        in_single_quote = true;
        colon_may_follow = false;
      } else if (value == '{' || value == '[') {
        ++depth;
        comment_may_start = true;
        colon_may_follow = false;
      } else if (value == ',' || (value == ':' && colon_may_follow)) {
        comment_may_start = true;
        colon_may_follow = false;
      } else if (value == '}' || value == ']') {
        if (depth == 0U) {
          return std::nullopt;
        }
        if (--depth == 0U) {
          return std::pair{*start, position + 1U};
        }
        comment_may_start = true;
        colon_may_follow = true;
      } else {
        colon_may_follow = false;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> finish_scalar(
      const ExactScalarSource& source, std::string_view path, YAML::Mark diagnostic_mark,
      config::ConfigResult<config::ConfigValue> result) {
    if (!result) {
      return tl::make_unexpected(contextualize(result.error(), path, source, diagnostic_mark));
    }
    if (spans_ != nullptr && !source.bytes.empty() && !path.empty() &&
        result->kind() != config::ConfigValueKind::kArray &&
        result->kind() != config::ConfigValueKind::kObject) {
      spans_->push_back(config::detail::LosslessScalarSpan{
          std::string{path}, result->kind(), source.begin, source.end()});
    }
    return result;
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_scalar(
      const YAML::Node& node, std::string_view path, YAML::Mark diagnostic_mark) {
    auto recovered = recover_scalar_source(node, path);
    if (!recovered) {
      return tl::make_unexpected(recovered.error());
    }
    const auto* source = std::get_if<ExactScalarSource>(&*recovered);
    if (source == nullptr) {
      const auto reason = std::get_if<UntrustedScalarSource>(&*recovered);
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            reason == nullptr
                                                ? "YAML scalar has no attributable source token"
                                                : scalar_source_failure_message(reason->reason),
                                            path,
                                            diagnostic_mark));
    }
    if (source->style == YamlScalarStyle::kSingleQuoted ||
        source->style == YamlScalarStyle::kDoubleQuoted) {
      return finish_scalar(
          *source, path, diagnostic_mark, config::ConfigValue::string(node.Scalar(), limits_));
    }
    const auto text = source->bytes;
    if (text == "null" || text == "~") {
      return finish_scalar(*source, path, diagnostic_mark, config::ConfigValue::null());
    }
    if (text == "true") {
      return finish_scalar(*source, path, diagnostic_mark, config::ConfigValue::boolean(true));
    }
    if (text == "false") {
      return finish_scalar(*source, path, diagnostic_mark, config::ConfigValue::boolean(false));
    }
    const bool numeric_like = config::detail::yaml_numeric_candidate(text, true);
    if (numeric_like) {
      if (!exact_json_number_grammar(text)) {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                              "unsupported YAML numeric scalar",
                                              path,
                                              *source,
                                              diagnostic_mark));
      }
      return finish_scalar(*source, path, diagnostic_mark, detail::parse_json(text, options_));
    }
    if (config::detail::unsupported_yaml_implicit(text)) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "unsupported YAML implicit scalar",
                                            path,
                                            *source,
                                            diagnostic_mark));
    }
    return finish_scalar(
        *source, path, diagnostic_mark, config::ConfigValue::string(node.Scalar(), limits_));
  }

  [[nodiscard]] config::ConfigResult<DecodedYamlKey> decode_key(const YAML::Node& node,
                                                                std::string_view parent_path,
                                                                bool alias_expansion) {
    const bool alias_occurrence = observe_alias(node);
    const auto diagnostic_mark =
        alias_expansion || alias_occurrence ? YAML::Mark::null_mark() : node.Mark();
    if (!node.IsScalar()) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "YAML mapping key must be a string",
                                            parent_path,
                                            diagnostic_mark));
    }
    auto recovered = recover_scalar_source(node, parent_path);
    if (!recovered) {
      return tl::make_unexpected(recovered.error());
    }
    const auto* source = std::get_if<ExactScalarSource>(&*recovered);
    if (source == nullptr) {
      const auto reason = std::get_if<UntrustedScalarSource>(&*recovered);
      return tl::make_unexpected(
          make_error(config::ConfigErrorCode::kUnsupportedValue,
                     reason == nullptr ? "YAML mapping key has no attributable source token"
                                       : scalar_source_failure_message(reason->reason, true),
                     parent_path,
                     diagnostic_mark));
    }
    if (source->style == YamlScalarStyle::kPlain &&
        (source->bytes == "null" || source->bytes == "~" || source->bytes == "true" ||
         source->bytes == "false" || config::detail::yaml_numeric_candidate(source->bytes) ||
         config::detail::unsupported_yaml_implicit(source->bytes))) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "YAML mapping key must be a string",
                                            parent_path,
                                            *source,
                                            diagnostic_mark));
    }
    if (node.Scalar().size() > options_.max_string_bytes) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kLimitExceeded,
                                            "YAML decoded key limit exceeded",
                                            parent_path,
                                            *source,
                                            diagnostic_mark));
    }
    auto checked = config::ConfigValue::string(node.Scalar(), limits_);
    if (!checked) {
      return tl::make_unexpected(
          contextualize(checked.error(), parent_path, *source, diagnostic_mark));
    }
    return DecodedYamlKey{node.Scalar(),
                          alias_occurrence,
                          source->style == YamlScalarStyle::kPlain && source->bytes == "<<"};
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_sequence(const YAML::Node& node,
                                                                           std::uint32_t depth,
                                                                           std::string_view path,
                                                                           bool alias_expansion) {
    const auto diagnostic_mark = alias_expansion ? YAML::Mark::null_mark() : node.Mark();
    if (node.size() > options_.max_container_entries) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kLimitExceeded,
                                            "YAML sequence entry limit exceeded",
                                            path,
                                            diagnostic_mark));
    }
    std::vector<config::ConfigValue> values;
    values.reserve(node.size());
    for (std::size_t index = 0; index < node.size(); ++index) {
      auto converted = convert(node[index],
                               depth + 1U,
                               append_pointer_token(path, std::to_string(index)),
                               alias_expansion);
      if (!converted) {
        return converted;
      }
      values.push_back(std::move(*converted));
    }
    auto result = config::ConfigValue::array(values, limits_);
    return result ? result
                  : config::ConfigResult<config::ConfigValue>{
                        tl::make_unexpected(contextualize(result.error(), path, diagnostic_mark))};
  }

  // Object conversion intentionally keeps validation, transactional assembly, and envelope
  // decoding in one path so no partially validated mapping can be published.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_map(const YAML::Node& node,
                                                                      std::uint32_t depth,
                                                                      std::string_view path,
                                                                      bool alias_expansion) {
    const auto diagnostic_mark = alias_expansion ? YAML::Mark::null_mark() : node.Mark();
    if (node.size() > options_.max_container_entries) {
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kLimitExceeded,
                                            "YAML mapping entry limit exceeded",
                                            path,
                                            diagnostic_mark));
    }
    std::vector<std::string> keys;
    std::vector<config::ConfigValue> values;
    std::unordered_set<std::string> unique_keys;
    keys.reserve(node.size());
    values.reserve(node.size());
    unique_keys.reserve(node.size());
    const std::size_t span_checkpoint = spans_ != nullptr ? spans_->size() : 0U;
    for (const auto& item : node) {
      auto key = decode_key(item.first, path, alias_expansion);
      if (!key) {
        return tl::make_unexpected(key.error());
      }
      const auto key_diagnostic_mark =
          alias_expansion || key->alias_occurrence ? YAML::Mark::null_mark() : item.first.Mark();
      const auto child_path = append_pointer_token(path, key->value);
      if (key->plain_merge_key) {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                              "YAML merge keys are unsupported",
                                              child_path,
                                              key_diagnostic_mark));
      }
      if (!unique_keys.insert(key->value).second) {
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kDuplicateKey,
                                              "duplicate YAML mapping key",
                                              child_path,
                                              key_diagnostic_mark));
      }
      auto converted = convert(item.second, depth + 1U, child_path, alias_expansion);
      if (!converted) {
        return converted;
      }
      keys.push_back(std::move(key->value));
      values.push_back(std::move(*converted));
    }
    std::vector<config::ConfigObjectEntry> entries;
    entries.reserve(keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
      entries.push_back({keys[index], values[index]});
    }
    auto object = config::ConfigValue::object(entries, limits_);
    if (!object) {
      return tl::make_unexpected(contextualize(object.error(), path, diagnostic_mark));
    }
    const auto object_view = object->view();
    bool reserved_envelope = false;
    if (object_view.size() == 1U) {
      auto reserved_key = object_view.key(0);
      reserved_envelope = reserved_key && (*reserved_key == "$holoscan.config.scalar/v1" ||
                                           *reserved_key == "$holoscan.config.object/v1");
    }
    auto decoded = detail::decode_reserved_envelope(std::move(*object), limits_);
    if (decoded && reserved_envelope && spans_ != nullptr) {
      // Reserved envelopes expose semantic values, not their private control fields. Retain a
      // whole-token span only when the backend map has one stable flow-collection extent.
      spans_->resize(span_checkpoint);
      if (!path.empty() && decoded->kind() != config::ConfigValueKind::kArray &&
          decoded->kind() != config::ConfigValueKind::kObject) {
        if (const auto span = flow_collection_span(node, path)) {
          spans_->push_back(config::detail::LosslessScalarSpan{
              std::string{path}, decoded->kind(), span->first, span->second});
        }
      }
    }
    return decoded ? decoded
                   : config::ConfigResult<config::ConfigValue>{tl::make_unexpected(
                         contextualize(decoded.error(), path, diagnostic_mark))};
  }

  // REQ: CONFIG-018
  // yaml-cpp is the sole collection-syntax authority. Each alias traversal constructs a fresh
  // owning subtree. The active path detects cycles; the observed identity set marks repeated nodes
  // so alias-bearing sources retain their fail-closed preservation policy.
  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert(const YAML::Node& node,
                                                                  std::uint32_t depth,
                                                                  std::string_view path,
                                                                  bool alias_expansion = false) {
    if (std::any_of(active_path_.begin(), active_path_.end(), [&](const YAML::Node& active) {
          return node.is(active);
        })) {
      // yaml-cpp does not retain an alias-use mark; preserve the exact semantic path and omit a
      // misleading anchor-definition location.
      return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                            "cyclic YAML aliases are unsupported",
                                            path,
                                            YAML::Mark::null_mark()));
    }
    const bool alias_occurrence = observe_alias(node);
    alias_expansion = alias_expansion || alias_occurrence;
    const auto diagnostic_mark = alias_expansion ? YAML::Mark::null_mark() : node.Mark();
    auto charged = charge(depth, path, diagnostic_mark);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    auto tag = validate_tag(node, path, diagnostic_mark);
    if (!tag) {
      return tl::make_unexpected(tag.error());
    }
    active_path_.push_back(node);
    struct ActivePathGuard {
      std::vector<YAML::Node>& path;
      ~ActivePathGuard() { path.pop_back(); }
    };
    const ActivePathGuard guard{active_path_};

    switch (node.Type()) {
      case YAML::NodeType::Null: {
        auto recovered = recover_scalar_source(node, path);
        if (!recovered) {
          return tl::make_unexpected(recovered.error());
        }
        if (std::holds_alternative<AbsentScalarSource>(*recovered)) {
          return config::ConfigValue::null();
        }
        if (const auto* untrusted = std::get_if<UntrustedScalarSource>(&*recovered)) {
          return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                                scalar_source_failure_message(untrusted->reason),
                                                path,
                                                diagnostic_mark));
        }
        const auto& source = std::get<ExactScalarSource>(*recovered);
        if (source.bytes != "null" && source.bytes != "~") {
          return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                                "unsupported YAML null spelling",
                                                path,
                                                source,
                                                diagnostic_mark));
        }
        return finish_scalar(source, path, diagnostic_mark, config::ConfigValue::null());
      }
      case YAML::NodeType::Scalar:
        return convert_scalar(node, path, diagnostic_mark);
      case YAML::NodeType::Sequence:
        return convert_sequence(node, depth, path, alias_expansion);
      case YAML::NodeType::Map:
        return convert_map(node, depth, path, alias_expansion);
      case YAML::NodeType::Undefined:
        return tl::make_unexpected(make_error(config::ConfigErrorCode::kParseError,
                                              "YAML backend produced an undefined node",
                                              path,
                                              diagnostic_mark));
    }
    return tl::make_unexpected(make_error(config::ConfigErrorCode::kUnsupportedValue,
                                          "YAML backend produced an unsupported node",
                                          path,
                                          diagnostic_mark));
  }

  std::string_view source_;
  std::string_view source_name_;
  ParseOptions options_;
  config::ConfigValueLimits limits_;
  std::vector<YAML::Node> active_path_;
  std::unordered_set<YAML::Node, NodeMarkHash, NodeIdentityEqual> observed_nodes_;
  std::uint32_t nodes_{0};
  std::vector<config::detail::LosslessScalarSpan>* spans_{nullptr};
  bool* contains_alias_output_{nullptr};
  bool contains_alias_{false};
};
#endif

void append_json_string(std::string& output, std::string_view value) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
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
          output.push_back(static_cast<char>(byte));
        }
    }
  }
  output.push_back('"');
}

[[nodiscard]] config::ConfigResult<void> add_output_size(std::size_t& total, std::size_t added,
                                                         const OutputLimits& limits) {
  if (added > limits.max_output_bytes - std::min(total, limits.max_output_bytes)) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "serialization output limit exceeded",
                                       config::ConfigOperationStage::kSerialize));
  }
  total += added;
  return {};
}

[[nodiscard]] config::ConfigResult<std::size_t> json_string_size(std::string_view value,
                                                                 const OutputLimits& limits) {
  if (value.size() > limits.max_string_bytes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "serialization string limit exceeded",
                                       config::ConfigOperationStage::kSerialize));
  }
  std::size_t result = 2U;
  for (const unsigned char byte : value) {
    const std::size_t width = byte < 0x20U ? 6U : (byte == '"' || byte == '\\') ? 2U : 1U;
    auto added = add_output_size(result, width, limits);
    if (!added) {
      return tl::make_unexpected(added.error());
    }
  }
  return result;
}

// This exhaustive recursive visitor keeps each ConfigValue kind adjacent to its exact canonical
// byte accounting, including the reserved-envelope collision form.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] config::ConfigResult<std::size_t> canonical_size(const config::ConfigView& value,
                                                               const OutputLimits& limits,
                                                               std::uint32_t depth,
                                                               std::uint32_t& nodes) {
  if (depth > limits.max_depth || ++nodes > limits.max_nodes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "serialization limit exceeded",
                                       config::ConfigOperationStage::kSerialize));
  }
  std::size_t result = 0;
  auto add = [&](std::size_t count) { return add_output_size(result, count, limits); };
  auto scalar_envelope = [&](std::string_view kind,
                             std::size_t text_size) -> config::ConfigResult<std::size_t> {
    constexpr std::string_view prefix{"{\"$holoscan.config.scalar/v1\":{\"kind\":\""};
    constexpr std::string_view middle{"\",\"value\":\""};
    constexpr std::string_view suffix{"\"}}"};
    auto status = add(prefix.size());
    if (status)
      status = add(kind.size());
    if (status)
      status = add(middle.size());
    if (status)
      status = add(text_size);
    if (status)
      status = add(suffix.size());
    return status ? config::ConfigResult<std::size_t>{result}
                  : config::ConfigResult<std::size_t>{tl::make_unexpected(status.error())};
  };

  switch (value.kind()) {
    case config::ConfigValueKind::kNull:
      return add(4U) ? config::ConfigResult<std::size_t>{result}
                     : config::ConfigResult<std::size_t>{
                           tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                                       "serialization output limit exceeded",
                                                       config::ConfigOperationStage::kSerialize))};
    case config::ConfigValueKind::kBool: {
      auto item = value.as_bool();
      if (!item)
        return tl::make_unexpected(item.error());
      auto status = add(*item ? 4U : 5U);
      return status ? config::ConfigResult<std::size_t>{result}
                    : config::ConfigResult<std::size_t>{tl::make_unexpected(status.error())};
    }
    case config::ConfigValueKind::kSignedInteger: {
      auto item = value.as_signed_integer();
      if (!item)
        return tl::make_unexpected(item.error());
      std::array<char, 32> text{};
      auto rendered = std::to_chars(text.data(), text.data() + text.size(), *item);
      return scalar_envelope("i64", static_cast<std::size_t>(rendered.ptr - text.data()));
    }
    case config::ConfigValueKind::kUnsignedInteger: {
      auto item = value.as_unsigned_integer();
      if (!item)
        return tl::make_unexpected(item.error());
      std::array<char, 32> text{};
      auto rendered = std::to_chars(text.data(), text.data() + text.size(), *item);
      return scalar_envelope("u64", static_cast<std::size_t>(rendered.ptr - text.data()));
    }
    case config::ConfigValueKind::kFloating: {
      auto item = value.as_floating();
      if (!item)
        return tl::make_unexpected(item.error());
      std::array<char, 128> text{};
      auto rendered =
          std::to_chars(text.data(), text.data() + text.size(), *item, std::chars_format::general);
      if (rendered.ec != std::errc()) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kUnsupportedValue,
                                           "unable to encode floating value",
                                           config::ConfigOperationStage::kSerialize));
      }
      return scalar_envelope("f64", static_cast<std::size_t>(rendered.ptr - text.data()));
    }
    case config::ConfigValueKind::kString: {
      auto item = value.as_string();
      if (!item)
        return tl::make_unexpected(item.error());
      return json_string_size(*item, limits);
    }
    case config::ConfigValueKind::kArray: {
      if (value.size() > limits.max_container_entries) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                           "serialization array limit exceeded",
                                           config::ConfigOperationStage::kSerialize));
      }
      auto status = add(2U + (value.size() == 0 ? 0U : value.size() - 1U));
      if (!status)
        return tl::make_unexpected(status.error());
      for (std::size_t index = 0; index < value.size(); ++index) {
        auto child = value.element(index);
        if (!child)
          return tl::make_unexpected(child.error());
        auto child_size = canonical_size(*child, limits, depth + 1U, nodes);
        if (!child_size)
          return tl::make_unexpected(child_size.error());
        status = add(*child_size);
        if (!status)
          return tl::make_unexpected(status.error());
      }
      return result;
    }
    case config::ConfigValueKind::kObject: {
      if (value.size() > limits.max_container_entries) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                           "serialization object limit exceeded",
                                           config::ConfigOperationStage::kSerialize));
      }
      bool collision = value.size() == 1U;
      auto first_key = collision ? value.key(0) : config::ConfigResult<std::string_view>{""};
      if (!first_key)
        return tl::make_unexpected(first_key.error());
      collision = collision && (*first_key == "$holoscan.config.scalar/v1" ||
                                *first_key == "$holoscan.config.object/v1");
      if (collision) {
        constexpr std::string_view prefix{"{\"$holoscan.config.object/v1\":[["};
        constexpr std::string_view suffix{"]]}"};
        auto key_size = json_string_size(*first_key, limits);
        auto child = value.element(0);
        if (!key_size)
          return tl::make_unexpected(key_size.error());
        if (!child)
          return tl::make_unexpected(child.error());
        auto child_size = canonical_size(*child, limits, depth + 1U, nodes);
        if (!child_size)
          return tl::make_unexpected(child_size.error());
        auto status = add(prefix.size());
        if (status)
          status = add(*key_size);
        if (status)
          status = add(1U + *child_size);
        if (status)
          status = add(suffix.size());
        return status ? config::ConfigResult<std::size_t>{result}
                      : config::ConfigResult<std::size_t>{tl::make_unexpected(status.error())};
      }
      auto status = add(2U + (value.size() == 0 ? 0U : value.size() - 1U));
      if (!status)
        return tl::make_unexpected(status.error());
      for (std::size_t index = 0; index < value.size(); ++index) {
        auto key = value.key(index);
        auto child = value.element(index);
        if (!key)
          return tl::make_unexpected(key.error());
        if (!child)
          return tl::make_unexpected(child.error());
        auto key_size = json_string_size(*key, limits);
        auto child_size = canonical_size(*child, limits, depth + 1U, nodes);
        if (!key_size)
          return tl::make_unexpected(key_size.error());
        if (!child_size)
          return tl::make_unexpected(child_size.error());
        status = add(*key_size + 1U);
        if (status)
          status = add(*child_size);
        if (!status)
          return tl::make_unexpected(status.error());
      }
      return result;
    }
  }
  return tl::make_unexpected(failure(config::ConfigErrorCode::kUnsupportedValue,
                                     "unknown configuration value kind",
                                     config::ConfigOperationStage::kSerialize));
}

[[nodiscard]] bool unsigned_less(std::string_view left, std::string_view right) noexcept {
  return std::lexicographical_compare(
      left.begin(), left.end(), right.begin(), right.end(), [](char lhs, char rhs) {
        return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
      });
}

// This exhaustive recursive writer mirrors canonical_size so sizing and serialization retain the
// same scalar-envelope and reserved-key collision semantics.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] config::ConfigResult<void> append_canonical(config::ConfigView value,
                                                          std::string& output,
                                                          const OutputLimits& limits,
                                                          std::uint32_t depth,
                                                          std::uint32_t& nodes) {
  if (depth > limits.max_depth || ++nodes > limits.max_nodes ||
      output.size() > limits.max_output_bytes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "serialization limit exceeded",
                                       config::ConfigOperationStage::kSerialize));
  }
  auto numeric_escape = [&](std::string_view kind, std::string_view text) {
    output += "{\"$holoscan.config.scalar/v1\":{\"kind\":\"";
    output += kind;
    output += "\",\"value\":";
    append_json_string(output, text);
    output += "}}";
  };
  switch (value.kind()) {
    case config::ConfigValueKind::kNull:
      output += "null";
      break;
    case config::ConfigValueKind::kBool: {
      auto item = value.as_bool();
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      output += *item ? "true" : "false";
      break;
    }
    case config::ConfigValueKind::kSignedInteger: {
      auto item = value.as_signed_integer();
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      numeric_escape("i64", std::to_string(*item));
      break;
    }
    case config::ConfigValueKind::kUnsignedInteger: {
      auto item = value.as_unsigned_integer();
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      numeric_escape("u64", std::to_string(*item));
      break;
    }
    case config::ConfigValueKind::kFloating: {
      auto item = value.as_floating();
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      std::array<char, 128> buffer{};
      const auto encoded = std::to_chars(
          buffer.data(), buffer.data() + buffer.size(), *item, std::chars_format::general);
      if (encoded.ec != std::errc()) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kUnsupportedValue,
                                           "unable to encode floating value",
                                           config::ConfigOperationStage::kSerialize));
      }
      numeric_escape("f64", std::string{buffer.data(), encoded.ptr});
      break;
    }
    case config::ConfigValueKind::kString: {
      auto item = value.as_string();
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      if (item->size() > limits.max_string_bytes) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                           "serialization string limit exceeded",
                                           config::ConfigOperationStage::kSerialize));
      }
      append_json_string(output, *item);
      break;
    }
    case config::ConfigValueKind::kArray: {
      if (value.size() > limits.max_container_entries) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                           "serialization array limit exceeded",
                                           config::ConfigOperationStage::kSerialize));
      }
      output.push_back('[');
      for (std::size_t index = 0; index < value.size(); ++index) {
        if (index != 0) {
          output.push_back(',');
        }
        auto child = value.element(index);
        if (!child) {
          return tl::make_unexpected(child.error());
        }
        auto appended = append_canonical(*child, output, limits, depth + 1U, nodes);
        if (!appended) {
          return appended;
        }
      }
      output.push_back(']');
      break;
    }
    case config::ConfigValueKind::kObject: {
      if (value.size() > limits.max_container_entries) {
        return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                           "serialization object limit exceeded",
                                           config::ConfigOperationStage::kSerialize));
      }
      std::vector<std::size_t> order(value.size());
      for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
      }
      std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const auto left_key = value.key(left);
        const auto right_key = value.key(right);
        return left_key && right_key && unsigned_less(*left_key, *right_key);
      });
      bool collision = value.size() == 1U;
      std::string_view only_key;
      if (collision) {
        auto key = value.key(0);
        collision =
            key && (*key == "$holoscan.config.scalar/v1" || *key == "$holoscan.config.object/v1");
        if (key) {
          only_key = *key;
        }
      }
      if (collision) {
        output += "{\"$holoscan.config.object/v1\":[[";
        append_json_string(output, only_key);
        output.push_back(',');
        auto child = value.element(0);
        if (!child) {
          return tl::make_unexpected(child.error());
        }
        auto appended = append_canonical(*child, output, limits, depth + 1U, nodes);
        if (!appended) {
          return appended;
        }
        output += "]]}";
      } else {
        output.push_back('{');
        for (std::size_t position = 0; position < order.size(); ++position) {
          if (position != 0) {
            output.push_back(',');
          }
          auto key = value.key(order[position]);
          auto child = value.element(order[position]);
          if (!key) {
            return tl::make_unexpected(key.error());
          }
          if (!child) {
            return tl::make_unexpected(child.error());
          }
          if (key->size() > limits.max_string_bytes) {
            return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                               "serialization key limit exceeded",
                                               config::ConfigOperationStage::kSerialize));
          }
          append_json_string(output, *key);
          output.push_back(':');
          auto appended = append_canonical(*child, output, limits, depth + 1U, nodes);
          if (!appended) {
            return appended;
          }
        }
        output.push_back('}');
      }
      break;
    }
  }
  if (output.size() > limits.max_output_bytes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "serialization output limit exceeded",
                                       config::ConfigOperationStage::kSerialize));
  }
  return {};
}

[[nodiscard]] config::ConfigResult<std::string> read_file(const std::filesystem::path& path,
                                                          std::size_t maximum_bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "unable to open configuration file",
                                       config::ConfigOperationStage::kParse,
                                       config::ConfigDestinationState::kNotApplicable,
                                       path.string()));
  }
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > maximum_bytes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "configuration file exceeds source limit",
                                       config::ConfigOperationStage::kParse));
  }
  input.seekg(0, std::ios::beg);
  std::string result(static_cast<std::size_t>(size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "unable to read configuration file",
                                       config::ConfigOperationStage::kParse));
  }
  return result;
}

#if !defined(_WIN32)
class UniqueFd {
 public:
  explicit UniqueFd(int value = -1) noexcept : value_(value) {}
  ~UniqueFd() {
    if (value_ >= 0) {
      close(value_);
    }
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) {
        close(value_);
      }
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

class ScopedSiblingCleanup {
 public:
  ScopedSiblingCleanup(int parent, const std::string& name) noexcept
      : parent_(parent), name_(name) {}
  ~ScopedSiblingCleanup() {
    if (armed_) {
      ::unlinkat(parent_, name_.c_str(), 0);
    }
  }
  ScopedSiblingCleanup(const ScopedSiblingCleanup&) = delete;
  ScopedSiblingCleanup& operator=(const ScopedSiblingCleanup&) = delete;
  void release() noexcept { armed_ = false; }

 private:
  int parent_;
  const std::string& name_;
  bool armed_{true};
};

[[nodiscard]] config::ConfigResult<UniqueFd> open_parent_directory_nofollow(
    const std::filesystem::path& path) {
  UniqueFd current{
      ::open(path.is_absolute() ? "/" : ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)};
  if (current.get() < 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "unable to open parent traversal root",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  for (const auto& component : path) {
    const auto text = component.string();
    if (text.empty() || text == "." || text == "/") {
      continue;
    }
    if (text == "..") {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kAtomicReplaceUnsupported,
                                         "parent traversal may not contain dot-dot components",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    UniqueFd next{
        ::openat(current.get(), text.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)};
    if (next.get() < 0) {
      const auto code = errno == EACCES ? config::ConfigErrorCode::kPermissionDenied
                        : errno == ELOOP || errno == ENOTDIR
                            ? config::ConfigErrorCode::kAtomicReplaceUnsupported
                            : config::ConfigErrorCode::kIoError;
      return tl::make_unexpected(failure(code,
                                         "unable to resolve no-follow parent directory component",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    current = std::move(next);
  }
  return current;
}

struct DestinationObservation {
  bool exists{false};
  struct stat metadata{};
  std::optional<config::RawSourceDigest> digest;
};

[[nodiscard]] bool same_identity(const struct stat& left, const struct stat& right) noexcept {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_mode == right.st_mode && left.st_size == right.st_size;
}

[[nodiscard]] bool same_observation(const DestinationObservation& left,
                                    const DestinationObservation& right) noexcept {
  if (left.exists != right.exists) {
    return false;
  }
  if (!left.exists) {
    return true;
  }
  return same_identity(left.metadata, right.metadata) && left.digest == right.digest;
}

[[nodiscard]] config::ConfigResult<DestinationObservation> observe_destination(
    int parent, std::string_view basename) {
  const std::string owned_name{basename};
  const UniqueFd input{::openat(parent, owned_name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (input.get() < 0) {
    if (errno == ENOENT) {
      return DestinationObservation{};
    }
    const auto code = errno == EACCES  ? config::ConfigErrorCode::kPermissionDenied
                      : errno == ELOOP ? config::ConfigErrorCode::kAtomicReplaceUnsupported
                                       : config::ConfigErrorCode::kIoError;
    return tl::make_unexpected(failure(code,
                                       "unable to open guarded destination",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }

  struct stat before{};
  if (::fstat(input.get(), &before) != 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "unable to inspect guarded destination",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  if (!S_ISREG(before.st_mode)) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kAtomicReplaceUnsupported,
                                       "destination is not a regular no-follow file",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  if (before.st_size < 0 || static_cast<std::uint64_t>(before.st_size) > kMaxOutputBytes) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "guarded destination exceeds package byte limit",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }

  std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = ::read(input.get(), bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                         "unable to read guarded destination",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    if (count == 0) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kSourceChanged,
                                         "guarded destination changed while reading",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    offset += static_cast<std::size_t>(count);
  }

  struct stat after{};
  if (::fstat(input.get(), &after) != 0 || !same_identity(before, after)) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kSourceChanged,
                                       "guarded destination identity changed while reading",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  auto digest = config::detail::raw_digest(bytes);
  if (!digest) {
    return tl::make_unexpected(digest.error());
  }
  return DestinationObservation{.exists = true, .metadata = after, .digest = *digest};
}

[[nodiscard]] config::ConfigResult<void> write_all(int descriptor, std::string_view bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                         "temporary configuration write failed",
                                         config::ConfigOperationStage::kWriteTemporary,
                                         config::ConfigDestinationState::kUnchanged));
    }
    offset += static_cast<std::size_t>(written);
  }
  return {};
}

[[nodiscard]] config::ConfigResult<UniqueFd> acquire_save_guard(int parent,
                                                                std::string_view basename,
                                                                std::chrono::milliseconds timeout) {
  const std::string lock_name = "." + std::string(basename) + ".holoscan-config.lock";
  UniqueFd guard{
      ::openat(parent, lock_name.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600)};
  if (guard.get() < 0) {
    config::ConfigErrorCode code = config::ConfigErrorCode::kIoError;
    if (errno == EACCES || errno == EPERM) {
      code = config::ConfigErrorCode::kPermissionDenied;
    } else if (errno == ELOOP) {
      code = config::ConfigErrorCode::kAtomicReplaceUnsupported;
    }
    return tl::make_unexpected(failure(code,
                                       "unable to open configuration save guard",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (::flock(guard.get(), LOCK_EX | LOCK_NB) != 0) {
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                         "configuration guard acquisition failed",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kGuardBusy,
                                         "configuration guard is busy",
                                         config::ConfigOperationStage::kAcquireGuard,
                                         config::ConfigDestinationState::kUnchanged));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return guard;
}

[[nodiscard]] config::ConfigResult<void> validate_expected_source(
    const DestinationObservation& initial, const std::optional<config::RawSourceDigest>& expected) {
  if (!expected) {
    return {};
  }
  if (!initial.exists) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kSourceChanged,
                                       "expected destination source is absent",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  if (initial.digest != expected) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kSourceChanged,
                                       "destination source digest changed",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  return {};
}

[[nodiscard]] config::ConfigResult<UniqueFd> create_temporary_file(int parent,
                                                                   const std::string& temporary) {
  UniqueFd output{::openat(
      parent, temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)};
  if (output.get() < 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "unable to create sibling temporary file",
                                       config::ConfigOperationStage::kWriteTemporary,
                                       config::ConfigDestinationState::kUnchanged));
  }
  return output;
}

[[nodiscard]] config::ConfigResult<void> write_temporary_file(const UniqueFd& output,
                                                              const DestinationObservation& initial,
                                                              std::string_view bytes) {
  if (initial.exists && ::fchmod(output.get(), initial.metadata.st_mode & 07777) != 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kPermissionDenied,
                                       "unable to preserve destination mode",
                                       config::ConfigOperationStage::kWriteTemporary,
                                       config::ConfigDestinationState::kUnchanged));
  }
  auto written = write_all(output.get(), bytes);
  if (!written) {
    return tl::make_unexpected(written.error());
  }
  if (::fsync(output.get()) != 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "temporary data flush failed",
                                       config::ConfigOperationStage::kWriteTemporary,
                                       config::ConfigDestinationState::kUnchanged));
  }
  return {};
}

[[nodiscard]] config::ConfigResult<void> verify_destination_unchanged(
    int parent, std::string_view basename, const DestinationObservation& initial) {
  auto final_observation = observe_destination(parent, basename);
  if (!final_observation) {
    return tl::make_unexpected(final_observation.error());
  }
  if (!same_observation(initial, final_observation.value())) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kSourceChanged,
                                       "destination changed before replacement",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
  }
  return {};
}

[[nodiscard]] config::ConfigResult<SaveResult> save_posix(
    const config::Document& document, const std::filesystem::path& path, std::string_view bytes,
    const SaveOptions& options, config::ConfigOperationStage& boundary_stage) {
  const auto parent_path =
      path.parent_path().empty() ? std::filesystem::path{"."} : path.parent_path();
  const std::string basename = path.filename().string();
  auto opened_parent = open_parent_directory_nofollow(parent_path);
  if (!opened_parent) {
    return tl::make_unexpected(opened_parent.error());
  }
  const UniqueFd parent = std::move(opened_parent.value());

  auto acquired_guard = acquire_save_guard(parent.get(), basename, options.guard_timeout);
  if (!acquired_guard) {
    return tl::make_unexpected(acquired_guard.error());
  }
  const UniqueFd guard = std::move(acquired_guard.value());

  auto observed = observe_destination(parent.get(), basename);
  if (!observed) {
    return tl::make_unexpected(observed.error());
  }
  const DestinationObservation initial = std::move(observed.value());
  auto expected = validate_expected_source(initial, options.expected_source_digest);
  if (!expected) {
    return tl::make_unexpected(expected.error());
  }

  static std::atomic<std::uint64_t> counter{0};
  boundary_stage = config::ConfigOperationStage::kWriteTemporary;
  const std::string temporary = "." + basename + ".holoscan-config.tmp." +
                                std::to_string(static_cast<unsigned long>(::getpid())) + "." +
                                std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
  auto created = create_temporary_file(parent.get(), temporary);
  if (!created) {
    return tl::make_unexpected(created.error());
  }
  const UniqueFd output = std::move(created.value());
  ScopedSiblingCleanup temporary_cleanup{parent.get(), temporary};
  auto written = write_temporary_file(output, initial, bytes);
  if (!written) {
    return tl::make_unexpected(written.error());
  }
  auto unchanged = verify_destination_unchanged(parent.get(), basename, initial);
  if (!unchanged) {
    return tl::make_unexpected(unchanged.error());
  }

  boundary_stage = config::ConfigOperationStage::kReplaceDestination;
  if (::renameat(parent.get(), temporary.c_str(), parent.get(), basename.c_str()) != 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "atomic destination replacement failed",
                                       config::ConfigOperationStage::kReplaceDestination,
                                       config::ConfigDestinationState::kUnchanged));
  }
  temporary_cleanup.release();
  boundary_stage = config::ConfigOperationStage::kFlushDestinationMetadata;
  if (options.durability == Durability::kDataAndMetadata && ::fsync(parent.get()) != 0) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "destination metadata flush failed",
                                       config::ConfigOperationStage::kFlushDestinationMetadata,
                                       config::ConfigDestinationState::kReplaced));
  }
  auto digest = config::detail::raw_digest(bytes);
  if (!digest) {
    return tl::make_unexpected(digest.error());
  }
  return SaveResult{
      .written_source_digest = digest.value(),
      .achieved_durability = options.durability,
      .achieved_preservation = config::ConfigIoAccess::state(document).requires_lossless
                                   ? Preservation::kRequireLossless
                                   : Preservation::kSemanticOnly};
}
#endif

[[nodiscard]] config::ConfigResult<config::Document> parse_document(std::string_view source,
                                                                    std::string_view source_name,
                                                                    Format format,
                                                                    ParseOptions options) {
  try {
    if (!valid_format(format) || !valid_parse_options(options)) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kInvalidArgument,
                                         "invalid format or parse options exceed package limits",
                                         config::ConfigOperationStage::kValidate,
                                         config::ConfigDestinationState::kNotApplicable,
                                         source_name));
    }
    if (source.size() > options.max_source_bytes) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                         "configuration source exceeds byte limit",
                                         config::ConfigOperationStage::kParse,
                                         config::ConfigDestinationState::kNotApplicable,
                                         source_name));
    }
    std::vector<config::detail::LosslessScalarSpan> spans;
    auto* retained_spans =
        options.preservation == Preservation::kRequireLossless ? &spans : nullptr;
    bool contains_yaml_alias = false;
    config::ConfigResult<config::ConfigValue> value =
        tl::make_unexpected(failure(config::ConfigErrorCode::kUnsupportedFormat,
                                    "selected configuration format is unavailable",
                                    config::ConfigOperationStage::kParse,
                                    config::ConfigDestinationState::kNotApplicable,
                                    source_name));
    if (format == Format::kJson) {
      value = detail::parse_json(source, options, retained_spans);
    } else {
#if HOLOSCAN_CONFIG_IO_USE_YAML_CPP
      value = YamlNodeConverter{source, source_name, options, retained_spans, &contains_yaml_alias}
                  .run();
#endif
    }
    if (!value) {
      return tl::make_unexpected(value.error());
    }

    auto raw = config::detail::raw_digest(source);
    if (!raw) {
      return tl::make_unexpected(raw.error());
    }
    auto owned_source = std::make_shared<const std::string>(source);
    auto encoded_source_format = static_cast<std::uint8_t>(format);
    if (contains_yaml_alias) {
      // Preserve ConfigIoAccess's existing exported signature. The high bit is an internal
      // transport flag and is masked before DocumentState records the public Format value.
      encoded_source_format |= config::detail::kYamlAliasSourceFlag;
    }
    return config::ConfigIoAccess::from_parsed(
        std::move(*value),
        *raw,
        encoded_source_format,
        std::move(owned_source),
        std::move(spans),
        options.preservation == Preservation::kRequireLossless);
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "configuration parse allocation failed",
                                       config::ConfigOperationStage::kParse,
                                       config::ConfigDestinationState::kNotApplicable,
                                       source_name));
  } catch (...) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kParseError,
                                       "configuration parser raised an unexpected exception",
                                       config::ConfigOperationStage::kParse,
                                       config::ConfigDestinationState::kNotApplicable,
                                       source_name));
  }
}

}  // namespace

config::ConfigResult<config::Document> parse(std::string_view source, Format format,
                                             ParseOptions options) {
  return parse_document(source, {}, format, options);
}

config::ConfigResult<config::Document> detail::parse_generated_canonical_json(
    std::string_view source, OutputLimits limits) {
  try {
    if (!valid_output_limits(limits)) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kInvalidArgument,
                                         "generated JSON limits exceed package limits",
                                         config::ConfigOperationStage::kValidate));
    }
    if (source.size() > limits.max_output_bytes) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                         "generated JSON exceeds output byte limit",
                                         config::ConfigOperationStage::kParse));
    }
    const ParseOptions options{
        .preservation = Preservation::kRequireLossless,
        .max_source_bytes = limits.max_output_bytes,
        .max_depth = limits.max_depth,
        .max_nodes = limits.max_nodes,
        .max_string_bytes = limits.max_string_bytes,
        .max_container_entries = limits.max_container_entries,
    };
    std::vector<config::detail::LosslessScalarSpan> spans;
    auto value = detail::parse_json(source, options, &spans);
    if (!value) {
      return tl::make_unexpected(value.error());
    }
    auto raw = config::detail::raw_digest(source);
    if (!raw) {
      return tl::make_unexpected(raw.error());
    }
    auto owned_source = std::make_shared<const std::string>(source);
    return config::ConfigIoAccess::from_parsed(std::move(*value),
                                               *raw,
                                               static_cast<std::uint8_t>(Format::kJson),
                                               std::move(owned_source),
                                               std::move(spans),
                                               true);
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "generated JSON parse allocation failed",
                                       config::ConfigOperationStage::kParse));
  } catch (...) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kParseError,
                                       "generated JSON parser raised an unexpected exception",
                                       config::ConfigOperationStage::kParse));
  }
}

config::ConfigResult<config::Document> load(const std::filesystem::path& path, Format format,
                                            ParseOptions options) {
  try {
    if (!valid_format(format) || !valid_parse_options(options)) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kInvalidArgument,
                                         "parse options exceed package limits",
                                         config::ConfigOperationStage::kValidate));
    }
    auto bytes = read_file(path, options.max_source_bytes);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    if (format == Format::kJson) {
      // Preserve the existing JSON diagnostic behavior; Task 31 adds source-name evidence only to
      // the amended YAML path.
      return parse(*bytes, format, options);
    }
    const auto source_name = path.string();
    return parse_document(*bytes, source_name, format, options);
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "configuration load allocation failed",
                                       config::ConfigOperationStage::kParse));
  } catch (const std::filesystem::filesystem_error& error) {
    const auto code = error.code() == std::errc::permission_denied
                          ? config::ConfigErrorCode::kPermissionDenied
                          : config::ConfigErrorCode::kIoError;
    return tl::make_unexpected(failure(code,
                                       "configuration load filesystem operation failed",
                                       config::ConfigOperationStage::kParse));
  } catch (...) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "configuration load raised an unexpected exception",
                                       config::ConfigOperationStage::kParse));
  }
}

config::ConfigResult<std::string> serialize(const config::Document& document, Format format,
                                            SerializeOptions options) {
  try {
    if (!valid_format(format) || !valid_output_limits(options.output_limits)) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kInvalidArgument,
                                         "serialize options exceed package limits",
                                         config::ConfigOperationStage::kValidate));
    }
    const auto& state = config::ConfigIoAccess::state(document);
    if (state.requires_lossless) {
      if (!state.lossless_available || !state.raw_source || !state.source_format ||
          *state.source_format != static_cast<std::uint8_t>(format) ||
          state.raw_source->size() > options.output_limits.max_output_bytes) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kLosslessPreservationUnavailable,
                    "lossless source representation is unavailable",
                    config::ConfigOperationStage::kSerialize));
      }
      return *state.raw_source;
    }
    std::string output;
    std::uint32_t measured_nodes = 0;
    auto measured = canonical_size(document.root(), options.output_limits, 1U, measured_nodes);
    if (!measured) {
      return tl::make_unexpected(measured.error());
    }
    output.reserve(*measured);
    std::uint32_t nodes = 0;
    auto appended = append_canonical(document.root(), output, options.output_limits, 1U, nodes);
    if (!appended) {
      return tl::make_unexpected(appended.error());
    }
    if (output.size() != *measured) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kCodecFailure,
                                         "canonical size preflight mismatch",
                                         config::ConfigOperationStage::kSerialize));
    }
    // Canonical JSON is intentionally also the deterministic YAML 1.2 semantic-subset rendering.
    return output;
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "configuration serialization allocation failed",
                                       config::ConfigOperationStage::kSerialize));
  } catch (...) {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kCodecFailure,
                                       "configuration serialization raised an unexpected exception",
                                       config::ConfigOperationStage::kSerialize));
  }
}

config::ConfigResult<SaveResult> save(const config::Document& document,
                                      const std::filesystem::path& path, Format format,
                                      SaveOptions options) {
  config::ConfigOperationStage boundary_stage = config::ConfigOperationStage::kValidate;
  try {
    if (!valid_output_limits(options.output_limits) || options.guard_timeout.count() < 0 ||
        options.guard_timeout.count() > 30000 || !valid_format(format) ||
        !valid_durability(options.durability) || path.empty() || path.filename().empty()) {
      return tl::make_unexpected(failure(config::ConfigErrorCode::kInvalidArgument,
                                         "invalid atomic-save options or destination",
                                         config::ConfigOperationStage::kValidate,
                                         config::ConfigDestinationState::kUnchanged));
    }
    boundary_stage = config::ConfigOperationStage::kSerialize;
    auto bytes = serialize(document, format, {.output_limits = options.output_limits});
    if (!bytes) {
      auto error = bytes.error();
      return tl::make_unexpected(
          config::ConfigError::make(error.code(),
                                    error.message(),
                                    error.path(),
                                    error.source(),
                                    error.excerpt(),
                                    error.operation_stage(),
                                    config::ConfigDestinationState::kUnchanged));
    }

#if defined(_WIN32)
    (void)options;
    return tl::make_unexpected(failure(config::ConfigErrorCode::kAtomicReplaceUnsupported,
                                       "native Windows atomic adapter is not built",
                                       config::ConfigOperationStage::kAcquireGuard,
                                       config::ConfigDestinationState::kUnchanged));
#else
    boundary_stage = config::ConfigOperationStage::kAcquireGuard;
    return save_posix(document, path, bytes.value(), options, boundary_stage);
#endif
  } catch (const std::bad_alloc&) {
    const auto state = boundary_stage == config::ConfigOperationStage::kFlushDestinationMetadata
                           ? config::ConfigDestinationState::kReplaced
                           : config::ConfigDestinationState::kUnchanged;
    return tl::make_unexpected(failure(config::ConfigErrorCode::kLimitExceeded,
                                       "configuration save allocation failed",
                                       boundary_stage,
                                       state));
  } catch (const std::filesystem::filesystem_error& error) {
    const auto code = error.code() == std::errc::permission_denied
                          ? config::ConfigErrorCode::kPermissionDenied
                          : config::ConfigErrorCode::kIoError;
    const auto state = boundary_stage == config::ConfigOperationStage::kFlushDestinationMetadata
                           ? config::ConfigDestinationState::kReplaced
                           : config::ConfigDestinationState::kUnchanged;
    return tl::make_unexpected(
        failure(code, "configuration save filesystem operation failed", boundary_stage, state));
  } catch (...) {
    const auto state = boundary_stage == config::ConfigOperationStage::kFlushDestinationMetadata
                           ? config::ConfigDestinationState::kReplaced
                           : config::ConfigDestinationState::kUnchanged;
    return tl::make_unexpected(failure(config::ConfigErrorCode::kIoError,
                                       "configuration save raised an unexpected exception",
                                       boundary_stage,
                                       state));
  }
}

}  // namespace holoscan::config_io
