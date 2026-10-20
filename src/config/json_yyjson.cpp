/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config/json_yyjson.hpp"
#include "config/reserved_envelope.hpp"

#include <yyjson.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace holoscan::config_io::detail {
namespace {

[[nodiscard]] config::ConfigError failure(config::ConfigErrorCode code, std::string_view message,
                                          std::string_view source = {},
                                          std::size_t position = 0U) noexcept {
  const std::size_t offset = std::min(position, source.size());
  const auto* excerpt =
      source.empty() ? nullptr : reinterpret_cast<const std::byte*>(source.data() + offset);
  return config::ConfigError::make(
      code,
      message,
      {},
      {},
      std::span<const std::byte>{excerpt, std::min<std::size_t>(source.size() - offset, 256U)},
      config::ConfigOperationStage::kParse);
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

class JsonSourceCursor {
 public:
  explicit JsonSourceCursor(std::string_view source) : source_(source) {}

  void skip_space() noexcept {
    while (position_ < source_.size() &&
           (source_[position_] == ' ' || source_[position_] == '\t' || source_[position_] == '\r' ||
            source_[position_] == '\n')) {
      ++position_;
    }
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }

  [[nodiscard]] config::ConfigResult<void> consume(char expected) {
    skip_space();
    if (position_ == source_.size() || source_[position_] != expected) {
      return diverged();
    }
    ++position_;
    return {};
  }

  [[nodiscard]] config::ConfigResult<void> consume_token(std::string_view token) {
    skip_space();
    if (source_.substr(position_, token.size()) != token) {
      return diverged();
    }
    position_ += token.size();
    return {};
  }

  [[nodiscard]] config::ConfigResult<void> consume_string() {
    skip_space();
    if (position_ == source_.size() || source_[position_] != '"') {
      return diverged();
    }
    ++position_;
    while (position_ < source_.size()) {
      const char value = source_[position_++];
      if (value == '"') {
        return {};
      }
      if (value != '\\') {
        continue;
      }
      if (position_ == source_.size()) {
        return diverged();
      }
      const char escaped = source_[position_++];
      if (escaped == 'u') {
        if (source_.size() - position_ < 4U) {
          return diverged();
        }
        position_ += 4U;
      }
    }
    return diverged();
  }

  [[nodiscard]] config::ConfigResult<void> finish() {
    skip_space();
    return position_ == source_.size() ? config::ConfigResult<void>{}
                                       : config::ConfigResult<void>{tl::make_unexpected(
                                             failure(config::ConfigErrorCode::kParseError,
                                                     "validated JSON cursor has trailing input",
                                                     source_,
                                                     position_))};
  }

 private:
  [[nodiscard]] config::ConfigResult<void> diverged() const {
    return tl::make_unexpected(failure(config::ConfigErrorCode::kParseError,
                                       "validated JSON token cursor diverged",
                                       source_,
                                       position_));
  }

  std::string_view source_;
  std::size_t position_{0};
};

class JsonConverter {
 public:
  JsonConverter(std::string_view source, ParseOptions options,
                std::vector<config::detail::LosslessScalarSpan>* spans, std::size_t source_offset)
      : options_(options),
        limits_(construction_limits(options)),
        spans_(spans),
        source_offset_(source_offset),
        cursor_(source) {}

  [[nodiscard]] config::ConfigResult<config::ConfigValue> run(yyjson_val* root) {
    auto value = convert(root, 1U, {});
    if (!value) {
      return value;
    }
    auto finished = cursor_.finish();
    if (!finished) {
      return tl::make_unexpected(finished.error());
    }
    return value;
  }

 private:
  [[nodiscard]] config::ConfigResult<void> charge(std::uint32_t depth) {
    if (depth > options_.max_depth || ++nodes_ > options_.max_nodes) {
      return tl::make_unexpected(
          failure(config::ConfigErrorCode::kLimitExceeded, "JSON parse limit exceeded"));
    }
    return {};
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> parse_number(
      std::string_view token) const {
    const bool floating = token.find_first_of(".eE") != std::string_view::npos;
    if (floating) {
      double value = 0.0;
      const auto parsed = std::from_chars(
          token.data(), token.data() + token.size(), value, std::chars_format::general);
      if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size() ||
          !std::isfinite(value)) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kOutOfRange, "JSON floating value is out of range"));
      }
      return config::ConfigValue::try_floating(value);
    }
    if (token.front() == '-') {
      std::int64_t value = 0;
      const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
      if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kOutOfRange, "JSON signed integer is out of range"));
      }
      return config::ConfigValue::try_signed_integer(value);
    }
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size()) {
      return tl::make_unexpected(
          failure(config::ConfigErrorCode::kOutOfRange, "JSON unsigned integer is out of range"));
    }
    if (value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return config::ConfigValue::try_signed_integer(static_cast<std::int64_t>(value));
    }
    return config::ConfigValue::try_unsigned_integer(value);
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_array(yyjson_val* value,
                                                                        std::uint32_t depth,
                                                                        std::string_view path) {
    const std::size_t size = yyjson_arr_size(value);
    if (size > options_.max_container_entries) {
      return tl::make_unexpected(
          failure(config::ConfigErrorCode::kLimitExceeded, "array entry limit exceeded"));
    }
    auto opened = cursor_.consume('[');
    if (!opened) {
      return tl::make_unexpected(opened.error());
    }
    std::vector<config::ConfigValue> values;
    values.reserve(size);
    yyjson_arr_iter iterator = yyjson_arr_iter_with(value);
    for (std::size_t index = 0; index < size; ++index) {
      if (index != 0U) {
        auto comma = cursor_.consume(',');
        if (!comma) {
          return tl::make_unexpected(comma.error());
        }
      }
      yyjson_val* child = yyjson_arr_iter_next(&iterator);
      auto converted =
          convert(child, depth + 1U, append_pointer_token(path, std::to_string(index)));
      if (!converted) {
        return converted;
      }
      values.push_back(std::move(*converted));
    }
    auto closed = cursor_.consume(']');
    if (!closed) {
      return tl::make_unexpected(closed.error());
    }
    return config::ConfigValue::try_array(values, limits_);
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert_object(yyjson_val* value,
                                                                         std::uint32_t depth,
                                                                         std::string_view path) {
    const std::size_t size = yyjson_obj_size(value);
    if (size > options_.max_container_entries) {
      return tl::make_unexpected(
          failure(config::ConfigErrorCode::kLimitExceeded, "object entry limit exceeded"));
    }
    auto opened = cursor_.consume('{');
    if (!opened) {
      return tl::make_unexpected(opened.error());
    }
    std::vector<std::string> keys;
    std::vector<config::ConfigValue> values;
    std::unordered_set<std::string> unique_keys;
    keys.reserve(size);
    values.reserve(size);
    unique_keys.reserve(size);
    yyjson_obj_iter iterator = yyjson_obj_iter_with(value);
    for (std::size_t index = 0; index < size; ++index) {
      if (index != 0U) {
        auto comma = cursor_.consume(',');
        if (!comma) {
          return tl::make_unexpected(comma.error());
        }
      }
      yyjson_val* key_value = yyjson_obj_iter_next(&iterator);
      yyjson_val* child = yyjson_obj_iter_get_val(key_value);
      const char* key_data = yyjson_get_str(key_value);
      const std::size_t key_size = yyjson_get_len(key_value);
      if (key_data == nullptr || key_size > options_.max_string_bytes) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kLimitExceeded, "decoded string limit exceeded"));
      }
      std::string key{key_data, key_size};
      if (!unique_keys.insert(key).second) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kDuplicateKey, "duplicate JSON object key"));
      }
      auto key_token = cursor_.consume_string();
      if (!key_token) {
        return tl::make_unexpected(key_token.error());
      }
      auto colon = cursor_.consume(':');
      if (!colon) {
        return tl::make_unexpected(colon.error());
      }
      auto converted = convert(child, depth + 1U, append_pointer_token(path, key));
      if (!converted) {
        return converted;
      }
      keys.push_back(std::move(key));
      values.push_back(std::move(*converted));
    }
    auto closed = cursor_.consume('}');
    if (!closed) {
      return tl::make_unexpected(closed.error());
    }
    std::vector<config::ConfigObjectEntry> entries;
    entries.reserve(keys.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
      entries.push_back({keys[index], values[index]});
    }
    auto object = config::ConfigValue::try_object(entries, limits_);
    return object ? decode_reserved_envelope(std::move(*object), limits_) : object;
  }

  [[nodiscard]] config::ConfigResult<config::ConfigValue> convert(yyjson_val* value,
                                                                  std::uint32_t depth,
                                                                  std::string_view path) {
    auto charged = charge(depth);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    cursor_.skip_space();
    const std::size_t begin = cursor_.position();
    const std::size_t span_checkpoint = spans_ != nullptr ? spans_->size() : 0U;
    config::ConfigResult<config::ConfigValue> result = tl::make_unexpected(
        failure(config::ConfigErrorCode::kParseError, "yyjson produced an unsupported value"));
    if (yyjson_is_null(value)) {
      auto consumed = cursor_.consume_token("null");
      result =
          consumed
              ? config::ConfigValue::null()
              : config::ConfigResult<config::ConfigValue>{tl::make_unexpected(consumed.error())};
    } else if (yyjson_is_bool(value)) {
      auto consumed = cursor_.consume_token(yyjson_get_bool(value) ? "true" : "false");
      result =
          consumed
              ? config::ConfigValue::try_boolean(yyjson_get_bool(value))
              : config::ConfigResult<config::ConfigValue>{tl::make_unexpected(consumed.error())};
    } else if (yyjson_is_str(value)) {
      const char* text = yyjson_get_str(value);
      const std::size_t size = yyjson_get_len(value);
      if (text == nullptr || size > options_.max_string_bytes) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kLimitExceeded, "decoded string limit exceeded"));
      }
      auto consumed = cursor_.consume_string();
      result =
          consumed
              ? config::ConfigValue::try_string(std::string_view{text, size}, limits_)
              : config::ConfigResult<config::ConfigValue>{tl::make_unexpected(consumed.error())};
    } else if (yyjson_is_raw(value)) {
      const char* raw = yyjson_get_raw(value);
      if (raw == nullptr) {
        return tl::make_unexpected(
            failure(config::ConfigErrorCode::kParseError, "yyjson omitted a raw number"));
      }
      const std::string_view token{raw, yyjson_get_len(value)};
      auto consumed = cursor_.consume_token(token);
      result =
          consumed
              ? parse_number(token)
              : config::ConfigResult<config::ConfigValue>{tl::make_unexpected(consumed.error())};
    } else if (yyjson_is_arr(value)) {
      result = convert_array(value, depth, path);
    } else if (yyjson_is_obj(value)) {
      result = convert_object(value, depth, path);
    }
    if (result && spans_ != nullptr && !path.empty() &&
        result->kind() != config::ConfigValueKind::kArray &&
        result->kind() != config::ConfigValueKind::kObject) {
      // Reserved scalar envelopes expose one user leaf, not their private control fields.
      spans_->resize(span_checkpoint);
      spans_->push_back(config::detail::LosslessScalarSpan{std::string{path},
                                                           result->kind(),
                                                           source_offset_ + begin,
                                                           source_offset_ + cursor_.position()});
    }
    return result;
  }

  ParseOptions options_;
  config::ConfigValueLimits limits_;
  std::vector<config::detail::LosslessScalarSpan>* spans_{nullptr};
  std::size_t source_offset_{0};
  std::uint32_t nodes_{0};
  JsonSourceCursor cursor_;
};

struct YyjsonDocumentDeleter {
  void operator()(yyjson_doc* document) const noexcept { yyjson_doc_free(document); }
};

}  // namespace

config::ConfigResult<config::ConfigValue> parse_json(
    std::string_view source, ParseOptions options,
    std::vector<config::detail::LosslessScalarSpan>* spans, std::size_t source_offset) {
  yyjson_read_err error{};
  constexpr yyjson_read_flag kReadFlags = YYJSON_READ_NUMBER_AS_RAW;
  const std::unique_ptr<yyjson_doc, YyjsonDocumentDeleter> document{yyjson_read_opts(
      const_cast<char*>(source.data()), source.size(), kReadFlags, nullptr, &error)};
  if (!document) {
    const auto code = error.code == YYJSON_READ_ERROR_MEMORY_ALLOCATION
                          ? config::ConfigErrorCode::kLimitExceeded
                          : config::ConfigErrorCode::kParseError;
    const std::string_view message = error.code == YYJSON_READ_ERROR_MEMORY_ALLOCATION
                                         ? "JSON parser allocation failed"
                                         : "input does not conform to strict JSON syntax";
    return tl::make_unexpected(failure(code, message, source, error.pos));
  }
  yyjson_val* root = yyjson_doc_get_root(document.get());
  if (root == nullptr) {
    return tl::make_unexpected(
        failure(config::ConfigErrorCode::kParseError, "JSON document has no root value", source));
  }
  return JsonConverter{source, options, spans, source_offset}.run(root);
}

}  // namespace holoscan::config_io::detail
