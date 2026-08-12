/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_catalog.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan {
namespace {

[[nodiscard]] constexpr bool ascii_edge_whitespace(unsigned char value) noexcept {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
         value == '\v';
}

[[nodiscard]] constexpr bool uri_reserved(unsigned char value) noexcept {
  switch (value) {
    case ':':
    case '/':
    case '?':
    case '#':
    case '[':
    case ']':
    case '@':
    case '!':
    case '$':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case ';':
    case '=':
      return true;
    default:
      return false;
  }
}

[[nodiscard]] constexpr int hex_digit(unsigned char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

[[nodiscard]] bool valid_utf8(std::string_view value) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7FU) {
      ++index;
      continue;
    }
    std::size_t length{};
    std::uint32_t code_point{};
    std::uint32_t minimum{};
    if ((first & 0xE0U) == 0xC0U) {
      length = 2U;
      code_point = first & 0x1FU;
      minimum = 0x80U;
    } else if ((first & 0xF0U) == 0xE0U) {
      length = 3U;
      code_point = first & 0x0FU;
      minimum = 0x800U;
    } else if ((first & 0xF8U) == 0xF0U) {
      length = 4U;
      code_point = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (length > value.size() - index) {
      return false;
    }
    for (std::size_t offset = 1U; offset < length; ++offset) {
      const auto continuation = static_cast<unsigned char>(value[index + offset]);
      if ((continuation & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3FU);
    }
    if (code_point < minimum || code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += length;
  }
  return true;
}

[[nodiscard]] bool valid_decoded_path_segment(std::string_view segment) noexcept {
  if (segment.empty() || segment == "." || segment == ".." ||
      ascii_edge_whitespace(static_cast<unsigned char>(segment.front())) ||
      ascii_edge_whitespace(static_cast<unsigned char>(segment.back()))) {
    return false;
  }
  return valid_utf8(segment) && std::none_of(segment.begin(), segment.end(), [](char character) {
           const auto value = static_cast<unsigned char>(character);
           return value < 0x20U || value == 0x7FU;
         });
}

[[nodiscard]] expected<std::string, Error> decode_path_segment(std::string_view encoded) {
  try {
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
      const auto value = static_cast<unsigned char>(encoded[index]);
      if (value == '%') {
        if (encoded.size() - index < 3U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const int high = hex_digit(static_cast<unsigned char>(encoded[index + 1U]));
        const int low = hex_digit(static_cast<unsigned char>(encoded[index + 2U]));
        if (high < 0 || low < 0) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        decoded.push_back(static_cast<char>((high << 4U) | low));
        index += 2U;
      } else {
        if (uri_reserved(value)) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        decoded.push_back(static_cast<char>(value));
      }
    }
    if (!valid_decoded_path_segment(decoded)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return decoded;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

[[nodiscard]] expected<std::vector<std::string>, Error> parse_path(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.back() == '/') {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    std::vector<std::string> result;
    std::size_t begin = 0U;
    while (begin < path.size()) {
      const std::size_t end = path.find('/', begin);
      const std::string_view segment =
          path.substr(begin, end == std::string_view::npos ? path.size() - begin : end - begin);
      auto decoded = decode_path_segment(segment);
      if (!decoded) {
        return make_unexpected(std::move(decoded.error()));
      }
      result.push_back(std::move(decoded).value());
      if (end == std::string_view::npos) {
        break;
      }
      begin = end + 1U;
    }
    return result;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

[[nodiscard]] expected<std::vector<std::string>, Error> copy_path_segments(
    std::span<const std::string> segments) {
  if (segments.empty() || std::any_of(segments.begin(), segments.end(), [](const auto& value) {
        return !valid_decoded_path_segment(value);
      })) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    return std::vector<std::string>{segments.begin(), segments.end()};
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

}  // namespace

expected<OperatorPath, Error> OperatorPath::parse(std::string_view path) {
  auto parsed = parse_path(path);
  if (!parsed) {
    return make_unexpected(std::move(parsed.error()));
  }
  return OperatorPath{std::move(parsed).value()};
}

expected<OperatorPath, Error> OperatorPath::from_segments(std::span<const std::string> segments) {
  auto copied = copy_path_segments(segments);
  if (!copied) {
    return make_unexpected(std::move(copied.error()));
  }
  return OperatorPath{std::move(copied).value()};
}

expected<ParamPath, Error> ParamPath::parse(std::string_view path) {
  auto parsed = parse_path(path);
  if (!parsed) {
    return make_unexpected(std::move(parsed.error()));
  }
  return ParamPath{std::move(parsed).value()};
}

expected<ParamPath, Error> ParamPath::from_segments(std::span<const std::string> segments) {
  auto copied = copy_path_segments(segments);
  if (!copied) {
    return make_unexpected(std::move(copied.error()));
  }
  return ParamPath{std::move(copied).value()};
}

expected<ParamFieldView, Error> ParameterCatalogView::field(ParamAddress address) const {
  const auto found = std::find_if(fields_.begin(), fields_.end(), [&](const auto& record) {
    return record.address == address;
  });
  if (found != fields_.end()) {
    return ParamFieldView{&*found};
  }
  if (!operators_.empty() && (address.operator_instance.plan_generation !=
                                  operators_.front().operator_instance.plan_generation ||
                              address.operator_instance.run_generation !=
                                  operators_.front().operator_instance.run_generation)) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

expected<ParamFieldView, Error> ParameterCatalogView::field(ParamInstanceId instance) const {
  const auto found = std::find_if(fields_.begin(), fields_.end(), [&](const auto& record) {
    return record.instance == instance;
  });
  return found == fields_.end()
             ? expected<ParamFieldView, Error>{make_unexpected(Error{ErrorCode::kNotFound})}
             : expected<ParamFieldView, Error>{ParamFieldView{&*found}};
}

expected<ParamAddress, Error> ParameterCatalogView::resolve(const OperatorPath& operator_path,
                                                            const ParamPath& param_path) const {
  const auto op = std::find_if(operators_.begin(), operators_.end(), [&](const auto& record) {
    return record.path == operator_path;
  });
  if (op == operators_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto field = std::find_if(fields_.begin(), fields_.end(), [&](const auto& record) {
    return record.address.operator_instance == op->operator_instance && record.path == param_path;
  });
  return field == fields_.end()
             ? expected<ParamAddress, Error>{make_unexpected(Error{ErrorCode::kNotFound})}
             : expected<ParamAddress, Error>{field->address};
}

}  // namespace holoscan
