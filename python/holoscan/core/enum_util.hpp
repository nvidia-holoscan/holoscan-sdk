/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cctype>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <nanobind/nanobind.h>
#include <magic_enum/magic_enum.hpp>

namespace holoscan::python {

namespace nb = nanobind;

inline std::string python_enum_value_name(std::string_view cpp_name) {
  if (!cpp_name.empty() && cpp_name.front() == 'k') {
    cpp_name.remove_prefix(1);
  }

  std::string python_name;
  python_name.reserve(cpp_name.size());
  for (const char c : cpp_name) {
    python_name.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return python_name;
}

template <typename E, typename... Extra>
nb::enum_<E> export_enum(nb::handle scope, const char* python_name, Extra&&... extra) {
  static_assert(std::is_enum_v<E>);

  nb::enum_<E> enum_type(scope, python_name, std::forward<Extra>(extra)...);
  for (const auto& [value, name] : magic_enum::enum_entries<E>()) {
    const std::string value_name = python_enum_value_name(name);
    enum_type.value(value_name.c_str(), value);
  }
  return enum_type;
}

template <typename E>
std::string enum_value_name(E value) {
  static_assert(std::is_enum_v<E>);

  const std::string_view name = magic_enum::enum_name(value);
  if (name.empty()) {
    return "unknown";
  }
  return python_enum_value_name(name);
}

template <typename E>
std::string cpp_enum_value_name(E value) {
  static_assert(std::is_enum_v<E>);

  const std::string_view name = magic_enum::enum_name(value);
  if (name.empty()) {
    return "unknown";
  }
  return std::string{name};
}

}  // namespace holoscan::python
