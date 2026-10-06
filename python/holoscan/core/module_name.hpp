/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <utility>

namespace holoscan::python {

/// Bind a native class while nanobind snapshots its canonical public module name.
template <typename T, typename... Bases, typename... Extra>
nanobind::class_<T, Bases...> bind_class_in_module(nanobind::module_& native_module,
                                                   const char* public_module, const char* name,
                                                   Extra&&... extra) {
  nanobind::object native_name = nanobind::borrow<nanobind::object>(native_module.attr("__name__"));
  native_module.attr("__name__") = public_module;
  try {
    auto result = nanobind::class_<T, Bases...>(native_module, name, std::forward<Extra>(extra)...);
    native_module.attr("__name__") = std::move(native_name);
    return result;
  } catch (...) {
    native_module.attr("__name__") = std::move(native_name);
    throw;
  }
}

/// Bind a native enum while nanobind snapshots its canonical public module name.
template <typename T, typename... Extra>
nanobind::enum_<T> bind_enum_in_module(nanobind::module_& native_module, const char* public_module,
                                       const char* name, Extra&&... extra) {
  nanobind::object native_name = nanobind::borrow<nanobind::object>(native_module.attr("__name__"));
  native_module.attr("__name__") = public_module;
  try {
    auto result = nanobind::enum_<T>(native_module, name, std::forward<Extra>(extra)...);
    native_module.attr("__name__") = std::move(native_name);
    return result;
  } catch (...) {
    native_module.attr("__name__") = std::move(native_name);
    throw;
  }
}

/// Bind native functions while nanobind snapshots their canonical public module name.
template <typename Bind>
void bind_functions_in_module(nanobind::module_& native_module, const char* public_module,
                              Bind&& bind) {
  nanobind::object native_name = nanobind::borrow<nanobind::object>(native_module.attr("__name__"));
  native_module.attr("__name__") = public_module;
  try {
    std::forward<Bind>(bind)();
  } catch (...) {
    native_module.attr("__name__") = std::move(native_name);
    throw;
  }
  native_module.attr("__name__") = std::move(native_name);
}

}  // namespace holoscan::python
