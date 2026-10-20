/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "errors.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <magic_enum/magic_enum.hpp>

#include <array>
#include <cctype>
#include <cstddef>
#include <exception>
#include <string>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>

#include "errors_pydoc.hpp"
#include "module_name.hpp"

namespace holoscan {
namespace {

constexpr auto kErrorCodes = magic_enum::enum_values<ErrorCode>();
constexpr std::size_t kErrorCodeCount = kErrorCodes.size();

consteval bool error_codes_are_dense() {
  for (std::size_t index = 0U; index < kErrorCodes.size(); ++index) {
    if (static_cast<std::size_t>(kErrorCodes[index]) != index) {
      return false;
    }
  }
  return true;
}

static_assert(error_codes_are_dense(), "Python ErrorCode lookup requires dense values from zero");

struct ExceptionState {
  std::array<nb::object, kErrorCodeCount> exception_types{};
  std::array<nb::object, kErrorCodeCount> compile_exception_types{};
  std::array<nb::object, kErrorCodeCount> error_codes{};
  nb::object compile_exception{};
};

std::string exception_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::kNotImplemented:
      return "HoloscanNotImplementedError";
    case ErrorCode::kTimeout:
      return "HoloscanTimeoutError";
    case ErrorCode::kInterrupted:
      return "HoloscanInterruptedError";
    default:
      break;
  }
  std::string_view name = error_code_name(code);
  if (!name.empty() && name.front() == 'k') {
    name.remove_prefix(1U);
  }
  std::string result{name};
  if (!name.ends_with("Error")) {
    result.append("Error");
  }
  return result;
}

// Convert the C++ spelling to the public Python enum spelling, for example
// ErrorCode::kInvalidArgument to ErrorCode.INVALID_ARGUMENT.
std::string python_error_code_name(ErrorCode code) {
  std::string_view name = error_code_name(code);
  if (!name.empty() && name.front() == 'k') {
    name.remove_prefix(1U);
  }
  std::string result;
  result.reserve(name.size() + 8U);
  for (std::size_t index = 0U; index < name.size(); ++index) {
    const unsigned char character = static_cast<unsigned char>(name[index]);
    if (index != 0U && std::isupper(character) != 0 &&
        std::islower(static_cast<unsigned char>(name[index - 1U])) != 0) {
      result.push_back('_');
    }
    result.push_back(static_cast<char>(std::toupper(character)));
  }
  return result;
}

std::string stable_token(const RuntimeError& error) {
  std::string_view message = error.what();
  const std::string_view code_name = error_code_name(error.code());
  if (message == code_name) {
    message = {};
  } else if (message.starts_with(code_name) && message.substr(code_name.size()).starts_with(": ")) {
    message.remove_prefix(code_name.size() + 2U);
  }
  std::size_t length{};
  while (length < message.size()) {
    const unsigned char character = static_cast<unsigned char>(message[length]);
    if (!(std::isupper(character) != 0 || std::isdigit(character) != 0 || character == '_')) {
      break;
    }
    ++length;
  }
  if (length != 0U) {
    return std::string{message.substr(0U, length)};
  }
  return python_error_code_name(error.code());
}

void set_python_error(const RuntimeError& error, PyObject* type, ExceptionState& state) {
  const std::size_t index = static_cast<std::size_t>(error.code());
  if (type == nullptr) {
    type = index < state.exception_types.size() && state.exception_types[index].is_valid()
               ? state.exception_types[index].ptr()
               : nullptr;
  }
  if (type == nullptr) {
    type = state.exception_types[static_cast<std::size_t>(ErrorCode::kFailure)].ptr();
  }
  if (type == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Holoscan exception type metadata is unavailable");
    return;
  }

  PyObject* instance = PyObject_CallFunction(type, "s", error.what());
  if (instance == nullptr) {
    return;
  }
  const std::string token = stable_token(error);
  if (token.size() > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
    Py_DECREF(instance);
    PyErr_SetString(PyExc_RuntimeError, "Holoscan exception token is too large");
    return;
  }
  PyObject* token_object =
      PyUnicode_FromStringAndSize(token.data(), static_cast<Py_ssize_t>(token.size()));
  if (token_object == nullptr) {
    Py_DECREF(instance);
    return;
  }
  PyObject* code = index < state.error_codes.size() && state.error_codes[index].is_valid()
                       ? state.error_codes[index].ptr()
                       : state.error_codes[static_cast<std::size_t>(ErrorCode::kFailure)].ptr();
  if (code == nullptr) {
    Py_DECREF(token_object);
    Py_DECREF(instance);
    PyErr_SetString(PyExc_RuntimeError, "Holoscan error-code metadata is unavailable");
    return;
  }
  if (PyObject_SetAttrString(instance, "code", code) != 0 ||
      PyObject_SetAttrString(instance, "token", token_object) != 0) {
    Py_DECREF(token_object);
    Py_DECREF(instance);
    return;
  }
  Py_DECREF(token_object);
  PyErr_SetObject(type, instance);
  Py_DECREF(instance);
}

void translate_runtime_error(const std::exception_ptr& exception, void* payload) {
  try {
    std::rethrow_exception(exception);
  } catch (const CompileError& error) {
    auto& state = *static_cast<ExceptionState*>(payload);
    const std::size_t index = static_cast<std::size_t>(error.code());
    PyObject* type = index < state.compile_exception_types.size() &&
                             state.compile_exception_types[index].is_valid()
                         ? state.compile_exception_types[index].ptr()
                         : state.compile_exception.ptr();
    set_python_error(error, type, state);
  } catch (const RuntimeError& error) {
    auto& state = *static_cast<ExceptionState*>(payload);
    set_python_error(error, nullptr, state);
  }
}

}  // namespace

void init_errors(nb::module_& module) {
  auto error_code_type = python::bind_enum_in_module<ErrorCode>(
      module, "holoscan.errors", "ErrorCode", nb::is_arithmetic(), doc::errors::kErrorCode);
  for (const ErrorCode code : kErrorCodes) {
    const std::string name = python_error_code_name(code);
    error_code_type.value(name.c_str(), code);
  }

  // nanobind retains translator payloads for the lifetime of its interpreter internals without a
  // payload cleanup hook. Keep owning references here so module teardown cannot leave borrowed
  // exception pointers behind; the state deliberately shares that translator lifetime.
  auto* state = new ExceptionState;
  constexpr std::string_view kPublicErrorModule = "holoscan.errors";
  const std::string base_name = fmt::format("{}.HoloscanError", kPublicErrorModule);
  PyObject* base = PyErr_NewException(base_name.c_str(), PyExc_Exception, nullptr);
  if (base == nullptr) {
    throw nb::python_error();
  }
  module.attr("HoloscanError") = nb::steal<nb::object>(base);
  module.attr("HoloscanError").attr("__doc__") = doc::errors::kHoloscanError;

  const std::string consumed_loan_name = fmt::format("{}.ConsumedLoanError", kPublicErrorModule);
  PyObject* consumed_loan =
      PyErr_NewException(consumed_loan_name.c_str(), module.attr("HoloscanError").ptr(), nullptr);
  if (consumed_loan == nullptr) {
    throw nb::python_error();
  }
  module.attr("ConsumedLoanError") = nb::steal<nb::object>(consumed_loan);

  const std::string compile_name = fmt::format("{}.CompileError", kPublicErrorModule);
  PyObject* compile_type =
      PyErr_NewException(compile_name.c_str(), module.attr("HoloscanError").ptr(), nullptr);
  if (compile_type == nullptr) {
    throw nb::python_error();
  }
  module.attr("CompileError") = nb::steal<nb::object>(compile_type);
  state->compile_exception = module.attr("CompileError");

  for (const ErrorCode code : kErrorCodes) {
    const std::size_t index = static_cast<std::size_t>(code);
    const std::string enum_name = python_error_code_name(code);
    state->error_codes[index] = module.attr("ErrorCode").attr(enum_name.c_str());
    if (code == ErrorCode::kSuccess) {
      state->exception_types[index] = module.attr("HoloscanError");
      continue;
    }
    const std::string class_name = exception_name(code);
    const std::string qualified_name = fmt::format("{}.{}", kPublicErrorModule, class_name);
    PyObject* type =
        PyErr_NewException(qualified_name.c_str(), module.attr("HoloscanError").ptr(), nullptr);
    if (type == nullptr) {
      throw nb::python_error();
    }
    module.attr(class_name.c_str()) = nb::steal<nb::object>(type);
    state->exception_types[index] = module.attr(class_name.c_str());
  }

  for (const ErrorCode code : kErrorCodes) {
    const std::size_t index = static_cast<std::size_t>(code);
    if (code == ErrorCode::kSuccess) {
      state->compile_exception_types[index] = state->compile_exception;
      continue;
    }
    const std::string qualified_name =
        fmt::format("{}._Compile{}", kPublicErrorModule, exception_name(code));
    const nb::tuple bases = nb::make_tuple(state->compile_exception, state->exception_types[index]);
    PyObject* type = PyErr_NewException(qualified_name.c_str(), bases.ptr(), nullptr);
    if (type == nullptr) {
      throw nb::python_error();
    }
    state->compile_exception_types[index] = nb::steal<nb::object>(type);
  }

  nb::register_exception_translator(&translate_runtime_error, state);
}

[[noreturn]] void throw_consumed_loan_error() {
  const nb::object errors = nb::module_::import_("holoscan.errors");
  const nb::object type = errors.attr("ConsumedLoanError");
  const nb::object instance = type("Tensor output loan has already been consumed");
  instance.attr("code") = errors.attr("ErrorCode").attr("INVALID_ARGUMENT");
  instance.attr("token") = "PYTHON_TENSOR_LOAN_CONSUMED";
  PyErr_SetObject(type.ptr(), instance.ptr());
  throw nb::python_error();
}

}  // namespace holoscan
