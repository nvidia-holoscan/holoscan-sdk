/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/hosted_parameter_snapshot.hpp>
#include <holoscan/core/lifecycle.hpp>

namespace holoscan {

namespace nb = nanobind;

// Marker base used to recognize Python-defined operator classes at Graph.op() dispatch.
class PyNativeOperatorBase {};

// Guards a callback-scoped Core object shared with Python and rejects use after invalidation.
template <typename Value>
class PyCallbackBorrow {
 public:
  explicit PyCallbackBorrow(Value& value) : state_(std::make_shared<State>(&value)) {}

  template <typename Function>
  decltype(auto) with_value(std::string_view expired_token, Function&& function) const {
    const std::lock_guard lock(state_->mutex);
    if (state_->value == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotReady, std::string{expired_token}}};
    }
    return std::invoke(std::forward<Function>(function), *state_->value);
  }

  void invalidate() noexcept {
    const std::lock_guard lock(state_->mutex);
    state_->value = nullptr;
  }

 private:
  struct State {
    explicit State(Value* initial) : value(initial) {}

    std::mutex mutex;
    Value* value{};
  };

  std::shared_ptr<State> state_;
};

// Owns a Python object while making refcount changes safe across callback and teardown threads.
class GilSafeObject {
 public:
  GilSafeObject() = default;
  // Construct only from a Python-facing path that already holds a valid GIL.
  explicit GilSafeObject(nb::handle value) : value_(value.ptr()) { Py_XINCREF(value_); }

  GilSafeObject(const GilSafeObject& other) : value_(other.value_) {
    if (value_ != nullptr && nb::is_alive()) {
      const nb::gil_scoped_acquire gil;
      if (gil.is_valid()) {
        Py_INCREF(value_);
      } else {
        value_ = nullptr;
      }
    }
  }

  GilSafeObject& operator=(const GilSafeObject& other) {
    if (this != &other) {
      GilSafeObject copy{other};
      swap(copy);
    }
    return *this;
  }

  GilSafeObject(GilSafeObject&& other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  GilSafeObject& operator=(GilSafeObject&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  ~GilSafeObject() { reset(); }

  [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }
  [[nodiscard]] nb::object borrow() const {
    return value_ == nullptr ? nb::none() : nb::borrow<nb::object>(nb::handle{value_});
  }

 private:
  void swap(GilSafeObject& other) noexcept { std::swap(value_, other.value_); }
  void reset() noexcept {
    if (value_ != nullptr && nb::is_alive()) {
      const nb::gil_scoped_acquire gil;
      if (gil.is_valid()) {
        Py_DECREF(value_);
      }
    }
    value_ = nullptr;
  }

  PyObject* value_{};
};

// Stores the Python lifecycle methods selected while configuring one hosted operator.
class PyLifecycleHookBindings {
 public:
  explicit PyLifecycleHookBindings(nb::handle instance);

  void reset() noexcept;
  void bind(LifecycleStage stage, std::string_view method, bool required);
  [[nodiscard]] bool has(LifecycleStage stage) const noexcept;
  [[nodiscard]] bool required(LifecycleStage stage) const noexcept;
  [[nodiscard]] const GilSafeObject& callback(LifecycleStage stage) const noexcept;

 private:
  static constexpr std::size_t kStageCount =
      static_cast<std::size_t>(LifecycleStage::kRelease) + 1U;

  GilSafeObject instance_;
  std::array<GilSafeObject, kStageCount> callbacks_;
  std::array<bool, kStageCount> required_{};
};

// Everything a hosted operator declaration needs, prepared before any Graph mutation so that the
// fallible Python work (class inspection, parameter validation, value copies) cannot follow a
// committed declaration.
struct PreparedNativeOperator {
  // Type-erased `const PyStateCapture` (private to native_operator.cpp): the validated Python
  // class, value snapshot, and port and parameter definitions that the hosted operator's factory
  // captures. Only prepare_native_operator() creates it and only add_native_operator() casts it
  // back, which keeps the capture type out of this header.
  std::shared_ptr<const void> capture;
  std::string type_name;
  HostedParameterSnapshot parameters;
  nb::dict values;
};

// Return the prepared declaration for a hosted operator class, or nullopt for any other type.
[[nodiscard]] std::optional<PreparedNativeOperator> prepare_native_operator(
    nb::handle type, const nb::kwargs& parameters);

// Declare a prepared hosted operator and harvest its ports as one atomic Graph mutation.
[[nodiscard]] authoring::RegisteredOperatorAddition add_native_operator(
    Graph& graph, std::string_view instance_name, const PreparedNativeOperator& prepared);

// Marks the start of interpreter exit, after the session drain. From then on, a hosted callback on
// a Runtime thread fails instead of entering the interpreter, and a Runtime thread that exits
// leaves its retained Python thread state to interpreter finalization.
void mark_python_exit_started() noexcept;

void init_native_operator(nb::module_& module);

}  // namespace holoscan
