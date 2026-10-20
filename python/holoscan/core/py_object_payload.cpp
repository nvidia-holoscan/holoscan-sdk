/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "py_object_payload.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/predicate_input_view.hpp>

#include "errors.hpp"

namespace holoscan {

PyObjectPayload::PyObjectPayload(nb::handle value) : value_(retain(value)) {}

nb::object PyObjectPayload::borrow() const {
  return value_ ? nb::borrow<nb::object>(nb::handle{value_.get()}) : nb::none();
}

std::shared_ptr<void> PyObjectPayload::share() const noexcept {
  return value_;
}

std::shared_ptr<PyObject> PyObjectPayload::retain(nb::handle value) {
  PyObject* const object = value.ptr();
  Py_INCREF(object);
  return std::shared_ptr<PyObject>{object, PythonDeleter{}};
}

void PyObjectPayload::PythonDeleter::operator()(PyObject* value) const noexcept {
  if (value == nullptr || !nb::is_alive()) {
    return;
  }
  try {
    const nb::gil_scoped_acquire gil;
    if (gil.is_valid()) {
      Py_DECREF(value);
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Interpreter teardown cannot safely report or retry a final reference release.
  }
}

void init_py_object_payload(nb::module_& module) {
  static std::once_flag flag;
  std::call_once(flag, [] {
    auto registered = authoring::authoring_registry().register_payload(
        authoring::detail::make_registered_payload<PyObjectPayload>("holoscan.PyObjectPayload"));
    if (!registered) {
      throw RuntimeError{std::move(registered).error()};
    }
  });
  module.def("_test_release_pyobject_payload_off_thread", [](const nb::callable& factory) {
    std::unique_ptr<PyObjectPayload> payload;
    {
      const nb::object value = factory();
      payload = std::make_unique<PyObjectPayload>(value);
    }
    const nb::gil_scoped_release release;
    std::thread worker{[owned = std::move(payload)]() mutable { owned.reset(); }};
    worker.join();
  });
}

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<PyObjectPayload>() noexcept {
  static const RuntimePortTypeOps ops{
      .payload_size = sizeof(PyObjectPayload),
      .payload_alignment = alignof(PyObjectPayload),
      .wire_capacity = 0U,
      .sample_size = sizeof(Sample<PyObjectPayload>),
      .sample_alignment = alignof(Sample<PyObjectPayload>),
      .predicate_view_size = sizeof(PredicateInputView<PyObjectPayload>),
      .predicate_view_alignment = alignof(PredicateInputView<PyObjectPayload>),
      .codec_family = RuntimePayloadCodecFamily::kProcessLocalObject,
      .payload_kind = PayloadKind::kInlineTyped,
      .inline_pod = false,
      .generic_loan_supported = false,
      .persistent_sample_storage = false,
      .generic_loan_slot_size_bytes = 0U,
      .encode_payload = nullptr,
      .construct_sample = nullptr,
      .construct_sample_from_value = [](void* storage,
                                        const void* payload,
                                        SampleMetadata metadata) noexcept -> expected<void, Error> {
        if (storage == nullptr || payload == nullptr ||
            reinterpret_cast<std::uintptr_t>(storage) % alignof(Sample<PyObjectPayload>) != 0U) {
          return make_unexpected(Error{ErrorCode::kReceiveError});
        }
        std::construct_at(static_cast<Sample<PyObjectPayload>*>(storage),
                          *static_cast<const PyObjectPayload*>(payload),
                          metadata);
        return {};
      },
      .destroy_sample =
          [](void* storage) noexcept {
            std::destroy_at(static_cast<Sample<PyObjectPayload>*>(storage));
          },
      .retain_sample_payload_for_deferred_release =
          [](const void* storage, std::shared_ptr<void>* retained) noexcept {
            *retained = static_cast<const Sample<PyObjectPayload>*>(storage)->data.share();
          },
      .sample_payload = [](const void* sample) noexcept -> const void* {
        return &static_cast<const Sample<PyObjectPayload>*>(sample)->data;
      },
      .sample_metadata = [](const void* sample) noexcept -> const SampleMetadata* {
        return &static_cast<const Sample<PyObjectPayload>*>(sample)->metadata;
      },
      .construct_predicate_view =
          [](void* storage,
             const void* sample,
             InputSelection selection,
             const std::uint64_t* active_generation,
             std::uint64_t expected_generation) noexcept -> expected<void, Error> {
        if (storage == nullptr || sample == nullptr || active_generation == nullptr ||
            reinterpret_cast<std::uintptr_t>(storage) %
                    alignof(PredicateInputView<PyObjectPayload>) !=
                0U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto view = make_runtime_predicate_input_view<PyObjectPayload>(
            PredicateInputBinding<PyObjectPayload>{
                .sample = static_cast<const Sample<PyObjectPayload>*>(sample),
                .selection = selection,
                .active_generation = active_generation,
                .expected_generation = expected_generation,
            });
        if (!view) {
          return make_unexpected(std::move(view).error());
        }
        std::construct_at(static_cast<PredicateInputView<PyObjectPayload>*>(storage),
                          std::move(*view));
        return {};
      },
      .destroy_predicate_view =
          [](void* storage) noexcept {
            std::destroy_at(static_cast<PredicateInputView<PyObjectPayload>*>(storage));
          },
      .construct_value = nullptr,
      .destroy_value = nullptr,
      .bind_input =
          [](void* declaration, RuntimeInputBinding* binding) noexcept {
            RuntimePortBindingAccess::bind_input(*static_cast<Input<PyObjectPayload>*>(declaration),
                                                 binding);
          },
      .unbind_input =
          [](void* declaration, RuntimeInputBinding* binding) noexcept {
            RuntimePortBindingAccess::unbind_input(
                *static_cast<Input<PyObjectPayload>*>(declaration), binding);
          },
      .bind_output =
          [](void* declaration, RuntimeOutputBinding* binding) noexcept {
            RuntimePortBindingAccess::bind_output(
                *static_cast<Output<PyObjectPayload>*>(declaration), binding);
          },
      .unbind_output =
          [](void* declaration, RuntimeOutputBinding* binding) noexcept {
            RuntimePortBindingAccess::unbind_output(
                *static_cast<Output<PyObjectPayload>*>(declaration), binding);
          },
  };
  return &ops;
}

}  // namespace detail
}  // namespace holoscan
