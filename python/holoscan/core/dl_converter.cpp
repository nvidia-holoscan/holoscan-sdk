/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dl_converter.hpp"

#include <cuda_runtime.h>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

#include "imported_tensor.hpp"
#include "integer_tuple.hpp"

namespace holoscan {

namespace {

struct DLPackConsumerStream {
  cudaStream_t stream = nullptr;
  bool synchronize = true;
};

constexpr const char* dlpack_c_exchange_api_attr_name{"__dlpack_c_exchange_api__"};
constexpr const char* dlpack_exchange_api_capsule_name{"dlpack_exchange_api"};

// Array protocol probes miss for most array types: CuPy and NumPy have no C exchange API, and an
// array-interface producer has no DLPack methods. Under the 3.12 limited API, PyObject_GetAttr and
// PyObject_HasAttr build and clear an AttributeError for every miss, which costs more than the
// rest of a host Tensor import. Python's getattr(obj, name, default) answers a miss without one.
nb::str interned_name(const char* name) {
  PyObject* const interned = PyUnicode_InternFromString(name);
  if (interned == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::str>(interned);
}

struct ProtocolAttributes {
  nb::object getattr;
  nb::object missing;
  nb::str dlpack_c_exchange_api = interned_name(dlpack_c_exchange_api_attr_name);
  nb::str dlpack = interned_name("__dlpack__");
  nb::str dlpack_device = interned_name("__dlpack_device__");
  nb::str cuda_array_interface = interned_name("__cuda_array_interface__");
  nb::str array_interface = interned_name("__array_interface__");
};

const ProtocolAttributes& protocol_attributes() {
  // Built once with the GIL held, by calls that keep it, and never destroyed: no Python
  // reference may be released after the interpreter finalizes.
  static const ProtocolAttributes* const attributes = [] {
    auto created = std::make_unique<ProtocolAttributes>();
    created->getattr = nb::module_::import_("builtins").attr("getattr");
    created->missing =
        nb::steal(PyObject_CallNoArgs(reinterpret_cast<PyObject*>(&PyBaseObject_Type)));
    if (!created->missing.is_valid()) {
      throw nb::python_error();
    }
    return created.release();
  }();
  return *attributes;
}

// Return `obj.<name>`, or an invalid object if it is missing. Like PyObject_HasAttr, a lenient
// lookup also reports any other error from the attribute as missing; a strict one raises it.
nb::object protocol_attribute(nb::handle obj, const nb::str& name, bool strict = false) {
  const auto& attributes = protocol_attributes();
  PyObject* const arguments[] = {obj.ptr(), name.ptr(), attributes.missing.ptr()};
  PyObject* const value = PyObject_Vectorcall(attributes.getattr.ptr(), arguments, 3, nullptr);
  if (value == nullptr) {
    if (strict) {
      throw nb::python_error();
    }
    PyErr_Clear();
    return {};
  }
  nb::object result = nb::steal(value);
  return result.is(attributes.missing) ? nb::object{} : result;
}

std::shared_ptr<Tensor> adopt_read_only_tensor(std::shared_ptr<DLManagedTensorContext> context) {
  if (context != nullptr) {
    context->tensor.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
  }
  return std::make_shared<Tensor>(std::move(context));
}

bool is_cuda_dlpack_device(DLDeviceType device_type) {
  return device_type == kDLCUDA || device_type == kDLCUDAManaged || device_type == kDLCUDAHost;
}

expected<DLDevice, Error> dldevice_from_array_interface_pointer(void* data_ptr, bool cuda) {
  if (!cuda) {
    return DLDevice{kDLCPU, 0};
  }
  return dldevice_from_pointer(data_ptr);
}

// Parse the consumer stream argument of `__dlpack__` per the array API standard and CUDA stream
// protocol: None / 1 = legacy default stream (synchronize), -1 = no synchronization, 2 =
// per-thread default, positive int = stream pointer, or __cuda_stream__() -> (0, stream pointer).
DLPackConsumerStream parse_dlpack_stream(const nb::object& stream_obj) {
  int64_t stream_id = 1;  // legacy default stream

  if (stream_obj.is_none()) {
    stream_id = 1;
  } else if (nb::isinstance<nb::int_>(stream_obj)) {
    stream_id = nb::cast<int64_t>(stream_obj);
    if (stream_id == 0 || stream_id < -1) {
      throw std::runtime_error(
          "Invalid stream, valid stream should be -1 (non-blocking), 1 (legacy default stream), 2 "
          "(per-thread default stream), or a positive integer (stream pointer)");
    }
  } else if (nb::hasattr(stream_obj, "__cuda_stream__")) {
    const nb::object stream_protocol = stream_obj.attr("__cuda_stream__");
    if (PyCallable_Check(stream_protocol.ptr()) == 0) {
      throw std::runtime_error("Invalid stream type: __cuda_stream__ must be callable");
    }
    const nb::object stream_info = stream_protocol();
    const Py_ssize_t stream_info_size = PySequence_Length(stream_info.ptr());
    if (stream_info_size < 0) {
      PyErr_Clear();
      throw std::runtime_error("__cuda_stream__ must return a sequence with 2 elements");
    }
    if (stream_info_size != 2) {
      throw std::runtime_error("__cuda_stream__ must return a sequence with 2 elements");
    }
    const nb::object stream_version = nb::steal(PySequence_GetItem(stream_info.ptr(), 0));
    const nb::object stream_handle = nb::steal(PySequence_GetItem(stream_info.ptr(), 1));
    const int64_t version = nb::cast<int64_t>(stream_version);
    if (version != 0) {
      throw std::runtime_error("__cuda_stream__ protocol version must be 0");
    }
    stream_id = nb::cast<int64_t>(stream_handle);
    if (stream_id < 0) {
      throw std::runtime_error("__cuda_stream__ stream pointer must be non-negative");
    }
  } else {
    throw std::runtime_error(
        "Invalid stream type: should be int, None, or object with callable __cuda_stream__");
  }

  if (stream_id == -1) {
    return DLPackConsumerStream{.stream = nullptr, .synchronize = false};
  }
  if (stream_id == 1) {
    return DLPackConsumerStream{.stream = nullptr, .synchronize = true};
  }
  if (stream_id == 2) {
    return DLPackConsumerStream{.stream = cudaStreamPerThread, .synchronize = true};
  }
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
  return DLPackConsumerStream{
      .stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(stream_id)),
      .synchronize = true};
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
}

// PyCapsule destructor: free the DLManagedTensor only if no consumer has claimed it (i.e. the
// capsule still carries its original name rather than the "used_*" variant).
void dlpack_capsule_destructor(PyObject* ptr) {
  const char* name = PyCapsule_GetName(ptr);
  if (name == nullptr) {
    return;
  }
  const bool is_versioned = (std::strcmp(name, dlpack_versioned_capsule_name) == 0);
  const char* valid_name = is_versioned ? dlpack_versioned_capsule_name : dlpack_capsule_name;
  if (PyCapsule_IsValid(ptr, valid_name) == 0) {
    return;
  }
  if (is_versioned) {
    auto* tensor = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(ptr, valid_name));
    if (tensor != nullptr && tensor->deleter != nullptr) {
      tensor->deleter(tensor);
    }
  } else {
    auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, valid_name));
    if (tensor != nullptr && tensor->deleter != nullptr) {
      tensor->deleter(tensor);
    }
  }
}

// Producer deleters may touch Python references and must run outside CUDA host callbacks.
template <typename Managed>
std::shared_ptr<TensorMemoryReference> imported_dlpack_reference(Managed* managed) {
  std::shared_ptr<void> owner{managed, [](void* pointer) {
                                auto* tensor = static_cast<Managed*>(pointer);
                                if (tensor->deleter != nullptr && nb::is_alive()) {
                                  const nb::gil_scoped_acquire gil;
                                  if (gil.is_valid()) {
                                    tensor->deleter(tensor);
                                  }
                                }
                              }};
  return make_python_tensor_memory_reference(std::move(owner), managed->dl_tensor.device);
}

DLPackConsumerStream import_stream(const nb::object& stream) {
  if (stream.is_none()) {
    if (const auto active = PyImportedTensorScope::stream()) {
      return DLPackConsumerStream{.stream = *active, .synchronize = true};
    }
  }
  const auto consumer = parse_dlpack_stream(stream);
  if (!consumer.synchronize) {
    throw std::runtime_error("Tensor import requires a readiness stream; stream=-1 is unsupported");
  }
  return consumer;
}

std::int64_t dlpack_stream_id(cudaStream_t stream) {
  if (stream == nullptr || stream == cudaStreamLegacy) {
    return 1;
  }
  if (stream == cudaStreamPerThread) {
    return 2;
  }
  return static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(stream));
}

[[nodiscard]] std::vector<std::int64_t> cast_int64_sequence(const nb::object& value,
                                                            const char* error_message) {
  PyObject* sequence = PySequence_Fast(value.ptr(), error_message);
  if (sequence == nullptr) {
    throw nb::python_error();
  }
  const nb::object sequence_owner = nb::steal(sequence);
  const Py_ssize_t size = PySequence_Size(sequence);
  if (size < 0) {
    throw nb::python_error();
  }
  std::vector<std::int64_t> result;
  result.reserve(static_cast<std::size_t>(size));
  for (Py_ssize_t index = 0; index < size; ++index) {
    PyObject* item = PySequence_GetItem(sequence, index);
    if (item == nullptr) {
      throw nb::python_error();
    }
    result.push_back(nb::cast<std::int64_t>(nb::steal(item)));
  }
  return result;
}

// Import a Tensor from an object exposing __array_interface__ (host) or __cuda_array_interface__
// (device). Both are zero-copy views; ArrayInterfaceMemoryReference keeps the producer alive.
// Keep protocol validation, metadata ownership, and CUDA readiness in one import path so
// the zero-copy Tensor retains exactly the producer whose interface was validated.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::shared_ptr<Tensor> tensor_from_array_interface(const nb::object& obj, bool cuda) {
  const char* attr_name = cuda ? "__cuda_array_interface__" : "__array_interface__";
  const nb::dict interface = nb::cast<nb::dict>(obj.attr(attr_name));
  PyObject* dict = interface.ptr();
  auto require = [&](const char* key) -> nb::object {
    PyObject* value = PyDict_GetItemString(dict, key);
    if (value == nullptr) {
      throw std::runtime_error(std::string("array interface is missing required key '") + key +
                               "'");
    }
    return nb::borrow(value);
  };

  auto shape = cast_int64_sequence(require("shape"), "array interface shape must be a sequence");
  auto typestr = nb::cast<std::string>(require("typestr"));
  auto data = nb::cast<nb::tuple>(require("data"));
  // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
  void* data_ptr = reinterpret_cast<void*>(nb::cast<uintptr_t>(nb::object(data[0])));

  auto maybe_dtype = dldatatype_from_typestr(typestr);
  if (!maybe_dtype) {
    throw std::runtime_error(runtime_error_message(maybe_dtype.error()));
  }
  auto maybe_device = dldevice_from_array_interface_pointer(data_ptr, cuda);
  if (!maybe_device) {
    throw std::runtime_error(runtime_error_message(maybe_device.error()));
  }

  // Array-interface strides are expressed in bytes and may be omitted or None for a
  // C-contiguous layout. DLPack uses element strides and, since DLPack 1.2, requires
  // explicit strides when ndim != 0, so materialize contiguous strides below.
  std::vector<int64_t> element_strides;
  PyObject* strides_obj = PyDict_GetItemString(dict, "strides");  // borrowed; may be null
  if (strides_obj != nullptr && strides_obj != Py_None) {
    element_strides =
        cast_int64_sequence(nb::borrow(strides_obj), "array interface strides must be a sequence");
    if (element_strides.size() != shape.size()) {
      throw std::runtime_error("array interface strides length must match shape rank");
    }
  }

  DLTensor local{};
  local.data = data_ptr;
  local.device = maybe_device.value();
  local.ndim = static_cast<int32_t>(shape.size());
  local.dtype = maybe_dtype.value();
  local.shape = shape.data();
  local.strides = nullptr;
  local.byte_offset = 0;

  if (strides_obj != nullptr && strides_obj != Py_None) {
    const int64_t elem_size = local.dtype.bits / 8;
    for (int64_t& stride : element_strides) {
      stride /= elem_size;
    }
  } else {
    calc_strides(local, element_strides, /*to_num_elements=*/true);
  }
  local.strides = element_strides.data();

  auto context = std::make_shared<DLManagedTensorContext>();
  Py_INCREF(obj.ptr());
  std::shared_ptr<void> owner{obj.ptr(), [](void* pointer) {
                                if (nb::is_alive()) {
                                  const nb::gil_scoped_acquire gil;
                                  if (gil.is_valid()) {
                                    Py_DECREF(static_cast<PyObject*>(pointer));
                                  }
                                }
                              }};
  context->memory_reference = make_python_tensor_memory_reference(std::move(owner), local.device);
  copy_dl_tensor_with_owned_metadata(*context, local);

  auto tensor = adopt_read_only_tensor(std::move(context));

  if (cuda) {
    // The CUDA array interface's optional `stream` is the producer's ready stream; record it so a
    // later export / host access can order against it. Absent or None -> default stream; 2 ->
    // per-thread default; >2 -> stream pointer (the array-API convention).
    cudaStream_t readiness = nullptr;
    PyObject* stream_obj = PyDict_GetItemString(dict, "stream");
    if (stream_obj != nullptr && stream_obj != Py_None) {
      const int64_t stream_id = nb::cast<int64_t>(nb::borrow(stream_obj));
      if (stream_id == 2) {
        readiness = cudaStreamPerThread;
      } else if (stream_id > 2) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
        readiness = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));
      }
    }
    if (!tensor->set_readiness_stream(readiness)) {
      throw std::runtime_error("Failed to record imported CUDA array readiness");
    }
    track_python_tensor(*tensor);
  }
  return tensor;
}

struct DLPackExportVersion {
  bool versioned{};
  std::uint32_t major{HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR};
  std::uint32_t minor{HOLOSCAN_DLPACK_IMPL_VERSION_MINOR};
};

struct DLPackManagedExport {
  void* tensor{};
  const char* capsule_name{};
  bool versioned{};
};

[[nodiscard]] DLPackExportVersion negotiate_dlpack_export_version(
    const std::optional<std::tuple<int, int>>& max_version) noexcept {
  DLPackExportVersion negotiated;
  if (!max_version.has_value()) {
    return negotiated;
  }
  const auto [major, minor] = *max_version;
  if (major < 1) {
    return negotiated;
  }
  negotiated.versioned = true;
  if (static_cast<std::uint32_t>(major) == HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR && minor >= 0 &&
      static_cast<std::uint32_t>(minor) < HOLOSCAN_DLPACK_IMPL_VERSION_MINOR) {
    negotiated.minor = static_cast<std::uint32_t>(minor);
  }
  return negotiated;
}

[[nodiscard]] std::optional<DLDevice> parse_dlpack_device_request(
    const std::optional<std::tuple<int, int>>& requested_device) {
  if (!requested_device.has_value()) {
    return std::nullopt;
  }
  const auto [device_type, device_id] = *requested_device;
  return DLDevice{static_cast<DLDeviceType>(device_type), device_id};
}

void synchronize_dlpack_export(const Tensor& tensor, const DLPackConsumerStream consumer_stream) {
  if (!consumer_stream.synchronize || !is_cuda_dlpack_device(tensor.device().device_type)) {
    return;
  }
  auto synchronized = tensor.synchronize_for_consumer_stream(consumer_stream.stream);
  if (!synchronized) {
    throw std::runtime_error(runtime_error_message(synchronized.error()));
  }
}

[[noreturn]] void throw_dlpack_export_error(const Error& error) {
  const std::string message = runtime_error_message(error);
  if (error.code == ErrorCode::kNotSupported) {
    throw nb::buffer_error(message.c_str());
  }
  throw std::runtime_error(message);
}

[[nodiscard]] DLPackManagedExport export_versioned_dlpack(
    Tensor& tensor, const DLPackExportVersion version,
    const std::optional<DLDevice>& requested_device) {
  auto exported = tensor.to_dlpack_versioned();
  if (!exported) {
    throw_dlpack_export_error(exported.error());
  }
  auto* managed = exported.value();
  managed->version.major = version.major;
  managed->version.minor = version.minor;
  if (requested_device.has_value()) {
    managed->dl_tensor.device = *requested_device;
  }
  constexpr std::uint64_t kExportFlags = 0U;
  managed->flags |= kExportFlags;
  return DLPackManagedExport{managed, dlpack_versioned_capsule_name, true};
}

[[nodiscard]] DLPackManagedExport export_unversioned_dlpack(
    Tensor& tensor, const std::optional<DLDevice>& requested_device) {
  auto exported = tensor.to_dlpack();
  if (!exported) {
    throw_dlpack_export_error(exported.error());
  }
  auto* managed = exported.value();
  if (requested_device.has_value()) {
    managed->dl_tensor.device = *requested_device;
  }
  return DLPackManagedExport{managed, dlpack_capsule_name, false};
}

void release_unclaimed_dlpack_export(const DLPackManagedExport& exported) {
  if (exported.versioned) {
    auto* managed = static_cast<DLManagedTensorVersioned*>(exported.tensor);
    managed->deleter(managed);
    return;
  }
  auto* managed = static_cast<DLManagedTensor*>(exported.tensor);
  managed->deleter(managed);
}

[[nodiscard]] nb::capsule make_dlpack_capsule(const DLPackManagedExport& exported) {
  PyObject* capsule =
      PyCapsule_New(exported.tensor, exported.capsule_name, dlpack_capsule_destructor);
  if (capsule == nullptr) {
    release_unclaimed_dlpack_export(exported);
    throw nb::python_error();
  }
  return nb::steal<nb::capsule>(capsule);
}

}  // namespace

nb::capsule py_dlpack(Tensor* tensor, const nb::object& stream,
                      std::optional<std::tuple<int, int>> max_version,
                      std::optional<std::tuple<int, int>> dl_device, std::optional<bool> copy) {
  // DLPack has no stream field. The Python protocol passes the consumer stream here, and Holoscan
  // inserts any needed dependency before producing the capsule.
  const auto consumer_stream = parse_dlpack_stream(stream);
  const DLPackExportVersion version = negotiate_dlpack_export_version(max_version);
  const std::optional<DLDevice> requested_device = parse_dlpack_device_request(dl_device);
  const DLDevice source_device = tensor->device();
  const bool device_copy =
      requested_device && (requested_device->device_type != source_device.device_type ||
                           requested_device->device_id != source_device.device_id);
  if (device_copy && copy.has_value() && !*copy) {
    throw nb::buffer_error("copy=False cannot satisfy a different dl_device request");
  }

  std::optional<Tensor> copied;
  Tensor* exported_tensor = tensor;
  if (copy.value_or(false) || device_copy) {
    const DLDevice target_device = requested_device.value_or(source_device);
    const bool asynchronous_copy =
        source_device.device_type == kDLCUDA || source_device.device_type == kDLCUDAManaged ||
        target_device.device_type == kDLCUDA || target_device.device_type == kDLCUDAManaged;
    if (asynchronous_copy && !consumer_stream.synchronize) {
      throw nb::buffer_error("stream=-1 cannot order an explicit Tensor copy");
    }
    expected<Tensor, Error> result;
    {
      const nb::gil_scoped_release release;
      result = tensor->copy_to(target_device, consumer_stream.stream);
    }
    if (!result) {
      throw_dlpack_export_error(result.error());
    }
    copied.emplace(std::move(*result));
    exported_tensor = &*copied;
  }
  synchronize_dlpack_export(*exported_tensor, consumer_stream);
  if (consumer_stream.synchronize) {
    track_python_tensor(*exported_tensor, consumer_stream.stream);
  } else {
    track_python_tensor_untracked_export(*exported_tensor);
  }

  const DLPackManagedExport exported =
      version.versioned ? export_versioned_dlpack(*exported_tensor, version, requested_device)
                        : export_unversioned_dlpack(*exported_tensor, requested_device);
  return make_dlpack_capsule(exported);
}

nb::tuple py_dlpack_device(Tensor* tensor) {
  const auto device = tensor->device();
  return nb::make_tuple(static_cast<int>(device.device_type), device.device_id);
}

namespace {

// Import through the producer's bound `__dlpack__` and, if it has one, `__dlpack_device__`.
std::shared_ptr<Tensor> import_dlpack(const nb::object& dlpack, const nb::object& dlpack_device,
                                      const nb::object& stream) {
  // Array API: query __dlpack_device__ first so we can negotiate an appropriate import stream.
  bool producer_is_cuda = false;
  if (dlpack_device.is_valid()) {
    auto device = nb::cast<nb::tuple>(dlpack_device());
    if (device.size() >= 1) {
      producer_is_cuda = is_cuda_dlpack_device(static_cast<DLDeviceType>(nb::cast<int>(device[0])));
    }
  }
  const auto consumer = producer_is_cuda ? import_stream(stream) : DLPackConsumerStream{};
  // Negotiate directly on the activation stream inside compute(), avoiding a default-stream
  // detour. Outside compute(), callers can pass any __cuda_stream__ object or stream integer.
  const nb::object capsule =
      producer_is_cuda ? dlpack(nb::arg("stream") = dlpack_stream_id(consumer.stream)) : dlpack();
  PyObject* cap = capsule.ptr();

  auto context = std::make_shared<DLManagedTensorContext>();
  if (PyCapsule_IsValid(cap, dlpack_versioned_capsule_name) != 0) {
    auto* managed = static_cast<DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(cap, dlpack_versioned_capsule_name));
    if (managed == nullptr) {
      throw std::runtime_error("invalid versioned DLPack capsule");
    }
    // Reject a newer major version without taking ownership: the struct layout may differ.
    if (managed->version.major > HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR) {
      throw std::runtime_error("unsupported DLPack major version");
    }
    PyCapsule_SetName(cap, used_dlpack_versioned_capsule_name);
    context->memory_reference = imported_dlpack_reference(managed);
    copy_dl_tensor_with_owned_metadata(*context, *managed);
  } else if (PyCapsule_IsValid(cap, dlpack_capsule_name) != 0) {
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, dlpack_capsule_name));
    if (managed == nullptr) {
      throw std::runtime_error("invalid DLPack capsule");
    }
    PyCapsule_SetName(cap, used_dlpack_capsule_name);
    context->memory_reference = imported_dlpack_reference(managed);
    copy_dl_tensor_with_owned_metadata(*context, managed->dl_tensor);
  } else {
    throw std::runtime_error("__dlpack__ did not return a valid DLPack capsule");
  }

  auto tensor = adopt_read_only_tensor(std::move(context));
  if (producer_is_cuda) {
    if (!tensor->set_readiness_stream(consumer.stream)) {
      throw std::runtime_error("Failed to record imported DLPack CUDA readiness");
    }
    track_python_tensor(*tensor);
  }
  return tensor;
}

}  // namespace

std::shared_ptr<Tensor> tensor_from_dlpack(const nb::object& obj, const nb::object& stream) {
  const auto& attributes = protocol_attributes();
  const nb::object dlpack = protocol_attribute(obj, attributes.dlpack);
  if (!dlpack.is_valid()) {
    throw std::runtime_error("object does not support the __dlpack__ protocol");
  }
  return import_dlpack(dlpack, protocol_attribute(obj, attributes.dlpack_device), stream);
}

namespace {

nb::handle type_of(const nb::object& obj) {
  return reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()));
}

bool has_dlpack_c_exchange_api(const nb::object& obj) {
  return protocol_attribute(type_of(obj), protocol_attributes().dlpack_c_exchange_api).is_valid();
}

std::shared_ptr<Tensor> try_tensor_from_dlpack_c_exchange(const nb::object& obj);

}  // namespace

bool is_tensor_like(const nb::object& obj) {
  const auto& attributes = protocol_attributes();
  return has_dlpack_c_exchange_api(obj) ||
         (protocol_attribute(obj, attributes.dlpack).is_valid() &&
          protocol_attribute(obj, attributes.dlpack_device).is_valid()) ||
         protocol_attribute(obj, attributes.cuda_array_interface).is_valid() ||
         protocol_attribute(obj, attributes.array_interface).is_valid();
}

nb::tuple tensor_shape_tuple(const Tensor& tensor) {
  return python::detail::scaled_integer_tuple(tensor.shape_span());
}

nb::tuple tensor_strides_tuple(const Tensor& tensor) {
  const int64_t item_size = static_cast<int64_t>(tensor.itemsize());
  const std::span<const int64_t> explicit_strides = tensor.dlpack_strides_span();
  if (!explicit_strides.empty()) {
    return python::detail::scaled_integer_tuple(explicit_strides, item_size);
  }

  const std::span<const int64_t> shape = tensor.shape_span();
  // Keep this implicit row-major rule aligned with calc_strides() in dlpack_utils.cpp. Building
  // each tuple entry directly avoids allocating a temporary C++ strides vector.
  return python::detail::integer_tuple(shape.size(), [&shape, item_size](std::size_t index) {
    int64_t stride = item_size;
    for (std::size_t trailing = index + 1; trailing < shape.size(); ++trailing) {
      stride *= shape[trailing];
    }
    return stride;
  });
}

namespace {

// Build the common array-interface dict (shared by the host and CUDA variants). `cuda` adds the
// `mask`/`stream` keys required by __cuda_array_interface__.
nb::dict build_array_interface(const Tensor& tensor, const void* data_pointer, bool cuda) {
  auto maybe_typestr = numpy_dtype(tensor.dtype());
  if (!maybe_typestr) {
    throw std::runtime_error(runtime_error_message(maybe_typestr.error()));
  }
  const char* typestr = maybe_typestr.value();

  nb::tuple shape = tensor_shape_tuple(tensor);

  // data is (address, read_only).
  nb::tuple data = nb::make_tuple(reinterpret_cast<uintptr_t>(data_pointer), tensor.is_read_only());

  // The array interface uses byte strides; DLPack stores element strides (or nullptr = contiguous).
  nb::object strides = nb::none();
  const std::span<const int64_t> explicit_strides = tensor.dlpack_strides_span();
  if (!explicit_strides.empty()) {
    strides = python::detail::scaled_integer_tuple(explicit_strides,
                                                   static_cast<int64_t>(tensor.itemsize()));
  }

  nb::list descr;
  descr.append(nb::make_tuple("", typestr));

  nb::dict interface;
  interface["shape"] = shape;
  interface["typestr"] = typestr;
  interface["data"] = data;
  interface["version"] = 3;
  interface["strides"] = strides;
  interface["descr"] = descr;
  if (cuda) {
    interface["mask"] = nb::none();
    interface["stream"] = 1;  // legacy default stream
  }
  return interface;
}

}  // namespace

nb::dict array_interface(const Tensor& tensor) {
  const void* const data_pointer = tensor.data();
  if (data_pointer == nullptr) {
    throw nb::attribute_error("__array_interface__");
  }
  const DLDeviceType device_type = tensor.device().device_type;
  if (device_type != kDLCPU && device_type != kDLCUDAHost) {
    throw nb::attribute_error("__array_interface__");
  }
  if (tensor.has_readiness_stream_contract()) {
    // The array interface carries no stream, and NumPy reads the bytes on the CPU as soon as it
    // has this dictionary. Wait on the host for any producer work first, as `__dlpack__` does for a
    // host consumer. A Tensor written on the host has nothing pending, so this returns at once.
    // Release the GIL: a CUDA-written Tensor can keep this thread waiting on the GPU.
    expected<void, Error> synchronized;
    {
      const nb::gil_scoped_release release;
      synchronized = tensor.synchronize_for_host();
    }
    if (!synchronized) {
      throw std::runtime_error(runtime_error_message(synchronized.error()));
    }
  }
  return build_array_interface(tensor, data_pointer, /*cuda=*/false);
}

nb::dict cuda_array_interface(const Tensor& tensor) {
  const void* const data_pointer = tensor.data();
  if (data_pointer == nullptr) {
    throw nb::attribute_error("__cuda_array_interface__");
  }
  const DLDeviceType device_type = tensor.device().device_type;
  if (device_type != kDLCUDA && device_type != kDLCUDAManaged) {
    throw nb::attribute_error("__cuda_array_interface__");
  }
  if (tensor.has_readiness_stream_contract()) {
    // The CUDA array interface's `stream` field below advertises legacy-default stream readiness.
    // If this Tensor tracks a different producer/readiness stream, make that statement true before
    // exposing the dictionary. Tensors without a readiness contract keep the historical behavior:
    // they export metadata only, and synchronization remains the producer's responsibility.
    auto sync_result = tensor.synchronize_for_consumer_stream(static_cast<cudaStream_t>(nullptr));
    if (!sync_result) {
      throw std::runtime_error(runtime_error_message(sync_result.error()));
    }
  }
  track_python_tensor_untracked_export(tensor);
  return build_array_interface(tensor, data_pointer, /*cuda=*/true);
}

std::shared_ptr<Tensor> as_tensor(const nb::object& obj, const nb::object& stream) {
  std::shared_ptr<Tensor> existing;
  if (nb::try_cast(obj, existing) && existing != nullptr && existing->is_read_only()) {
    if (!stream.is_none() && is_cuda_dlpack_device(existing->device().device_type)) {
      const auto consumer = import_stream(stream);
      const auto synchronized = existing->synchronize_for_consumer_stream(consumer.stream);
      if (!synchronized) {
        throw std::runtime_error(runtime_error_message(synchronized.error()));
      }
    }
    track_python_tensor(*existing);
    return existing;
  }
  // Protocol precedence: DLPack C exchange first, then Python DLPack, then
  // __cuda_array_interface__, then __array_interface__.
  // All paths preserve the data pointer, but their synchronization and lifetime information
  // differs. Prefer Python DLPack over array interfaces for its explicit consumer stream:
  //  - Ownership/lifetime: DLPack transfers ownership (the imported Tensor owns the data via the
  //    capsule deleter), so it stays valid even if the source object is dropped; the array
  //    interfaces are views that require us to keep the producer object alive.
  //  - Stream semantics: the readiness-stream contract is built around DLPack's
  //    __dlpack__(stream=...) negotiation; the CUDA array interface's stream field is looser.
  //  - Metadata/standardization: DLPack (versioned) carries the read-only/copied flags and a
  //    version and is the array-API-endorsed interchange path; the array interfaces are the older
  //    convention.
  // This flips the 4.x default, which preferred __cuda_array_interface__.
  const auto prepare = [&](std::shared_ptr<Tensor> tensor) {
    if (!stream.is_none() && is_cuda_dlpack_device(tensor->device().device_type)) {
      const auto consumer = import_stream(stream);
      const auto synchronized = tensor->synchronize_for_consumer_stream(consumer.stream);
      if (!synchronized) {
        throw std::runtime_error(runtime_error_message(synchronized.error()));
      }
      // Import queues only a readiness wait here. Actual consumers register on access/export;
      // borrowing this stream until the Tensor is released would outlive temporary import streams.
    }
    return tensor;
  };
  if (auto tensor = try_tensor_from_dlpack_c_exchange(obj)) {
    return prepare(std::move(tensor));
  }
  const auto& attributes = protocol_attributes();
  if (const nb::object dlpack = protocol_attribute(obj, attributes.dlpack); dlpack.is_valid()) {
    if (const nb::object dlpack_device = protocol_attribute(obj, attributes.dlpack_device);
        dlpack_device.is_valid()) {
      return import_dlpack(dlpack, dlpack_device, stream);
    }
  }
  if (protocol_attribute(obj, attributes.cuda_array_interface).is_valid()) {
    return prepare(tensor_from_array_interface(obj, /*cuda=*/true));
  }
  if (protocol_attribute(obj, attributes.array_interface).is_valid()) {
    return tensor_from_array_interface(obj, /*cuda=*/false);
  }
  throw std::runtime_error(
      "object does not support __dlpack_c_exchange_api__, __dlpack__, "
      "__cuda_array_interface__, or __array_interface__");
}

// ---------------------------------------------------------------------------------------------
// DLPack 1.3 fast C exchange API (https://dmlc.github.io/dlpack/latest/python_spec.html). Publishes
// a __dlpack_c_exchange_api__ capsule on the Tensor type so peer C extensions can exchange without
// per-call PyCapsule/Python overhead. Ported from the 4.x pybind11 implementation; the 5.x version
// uses the GIL-safe versioned memory reference for imports (no PyTensor subclass).
// ---------------------------------------------------------------------------------------------
namespace {

// Wrap a producer-owned DLManagedTensorVersioned into a holoscan Tensor, taking ownership: the
// GIL-acquiring memory reference is constructed first so the producer's deleter runs (via RAII)
// even if a later step throws.
std::shared_ptr<Tensor> wrap_versioned_owning(DLManagedTensorVersioned* managed) {
  auto memory_reference = imported_dlpack_reference(managed);
  if (managed->version.major > HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR) {
    throw std::runtime_error("unsupported DLPack major version");
  }
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = std::move(memory_reference);
  copy_dl_tensor_with_owned_metadata(*context, *managed);
  return adopt_read_only_tensor(std::move(context));
}

std::shared_ptr<Tensor> try_tensor_from_dlpack_c_exchange(const nb::object& obj) {
  const nb::object api_capsule_obj =
      protocol_attribute(type_of(obj), protocol_attributes().dlpack_c_exchange_api, true);
  if (!api_capsule_obj.is_valid()) {
    return nullptr;
  }
  auto* api = static_cast<DLPackExchangeAPI*>(
      PyCapsule_GetPointer(api_capsule_obj.ptr(), dlpack_exchange_api_capsule_name));
  if (api == nullptr) {
    throw nb::python_error();
  }
  if (api->header.version.major != DLPACK_MAJOR_VERSION) {
    throw std::runtime_error("DLPack C exchange API major version mismatch");
  }
  if (api->managed_tensor_from_py_object_no_sync == nullptr ||
      api->current_work_stream == nullptr) {
    throw std::runtime_error("DLPack C exchange API table is missing required callbacks");
  }

  DLManagedTensorVersioned* managed_tensor = nullptr;
  if (api->managed_tensor_from_py_object_no_sync(obj.ptr(), &managed_tensor) != 0) {
    if (PyErr_Occurred() != nullptr) {
      throw nb::python_error();
    }
    throw std::runtime_error("DLPack C exchange managed export failed");
  }
  if (managed_tensor == nullptr) {
    throw std::runtime_error("DLPack C exchange managed export returned null");
  }

  auto tensor = wrap_versioned_owning(managed_tensor);
  const auto device = tensor->device();
  if (is_cuda_dlpack_device(device.device_type)) {
    void* current_stream = nullptr;
    if (api->current_work_stream(device.device_type, device.device_id, &current_stream) != 0) {
      if (PyErr_Occurred() != nullptr) {
        throw nb::python_error();
      }
      throw std::runtime_error("DLPack C exchange current stream query failed");
    }
    if (!tensor->set_readiness_stream(static_cast<cudaStream_t>(current_stream))) {
      throw std::runtime_error("Failed to record C exchange CUDA readiness");
    }
    track_python_tensor(*tensor);
  }
  return tensor;
}

expected<void, Error> prepare_tensor_for_dlpack_c_exchange_export(const Tensor& tensor) {
  const auto device = tensor.device();
  if (!is_cuda_dlpack_device(device.device_type)) {
    return {};
  }
  // The DLPack C exchange API's current_work_stream callback is type-level, not object-specific.
  // Holoscan does not have a global current stream per device, so it reports the legacy default
  // stream and makes that reported stream safe for readiness-aware tensors before export.
  // Common paths have a readiness contract and should succeed:
  // - MemoryBuffer-backed Holoscan tensors fall back to MemoryBuffer::stream() when unset.
  // - Python DLPack import records the negotiated producer-readiness stream.
  // - CUDA array interface import records the producer stream, or legacy default when absent.
  // - C exchange import records the producer's current_work_stream() result.
  // Failure is reserved for externally owned or custom memory references whose readiness cannot be
  // represented safely.
  auto synchronized = tensor.synchronize_for_consumer_stream(static_cast<cudaStream_t>(nullptr));
  if (synchronized) {
    track_python_tensor_untracked_export(tensor);
  }
  return synchronized;
}

expected<void, Error> prepare_tensor_for_unversioned_dlpack_c_exchange_export(
    const Tensor& tensor) {
  if (tensor.is_read_only()) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "Unversioned DLPack C exchange cannot represent read-only Tensor authority"});
  }
  return prepare_tensor_for_dlpack_c_exchange_export(tensor);
}

void set_dlpack_c_exchange_export_error(const Error& error) {
  const std::string message = runtime_error_message(error);
  if (error.code == ErrorCode::kNotImplemented || error.code == ErrorCode::kNotSupported) {
    PyErr_SetString(PyExc_BufferError, message.c_str());
    return;
  }
  PyErr_SetString(PyExc_RuntimeError, message.c_str());
}

int holoscan_dlpack_managed_tensor_allocator([[maybe_unused]] DLTensor* prototype,
                                             DLManagedTensorVersioned** out, void* error_ctx,
                                             void (*set_error)(void* error_ctx, const char* kind,
                                                               const char* message)) noexcept {
  if (out != nullptr) {
    *out = nullptr;
  }
  if (set_error != nullptr) {
    set_error(error_ctx, "BufferError", "Holoscan Tensor does not support DLPack allocation");
  }
  return -1;
}

int holoscan_dlpack_managed_tensor_from_py_object_no_sync(void* py_object,
                                                          DLManagedTensorVersioned** out) noexcept {
  if (out == nullptr) {
    PyErr_SetString(PyExc_ValueError, "DLPack output pointer must not be null");
    return -1;
  }
  *out = nullptr;
  try {
    std::shared_ptr<Tensor> tensor;
    if (!nb::try_cast(nb::handle(static_cast<PyObject*>(py_object)), tensor) || !tensor) {
      PyErr_SetString(PyExc_TypeError, "expected a holoscan.Tensor object");
      return -1;
    }
    auto sync_result = prepare_tensor_for_dlpack_c_exchange_export(*tensor);
    if (!sync_result) {
      set_dlpack_c_exchange_export_error(sync_result.error());
      return -1;
    }
    auto exported = tensor->to_dlpack_versioned();
    if (!exported) {
      set_dlpack_c_exchange_export_error(exported.error());
      return -1;
    }
    *out = exported.value();
    return 0;
  } catch (nb::python_error& error) {
    error.restore();
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
  return -1;
}

int holoscan_dlpack_managed_tensor_to_py_object_no_sync(DLManagedTensorVersioned* tensor,
                                                        void** out_py_object) noexcept {
  if (out_py_object == nullptr) {
    if (tensor != nullptr && tensor->deleter != nullptr) {
      tensor->deleter(tensor);
    }
    PyErr_SetString(PyExc_ValueError, "Python object output pointer must not be null");
    return -1;
  }
  *out_py_object = nullptr;
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_ValueError, "DLPack tensor must not be null");
    return -1;
  }
  try {
    // wrap_versioned_owning takes ownership; on any throw its memory reference frees the producer
    // tensor via RAII, so no manual deleter call is needed here.
    nb::object py_object = nb::cast(wrap_versioned_owning(tensor));
    *out_py_object = py_object.release().ptr();
    return 0;
  } catch (nb::python_error& error) {
    error.restore();
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
  return -1;
}

int holoscan_dlpack_dltensor_from_py_object_no_sync(void* py_object, DLTensor* out) noexcept {
  if (out == nullptr) {
    PyErr_SetString(PyExc_ValueError, "DLPack output tensor must not be null");
    return -1;
  }
  try {
    std::shared_ptr<Tensor> tensor;
    if (!nb::try_cast(nb::handle(static_cast<PyObject*>(py_object)), tensor) || !tensor) {
      PyErr_SetString(PyExc_TypeError, "expected a holoscan.Tensor object");
      return -1;
    }
    auto sync_result = prepare_tensor_for_unversioned_dlpack_c_exchange_export(*tensor);
    if (!sync_result) {
      set_dlpack_c_exchange_export_error(sync_result.error());
      return -1;
    }
    const auto shape = tensor->shape_span();
    const auto strides = tensor->dlpack_strides_span();
    const auto byte_offset = tensor->byte_offset();
    const auto logical_address = reinterpret_cast<uintptr_t>(tensor->data());
    if (logical_address < byte_offset) {
      PyErr_SetString(PyExc_BufferError, "cannot export invalid Holoscan Tensor data");
      return -1;
    }
    out->data = reinterpret_cast<void*>(logical_address - byte_offset);
    out->device = tensor->device();
    out->ndim = tensor->ndim();
    out->dtype = tensor->dtype();
    out->shape = shape.empty() ? nullptr : const_cast<int64_t*>(shape.data());
    out->strides = strides.empty() ? nullptr : const_cast<int64_t*>(strides.data());
    out->byte_offset = byte_offset;
    return 0;
  } catch (nb::python_error& error) {
    error.restore();
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
  return -1;
}

int holoscan_dlpack_current_work_stream([[maybe_unused]] DLDeviceType device_type,
                                        [[maybe_unused]] int32_t device_id,
                                        void** out_current_stream) noexcept {
  if (out_current_stream == nullptr) {
    PyErr_SetString(PyExc_ValueError, "DLPack stream output pointer must not be null");
    return -1;
  }
  // This callback is type-level and cannot describe per-Tensor readiness streams. Holoscan reports
  // the legacy default stream and orders readiness-aware exports onto it in the tensor callbacks.
  *out_current_stream = nullptr;
  return 0;
}

const DLPackExchangeAPI* holoscan_dlpack_exchange_api() {
  static const DLPackExchangeAPI api{
      DLPackExchangeAPIHeader{DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}, nullptr},
      holoscan_dlpack_managed_tensor_allocator,
      holoscan_dlpack_managed_tensor_from_py_object_no_sync,
      holoscan_dlpack_managed_tensor_to_py_object_no_sync,
      holoscan_dlpack_dltensor_from_py_object_no_sync,
      holoscan_dlpack_current_work_stream};
  return &api;
}

nb::object dlpack_array_to_tuple_or_none(const int64_t* array, int32_t ndim) {
  if (array == nullptr) {
    return nb::none();
  }
  const std::size_t size = ndim > 0 ? static_cast<std::size_t>(ndim) : 0U;
  return python::detail::scaled_integer_tuple(std::span<const int64_t>{array, size});
}

}  // namespace

nb::capsule dlpack_c_exchange_api_capsule() {
  return nb::capsule(holoscan_dlpack_exchange_api(), dlpack_exchange_api_capsule_name);
}

nb::dict test_tensor_metadata_projection(const Tensor& backing, const nb::object& shape,
                                         const nb::object& strides, DLDataType dtype,
                                         std::uint64_t byte_offset) {
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = cast_int64_sequence(shape, "test Tensor shape must be a sequence");
  if (context->dl_shape.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw nb::value_error("test Tensor shape rank exceeds the DLPack limit");
  }

  if (!strides.is_none()) {
    context->dl_strides =
        cast_int64_sequence(strides, "test Tensor strides must be a sequence or None");
    if (context->dl_strides.size() != context->dl_shape.size()) {
      throw nb::value_error("test Tensor strides length must match shape rank");
    }
  }

  DLTensor& tensor = context->tensor.dl_tensor;
  tensor.data = const_cast<void*>(backing.data());
  tensor.device = backing.device();
  tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
  tensor.dtype = dtype;
  tensor.shape = context->dl_shape.empty() ? nullptr : context->dl_shape.data();
  tensor.strides = context->dl_strides.empty() ? nullptr : context->dl_strides.data();
  tensor.byte_offset = byte_offset;
  context->tensor.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;

  const Tensor projection{std::move(context)};
  nb::dict result;
  result["shape"] = tensor_shape_tuple(projection);
  result["strides"] = tensor_strides_tuple(projection);
  result["itemsize"] = projection.itemsize();
  result["data"] = reinterpret_cast<uintptr_t>(projection.data());
  result["array_interface"] = array_interface(projection);
  return result;
}

nb::dict test_dlpack_c_exchange_api(const nb::object& tensor_obj) {
  PyObject* api_capsule = PyObject_GetAttrString(
      reinterpret_cast<PyObject*>(Py_TYPE(tensor_obj.ptr())), "__dlpack_c_exchange_api__");
  if (api_capsule == nullptr) {
    throw nb::python_error();
  }
  const nb::object api_capsule_obj = nb::steal(api_capsule);
  auto* api = static_cast<DLPackExchangeAPI*>(
      PyCapsule_GetPointer(api_capsule_obj.ptr(), dlpack_exchange_api_capsule_name));
  if (api == nullptr) {
    throw nb::python_error();
  }
  if (api->header.version.major != DLPACK_MAJOR_VERSION) {
    throw std::runtime_error("DLPack C exchange API major version mismatch");
  }
  if (api->managed_tensor_from_py_object_no_sync == nullptr ||
      api->managed_tensor_to_py_object_no_sync == nullptr ||
      api->dltensor_from_py_object_no_sync == nullptr || api->current_work_stream == nullptr) {
    throw std::runtime_error("DLPack C exchange API table is missing required callbacks");
  }

  DLManagedTensorVersioned* managed_tensor = nullptr;
  if (api->managed_tensor_from_py_object_no_sync(tensor_obj.ptr(), &managed_tensor) != 0) {
    throw nb::python_error();
  }
  if (managed_tensor == nullptr) {
    throw std::runtime_error("DLPack C exchange managed export returned null");
  }
  auto free_managed = [&managed_tensor]() {
    if (managed_tensor != nullptr && managed_tensor->deleter != nullptr) {
      managed_tensor->deleter(managed_tensor);
    }
  };

  DLTensor stack_tensor{};
  bool stack_export_supported = true;
  if (api->dltensor_from_py_object_no_sync(tensor_obj.ptr(), &stack_tensor) != 0) {
    if (PyErr_ExceptionMatches(PyExc_BufferError) == 0) {
      free_managed();
      throw nb::python_error();
    }
    // A bare DLTensor has no read-only flag. The C exchange callback therefore rejects a
    // read-only Tensor rather than silently granting write authority; retain that outcome in the
    // test result while continuing to exercise the versioned managed exchange path.
    PyErr_Clear();
    stack_export_supported = false;
  }

  void* current_stream = reinterpret_cast<void*>(0x1);
  if (api->current_work_stream(managed_tensor->dl_tensor.device.device_type,
                               managed_tensor->dl_tensor.device.device_id,
                               &current_stream) != 0) {
    free_managed();
    throw nb::python_error();
  }

  DLManagedTensorVersioned* import_tensor = nullptr;
  if (api->managed_tensor_from_py_object_no_sync(tensor_obj.ptr(), &import_tensor) != 0) {
    free_managed();
    throw nb::python_error();
  }
  void* imported_py_object_ptr = nullptr;
  if (api->managed_tensor_to_py_object_no_sync(import_tensor, &imported_py_object_ptr) != 0) {
    free_managed();
    throw nb::python_error();
  }
  if (imported_py_object_ptr == nullptr) {
    free_managed();
    throw std::runtime_error("DLPack C exchange import returned null Python object");
  }
  const nb::object imported_py_object =
      nb::steal(reinterpret_cast<PyObject*>(imported_py_object_ptr));

  nb::dict result;
  result["api_version"] = nb::make_tuple(api->header.version.major, api->header.version.minor);
  result["managed_version"] =
      nb::make_tuple(managed_tensor->version.major, managed_tensor->version.minor);
  result["managed_flags"] = managed_tensor->flags;
  result["device"] = nb::make_tuple(static_cast<int>(managed_tensor->dl_tensor.device.device_type),
                                    managed_tensor->dl_tensor.device.device_id);
  result["ndim"] = managed_tensor->dl_tensor.ndim;
  result["shape"] = dlpack_array_to_tuple_or_none(managed_tensor->dl_tensor.shape,
                                                  managed_tensor->dl_tensor.ndim);
  result["strides"] = dlpack_array_to_tuple_or_none(managed_tensor->dl_tensor.strides,
                                                    managed_tensor->dl_tensor.ndim);
  result["byte_offset"] = managed_tensor->dl_tensor.byte_offset;
  result["data"] = reinterpret_cast<uintptr_t>(managed_tensor->dl_tensor.data);
  result["stack_export_supported"] = stack_export_supported;
  if (stack_export_supported) {
    result["stack_data"] = reinterpret_cast<uintptr_t>(stack_tensor.data);
    result["stack_shape"] = dlpack_array_to_tuple_or_none(stack_tensor.shape, stack_tensor.ndim);
    result["stack_strides"] =
        dlpack_array_to_tuple_or_none(stack_tensor.strides, stack_tensor.ndim);
  } else {
    result["stack_data"] = nb::none();
    result["stack_shape"] = nb::none();
    result["stack_strides"] = nb::none();
  }
  result["stream"] = reinterpret_cast<uintptr_t>(current_stream);
  result["imported_shape"] = imported_py_object.attr("shape");
  result["imported_strides"] = imported_py_object.attr("strides");

  free_managed();
  return result;
}

}  // namespace holoscan
