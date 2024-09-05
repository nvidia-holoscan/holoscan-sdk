/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "dl_converter.hpp"
#include "gxf/std/dlpack_utils.hpp"  // DLDeviceFromPointer, DLDataTypeFromTypeString
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "kwarg_handling.hpp"
#include "tensor.hpp"
#include "tensor_pydoc.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace {

static constexpr const char* dlpack_capsule_name{"dltensor"};
static constexpr const char* used_dlpack_capsule_name{"used_dltensor"};
}  // namespace

namespace holoscan {

void init_tensor(py::module_& m) {
  // DLPack data structures
  py::enum_<DLDeviceType>(m, "DLDeviceType", py::module_local())  //
      .value("DLCPU", kDLCPU)                                     //
      .value("DLCUDA", kDLCUDA)                                   //
      .value("DLCUDAHOST", kDLCUDAHost)                           //
      .value("DLCUDAMANAGED", kDLCUDAManaged);

  py::class_<DLDevice>(m, "DLDevice", py::module_local(), doc::DLDevice::doc_DLDevice)
      .def(py::init<DLDeviceType, int32_t>())
      .def_readonly("device_type", &DLDevice::device_type, doc::DLDevice::doc_device_type)
      .def_readonly("device_id", &DLDevice::device_id, doc::DLDevice::doc_device_id)
      .def(
          "__repr__",
          [](const DLDevice& device) {
            return fmt::format(
                "<DLDevice device_type:{} device_id:{}>", device.device_type, device.device_id);
          },
          R"doc(Return repr(self).)doc");

  // Tensor Class
  py::class_<Tensor, std::shared_ptr<Tensor>>(
      m, "Tensor", py::dynamic_attr(), doc::Tensor::doc_Tensor)
      .def(py::init<>(), doc::Tensor::doc_Tensor)
      .def_property_readonly("ndim", &PyTensor::ndim, doc::Tensor::doc_ndim)
      .def_property_readonly(
          "shape",
          [](const Tensor& tensor) { return vector2pytuple<py::int_>(tensor.shape()); },
          doc::Tensor::doc_shape)
      .def_property_readonly(
          "strides",
          [](const Tensor& tensor) { return vector2pytuple<py::int_>(tensor.strides()); },
          doc::Tensor::doc_strides)
      .def_property_readonly("size", &PyTensor::size, doc::Tensor::doc_size)
      .def_property_readonly("dtype", &PyTensor::dtype, doc::Tensor::doc_dtype)
      .def_property_readonly("itemsize", &PyTensor::itemsize, doc::Tensor::doc_itemsize)
      .def_property_readonly("nbytes", &PyTensor::nbytes, doc::Tensor::doc_nbytes)
      .def_property_readonly(
          "data",
          [](const Tensor& t) {
            return static_cast<int64_t>(reinterpret_cast<uintptr_t>(t.data()));
          },
          doc::Tensor::doc_data)
      .def_property_readonly("device", &PyTensor::device, doc::Tensor::doc_device)
      .def("is_contiguous", &PyTensor::is_contiguous, doc::Tensor::doc_is_contiguous)
      // DLPack protocol
      .def("__dlpack__", &PyTensor::dlpack, "stream"_a = py::none(), doc::Tensor::doc_dlpack)
      .def("__dlpack_device__", &PyTensor::dlpack_device, doc::Tensor::doc_dlpack_device);

  py::class_<PyTensor, Tensor, std::shared_ptr<PyTensor>>(m, "PyTensor", doc::Tensor::doc_Tensor)
      .def_static("as_tensor", &PyTensor::as_tensor, "obj"_a, doc::Tensor::doc_as_tensor)
      .def_static(
          "from_dlpack", &PyTensor::from_dlpack_pyobj, "obj"_a, doc::Tensor::doc_from_dlpack);

  py::enum_<DLDataTypeCode>(m, "DLDataTypeCode", py::module_local())
      .value("DLINT", kDLInt)
      .value("DLUINT", kDLUInt)
      .value("DLFLOAT", kDLFloat)
      .value("DLOPAQUEHANDLE", kDLOpaqueHandle)
      .value("DLBFLOAT", kDLBfloat)
      .value("DLCOMPLEX", kDLComplex);

  py::class_<DLDataType, std::shared_ptr<DLDataType>>(m, "DLDataType", py::module_local())
      .def_readwrite("code", &DLDataType::code)
      .def_readwrite("bits", &DLDataType::bits)
      .def_readwrite("lanes", &DLDataType::lanes)
      .def(
          "__repr__",
          [](const DLDataType& dtype) {
            return fmt::format("<DLDataType: code={}, bits={}, lanes={}>",
                               dldatatypecode_namemap.at(static_cast<DLDataTypeCode>(dtype.code)),
                               dtype.bits,
                               dtype.lanes);
          },
          R"doc(Return repr(self).)doc");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyDLManagedMemoryBuffer definition
////////////////////////////////////////////////////////////////////////////////////////////////////

PyDLManagedMemoryBuffer::PyDLManagedMemoryBuffer(DLManagedTensor* self) : self_(self) {}

PyDLManagedMemoryBuffer::~PyDLManagedMemoryBuffer() {
  // Add the DLManagedTensor pointer to the queue for asynchronous deletion.
  // Without this, the deleter function will be called immediately, which can cause deadlock
  // when the deleter function is called from another non-python thread with GXF runtime mutex
  // acquired (issue 4293741).
  LazyDLManagedTensorDeleter::add(self_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// LazyDLManagedTensorDeleter
////////////////////////////////////////////////////////////////////////////////////////////////////

LazyDLManagedTensorDeleter::LazyDLManagedTensorDeleter() {
  // Use std::memory_order_relaxed because there are no other memory operations that need to be
  // synchronized with the fetch_add operation.
  if (s_instance_count.fetch_add(1, std::memory_order_relaxed) == 0) {
    // Wait until both s_stop and s_is_running are false (busy-waiting).
    // s_stop being true indicates that the previous deleter thread is still in the process
    // of deleting the object.
    while (true) {
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_stop && !s_is_running) { break; }
      }
      // Yield to other threads
      std::this_thread::yield();
    }

    std::lock_guard<std::mutex> lock(s_mutex);
    // Register pthread_atfork() and std::atexit() handlers (registered only once)
    //
    // Note: Issue 4318040
    // When fork() is called in a multi-threaded program, the child process will only have
    // the thread that called fork().
    // Other threads from the parent process won't be running in the child.
    // This can lead to deadlocks if a condition variable or mutex was being waited upon by another
    // thread at the time of the fork.
    // To avoid this, we register pthread_atfork() handlers to acquire all necessary locks in
    // the pre-fork handler and release them in both post-fork handlers, ensuring no mutex or
    // condition variable remains locked in the child.
    if (!s_pthread_atfork_registered) {
      pthread_atfork(on_fork_prepare, on_fork_parent, on_fork_child);
      s_pthread_atfork_registered = true;
      // Register on_exit() to be called when the application exits.
      // Note that the child process will not call on_exit() when fork() is called and exit() is
      // called in the child process.
      std::atexit(on_exit);
    }

    s_is_running = true;
    s_thread = std::thread(run);
    // Detach the thread so that it can be stopped when the application exits
    //
    // Note: Issue 4318040
    // According to the C++ Core Guidelines in CP.24 and CP.26
    // (https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines), std::detach() is generally
    // discouraged.
    // In C++ 20, std::jthread will be introduced to replace std::thread, and std::thread::detach()
    // will be deprecated.
    // However, std::jthread is not available in C++ 17 and we need to use std::thread::detach()
    // for now, with a synchronization mechanism to wait for the thread to finish itself,
    // instead of introducing a new dependency like https://github.com/martinmoene/jthread-lite.
    s_thread.detach();
  }
}

LazyDLManagedTensorDeleter::~LazyDLManagedTensorDeleter() {
  try {
    release();
  } catch (const std::exception& e) {}  // ignore potential fmt::v8::format_error
}

void LazyDLManagedTensorDeleter::add(DLManagedTensor* dl_managed_tensor_ptr) {
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_dlmanaged_tensors_queue.push(dl_managed_tensor_ptr);
  }
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::run() {
  while (true) {
    std::unique_lock<std::mutex> lock(s_mutex);

    s_cv.wait(lock, [] {
      return s_stop || !s_dlmanaged_tensors_queue.empty() || s_cv_do_not_wait_thread;
    });

    // Check if the thread should stop. If queue is not empty, process the queue.
    if (s_stop && s_dlmanaged_tensors_queue.empty()) { break; }

    // Check if the condition variable should not wait for the thread so that fork() can be called
    // without deadlock.
    if (s_cv_do_not_wait_thread) { continue; }

    // move queue onto the local stack before releasing the lock
    std::queue<DLManagedTensor*> local_queue;
    local_queue.swap(s_dlmanaged_tensors_queue);

    lock.unlock();
    // Call the deleter function for each pointer in the queue
    while (!local_queue.empty()) {
      auto dl_managed_tensor_ptr = local_queue.front();
      // Note: the deleter function can be nullptr (e.g. when the tensor is created from
      // __cuda_array_interface__ protocol)
      if (dl_managed_tensor_ptr && dl_managed_tensor_ptr->deleter != nullptr) {
        // Call the deleter function with GIL acquired
        py::gil_scoped_acquire scope_guard;
        dl_managed_tensor_ptr->deleter(dl_managed_tensor_ptr);
      }
      local_queue.pop();
    }
  }

  // Set the flag to indicate that the thread has stopped
  s_is_running = false;

  HOLOSCAN_LOG_DEBUG("LazyDLManagedTensorDeleter thread finished");
}

void LazyDLManagedTensorDeleter::release() {
  // Use std::memory_order_relaxed because there are no other memory operations that need to be
  // synchronized with the fetch_sub operation.
  if (s_instance_count.fetch_sub(1, std::memory_order_relaxed) == 1) {
    {
      std::lock_guard<std::mutex> lock(s_mutex);
      s_stop = true;
    }
    s_cv.notify_all();
    HOLOSCAN_LOG_DEBUG("Waiting for LazyDLManagedTensorDeleter thread to stop");
    // Wait until the thread has stopped
    while (true) {
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_is_running) { break; }
      }
      // Yield to other threads
      std::this_thread::yield();
    }
    HOLOSCAN_LOG_DEBUG("LazyDLManagedTensorDeleter thread stopped");
    {
      std::lock_guard<std::mutex> lock(s_mutex);
      s_stop = false;
    }
  }
}

void LazyDLManagedTensorDeleter::on_exit() {
  HOLOSCAN_LOG_DEBUG("LazyDLManagedTensorDeleter::on_exit() called");
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_stop = true;
  }
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::on_fork_prepare() {
  s_mutex.lock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = true;
  s_cv.notify_all();
}

void LazyDLManagedTensorDeleter::on_fork_parent() {
  s_mutex.unlock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = false;
}

void LazyDLManagedTensorDeleter::on_fork_child() {
  s_mutex.unlock();
  LazyDLManagedTensorDeleter::s_cv_do_not_wait_thread = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyTensor definition
////////////////////////////////////////////////////////////////////////////////////////////////////

PyTensor::PyTensor(std::shared_ptr<DLManagedTensorContext>& ctx) : Tensor(ctx) {}

PyTensor::PyTensor(DLManagedTensor* dl_managed_tensor_ptr) {
  dl_ctx_ = std::make_shared<DLManagedTensorContext>();
  // Create PyDLManagedMemoryBuffer to hold the DLManagedTensor and acquire GIL before calling
  // the deleter function
  dl_ctx_->memory_ref = std::make_shared<PyDLManagedMemoryBuffer>(dl_managed_tensor_ptr);

  auto& dl_managed_tensor = dl_ctx_->tensor;
  dl_managed_tensor = *dl_managed_tensor_ptr;
}

py::object PyTensor::as_tensor(const py::object& obj) {
  // This method could have been used as a constructor for the PyTensor class, but it was not
  // possible to get the py::object to be passed to the constructor. Instead, this method is used
  // to create a py::object from PyTensor object and set array interface on it.
  //
  //    // Note: this does not work, as the py::object is not passed to the constructor
  //    .def(py::init(&PyTensor::py_create), doc::Tensor::doc_Tensor);
  //
  //       include/pybind11/detail/init.h:86:19: error: static assertion failed: pybind11::init():
  //       init function must return a compatible pointer, holder, or value
  //       86 |     static_assert(!std::is_same<Class, Class>::value /* always false */,
  //
  //    // See https://github.com/pybind/pybind11/issues/2984 for more details
  std::shared_ptr<PyTensor> tensor;

  if (py::hasattr(obj, "__cuda_array_interface__")) {
    tensor = PyTensor::from_cuda_array_interface(obj);
  } else if (py::hasattr(obj, "__dlpack__") && py::hasattr(obj, "__dlpack_device__")) {
    tensor = PyTensor::from_dlpack(obj);
  } else if (py::hasattr(obj, "__array_interface__")) {
    tensor = PyTensor::from_array_interface(obj);
  } else {
    throw std::runtime_error("Unsupported Python object type");
  }
  py::object py_tensor = py::cast(tensor);

  // Set array interface attributes
  set_array_interface(py_tensor, tensor->dl_ctx());
  return py_tensor;
}

py::object PyTensor::from_dlpack_pyobj(const py::object& obj) {
  std::shared_ptr<PyTensor> tensor;
  if (py::hasattr(obj, "__dlpack__") && py::hasattr(obj, "__dlpack_device__")) {
    tensor = PyTensor::from_dlpack(obj);
  } else {
    throw std::runtime_error("Unsupported Python object type");
  }
  py::object py_tensor = py::cast(tensor);

  // Set array interface attributes
  set_array_interface(py_tensor, tensor->dl_ctx());
  return py_tensor;
}

std::shared_ptr<PyTensor> PyTensor::from_array_interface(const py::object& obj, bool cuda) {
  auto memory_buf = std::make_shared<ArrayInterfaceMemoryBuffer>();
  memory_buf->obj_ref = obj;  // hold obj to prevent it from being garbage collected

  const char* interface_name = cuda ? "__cuda_array_interface__" : "__array_interface__";
  auto array_interface = obj.attr(interface_name).cast<py::dict>();

  // Process mandatory entries
  memory_buf->dl_shape = array_interface["shape"].cast<std::vector<int64_t>>();
  auto& shape = memory_buf->dl_shape;
  auto typestr = array_interface["typestr"].cast<std::string>();
  if (!cuda) {
    if (!array_interface.contains("data")) {
      throw std::runtime_error(
          "Array interface data entry is missing (buffer interface) which is not supported ");
    }
    auto data_obj = array_interface["data"];
    if (data_obj.is_none()) {
      throw std::runtime_error(
          "Array interface data entry is None (buffer interface) which is not supported");
    }
    if (!py::isinstance<py::tuple>(data_obj)) {
      throw std::runtime_error(
          "Array interface data entry is not a tuple (buffer interface) which is not supported");
    }
  }
  auto data_array = array_interface["data"].cast<std::vector<int64_t>>();
  auto data_ptr = reinterpret_cast<void*>(data_array[0]);
  // bool data_readonly = data_array[1] > 0;
  // auto version = array_interface["version"].cast<int64_t>();

  auto maybe_dldatatype = nvidia::gxf::DLDataTypeFromTypeString(typestr);
  if (!maybe_dldatatype) {
    throw std::runtime_error("Unable to determine DLDataType from NumPy typestr");
  }
  auto maybe_device = nvidia::gxf::DLDeviceFromPointer(data_ptr);
  if (!maybe_device) { throw std::runtime_error("Unable to determine DLDevice from data pointer"); }
  DLTensor local_dl_tensor{
      .data = data_ptr,
      .device = maybe_device.value(),
      .ndim = static_cast<int32_t>(shape.size()),
      .dtype = maybe_dldatatype.value(),
      .shape = shape.data(),
      .strides = nullptr,
      .byte_offset = 0,
  };

  // Process 'optional' entries
  py::object strides_obj = py::none();
  if (array_interface.contains("strides")) { strides_obj = array_interface["strides"]; }
  auto& strides = memory_buf->dl_strides;
  if (strides_obj.is_none()) {
    calc_strides(local_dl_tensor, strides, true);
  } else {
    strides = strides_obj.cast<std::vector<int64_t>>();
    // The array interface's stride is using bytes, not element size, so we need to divide it by
    // the element size.
    int64_t elem_size = local_dl_tensor.dtype.bits / 8;
    for (auto& stride : strides) { stride /= elem_size; }
  }
  local_dl_tensor.strides = strides.data();

  // We do not process 'descr', 'mask', and 'offset' entries

  if (cuda) {
    // Process 'stream' entry
    py::object stream_obj = py::none();
    if (array_interface.contains("stream")) { stream_obj = array_interface["stream"]; }

    int64_t stream_id = 1;  // legacy default stream
    cudaStream_t stream_ptr = nullptr;
    if (stream_obj.is_none()) {
      stream_id = -1;
    } else {
      stream_id = stream_obj.cast<int64_t>();
    }
    if (stream_id < -1) {
      throw std::runtime_error(
          "Invalid stream, valid stream should be  None (no synchronization), 1 (legacy default "
          "stream), 2 "
          "(per-thread defaultstream), or a positive integer (stream pointer)");
    } else if (stream_id <= 2) {
      stream_ptr = nullptr;
    } else {
      stream_ptr = reinterpret_cast<cudaStream_t>(stream_id);
    }

    cudaStream_t curr_stream_ptr = nullptr;  // legacy stream

    if (stream_id >= 0 && curr_stream_ptr != stream_ptr) {
      cudaEvent_t curr_stream_event;
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaEventCreateWithFlags(&curr_stream_event, cudaEventDisableTiming),
          "Failure during call to cudaEventCreateWithFlags");
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(curr_stream_event, stream_ptr),
                                     "Failure during call to cudaEventRecord");
      // Make current stream (curr_stream_ptr) to wait until the given stream (stream_ptr)
      // is finished. This is a reverse of py_dlpack() method.
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamWaitEvent(curr_stream_ptr, curr_stream_event, 0),
                                     "Failure during call to cudaStreamWaitEvent");
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(curr_stream_event),
                                     "Failure during call to cudaEventDestroy");
    }
  }
  // Create DLManagedTensor object
  auto dl_managed_tensor_ctx = new DLManagedTensorContext;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = memory_buf;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorContext*>(self->manager_ctx);
    // Note: since 'memory_ref' is maintaining python object reference, we should acquire GIL in
    // case this function is called from another non-python thread, before releasing 'memory_ref'.
    py::gil_scoped_acquire scope_guard;
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = local_dl_tensor;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(&dl_managed_tensor);

  return tensor;
}

std::shared_ptr<PyTensor> PyTensor::from_dlpack(const py::object& obj) {
  // Pybind11 doesn't have a way to get/set a pointer with a name so we have to use the C API
  // for efficiency.
  // auto dlpack_capsule = py::reinterpret_borrow<py::capsule>(obj.attr("__dlpack__")());
  auto dlpack_device_func = obj.attr("__dlpack_device__");

  // We don't handle backward compatibility with older versions of DLPack
  if (dlpack_device_func.is_none()) { throw std::runtime_error("DLPack device is not set"); }

  auto dlpack_device = py::cast<py::tuple>(dlpack_device_func());
  // https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv48DLDevice
  DLDeviceType device_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
  int32_t device_id = dlpack_device[1].cast<int32_t>();

  DLDevice device = {device_type, device_id};

  auto dlpack_func = obj.attr("__dlpack__");
  py::capsule dlpack_capsule;

  // TOIMPROVE: need to get current stream pointer and call with the stream
  // https://github.com/dmlc/dlpack/issues/57 this thread was good to understand the differences
  // between __cuda_array_interface__ and __dlpack__ on life cycle/stream handling.
  // In DLPack, the client of the memory notify to the producer that the client will use the
  // client stream (`stream_ptr`) to consume the memory. It's the producer's responsibility to
  // make sure that the client stream wait for the producer stream to finish producing the memory.
  // The producer stream is the stream that the producer used to produce the memory. The producer
  // can then use this information to decide whether to use the same stream to produce the memory
  // or to use a different stream.
  // In __cuda_array_interface__, both producer and consumer are responsible for managing the
  // streams. The producer can use the `stream` field to specify the stream that the producer used
  // to produce the memory. The consumer can use the `stream` field to synchronize with the
  // producer stream. (please see
  // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization)
  switch (device_type) {
    case kDLCUDA:
    case kDLCUDAManaged: {
      py::int_ stream_ptr(1);  // legacy stream
      dlpack_capsule = py::reinterpret_borrow<py::capsule>(dlpack_func("stream"_a = stream_ptr));
      break;
    }
    case kDLCPU:
    case kDLCUDAHost: {
      dlpack_capsule = py::reinterpret_borrow<py::capsule>(dlpack_func());
      break;
    }
    default:
      throw std::runtime_error(fmt::format("Unsupported device type: {}", device_type));
  }

  // Note: we should keep the reference to the capsule object (`dlpack_obj`) while working with
  // PyObject* pointer. Otherwise, the capsule can be deleted and the pointers will be invalid.
  py::object dlpack_obj = dlpack_func();

  PyObject* dlpack_capsule_ptr = dlpack_obj.ptr();

  if (!PyCapsule_IsValid(dlpack_capsule_ptr, dlpack_capsule_name)) {
    const char* capsule_name = PyCapsule_GetName(dlpack_capsule_ptr);
    throw std::runtime_error(
        fmt::format("Received an invalid DLPack capsule ('{}'). You might have already consumed "
                    "the DLPack capsule.",
                    capsule_name));
  }

  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(PyCapsule_GetPointer(dlpack_capsule_ptr, dlpack_capsule_name));

  // Set device
  dl_managed_tensor->dl_tensor.device = device;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(dl_managed_tensor);

  // Set the capsule name to 'used_dltensor' so that it will not be consumed again.
  PyCapsule_SetName(dlpack_capsule_ptr, used_dlpack_capsule_name);

  // Steal the ownership of the capsule so that it will not be destroyed when the capsule object
  // goes out of scope.
  PyCapsule_SetDestructor(dlpack_capsule_ptr, nullptr);

  return tensor;
}

py::capsule PyTensor::dlpack(const py::object& obj, py::object stream) {
  auto tensor = py::cast<std::shared_ptr<Tensor>>(obj);
  if (!tensor) { throw std::runtime_error("Failed to cast to Tensor"); }
  // Do not copy 'obj' or a shared pointer here in the lambda expression's initializer, otherwise
  // the refcount of it will be increased by 1 and prevent the object from being destructed. Use a
  // raw pointer here instead.
  return py_dlpack(tensor.get(), std::move(stream));
}

py::tuple PyTensor::dlpack_device(const py::object& obj) {
  auto tensor = py::cast<std::shared_ptr<Tensor>>(obj);
  if (!tensor) { throw std::runtime_error("Failed to cast to Tensor"); }
  // Do not copy 'obj' or a shared pointer here in the lambda expression's initializer, otherwise
  // the refcount of it will be increased by 1 and prevent the object from being destructed. Use a
  // raw pointer here instead.
  return py_dlpack_device(tensor.get());
}

bool is_tensor_like(py::object value) {
  return ((py::hasattr(value, "__dlpack__") && py::hasattr(value, "__dlpack_device__")) ||
          py::isinstance<holoscan::PyTensor>(value) ||
          py::hasattr(value, "__cuda_array_interface__") ||
          py::hasattr(value, "__array_interface__"));
}

}  // namespace holoscan
