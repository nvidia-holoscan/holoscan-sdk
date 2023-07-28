/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/chrono.h>  // will include timedelta.h for us
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "./resources_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/resources/gxf/receiver.hpp"
#include "holoscan/core/resources/gxf/serialization_buffer.hpp"
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"
#include "holoscan/core/resources/gxf/transmitter.hpp"
#include "holoscan/core/resources/gxf/ucx_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_receiver.hpp"
#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"
#include "holoscan/core/resources/gxf/ucx_transmitter.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

int64_t get_duration_ns(const py::object& duration) {
  if (py::isinstance<py::int_>(duration)) {
    return py::cast<int64_t>(duration);
  } else {
    // Must acquire GIL before calling C API functions like PyDelta_Check
    py::gil_scoped_acquire scope_guard;

    // Must initialize PyDateTime_IMPORT here in order to be able to use PyDelta_Check below
    // see: https://docs.python.org/3/c-api/datetime.html?highlight=pydelta_check#datetime-objects
    if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

    if (PyDelta_Check(duration.ptr())) {
      // timedelta stores integer days, seconds, microseconds
      int64_t days, seconds, microseconds;
      days = PyDateTime_DELTA_GET_DAYS(duration.ptr());
      seconds = PyDateTime_DELTA_GET_SECONDS(duration.ptr());
      if (days) {
        int seconds_per_day = 24 * 3600;
        seconds += days * seconds_per_day;
      }
      microseconds = PyDateTime_DELTA_GET_MICROSECONDS(duration.ptr());
      if (seconds) { microseconds += 1000000 * seconds; }
      int64_t delta_ns = 1000 * microseconds;
      return delta_ns;
    } else {
      throw std::runtime_error("expected an integer or datetime.timedelta type");
    }
  }
}

// PyAllocator trampoline class: provides override for virtual function is_available

class PyAllocator : public Allocator {
 public:
  /* Inherit the constructors */
  using Allocator::Allocator;

  /* Trampolines (need one for each virtual function) */
  bool is_available(uint64_t size) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(bool, Allocator, is_available, size);
  }
};

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the resource.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the resource's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_resource<ResourceT>
 */

class PyBlockMemoryPool : public BlockMemoryPool {
 public:
  /* Inherit the constructors */
  using BlockMemoryPool::BlockMemoryPool;

  // Define a constructor that fully initializes the object.
  PyBlockMemoryPool(Fragment* fragment, int32_t storage_type, uint64_t block_size,
                    uint64_t num_blocks, const std::string& name = "block_memory_pool")
      : BlockMemoryPool(ArgList{Arg{"storage_type", storage_type},
                                Arg{"block_size", block_size},
                                Arg{"num_blocks", num_blocks}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyCudaStreamPool : public CudaStreamPool {
 public:
  /* Inherit the constructors */
  using CudaStreamPool::CudaStreamPool;

  // Define a constructor that fully initializes the object.
  PyCudaStreamPool(Fragment* fragment, int32_t dev_id, uint32_t stream_flags,
                   int32_t stream_priority, uint32_t reserved_size, uint32_t max_size,
                   const std::string& name = "cuda_stream_pool")
      : CudaStreamPool(ArgList{
            Arg{"dev_id", dev_id},
            Arg{"stream_flags", stream_flags},
            Arg{"stream_priority", stream_priority},
            Arg{"reserved_size", reserved_size},
            Arg{"max_size", max_size},
        }) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUnboundedAllocator : public UnboundedAllocator {
 public:
  /* Inherit the constructors */
  using UnboundedAllocator::UnboundedAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyUnboundedAllocator(Fragment* fragment, const std::string& name = "cuda_stream_pool")
      : UnboundedAllocator() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PySerializationBuffer : public SerializationBuffer {
 public:
  /* Inherit the constructors */
  using SerializationBuffer::SerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PySerializationBuffer(Fragment* fragment,
                                 std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                 size_t buffer_size = kDefaultSerializationBufferSize,
                                 const std::string& name = "serialization_buffer")
      : SerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxSerializationBuffer : public UcxSerializationBuffer {
 public:
  /* Inherit the constructors */
  using UcxSerializationBuffer::UcxSerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxSerializationBuffer(Fragment* fragment,
                                    std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                    size_t buffer_size = kDefaultSerializationBufferSize,
                                    const std::string& name = "serialization_buffer")
      : UcxSerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxComponentSerializer : public UcxComponentSerializer {
 public:
  /* Inherit the constructors */
  using UcxComponentSerializer::UcxComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxComponentSerializer(Fragment* fragment,
                                    std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                    const std::string& name = "ucx_component_serializer")
      : UcxComponentSerializer() {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxHoloscanComponentSerializer : public UcxHoloscanComponentSerializer {
 public:
  /* Inherit the constructors */
  using UcxHoloscanComponentSerializer::UcxHoloscanComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxHoloscanComponentSerializer(
      Fragment* fragment, std::shared_ptr<holoscan::Allocator> allocator = nullptr,
      const std::string& name = "ucx_holoscan_component_serializer")
      : UcxHoloscanComponentSerializer() {
    if (allocator) { this->add_arg(Arg{"allocator", allocator}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxEntitySerializer : public UcxEntitySerializer {
 public:
  /* Inherit the constructors */
  using UcxEntitySerializer::UcxEntitySerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxEntitySerializer(
      Fragment* fragment,
      // std::vector<std::shared_ptr<holoscan::Resource>> component_serializers = {},
      bool verbose_warning = false, const std::string& name = "ucx_entity_buffer")
      : UcxEntitySerializer(ArgList{
            Arg{"verbose_warning", verbose_warning},
        }) {
    // if (component_serializers.size() == 0) { this->add_arg(Arg{"component_serializers",
    // component_serializers}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyDoubleBufferReceiver : public DoubleBufferReceiver {
 public:
  /* Inherit the constructors */
  using DoubleBufferReceiver::DoubleBufferReceiver;

  // Define a constructor that fully initializes the object.
  PyDoubleBufferReceiver(Fragment* fragment, uint64_t capacity = 1UL, uint64_t policy = 2UL,
                         const std::string& name = "double_buffer_receiver")
      : DoubleBufferReceiver(ArgList{Arg{"capacity", capacity}, Arg{"policy", policy}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyDoubleBufferTransmitter : public DoubleBufferTransmitter {
 public:
  /* Inherit the constructors */
  using DoubleBufferTransmitter::DoubleBufferTransmitter;

  // Define a constructor that fully initializes the object.
  PyDoubleBufferTransmitter(Fragment* fragment, uint64_t capacity = 1UL, uint64_t policy = 2UL,
                            const std::string& name = "double_buffer_transmitter")
      : DoubleBufferTransmitter(ArgList{Arg{"capacity", capacity}, Arg{"policy", policy}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxReceiver : public UcxReceiver {
 public:
  /* Inherit the constructors */
  using UcxReceiver::UcxReceiver;

  // Define a constructor that fully initializes the object.
  PyUcxReceiver(Fragment* fragment, std::shared_ptr<UcxSerializationBuffer> buffer = nullptr,
                uint64_t capacity = 1UL, uint64_t policy = 2UL,
                const std::string& address = std::string("0.0.0.0"), int32_t port = kDefaultUcxPort,
                const std::string& name = "ucx_receiver")
      : UcxReceiver(ArgList{Arg{"capacity", capacity},
                            Arg{"policy", policy},
                            Arg{"address", address},
                            Arg{"port", port}}) {
    if (buffer) { this->add_arg(Arg{"buffer", buffer}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyUcxTransmitter : public UcxTransmitter {
 public:
  /* Inherit the constructors */
  using UcxTransmitter::UcxTransmitter;

  // Define a constructor that fully initializes the object.
  PyUcxTransmitter(Fragment* fragment, std::shared_ptr<UcxSerializationBuffer> buffer = nullptr,
                   uint64_t capacity = 1UL, uint64_t policy = 2UL,
                   const std::string& receiver_address = std::string("0.0.0.0"),
                   int32_t port = kDefaultUcxPort, int32_t maximum_connection_retries = 10,
                   const std::string& name = "ucx_transmitter")
      : UcxTransmitter(ArgList{Arg{"capacity", capacity},
                               Arg{"policy", policy},
                               Arg{"receiver_address", receiver_address},
                               Arg{"port", port},
                               Arg{"maximum_connection_retries", maximum_connection_retries}}) {
    if (buffer) { this->add_arg(Arg{"buffer", buffer}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyStdComponentSerializer : public StdComponentSerializer {
 public:
  /* Inherit the constructors */
  using StdComponentSerializer::StdComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyStdComponentSerializer(Fragment* fragment,
                                    const std::string& name = "std_component_serializer")
      : StdComponentSerializer() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyVideoStreamSerializer : public VideoStreamSerializer {
 public:
  /* Inherit the constructors */
  using VideoStreamSerializer::VideoStreamSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyVideoStreamSerializer(Fragment* fragment,
                                   const std::string& name = "video_stream_serializer")
      : VideoStreamSerializer() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyRealtimeClock : public RealtimeClock {
 public:
  /* Inherit the constructors */
  using RealtimeClock::RealtimeClock;

  // Define a constructor that fully initializes the object.
  explicit PyRealtimeClock(Fragment* fragment, double initial_time_offset = 0.0,
                           double initial_time_scale = 1.0, bool use_time_since_epoch = false,
                           const std::string& name = "realtime_clock")
      : RealtimeClock(ArgList{Arg{"initial_time_offset", initial_time_offset},
                              Arg{"initial_time_scale", initial_time_scale},
                              Arg{"use_time_since_epoch", use_time_since_epoch}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }

  /* Trampolines (need one for each virtual function) */
  double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, RealtimeClock, time);
  }
  int64_t timestamp() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(int64_t, RealtimeClock, timestamp);
  }
  void sleep_for(int64_t duration_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, RealtimeClock, sleep_for, duration_ns);
  }
  void sleep_until(int64_t target_time_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, RealtimeClock, sleep_until, target_time_ns);
  }
};

class PyManualClock : public ManualClock {
 public:
  /* Inherit the constructors */
  using ManualClock::ManualClock;

  // Define a constructor that fully initializes the object.
  explicit PyManualClock(Fragment* fragment, int64_t initial_timestamp = 0LL,
                         const std::string& name = "manual_clock")
      : ManualClock(Arg{"initial_timestamp", initial_timestamp}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }

  /* Trampolines (need one for each virtual function) */
  double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, ManualClock, time);
  }
  int64_t timestamp() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(int64_t, ManualClock, timestamp);
  }
  void sleep_for(int64_t duration_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, ManualClock, sleep_for, duration_ns);
  }
  void sleep_until(int64_t target_time_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, ManualClock, sleep_until, target_time_ns);
  }
};

PYBIND11_MODULE(_resources, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _resources
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<MemoryStorageType>(m, "MemoryStorageType")
      .value("HOST", MemoryStorageType::kHost)
      .value("DEVICE", MemoryStorageType::kDevice)
      .value("SYSTEM", MemoryStorageType::kSystem);

  py::class_<Allocator, PyAllocator, gxf::GXFResource, std::shared_ptr<Allocator>>(
      m, "Allocator", doc::Allocator::doc_Allocator)
      .def(py::init<>(), doc::Allocator::doc_Allocator)
      .def_property_readonly(
          "gxf_typename", &Allocator::gxf_typename, doc::Allocator::doc_gxf_typename)
      .def("is_available", &Allocator::is_available, "size"_a, doc::Allocator::doc_is_available)
      .def("allocate", &Allocator::allocate, "size"_a, "type"_a, doc::Allocator::doc_allocate)
      .def("free", &Allocator::free, "pointer"_a, doc::Allocator::doc_free);
  // TODO(grelee): for allocate / free how does std::byte* get cast to/from Python?

  py::class_<BlockMemoryPool, PyBlockMemoryPool, Allocator, std::shared_ptr<BlockMemoryPool>>(
      m, "BlockMemoryPool", doc::BlockMemoryPool::doc_BlockMemoryPool)
      .def(py::init<Fragment*, int32_t, uint64_t, uint64_t, const std::string&>(),
           "fragment"_a,
           "storage_type"_a,
           "block_size"_a,
           "num_blocks"_a,
           "name"_a = "block_memory_pool",
           doc::BlockMemoryPool::doc_BlockMemoryPool_python)
      .def_property_readonly(
          "gxf_typename", &BlockMemoryPool::gxf_typename, doc::BlockMemoryPool::doc_gxf_typename)
      .def("setup", &BlockMemoryPool::setup, "spec"_a, doc::BlockMemoryPool::doc_setup);

  py::class_<CudaStreamPool, PyCudaStreamPool, Allocator, std::shared_ptr<CudaStreamPool>>(
      m, "CudaStreamPool", doc::CudaStreamPool::doc_CudaStreamPool)
      .def(
          py::init<Fragment*, int32_t, uint32_t, int32_t, uint32_t, uint32_t, const std::string&>(),
          "fragment"_a,
          "dev_id"_a,
          "stream_flags"_a,
          "stream_priority"_a,
          "reserved_size"_a,
          "max_size"_a,
          "name"_a = "cuda_stream_pool"s,
          doc::CudaStreamPool::doc_CudaStreamPool_python)
      .def_property_readonly(
          "gxf_typename", &CudaStreamPool::gxf_typename, doc::CudaStreamPool::doc_gxf_typename)
      .def("setup", &CudaStreamPool::setup, "spec"_a, doc::CudaStreamPool::doc_setup);

  py::class_<SerializationBuffer,
             PySerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<SerializationBuffer>>(
      m, "SerializationBuffer", doc::SerializationBuffer::doc_SerializationBuffer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, size_t, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = "serialization_buffer"s,
           doc::SerializationBuffer::doc_SerializationBuffer_python)
      .def_property_readonly("gxf_typename",
                             &SerializationBuffer::gxf_typename,
                             doc::SerializationBuffer::doc_gxf_typename)
      .def("setup", &SerializationBuffer::setup, "spec"_a, doc::SerializationBuffer::doc_setup);

  py::class_<UcxSerializationBuffer,
             PyUcxSerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<UcxSerializationBuffer>>(
      m, "UcxSerializationBuffer", doc::UcxSerializationBuffer::doc_UcxSerializationBuffer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, size_t, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = "serialization_buffer"s,
           doc::UcxSerializationBuffer::doc_UcxSerializationBuffer_python)
      .def_property_readonly("gxf_typename",
                             &UcxSerializationBuffer::gxf_typename,
                             doc::UcxSerializationBuffer::doc_gxf_typename)
      .def("setup",
           &UcxSerializationBuffer::setup,
           "spec"_a,
           doc::UcxSerializationBuffer::doc_setup);

  py::class_<UcxComponentSerializer,
             PyUcxComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxComponentSerializer>>(
      m, "UcxComponentSerializer", doc::UcxComponentSerializer::doc_UcxComponentSerializer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "name"_a = "ucx_component_serializer"s,
           doc::UcxComponentSerializer::doc_UcxComponentSerializer_python)
      .def_property_readonly("gxf_typename",
                             &UcxComponentSerializer::gxf_typename,
                             doc::UcxComponentSerializer::doc_gxf_typename)
      .def("setup",
           &UcxComponentSerializer::setup,
           "spec"_a,
           doc::UcxComponentSerializer::doc_setup);

  py::class_<UcxHoloscanComponentSerializer,
             PyUcxHoloscanComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxHoloscanComponentSerializer>>(
      m,
      "UcxHoloscanComponentSerializer",
      doc::UcxHoloscanComponentSerializer::doc_UcxHoloscanComponentSerializer)
      .def(py::init<Fragment*, std::shared_ptr<holoscan::Allocator>, const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "name"_a = "ucx_component_serializer"s,
           doc::UcxHoloscanComponentSerializer::doc_UcxHoloscanComponentSerializer_python)
      .def_property_readonly("gxf_typename",
                             &UcxHoloscanComponentSerializer::gxf_typename,
                             doc::UcxHoloscanComponentSerializer::doc_gxf_typename)
      .def("setup",
           &UcxHoloscanComponentSerializer::setup,
           "spec"_a,
           doc::UcxHoloscanComponentSerializer::doc_setup);

  py::class_<UcxEntitySerializer,
             PyUcxEntitySerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxEntitySerializer>>(
      m, "UcxEntitySerializer", doc::UcxEntitySerializer::doc_UcxEntitySerializer)
      .def(py::init<Fragment*,
                    // std::vector<std::shared_ptr<holoscan::Resource>>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           // "component_serializers"_a = std::vector<std::shared_ptr<holoscan::Resource>>{},
           "verbose_warning"_a = false,
           "name"_a = "ucx_entity_serializer"s,
           doc::UcxEntitySerializer::doc_UcxEntitySerializer_python)
      .def_property_readonly("gxf_typename",
                             &UcxEntitySerializer::gxf_typename,
                             doc::UcxEntitySerializer::doc_gxf_typename)
      .def("setup", &UcxEntitySerializer::setup, "spec"_a, doc::UcxEntitySerializer::doc_setup);

  py::class_<UnboundedAllocator,
             PyUnboundedAllocator,
             Allocator,
             std::shared_ptr<UnboundedAllocator>>(
      m, "UnboundedAllocator", doc::UnboundedAllocator::doc_UnboundedAllocator)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "unbounded_allocator"s,
           doc::UnboundedAllocator::doc_UnboundedAllocator_python)
      .def_property_readonly("gxf_typename",
                             &UnboundedAllocator::gxf_typename,
                             doc::UnboundedAllocator::doc_gxf_typename)
      .def("setup", &UnboundedAllocator::setup, "spec"_a, doc::UnboundedAllocator::doc_setup);

  py::class_<Receiver, gxf::GXFResource, std::shared_ptr<Receiver>>(
      m, "Receiver", doc::Receiver::doc_Receiver)
      .def(py::init<>(), doc::Receiver::doc_Receiver)
      .def_property_readonly(
          "gxf_typename", &Receiver::gxf_typename, doc::Receiver::doc_gxf_typename);

  py::class_<DoubleBufferReceiver,
             PyDoubleBufferReceiver,
             Receiver,
             std::shared_ptr<DoubleBufferReceiver>>(
      m, "DoubleBufferReceiver", doc::DoubleBufferReceiver::doc_DoubleBufferReceiver)
      .def(py::init<Fragment*, uint64_t, uint64_t, const std::string&>(),
           "fragment"_a,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "name"_a = "double_buffer_receiver"s,
           doc::DoubleBufferReceiver::doc_DoubleBufferReceiver_python)
      .def_property_readonly("gxf_typename",
                             &DoubleBufferReceiver::gxf_typename,
                             doc::DoubleBufferReceiver::doc_gxf_typename)
      .def("setup", &DoubleBufferReceiver::setup, "spec"_a, doc::DoubleBufferReceiver::doc_setup);

  py::class_<UcxReceiver, PyUcxReceiver, Receiver, std::shared_ptr<UcxReceiver>>(
      m, "UcxReceiver", doc::UcxReceiver::doc_UcxReceiver)
      .def(py::init<Fragment*,
                    std::shared_ptr<UcxSerializationBuffer>,
                    uint64_t,
                    uint64_t,
                    const std::string&,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "buffer"_a = nullptr,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "address"_a = std::string("0.0.0.0"),
           "port"_a = kDefaultUcxPort,
           "name"_a = "ucx_receiver"s,
           doc::UcxReceiver::doc_UcxReceiver_python)
      .def_property_readonly(
          "gxf_typename", &UcxReceiver::gxf_typename, doc::UcxReceiver::doc_gxf_typename)
      .def("setup", &UcxReceiver::setup, "spec"_a, doc::UcxReceiver::doc_setup);

  py::class_<Transmitter, gxf::GXFResource, std::shared_ptr<Transmitter>>(
      m, "Transmitter", doc::Transmitter::doc_Transmitter)
      .def(py::init<>(), doc::Transmitter::doc_Transmitter)
      .def_property_readonly(
          "gxf_typename", &Transmitter::gxf_typename, doc::Transmitter::doc_gxf_typename);

  py::class_<DoubleBufferTransmitter,
             PyDoubleBufferTransmitter,
             Transmitter,
             std::shared_ptr<DoubleBufferTransmitter>>(
      m, "DoubleBufferTransmitter", doc::DoubleBufferTransmitter::doc_DoubleBufferTransmitter)
      .def(py::init<Fragment*, uint64_t, uint64_t, const std::string&>(),
           "fragment"_a,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "name"_a = "double_buffer_transmitter"s,
           doc::DoubleBufferTransmitter::doc_DoubleBufferTransmitter_python)
      .def_property_readonly("gxf_typename",
                             &DoubleBufferTransmitter::gxf_typename,
                             doc::DoubleBufferTransmitter::doc_gxf_typename)
      .def("setup",
           &DoubleBufferTransmitter::setup,
           "spec"_a,
           doc::DoubleBufferTransmitter::doc_setup);

  py::class_<UcxTransmitter, PyUcxTransmitter, Transmitter, std::shared_ptr<UcxTransmitter>>(
      m, "UcxTransmitter", doc::UcxTransmitter::doc_UcxTransmitter)
      .def(py::init<Fragment*,
                    std::shared_ptr<UcxSerializationBuffer>,
                    uint64_t,
                    uint64_t,
                    const std::string&,
                    int32_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "buffer"_a = nullptr,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "receiver_address"_a = std::string("0.0.0.0"),
           "port"_a = kDefaultUcxPort,
           "maximum_connection_retries"_a = 10,
           "name"_a = "ucx_transmitter"s,
           doc::UcxTransmitter::doc_UcxTransmitter_python)
      .def_property_readonly(
          "gxf_typename", &UcxTransmitter::gxf_typename, doc::UcxTransmitter::doc_gxf_typename)
      .def("setup", &UcxTransmitter::setup, "spec"_a, doc::UcxTransmitter::doc_setup);

  py::class_<StdComponentSerializer,
             PyStdComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<StdComponentSerializer>>(
      m, "StdComponentSerializer", doc::StdComponentSerializer::doc_StdComponentSerializer)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "standard_component_serializer"s,
           doc::StdComponentSerializer::doc_StdComponentSerializer_python)
      .def_property_readonly("gxf_typename",
                             &StdComponentSerializer::gxf_typename,
                             doc::StdComponentSerializer::doc_gxf_typename)
      .def(
          "setup", &StdComponentSerializer::setup, "spec"_a, doc::StdComponentSerializer::doc_setup)
      .def("initialize",
           &StdComponentSerializer::initialize,
           doc::StdComponentSerializer::doc_initialize);

  py::class_<VideoStreamSerializer,
             PyVideoStreamSerializer,
             gxf::GXFResource,
             std::shared_ptr<VideoStreamSerializer>>(
      m, "VideoStreamSerializer", doc::VideoStreamSerializer::doc_VideoStreamSerializer)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "video_stream_serializer"s,
           doc::VideoStreamSerializer::doc_VideoStreamSerializer_python)
      .def_property_readonly("gxf_typename",
                             &VideoStreamSerializer::gxf_typename,
                             doc::VideoStreamSerializer::doc_gxf_typename)
      .def("setup", &VideoStreamSerializer::setup, "spec"_a, doc::VideoStreamSerializer::doc_setup)
      .def("initialize",
           &VideoStreamSerializer::initialize,
           doc::VideoStreamSerializer::doc_initialize);

  py::class_<Clock, gxf::GXFResource, std::shared_ptr<Clock>>(m, "Clock", doc::Clock::doc_Clock);

  py::class_<RealtimeClock, PyRealtimeClock, Clock, std::shared_ptr<RealtimeClock>>(
      m, "RealtimeClock", doc::RealtimeClock::doc_RealtimeClock)
      .def(py::init<Fragment*, double, double, bool, const std::string&>(),
           "fragment"_a,
           "initial_time_offset"_a = 0.0,
           "initial_time_scale"_a = 1.0,
           "use_time_since_epoch"_a = false,
           "name"_a = "realtime_clock"s,
           doc::RealtimeClock::doc_RealtimeClock_python)
      .def_property_readonly(
          "gxf_typename", &RealtimeClock::gxf_typename, doc::RealtimeClock::doc_gxf_typename)
      .def("setup", &RealtimeClock::setup, "spec"_a, doc::RealtimeClock::doc_setup)
      .def("time", &RealtimeClock::time, doc::Clock::doc_time)
      .def("timestamp", &RealtimeClock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](RealtimeClock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &RealtimeClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until)
      .def("set_time_scale",
           &RealtimeClock::set_time_scale,
           "time_scale"_a,
           doc::RealtimeClock::doc_set_time_scale);

  py::class_<ManualClock, PyManualClock, Clock, std::shared_ptr<ManualClock>>(
      m, "ManualClock", doc::ManualClock::doc_ManualClock)
      .def(py::init<Fragment*, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_timestamp"_a = 0LL,
           "name"_a = "realtime_clock"s,
           doc::ManualClock::doc_ManualClock_python)
      .def_property_readonly(
          "gxf_typename", &ManualClock::gxf_typename, doc::ManualClock::doc_gxf_typename)
      .def("setup", &ManualClock::setup, "spec"_a, doc::ManualClock::doc_setup)
      .def("time", &ManualClock::time, doc::Clock::doc_time)
      .def("timestamp", &ManualClock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](ManualClock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &ManualClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until);
}  // PYBIND11_MODULE
}  // namespace holoscan
