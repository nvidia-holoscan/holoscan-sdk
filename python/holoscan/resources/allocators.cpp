/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "./allocators_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

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
                    uint64_t num_blocks, int32_t dev_id = 0,
                    const std::string& name = "block_memory_pool")
      : BlockMemoryPool(ArgList{Arg{"storage_type", storage_type},
                                Arg{"block_size", block_size},
                                Arg{"num_blocks", num_blocks},
                                Arg{"dev_id", dev_id}}) {
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

void init_allocators(py::module_& m) {
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
      .def(py::init<Fragment*, int32_t, uint64_t, uint64_t, int32_t, const std::string&>(),
           "fragment"_a,
           "storage_type"_a,
           "block_size"_a,
           "num_blocks"_a,
           "dev_id"_a = 0,
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
}
}  // namespace holoscan
