/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "./allocators_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_green_context.hpp"
#include "holoscan/core/resources/gxf/cuda_green_context_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/rmm_allocator.hpp"
#include "holoscan/core/resources/gxf/stream_ordered_allocator.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

namespace {
// constants copied from rmm_allocator.cpp
// kPoolInitialSize, kPoolMaxSize copied from rmm_allocator.cpp
#ifdef __aarch64__
constexpr const char* kPoolInitialSize = "8MB";  // 8 MB initial pool size
constexpr const char* kPoolMaxSize = "16MB";
#else
constexpr const char* kPoolInitialSize = "16MB";  // 16 MB initial pool size
constexpr const char* kPoolMaxSize = "32MB";
#endif
constexpr const char* kReleaseThreshold = "4MB";  // 4MB release threshold

// Default empty vector for sms_per_partition
static const std::vector<uint32_t> kDefaultSmsPerPartition = {};

}  // namespace

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
    setup(*spec_);
  }
};

class PyUnboundedAllocator : public UnboundedAllocator {
 public:
  /* Inherit the constructors */
  using UnboundedAllocator::UnboundedAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyUnboundedAllocator(Fragment* fragment, const std::string& name = "cuda_stream_pool") {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyCudaGreenContextPool : public CudaGreenContextPool {
 public:
  /* Inherit the constructors */
  using CudaGreenContextPool::CudaGreenContextPool;

  // Define a constructor that fully initializes the object.
  explicit PyCudaGreenContextPool(Fragment* fragment, int32_t dev_id = 0, uint32_t flags = 0,
                                  uint32_t num_partitions = 0,
                                  const std::vector<uint32_t>& sms_per_partition = {},
                                  int32_t default_context_index = -1,
                                  uint32_t min_sm_size = 2,
                                  const std::string& name = "cuda_green_context_pool")
      : CudaGreenContextPool(ArgList{
            Arg{"dev_id", dev_id},
            Arg{"flags", flags},
            Arg{"num_partitions", num_partitions},
            Arg{"sms_per_partition", sms_per_partition},
            Arg{"default_context_index", default_context_index},
            Arg{"min_sm_size", min_sm_size},
        }) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyCudaGreenContext : public CudaGreenContext {
 public:
  /* Inherit the constructors */
  using CudaGreenContext::CudaGreenContext;

  // Define a constructor that fully initializes the object.
  explicit PyCudaGreenContext(
      Fragment* fragment, std::shared_ptr<CudaGreenContextPool> cuda_green_context_pool = nullptr,
      int32_t index = -1, const std::string& name = "cuda_green_context")
      : CudaGreenContext(cuda_green_context_pool, index) {
    name_ = name;
    fragment_ = fragment;
    if (cuda_green_context_pool) {
      cuda_green_context_pool->initialize();
    }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyCudaStreamPool : public CudaStreamPool {
 public:
  /* Inherit the constructors */
  using CudaStreamPool::CudaStreamPool;

  // Define a constructor that fully initializes the object.
  explicit PyCudaStreamPool(Fragment* fragment, int32_t dev_id = 0, uint32_t stream_flags = 0,
                            int32_t stream_priority = 0, uint32_t reserved_size = 1,
                            uint32_t max_size = 0,
                            std::shared_ptr<CudaGreenContext> cuda_green_context = nullptr,
                            const std::string& name = "cuda_stream_pool")
      : CudaStreamPool(dev_id, stream_flags, stream_priority, reserved_size, max_size,
                       cuda_green_context) {
    name_ = name;
    fragment_ = fragment;
    if (cuda_green_context) {
      cuda_green_context->initialize();
    }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyRMMAllocator : public RMMAllocator {
 public:
  /* Inherit the constructors */
  using RMMAllocator::RMMAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyRMMAllocator(
      Fragment* fragment,
      const std::string& device_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& device_memory_max_size = std::string(kPoolMaxSize),
      const std::string& host_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& host_memory_max_size = std::string(kPoolMaxSize), int32_t dev_id = 0,
      const std::string& name = "rmm_pool")
      : RMMAllocator(ArgList{Arg{"device_memory_initial_size", device_memory_initial_size},
                             Arg{"device_memory_max_size", device_memory_max_size},
                             Arg{"host_memory_initial_size", host_memory_initial_size},
                             Arg{"host_memory_max_size", host_memory_max_size},
                             Arg{"dev_id", dev_id}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyStreamOrderedAllocator : public StreamOrderedAllocator {
 public:
  /* Inherit the constructors */
  using StreamOrderedAllocator::StreamOrderedAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyStreamOrderedAllocator(
      Fragment* fragment,
      const std::string& device_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& device_memory_max_size = std::string(kPoolMaxSize),
      const std::string& release_threshold = std::string(kReleaseThreshold), int32_t dev_id = 0,
      const std::string& name = "stream_ordered_allocator")
      : StreamOrderedAllocator(
            ArgList{Arg{"device_memory_initial_size", device_memory_initial_size},
                    Arg{"device_memory_max_size", device_memory_max_size},
                    Arg{"release_threshold", release_threshold},
                    Arg{"dev_id", dev_id}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_allocators(py::module_& m) {
  py::enum_<MemoryStorageType>(m, "MemoryStorageType")
      .value("HOST", MemoryStorageType::kHost)
      .value("DEVICE", MemoryStorageType::kDevice)
      .value("CUDA_MANAGED", MemoryStorageType::kCudaManaged)
      .value("SYSTEM", MemoryStorageType::kSystem);

  py::class_<Allocator, gxf::GXFResource, std::shared_ptr<Allocator>>(
      m, "Allocator", doc::Allocator::doc_Allocator)
      .def(py::init<>(), doc::Allocator::doc_Allocator)
      .def("is_available", &Allocator::is_available, "size"_a, doc::Allocator::doc_is_available)
      .def("allocate", &Allocator::allocate, "size"_a, "type"_a, doc::Allocator::doc_allocate)
      .def("free", &Allocator::free, "pointer"_a, doc::Allocator::doc_free)
      .def_property_readonly("block_size", &Allocator::block_size, doc::Allocator::doc_block_size);
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
           doc::BlockMemoryPool::doc_BlockMemoryPool);

  py::class_<CudaGreenContextPool,
             PyCudaGreenContextPool,
             gxf::GXFResource,
             std::shared_ptr<CudaGreenContextPool>>(
      m, "CudaGreenContextPool", doc::CudaGreenContextPool::doc_CudaGreenContextPool)
      .def(py::init<Fragment*,
                    int32_t,
                    uint32_t,
                    uint32_t,
                    std::vector<uint32_t>,
                    int32_t,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "dev_id"_a = 0,
           "flags"_a = 0U,
           "num_partitions"_a = 0U,
           "sms_per_partition"_a = py::cast(std::vector<uint32_t>{}),
           "default_context_index"_a = -1,
           "min_sm_size"_a = 2U,
           "name"_a = "cuda_green_context_pool",
           doc::CudaGreenContextPool::doc_CudaGreenContextPool);

  py::class_<CudaGreenContext,
             PyCudaGreenContext,
             gxf::GXFResource,
             std::shared_ptr<CudaGreenContext>>(
      m, "CudaGreenContext", doc::CudaGreenContext::doc_CudaGreenContext)
      .def(py::init<Fragment*,
                    std::shared_ptr<CudaGreenContextPool>,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "cuda_green_context_pool"_a = nullptr,
           "index"_a = -1,
           "name"_a = "cuda_green_context",
           doc::CudaGreenContext::doc_CudaGreenContext);

  py::class_<CudaStreamPool, PyCudaStreamPool, Allocator, std::shared_ptr<CudaStreamPool>>(
      m, "CudaStreamPool", doc::CudaStreamPool::doc_CudaStreamPool)
      .def(py::init<Fragment*,
                    int32_t,
                    uint32_t,
                    int32_t,
                    uint32_t,
                    uint32_t,
                    std::shared_ptr<CudaGreenContext>,
                    const std::string&>(),
           "fragment"_a,
           "dev_id"_a = 0,
           "stream_flags"_a = 0U,
           "stream_priority"_a = 0,
           "reserved_size"_a = 1U,
           "max_size"_a = 0U,
           "cuda_green_context"_a = nullptr,
           "name"_a = "cuda_stream_pool"s,
           doc::CudaStreamPool::doc_CudaStreamPool);

  py::class_<UnboundedAllocator,
             PyUnboundedAllocator,
             Allocator,
             std::shared_ptr<UnboundedAllocator>>(
      m, "UnboundedAllocator", doc::UnboundedAllocator::doc_UnboundedAllocator)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "unbounded_allocator"s,
           doc::UnboundedAllocator::doc_UnboundedAllocator);

  py::class_<CudaAllocator, Allocator, std::shared_ptr<CudaAllocator>>(
      m, "CudaAllocator", doc::CudaAllocator::doc_CudaAllocator)
      .def(py::init<>(), doc::CudaAllocator::doc_CudaAllocator)
      // Haven't wrapped cudaStream_t yet from Python
      // .def("allocate_async",
      //      &CudaAllocator::allocate_async,
      //      "size"_a,
      //      "stream"_a,
      //      doc::CudaAllocator::doc_allocate_async)
      // .def("free_async",
      //      &CudaAllocator::free_async,
      //      "pointer"_a,
      //      "stream"_a,
      //      doc::CudaAllocator::doc_free_async)
      .def_property_readonly(
          "pool_size", &CudaAllocator::pool_size, doc::CudaAllocator::doc_pool_size);

  py::class_<RMMAllocator, PyRMMAllocator, CudaAllocator, std::shared_ptr<RMMAllocator>>(
      m, "RMMAllocator", doc::RMMAllocator::doc_RMMAllocator)
      .def(py::init<Fragment*,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "device_memory_initial_size"_a = std::string(kPoolInitialSize),
           "device_memory_max_size"_a = std::string(kPoolMaxSize),
           "host_memory_initial_size"_a = std::string(kPoolInitialSize),
           "host_memory_max_size"_a = std::string(kPoolMaxSize),
           "dev_id"_a = 0,
           "name"_a = "rmm_pool",
           doc::RMMAllocator::doc_RMMAllocator);

  py::class_<StreamOrderedAllocator,
             PyStreamOrderedAllocator,
             CudaAllocator,
             std::shared_ptr<StreamOrderedAllocator>>(
      m, "StreamOrderedAllocator", doc::StreamOrderedAllocator::doc_StreamOrderedAllocator)
      .def(py::init<Fragment*,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "device_memory_initial_size"_a = std::string(kPoolInitialSize),
           "device_memory_max_size"_a = std::string(kPoolMaxSize),
           "release_threshold"_a = std::string(kReleaseThreshold),
           "dev_id"_a = 0,
           "name"_a = "stream_ordered_allocator",
           doc::StreamOrderedAllocator::doc_StreamOrderedAllocator);
}
}  // namespace holoscan
