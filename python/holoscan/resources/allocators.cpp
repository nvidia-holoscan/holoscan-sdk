/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>  // CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/block_memory_pool.hpp>
#include <holoscan/core/resources/gxf/cuda_allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_green_context.hpp>
#include <holoscan/core/resources/gxf/cuda_green_context_pool.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/rmm_allocator.hpp>
#include <holoscan/core/resources/gxf/stream_ordered_allocator.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./allocators_pydoc.hpp"

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
  PyBlockMemoryPool(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                    int32_t storage_type, uint64_t block_size, uint64_t num_blocks,
                    int32_t dev_id = 0,
                    const std::string& name = resource_default_name_v<BlockMemoryPool>)
      : BlockMemoryPool(ArgList{Arg{"storage_type", storage_type},
                                Arg{"block_size", block_size},
                                Arg{"num_blocks", num_blocks},
                                Arg{"dev_id", dev_id}}) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyUnboundedAllocator : public UnboundedAllocator {
 public:
  /* Inherit the constructors */
  using UnboundedAllocator::UnboundedAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyUnboundedAllocator(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      const std::string& name = resource_default_name_v<UnboundedAllocator>) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyCudaGreenContextPool : public CudaGreenContextPool {
 public:
  /* Inherit the constructors */
  using CudaGreenContextPool::CudaGreenContextPool;

  // Define a constructor that fully initializes the object.
  explicit PyCudaGreenContextPool(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, int32_t dev_id = 0,
      uint32_t flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, uint32_t num_partitions = 0,
      const std::vector<uint32_t>& sms_per_partition = {}, int32_t default_context_index = -1,
      uint32_t min_sm_size = 2,
      const std::string& name = resource_default_name_v<CudaGreenContextPool>)
      : CudaGreenContextPool(ArgList{
            Arg{"dev_id", dev_id},
            Arg{"green_context_flags", flags},
            Arg{"num_partitions", num_partitions},
            Arg{"sms_per_partition", sms_per_partition},
            Arg{"default_context", default_context_index},
            Arg{"min_sm_count", min_sm_size},
        }) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyCudaGreenContext : public CudaGreenContext {
 public:
  /* Inherit the constructors */
  using CudaGreenContext::CudaGreenContext;

  // Define a constructor that fully initializes the object.
  explicit PyCudaGreenContext(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<CudaGreenContextPool> cuda_green_context_pool = nullptr, int32_t index = -1,
      const std::string& nvtx_identifier = "defaultGreenContext",
      const std::string& name = resource_default_name_v<CudaGreenContext>)
      : CudaGreenContext(std::move(cuda_green_context_pool), index, nvtx_identifier) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyCudaStreamPool : public CudaStreamPool {
 public:
  /* Inherit the constructors */
  using CudaStreamPool::CudaStreamPool;

  // Define a constructor that fully initializes the object.
  explicit PyCudaStreamPool(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                            int32_t dev_id = 0, uint32_t stream_flags = 0,
                            int32_t stream_priority = 0, uint32_t reserved_size = 1,
                            uint32_t max_size = 0,
                            std::shared_ptr<CudaGreenContext> cuda_green_context = nullptr,
                            const std::string& nvtx_identifier = "nvtx_stream_pool",
                            const std::string& name = resource_default_name_v<CudaStreamPool>)
      : CudaStreamPool(dev_id, stream_flags, stream_priority, reserved_size, max_size,
                       std::move(cuda_green_context), nvtx_identifier) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyRMMAllocator : public RMMAllocator {
 public:
  /* Inherit the constructors */
  using RMMAllocator::RMMAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyRMMAllocator(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      const std::string& device_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& device_memory_max_size = std::string(kPoolMaxSize),
      const std::string& host_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& host_memory_max_size = std::string(kPoolMaxSize), int32_t dev_id = 0,
      const std::string& name = resource_default_name_v<RMMAllocator>)
      : RMMAllocator(ArgList{Arg{"device_memory_initial_size", device_memory_initial_size},
                             Arg{"device_memory_max_size", device_memory_max_size},
                             Arg{"host_memory_initial_size", host_memory_initial_size},
                             Arg{"host_memory_max_size", host_memory_max_size},
                             Arg{"dev_id", dev_id}}) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyStreamOrderedAllocator : public StreamOrderedAllocator {
 public:
  /* Inherit the constructors */
  using StreamOrderedAllocator::StreamOrderedAllocator;

  // Define a constructor that fully initializes the object.
  explicit PyStreamOrderedAllocator(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      const std::string& device_memory_initial_size = std::string(kPoolInitialSize),
      const std::string& device_memory_max_size = std::string(kPoolMaxSize),
      const std::string& release_threshold = std::string(kReleaseThreshold), int32_t dev_id = 0,
      const std::string& name = resource_default_name_v<StreamOrderedAllocator>)
      : StreamOrderedAllocator(
            ArgList{Arg{"device_memory_initial_size", device_memory_initial_size},
                    Arg{"device_memory_max_size", device_memory_max_size},
                    Arg{"release_threshold", release_threshold},
                    Arg{"dev_id", dev_id}}) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
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
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    int32_t,
                    uint64_t,
                    uint64_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "storage_type"_a,
           "block_size"_a,
           "num_blocks"_a,
           "dev_id"_a = 0,
           "name"_a = std::string(resource_default_name_v<BlockMemoryPool>),
           doc::BlockMemoryPool::doc_BlockMemoryPool);

  py::class_<CudaGreenContextPool,
             PyCudaGreenContextPool,
             gxf::GXFResource,
             std::shared_ptr<CudaGreenContextPool>>(
      m, "CudaGreenContextPool", doc::CudaGreenContextPool::doc_CudaGreenContextPool)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    int32_t,
                    uint32_t,
                    uint32_t,
                    std::vector<uint32_t>,
                    int32_t,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "dev_id"_a = 0,
           // Cast the enum to uint32_t: pybind11's ``py::arg::operator=`` templates on the
           // value type and cannot auto-convert the ``CUdev_SM_resource_split_flags_enum``
           // enum into a Python object, which would fail module import with
           // ``ImportError: arg(): could not convert default argument into a Python object``.
           "flags"_a = static_cast<uint32_t>(CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING),
           "num_partitions"_a = 0U,
           "sms_per_partition"_a = py::cast(std::vector<uint32_t>{}),
           "default_context_index"_a = -1,
           "min_sm_size"_a = 2U,
           "name"_a = std::string(resource_default_name_v<CudaGreenContextPool>),
           doc::CudaGreenContextPool::doc_CudaGreenContextPool)
      .def_static("is_partitioning_supported",
                  &CudaGreenContextPool::is_partitioning_supported,
                  "dev_id"_a,
                  "min_sm_count"_a,
                  "sms_per_partition"_a,
                  doc::CudaGreenContextPool::doc_is_partitioning_supported);

  py::class_<CudaGreenContext,
             PyCudaGreenContext,
             gxf::GXFResource,
             std::shared_ptr<CudaGreenContext>>(
      m, "CudaGreenContext", doc::CudaGreenContext::doc_CudaGreenContext)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<CudaGreenContextPool>,
                    int32_t,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "cuda_green_context_pool"_a = nullptr,
           "index"_a = -1,
           "nvtx_identifier"_a = "nvtx_green_context",
           "name"_a = std::string(resource_default_name_v<CudaGreenContext>),
           doc::CudaGreenContext::doc_CudaGreenContext);

  py::class_<CudaStreamPool, PyCudaStreamPool, Allocator, std::shared_ptr<CudaStreamPool>>(
      m, "CudaStreamPool", doc::CudaStreamPool::doc_CudaStreamPool)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    int32_t,
                    uint32_t,
                    int32_t,
                    uint32_t,
                    uint32_t,
                    std::shared_ptr<CudaGreenContext>,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "dev_id"_a = 0,
           "stream_flags"_a = 0U,
           "stream_priority"_a = 0,
           "reserved_size"_a = 1U,
           "max_size"_a = 0U,
           "cuda_green_context"_a = nullptr,
           "nvtx_identifier"_a = "nvtx_stream_pool",
           "name"_a = std::string(resource_default_name_v<CudaStreamPool>),
           doc::CudaStreamPool::doc_CudaStreamPool);

  py::class_<UnboundedAllocator,
             PyUnboundedAllocator,
             Allocator,
             std::shared_ptr<UnboundedAllocator>>(
      m, "UnboundedAllocator", doc::UnboundedAllocator::doc_UnboundedAllocator)
      .def(py::init<std::variant<Fragment*, Subgraph*>, const std::string&>(),
           "fragment"_a,
           "name"_a = std::string(resource_default_name_v<UnboundedAllocator>),
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
      .def(py::init<std::variant<Fragment*, Subgraph*>,
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
           "name"_a = std::string(resource_default_name_v<RMMAllocator>),
           doc::RMMAllocator::doc_RMMAllocator);

  py::class_<StreamOrderedAllocator,
             PyStreamOrderedAllocator,
             CudaAllocator,
             std::shared_ptr<StreamOrderedAllocator>>(
      m, "StreamOrderedAllocator", doc::StreamOrderedAllocator::doc_StreamOrderedAllocator)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
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
           "name"_a = std::string(resource_default_name_v<StreamOrderedAllocator>),
           doc::StreamOrderedAllocator::doc_StreamOrderedAllocator);
}
}  // namespace holoscan
