/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/detail/transport/backend.hpp>
#include <holomq/device_buffer_allocator.hpp>
#include "buffer_pool/device/device_buffer_pool_impl.hpp"
#include "buffer_pool/device/device_const_buffer_impl.hpp"
#include "buffer_pool/device/vmm_exported_pool.hpp"
#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"
#include "context_impl.hpp"
#include "cuda/context.hpp"
#include "fd_pass/server.hpp"
#include "fd_pass/sockaddr.hpp"
#include "vmm_pool_import_cache.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::uint32_t kDefaultBufferCount = 4U;
constexpr std::size_t kBufferBytes = 256U;

[[nodiscard]] std::optional<std::string> cuda_error_message(const CUresult result) {
  if (result == CUDA_SUCCESS) {
    return std::nullopt;
  }
  const char* name = nullptr;
  const char* msg = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &msg));
  return std::string(name != nullptr ? name : "?") + " (" + (msg != nullptr ? msg : "?") + ")";
}

[[nodiscard]] std::string make_socket_path() {
  static std::atomic<std::uint64_t> next_socket_id{0U};
  const std::uint64_t socket_id = next_socket_id.fetch_add(1U, std::memory_order_relaxed);
  std::string path = "/tmp/holomq_vmm_" + std::to_string(static_cast<long>(::getpid())) + "_" +
                     std::to_string(socket_id) + ".sock";
  if (path.size() > holomq::fd_pass::endpoint_max_length()) {
    return {};
  }
  return path;
}

void cleanup_socket_path(const std::string& socket_path) {
  if (holomq::fd_pass::endpoint_uses_filesystem_path()) {
    static_cast<void>(::unlink(socket_path.c_str()));
  }
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::Buffer& buffer) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(buffer.ptr()));
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::ConstBuffer& buffer) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(buffer.ptr()));
}

[[nodiscard]] bool write_pattern(const CUdeviceptr device_ptr, const std::size_t size_bytes,
                                 const std::uint32_t seed, std::string* error_out) {
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(word_count);
  for (std::size_t i = 0; i < word_count; ++i) {
    host[i] = seed + static_cast<std::uint32_t>(i);
  }
  const CUresult result = cuMemcpyHtoD(device_ptr, host.data(), size_bytes);
  if (result != CUDA_SUCCESS && error_out != nullptr) {
    *error_out = "cuMemcpyHtoD failed: " + cuda_error_message(result).value_or("unknown");
  }
  return result == CUDA_SUCCESS;
}

[[nodiscard]] bool read_and_verify(const CUdeviceptr device_ptr, const std::size_t size_bytes,
                                   const std::uint32_t seed, std::string* error_out) {
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(word_count);
  const CUresult copy_result = cuMemcpyDtoH(host.data(), device_ptr, size_bytes);
  if (copy_result != CUDA_SUCCESS) {
    if (error_out != nullptr) {
      *error_out = "cuMemcpyDtoH failed: " + cuda_error_message(copy_result).value_or("unknown");
    }
    return false;
  }
  for (std::size_t i = 0; i < word_count; ++i) {
    const std::uint32_t expected = seed + static_cast<std::uint32_t>(i);
    if (host[i] != expected) {
      if (error_out != nullptr) {
        *error_out = "verify mismatch at word " + std::to_string(i) + " (got " +
                     std::to_string(host[i]) + ", expected " + std::to_string(expected) + ")";
      }
      return false;
    }
  }
  return true;
}

void skip_unless_cuda_available() {
  const CUresult init_result = cuInit(0);
  if (init_result != CUDA_SUCCESS) {
    GTEST_SKIP() << "CUDA driver not available: "
                 << cuda_error_message(init_result).value_or("unknown");
  }

  CUdevice device{};
  const CUresult device_result = cuDeviceGet(&device, 0);
  if (device_result != CUDA_SUCCESS) {
    GTEST_SKIP() << "No CUDA device available: "
                 << cuda_error_message(device_result).value_or("unknown");
  }
}

[[nodiscard]] holomq::cuda::Context make_cuda_context() {
  CUdevice device{};
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return {};
  }
  return holomq::cuda::Context{device};
}

}  // namespace

TEST(CudaVmmImportTest, PoolCookieIsUniqueAcrossPoolRecreate) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  constexpr std::size_t kRegionBytes = 4096U;
  const holomq::DeviceBufferAllocator allocator = holomq::default_device_buffer_allocator();

  auto first_pool_or = holomq::VmmExportedPool::create(allocator, kRegionBytes);
  if (!first_pool_or.has_value()) {
    ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent retry failed";
    first_pool_or = holomq::VmmExportedPool::create(allocator, kRegionBytes);
  }
  if (!first_pool_or.has_value()) {
    GTEST_SKIP() << "VMM pool allocator unavailable in this CUDA configuration";
  }
  const std::uint64_t first_cookie = first_pool_or->cookie();
  const void* const first_ptr = first_pool_or->ptr();
  ASSERT_NE(first_cookie, 0U);
  first_pool_or.reset();

  auto second_pool_or = holomq::VmmExportedPool::create(allocator, kRegionBytes);
  ASSERT_TRUE(second_pool_or.has_value()) << "second VmmExportedPool::create failed";
  EXPECT_NE(second_pool_or->cookie(), first_cookie)
      << "pool_cookie must not repeat after pool destroy/recreate";
  if (second_pool_or->ptr() == first_ptr) {
    EXPECT_NE(second_pool_or->cookie(), first_cookie)
        << "pool_cookie must not alias a reused CUDA VA";
  }
}

// Verify concurrent first access creates one imported mapping and shares it with every caller.
TEST(CudaVmmImportTest, ConcurrentFirstImportIsSingleFlight) {
  using holomq::VmmPoolImportCache;
  using holomq::VmmPoolImportDescriptor;
  using holomq::fd_pass::Server;

  skip_unless_cuda_available();

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty()) << "failed to build socket path";

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  constexpr std::size_t kRegionBytes = 4096U;
  auto exported_pool_or =
      holomq::VmmExportedPool::create(holomq::default_device_buffer_allocator(), kRegionBytes);
  if (!exported_pool_or.has_value()) {
    GTEST_SKIP() << "VMM pool allocator unavailable in this CUDA configuration";
  }
  holomq::VmmExportedPool exported_pool = std::move(exported_pool_or).value();

  auto transport_or = holomq::detail::transport::create_runtime();
  ASSERT_TRUE(transport_or.has_value()) << "create_runtime failed";
  const auto session = std::make_shared<holomq::ContextImpl>(
      std::move(transport_or).value(), socket_path, std::nullopt);
  exported_pool.register_fd_export(session);

  std::atomic<std::uint32_t> fd_request_count{0U};
  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value()) << "Server::create failed";
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([session, &fd_request_count](const std::uint64_t pool_cookie) {
    fd_request_count.fetch_add(1U, std::memory_order_relaxed);
    return session->vmm_pool_export_registry().dup_exported_fd(pool_cookie);
  }));

  VmmPoolImportDescriptor descriptor{};
  descriptor.pool_cookie = exported_pool.cookie();
  descriptor.slot_stride_bytes = kRegionBytes;
  descriptor.pool_total_bytes = exported_pool.size();
  descriptor.fd_pass_socket_path = socket_path.c_str();
  CUdevice expected_device{};
  ASSERT_EQ(cuCtxGetDevice(&expected_device), CUDA_SUCCESS);
  descriptor.cuda_device_ordinal = static_cast<std::int32_t>(expected_device);

  constexpr std::size_t kThreadCount = 8U;
  VmmPoolImportCache import_cache;
  std::vector<std::optional<holomq::VmmPoolImportBase>> imports(kThreadCount);
  std::vector<CUresult> set_context_results(kThreadCount, CUDA_ERROR_UNKNOWN);
  std::atomic<std::size_t> ready_count{0U};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  // Stage every importer before releasing them together against an initially empty cache.
  for (std::size_t i = 0U; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] {
      set_context_results[i] = cuCtxSetCurrent(cuda_ctx.get());
      ready_count.fetch_add(1U, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      imports[i] = import_cache.import_pool_base(descriptor);
    });
  }
  while (ready_count.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_TRUE(imports[0].has_value());
  // All callers must observe the same mapping, produced by exactly one FD-pass request.
  for (std::size_t i = 0U; i < kThreadCount; ++i) {
    EXPECT_EQ(set_context_results[i], CUDA_SUCCESS);
    ASSERT_TRUE(imports[i].has_value());
    EXPECT_EQ(imports[i]->base, imports[0]->base);
    EXPECT_EQ(imports[i]->pool_total_bytes, imports[0]->pool_total_bytes);
    EXPECT_EQ(imports[i]->slot_stride_bytes, imports[0]->slot_stride_bytes);
    EXPECT_EQ(imports[i]->cuda_device_ordinal, descriptor.cuda_device_ordinal);
  }
  EXPECT_EQ(fd_request_count.load(std::memory_order_relaxed), 1U);

  VmmPoolImportDescriptor mismatched_descriptor = descriptor;
  mismatched_descriptor.cuda_device_ordinal = descriptor.cuda_device_ordinal + 1;
  EXPECT_FALSE(import_cache.import_pool_base(mismatched_descriptor).has_value())
      << "a cache hit must reject a wire ordinal that differs from the imported allocation";
  EXPECT_EQ(fd_request_count.load(std::memory_order_relaxed), 1U)
      << "cache-hit validation must not request another exported FD";

  VmmPoolImportCache mismatched_first_import_cache;
  EXPECT_FALSE(mismatched_first_import_cache.import_pool_base(mismatched_descriptor).has_value())
      << "a first import must reject a wire ordinal that differs from allocation properties";
  EXPECT_EQ(fd_request_count.load(std::memory_order_relaxed), 2U);

  server.stop();
  cleanup_socket_path(socket_path);
}

TEST(CudaVmmImportTest, ExportsAndImportsDeviceBuffers) {
  using holomq::VmmPoolImportCache;
  using holomq::fd_pass::Server;

  skip_unless_cuda_available();

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty()) << "failed to build socket path";

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  auto transport_or = holomq::detail::transport::create_runtime();
  ASSERT_TRUE(transport_or.has_value()) << "create_runtime failed";
  const auto session = std::make_shared<holomq::ContextImpl>(
      std::move(transport_or).value(), socket_path, std::nullopt);
  auto pool_or = holomq::BufferPool::create<holomq::DeviceBufferPoolImpl>(
      kBufferBytes, kDefaultBufferCount, socket_path, session);
  ASSERT_TRUE(pool_or.has_value()) << "BufferPool::create failed";

  holomq::BufferPool pool = std::move(pool_or).value();
  const std::size_t buffer_bytes = pool.buffer_bytes();
  ASSERT_EQ(pool.buffer_count(), kDefaultBufferCount);

  CUdevice expected_device{};
  ASSERT_EQ(cuDeviceGet(&expected_device, 0), CUDA_SUCCESS);
  const auto pool_ordinal = pool.cuda_device_ordinal();
  ASSERT_TRUE(pool_ordinal.has_value());
  EXPECT_EQ(pool_ordinal.value(), static_cast<int32_t>(expected_device));

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value()) << "Server::create failed";
  Server server = std::move(server_or).value();

  const bool started = server.start([session](std::uint64_t pool_cookie) {
    return session->vmm_pool_export_registry().dup_exported_fd(pool_cookie);
  });
  ASSERT_TRUE(started) << "Server::start failed";

  std::vector<holomq::Buffer> owner_buffers;
  owner_buffers.reserve(kDefaultBufferCount);
  for (std::uint32_t i = 0U; i < kDefaultBufferCount; ++i) {
    auto owner_buffer_or = pool.acquire();
    ASSERT_TRUE(owner_buffer_or.has_value()) << "acquire failed while filling pool at slot " << i;
    owner_buffers.push_back(std::move(owner_buffer_or).value());
  }

  EXPECT_FALSE(pool.acquire().has_value()) << "expected acquire to fail after exhausting pool";

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS)
      << "cuStreamCreate failed";

  VmmPoolImportCache import_cache{};
  holomq::RefCountSegmentCache ref_count_segment_cache{};

  for (std::uint32_t i = 0U; i < kDefaultBufferCount; ++i) {
    const std::uint32_t seed = 0xB0000000U + i;
    holomq::Buffer& owner_buffer = owner_buffers[i];

    std::string error;
    ASSERT_TRUE(write_pattern(buffer_device_ptr(owner_buffer), buffer_bytes, seed, &error))
        << error;

    const auto exported = owner_buffer.export_handle(stream);
    ASSERT_TRUE(exported.has_value()) << "Buffer::export_handle failed";
    owner_buffer = holomq::Buffer{};

    auto imported = holomq::DeviceConstBufferImpl::create(
        exported.value(), stream, import_cache, ref_count_segment_cache);
    ASSERT_TRUE(imported.has_value()) << "DeviceConstBufferImpl::create failed";

    holomq::ConstBuffer imported_buffer = std::move(imported).value();
    EXPECT_EQ(imported_buffer.memory_type(), holomq::BufferMemoryType::Device);
    ASSERT_TRUE(imported_buffer.cuda_device_ordinal().has_value());
    EXPECT_EQ(imported_buffer.cuda_device_ordinal().value(), pool_ordinal.value());
    if (i == 0U) {
      const auto& wire = holomq::detail::BufferHandleWireAccess::device_wire(exported.value());
      holomq::VmmPoolImportDescriptor mismatched_descriptor{};
      mismatched_descriptor.pool_cookie = wire.pool_cookie;
      mismatched_descriptor.slot_stride_bytes = wire.slot_stride_bytes;
      mismatched_descriptor.pool_total_bytes = wire.pool_total_bytes;
      mismatched_descriptor.fd_pass_socket_path = wire.fd_pass_socket_path;
      mismatched_descriptor.cuda_device_ordinal = pool_ordinal.value() + 1;
      EXPECT_FALSE(import_cache.import_pool_base(mismatched_descriptor).has_value())
          << "a warm cache hit must reject a mismatched wire ordinal";
    }
    ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS) << "cuStreamSynchronize failed";
    ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported_buffer), buffer_bytes, seed, &error))
        << error;

    imported_buffer = holomq::ConstBuffer{};
    ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS)
        << "cuStreamSynchronize after imported buffer release failed";
  }

  static_cast<void>(cuStreamDestroy(stream));
  owner_buffers.clear();
  server.stop();
  cleanup_socket_path(socket_path);
}

// Location attribution for imported VMM allocations. cuMemGetAllocationPropertiesFromHandle does
// not round-trip creation properties for an imported handle: on coherent/UMA platforms a pool
// created as DEVICE reads back as HOST_NUMA (measured on GB10 / DGX Spark as DEVICE/0 ->
// HOST_NUMA/0), so location.id must not be treated as a device ordinal.

TEST(CudaVmmImportTest, DeviceLocationSuppliesTheOrdinalDirectly) {
  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = 3;

  const auto without_expected = holomq::detail::device_ordinal_for_location(location, -1);
  ASSERT_TRUE(without_expected.has_value());
  EXPECT_EQ(*without_expected, 3);

  const auto with_expected = holomq::detail::device_ordinal_for_location(location, 3);
  ASSERT_TRUE(with_expected.has_value());
  EXPECT_EQ(*with_expected, 3);

  EXPECT_FALSE(holomq::detail::device_ordinal_for_location(location, 2).has_value())
      << "a DEVICE location must match a nonnegative expected ordinal";
}

TEST(CudaVmmImportTest, DeviceLocationRejectsNegativeId) {
  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = -1;
  EXPECT_FALSE(holomq::detail::device_ordinal_for_location(location, -1).has_value());
}

TEST(CudaVmmImportTest, UnattributableLocationTypesAreRejected) {
  for (const auto location_type : {CU_MEM_LOCATION_TYPE_INVALID,
                                   CU_MEM_LOCATION_TYPE_HOST,
                                   CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT}) {
    CUmemLocation location{};
    location.type = location_type;
    location.id = 0;
    EXPECT_FALSE(holomq::detail::device_ordinal_for_location(location, 0).has_value())
        << "location type " << static_cast<int>(location_type) << " must not resolve to a device";
  }

  CUmemLocation negative_host_numa{};
  negative_host_numa.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  negative_host_numa.id = -1;
  EXPECT_FALSE(holomq::detail::device_ordinal_for_location(negative_host_numa, -1).has_value());
}

TEST(CudaVmmImportTest, HostNumaLocationResolvesThroughHostNumaId) {
  skip_unless_cuda_available();

  CUdevice device{};
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  int host_numa_id = -1;
  if (cuDeviceGetAttribute(&host_numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device) !=
          CUDA_SUCCESS ||
      host_numa_id < 0) {
    GTEST_SKIP() << "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID unavailable on this platform";
  }

  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  location.id = host_numa_id;

  // Confirm branch: the publisher's ordinal names the device, which stays unambiguous even when
  // several devices share one host NUMA node.
  const auto confirmed = holomq::detail::device_ordinal_for_location(location, 0);
  ASSERT_TRUE(confirmed.has_value()) << "device 0 must be attributable to its own host NUMA node";
  EXPECT_EQ(*confirmed, 0);

  // A node that no device reports must not resolve, under either branch.
  CUmemLocation foreign{};
  foreign.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  foreign.id = host_numa_id + 1000;
  EXPECT_FALSE(holomq::detail::device_ordinal_for_location(foreign, 0).has_value());
  EXPECT_FALSE(holomq::detail::device_ordinal_for_location(foreign, -1).has_value());

  // Scan branch (expected_ordinal == -1, the host split-generation pool path). It resolves only
  // when exactly one device sits on the node; more than one is ambiguous and fails closed.
  int device_count = 0;
  ASSERT_EQ(cuDeviceGetCount(&device_count), CUDA_SUCCESS);
  int devices_on_node = 0;
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    CUdevice candidate{};
    int candidate_node = -1;
    if (cuDeviceGet(&candidate, ordinal) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&candidate_node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, candidate) ==
            CUDA_SUCCESS &&
        candidate_node == host_numa_id) {
      ++devices_on_node;
    }
  }
  const auto scanned = holomq::detail::device_ordinal_for_location(location, -1);
  if (devices_on_node == 1) {
    ASSERT_TRUE(scanned.has_value());
    EXPECT_EQ(*scanned, 0);
  } else {
    EXPECT_FALSE(scanned.has_value()) << devices_on_node << " devices share host NUMA node "
                                      << host_numa_id << "; an ambiguous node must fail closed";
  }
}

TEST(CudaVmmImportTest, ImportedAllocationLocationAttributesToTheCreatingDevice) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx)) << "cuCtxCreate failed";
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS) << "cuCtxSetCurrent failed";

  CUdevice device{};
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device);

  std::size_t granularity = 0;
  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
          CUDA_SUCCESS ||
      granularity == 0U) {
    GTEST_SKIP() << "VMM allocation granularity unavailable in this CUDA configuration";
  }

  CUmemGenericAllocationHandle handle{};
  if (cuMemCreate(&handle, granularity, &prop, 0) != CUDA_SUCCESS) {
    GTEST_SKIP() << "VMM pool allocator unavailable in this CUDA configuration";
  }

  int fd = -1;
  const CUresult exported =
      cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (exported != CUDA_SUCCESS) {
    static_cast<void>(cuMemRelease(handle));
    GTEST_SKIP() << "POSIX-FD VMM export unsupported in this CUDA configuration";
  }

  CUmemGenericAllocationHandle imported_handle{};
  const CUresult imported =
      cuMemImportFromShareableHandle(&imported_handle,
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(fd)),
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  ASSERT_EQ(imported, CUDA_SUCCESS) << "cuMemImportFromShareableHandle failed";

  CUmemAllocationProp imported_prop{};
  ASSERT_EQ(cuMemGetAllocationPropertiesFromHandle(&imported_prop, imported_handle), CUDA_SUCCESS);
  EXPECT_EQ(imported_prop.type, CU_MEM_ALLOCATION_TYPE_PINNED)
      << "allocation type must survive export/import";

  // The regression guard: whatever location the driver reports for the imported handle -- DEVICE
  // on discrete platforms, HOST_NUMA on coherent ones -- must attribute back to device 0.
  const auto resolved = holomq::detail::device_ordinal_for_location(imported_prop.location, 0);
  EXPECT_TRUE(resolved.has_value())
      << "imported location (type=" << static_cast<int>(imported_prop.location.type)
      << ", id=" << imported_prop.location.id << ") must attribute to the creating device";
  if (resolved.has_value()) {
    EXPECT_EQ(*resolved, 0);
  }

  static_cast<void>(cuMemRelease(imported_handle));
  static_cast<void>(cuMemRelease(handle));
  static_cast<void>(::close(fd));
}
