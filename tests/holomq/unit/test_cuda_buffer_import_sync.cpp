/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer.hpp>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include <holomq/logger.hpp>
#include "../../common/allocation_counter.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/stream_lifetime_access.hpp"
#include "buffer_pool/sync/stream_release.hpp"
#include "cuda/context.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::uint32_t kBufferCount = 4U;
constexpr std::size_t kBufferBytes = 256U;
constexpr std::uint32_t kPatternSeed = 0xC0FF0000U;
thread_local bool in_release_callback{};

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
  std::string path = "/tmp/holomq_bis_" + std::to_string(static_cast<long>(::getpid())) + "_" +
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

// Pinned host payloads are registered with CU_MEMHOSTREGISTER_DEVICEMAP. Their device address
// equals the host address only on devices that report
// CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, so resolve the device alias rather
// than casting. Device allocations resolve to themselves.
[[nodiscard]] CUdeviceptr device_address(const void* pointer) noexcept {
  const auto address = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer));
  CUdeviceptr device{};
  return cuPointerGetAttribute(&device, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, address) ==
                 CUDA_SUCCESS
             ? device
             : address;
}

// Failure-only context for a rejected device access to a buffer payload.
[[nodiscard]] std::string describe_device_address(const void* pointer) {
  const auto address = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer));
  unsigned int memory_type{};
  const CUresult type_result =
      cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address);
  CUdeviceptr alias{};
  const CUresult alias_result =
      cuPointerGetAttribute(&alias, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, address);
  CUdevice device{};
  int same_address = -1;
  if (cuCtxGetDevice(&device) == CUDA_SUCCESS) {
    static_cast<void>(cuDeviceGetAttribute(
        &same_address, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device));
  }
  std::ostringstream out;
  out << "payload=" << pointer << " memory_type="
      << (type_result == CUDA_SUCCESS ? std::to_string(memory_type)
                                      : cuda_error_message(type_result).value_or("?"))
      << " device_alias=";
  if (alias_result == CUDA_SUCCESS) {
    out << reinterpret_cast<const void*>(static_cast<std::uintptr_t>(alias));
  } else {
    out << cuda_error_message(alias_result).value_or("?");
  }
  out << " can_use_host_pointer_for_registered_mem=" << same_address;
  return out.str();
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::ConstBuffer& buffer) noexcept {
  return device_address(buffer.ptr());
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::Buffer& buffer) noexcept {
  return device_address(buffer.ptr());
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

[[nodiscard]] bool write_pattern_async(const CUdeviceptr device_ptr, const std::size_t size_bytes,
                                       const std::uint32_t seed, CUstream stream,
                                       std::string* error_out) {
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(word_count);
  for (std::size_t i = 0; i < word_count; ++i) {
    host[i] = seed + static_cast<std::uint32_t>(i);
  }
  const CUresult result = cuMemcpyHtoDAsync(device_ptr, host.data(), size_bytes, stream);
  if (result != CUDA_SUCCESS && error_out != nullptr) {
    *error_out = "cuMemcpyHtoDAsync failed: " + cuda_error_message(result).value_or("unknown");
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
        *error_out = "verify mismatch at word " + std::to_string(i);
      }
      return false;
    }
  }
  return true;
}

struct ReleaseGate {
  std::uint32_t* word{};
  CUdeviceptr device_word{};
  CUstream stream{};
  bool create(CUstream target) {
    stream = target;
    if (cuMemHostAlloc(reinterpret_cast<void**>(&word), sizeof(*word), CU_MEMHOSTALLOC_DEVICEMAP) !=
        CUDA_SUCCESS) {
      return false;
    }
    *word = 0U;
    return cuMemHostGetDevicePointer(&device_word, word, 0) == CUDA_SUCCESS;
  }
  void release() noexcept {
    if (word != nullptr) {
      std::atomic_ref<std::uint32_t>(*word).store(1U, std::memory_order_release);
    }
  }
  ~ReleaseGate() {
    if (word != nullptr) {
      release();
      if (stream != nullptr) {
        static_cast<void>(cuStreamSynchronize(stream));
      }
      static_cast<void>(cuMemFreeHost(word));
    }
  }
};

struct ExportFixture {
  holomq::Context context;
  holomq::BufferPool pool;
  holomq::BufferHandle handle;
  CUstream stream{};
  std::string socket_path;
};

[[nodiscard]] std::optional<ExportFixture> make_export_fixture() {
  const std::string socket_path = make_socket_path();
  if (socket_path.empty()) {
    return std::nullopt;
  }

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  if (!context_or.has_value()) {
    return std::nullopt;
  }
  holomq::Context context = std::move(context_or).value();

  auto pool_or = context.create_device_buffer_pool(kBufferBytes, kBufferCount);
  if (!pool_or.has_value()) {
    return std::nullopt;
  }
  holomq::BufferPool pool = std::move(pool_or).value();

  auto buffer_or = pool.acquire();
  if (!buffer_or.has_value()) {
    return std::nullopt;
  }
  holomq::Buffer buffer = std::move(buffer_or).value();

  CUstream stream{};
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    return std::nullopt;
  }

  std::string error;
  if (!write_pattern(buffer_device_ptr(buffer), buffer.size_bytes(), kPatternSeed, &error)) {
    static_cast<void>(cuStreamDestroy(stream));
    return std::nullopt;
  }

  const auto exported_or = buffer.export_handle(stream);
  if (!exported_or.has_value()) {
    static_cast<void>(cuStreamDestroy(stream));
    return std::nullopt;
  }

  return ExportFixture{
      std::move(context), std::move(pool), exported_or.value(), stream, socket_path};
}

}  // namespace

TEST(CudaBufferImportSyncTest, UnsynchronizedImportThenDeferredWait) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  auto fixture_or = make_export_fixture();
  ASSERT_TRUE(fixture_or.has_value()) << "failed to set up export fixture";
  ExportFixture fixture = std::move(fixture_or).value();

  EXPECT_FALSE(fixture.context.import_buffer(fixture.handle, nullptr).has_value())
      << "import_buffer must fail with null stream when an export fence is required";

  auto unsync_or = fixture.context.import_buffer_unsynchronized(fixture.handle);
  ASSERT_TRUE(unsync_or.has_value()) << "import_buffer_unsynchronized failed";
  holomq::ConstBuffer buffer = std::move(unsync_or).value();
  EXPECT_TRUE(buffer.has_producer_readiness_fence());

  EXPECT_FALSE(buffer.enqueue_ready_wait(nullptr))
      << "enqueue_ready_wait must fail with null stream when an export fence is required";

  ASSERT_TRUE(buffer.enqueue_ready_wait(fixture.stream))
      << "enqueue_ready_wait failed on valid stream";
  EXPECT_TRUE(buffer.enqueue_ready_wait(fixture.stream)) << "enqueue_ready_wait must be idempotent";

  std::string error;
  ASSERT_EQ(cuStreamSynchronize(fixture.stream), CUDA_SUCCESS);
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(buffer), kBufferBytes, kPatternSeed, &error))
      << error;

  buffer = holomq::ConstBuffer{};
  ASSERT_EQ(cuStreamSynchronize(fixture.stream), CUDA_SUCCESS);
  static_cast<void>(cuStreamDestroy(fixture.stream));
  cleanup_socket_path(fixture.socket_path);
}

TEST(CudaBufferImportSyncTest, ImportBufferMatchesDeferredWait) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  auto fixture_or = make_export_fixture();
  ASSERT_TRUE(fixture_or.has_value()) << "failed to set up export fixture";
  ExportFixture fixture = std::move(fixture_or).value();

  auto direct_or = fixture.context.import_buffer(fixture.handle, fixture.stream);
  ASSERT_TRUE(direct_or.has_value()) << "import_buffer failed";
  holomq::ConstBuffer direct = std::move(direct_or).value();

  auto deferred_or = fixture.context.import_buffer_unsynchronized(fixture.handle);
  ASSERT_TRUE(deferred_or.has_value()) << "import_buffer_unsynchronized failed";
  holomq::ConstBuffer deferred = std::move(deferred_or).value();
  ASSERT_TRUE(deferred.enqueue_ready_wait(fixture.stream));

  EXPECT_EQ(buffer_device_ptr(direct), buffer_device_ptr(deferred))
      << "deferred import should map the same device payload address";

  std::string error;
  ASSERT_EQ(cuStreamSynchronize(fixture.stream), CUDA_SUCCESS);
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(deferred), kBufferBytes, kPatternSeed, &error))
      << error;

  deferred = holomq::ConstBuffer{};
  direct = holomq::ConstBuffer{};
  ASSERT_EQ(cuStreamSynchronize(fixture.stream), CUDA_SUCCESS);
  static_cast<void>(cuStreamDestroy(fixture.stream));
  cleanup_socket_path(fixture.socket_path);
}

TEST(CudaBufferImportSyncTest, UnsynchronizedExportHasNoExportFence) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  auto pool_or = context_or->create_device_buffer_pool(kBufferBytes, kBufferCount);
  ASSERT_TRUE(pool_or.has_value());

  auto buffer_or = pool_or->acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();

  const auto exported_or = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(exported_or.has_value());
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(exported_or.value()), 0U);

  auto imported_or = context_or->import_buffer_unsynchronized(exported_or.value());
  ASSERT_TRUE(imported_or.has_value());
  EXPECT_FALSE(imported_or->has_producer_readiness_fence());

  EXPECT_FALSE(buffer.enqueue_ready_set(nullptr));
  EXPECT_FALSE(buffer.export_handle(nullptr).has_value());
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredOwnerReleaseCanRemainGenerationFree) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());
  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  auto pool_or = context_or->create_device_buffer_pool(kBufferBytes, kBufferCount);
  ASSERT_TRUE(pool_or.has_value());
  auto buffer_or = pool_or->acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  CUstream stream{};
  CUstream other_stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&other_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  // Key setup: arm owner release on one stream without recording a consumer-visible generation.
  ASSERT_TRUE(buffer.defer_release_on_stream(stream));
  EXPECT_TRUE(buffer.defer_release_on_stream(stream));
  EXPECT_FALSE(buffer.defer_release_on_stream(other_stream));
  auto sibling_handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(sibling_handle.has_value());
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(*sibling_handle), 0U);

  // Key assertion: the same lease can still become the joined-readiness leader on its release
  // stream, while its prior generation-free export remains unchanged.
  ASSERT_TRUE(buffer.enqueue_ready_set(stream));
  auto leader_handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(leader_handle.has_value());
  EXPECT_NE(holomq::detail::BufferHandleWireAccess::ready_generation(*leader_handle), 0U);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(*sibling_handle), 0U);

  buffer = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(other_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, SplitExportMatchesCombinedExportHandle) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  holomq::Context context = std::move(context_or).value();

  auto pool_or = context.create_device_buffer_pool(kBufferBytes, kBufferCount);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  auto combined_buffer_or = pool.acquire();
  ASSERT_TRUE(combined_buffer_or.has_value());
  holomq::Buffer combined_buffer = std::move(combined_buffer_or).value();
  std::string error;
  ASSERT_TRUE(write_pattern(
      buffer_device_ptr(combined_buffer), combined_buffer.size_bytes(), kPatternSeed, &error))
      << error;
  const auto combined_or = combined_buffer.export_handle(stream);
  ASSERT_TRUE(combined_or.has_value());

  auto split_buffer_or = pool.acquire();
  ASSERT_TRUE(split_buffer_or.has_value());
  holomq::Buffer split_buffer = std::move(split_buffer_or).value();
  ASSERT_TRUE(write_pattern(
      buffer_device_ptr(split_buffer), split_buffer.size_bytes(), kPatternSeed + 1U, &error))
      << error;
  ASSERT_TRUE(split_buffer.enqueue_ready_set(stream));
  EXPECT_TRUE(split_buffer.enqueue_ready_set(stream)) << "enqueue_ready_set must be idempotent";
  const auto split_or = split_buffer.export_handle_unsynchronized();
  ASSERT_TRUE(split_or.has_value());

  EXPECT_NE(holomq::detail::BufferHandleWireAccess::ready_generation(combined_or.value()), 0U);
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(combined_or.value()),
            holomq::detail::BufferHandleWireAccess::ready_generation(split_or.value()));

  auto imported_or = context.import_buffer(split_or.value(), stream);
  ASSERT_TRUE(imported_or.has_value()) << "split export path must import with fence";
  holomq::ConstBuffer imported = std::move(imported_or).value();

  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported), kBufferBytes, kPatternSeed + 1U, &error))
      << error;

  imported = holomq::ConstBuffer{};
  split_buffer = holomq::Buffer{};
  combined_buffer = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, MonotonicGenerationOnSlotReuse) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  holomq::Context context = std::move(context_or).value();

  constexpr std::uint32_t kOneSlot = 1U;
  constexpr std::uint32_t kSeedA = 0xA0000000U;
  constexpr std::uint32_t kSeedB = 0xB0000000U;

  auto pool_or = context.create_device_buffer_pool(kBufferBytes, kOneSlot);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  CUstream pub_stream{};
  CUstream sub_stream{};
  ASSERT_EQ(cuStreamCreate(&pub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&sub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  std::string error;

  auto buffer_a_or = pool.acquire();
  ASSERT_TRUE(buffer_a_or.has_value());
  holomq::Buffer buffer_a = std::move(buffer_a_or).value();
  ASSERT_TRUE(write_pattern_async(
      buffer_device_ptr(buffer_a), buffer_a.size_bytes(), kSeedA, pub_stream, &error))
      << error;
  const auto handle_a_or = buffer_a.export_handle(pub_stream);
  ASSERT_TRUE(handle_a_or.has_value());
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(handle_a_or.value()), 1U);

  auto imported_a_or = context.import_buffer(handle_a_or.value(), sub_stream);
  ASSERT_TRUE(imported_a_or.has_value());
  holomq::ConstBuffer imported_a = std::move(imported_a_or).value();
  ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported_a), kBufferBytes, kSeedA, &error))
      << error;

  imported_a = holomq::ConstBuffer{};
  buffer_a = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);

  auto buffer_b_or = pool.acquire();
  ASSERT_TRUE(buffer_b_or.has_value());
  holomq::Buffer buffer_b = std::move(buffer_b_or).value();
  ASSERT_TRUE(write_pattern_async(
      buffer_device_ptr(buffer_b), buffer_b.size_bytes(), kSeedB, pub_stream, &error))
      << error;
  const auto handle_b_or = buffer_b.export_handle(pub_stream);
  ASSERT_TRUE(handle_b_or.has_value());
  EXPECT_EQ(holomq::detail::BufferHandleWireAccess::ready_generation(handle_b_or.value()), 2U);

  auto imported_b_or = context.import_buffer(handle_b_or.value(), sub_stream);
  ASSERT_TRUE(imported_b_or.has_value());
  holomq::ConstBuffer imported_b = std::move(imported_b_or).value();
  ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported_b), kBufferBytes, kSeedB, &error))
      << error;

  imported_b = holomq::ConstBuffer{};
  buffer_b = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(sub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

// A reusable generic device import can move between slots in one already-mapped VMM pool.
TEST(CudaBufferImportSyncTest, GenericDeviceImportRebindsWithinTheSamePool) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  holomq::Context context = std::move(context_or).value();
  auto pool_or = context.create_device_buffer_pool(kBufferBytes, 2U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  auto first_buffer_or = pool.acquire();
  auto second_buffer_or = pool.acquire();
  ASSERT_TRUE(first_buffer_or.has_value());
  ASSERT_TRUE(second_buffer_or.has_value());
  holomq::Buffer first_buffer = std::move(first_buffer_or).value();
  holomq::Buffer second_buffer = std::move(second_buffer_or).value();
  std::string error;
  ASSERT_TRUE(write_pattern(buffer_device_ptr(first_buffer), kBufferBytes, kPatternSeed, &error))
      << error;
  ASSERT_TRUE(
      write_pattern(buffer_device_ptr(second_buffer), kBufferBytes, kPatternSeed + 1U, &error))
      << error;
  const auto first_handle = first_buffer.export_handle_unsynchronized();
  const auto second_handle = second_buffer.export_handle_unsynchronized();
  ASSERT_TRUE(first_handle.has_value());
  ASSERT_TRUE(second_handle.has_value());

  // Key setup: release the first imported slot while retaining its device-pool mappings.
  auto imported_or = context.import_buffer_unsynchronized(*first_handle);
  ASSERT_TRUE(imported_or.has_value());
  holomq::ConstBuffer imported = std::move(imported_or).value();
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported), kBufferBytes, kPatternSeed, &error))
      << error;
  imported.release();
  EXPECT_FALSE(static_cast<bool>(imported));
  EXPECT_EQ(imported.memory_type(), holomq::BufferMemoryType::None);
  EXPECT_EQ(imported.size_bytes(), 0U);
  EXPECT_EQ(imported.ptr(), nullptr);

  // Key assertion: the same control targets the second device slot and exposes its contents.
  ASSERT_TRUE(context.rebind_buffer_unsynchronized(*second_handle, imported).has_value());
  ASSERT_TRUE(read_and_verify(buffer_device_ptr(imported), kBufferBytes, kPatternSeed + 1U, &error))
      << error;
  imported.release();

  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredRefReleaseHoldsSlotUntilStreamCompletes) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  holomq::Context context = std::move(context_or).value();

  constexpr std::uint32_t kOneSlot = 1U;
  constexpr std::size_t kLargeBufferBytes = 4U * 1024U * 1024U;
  constexpr std::uint32_t kSeed = 0xD0000000U;

  auto pool_or = context.create_device_buffer_pool(kLargeBufferBytes, kOneSlot);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  CUstream pub_stream{};
  ASSERT_EQ(cuStreamCreate(&pub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  std::string error;

  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  ASSERT_TRUE(write_pattern_async(
      buffer_device_ptr(buffer), buffer.size_bytes(), kSeed, pub_stream, &error))
      << error;
  {
    // Hold pub_stream so the export's ready set, and the deferred release queued behind it, stay
    // pending until the gate opens. A copy from pageable memory can complete before it returns,
    // so an ungated stream can drain before the query below; the gate follows the copy, which
    // would otherwise wait for it.
    ReleaseGate gate;
    ASSERT_TRUE(gate.create(pub_stream));
    ASSERT_EQ(cuStreamWaitValue32(pub_stream, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
              CUDA_SUCCESS);
    ASSERT_TRUE(buffer.export_handle(pub_stream).has_value());

    // Drop the owner buffer without synchronizing pub_stream. Deferred ref release keeps the SHM
    // ref-count elevated until prior stream work and the host function complete.
    buffer = holomq::Buffer{};

    ASSERT_EQ(cuStreamQuery(pub_stream), CUDA_ERROR_NOT_READY)
        << "deferred ref release must be pending on pub_stream after buffer drop";

    EXPECT_FALSE(pool.acquire().has_value())
        << "slot must stay leased until deferred release runs on pub_stream";

    gate.release();
    ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);
  }

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredRefReleaseSurvivesPublisherTeardownBeforeStreamSync) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  CUstream pub_stream{};

  auto context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context_or.has_value());
  holomq::Context context = std::move(context_or).value();

  constexpr std::uint32_t kOneSlot = 1U;
  constexpr std::size_t kLargeBufferBytes = 4U * 1024U * 1024U;
  constexpr std::uint32_t kSeed = 0xE0000000U;

  auto pool_or = context.create_device_buffer_pool(kLargeBufferBytes, kOneSlot);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  ASSERT_EQ(cuStreamCreate(&pub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  std::string error;
  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  ASSERT_TRUE(write_pattern_async(
      buffer_device_ptr(buffer), buffer.size_bytes(), kSeed, pub_stream, &error))
      << error;
  {
    // Hold pub_stream behind the pageable copy, as in the test above, so the release below stays
    // pending until the gate opens.
    ReleaseGate gate;
    ASSERT_TRUE(gate.create(pub_stream));
    ASSERT_EQ(cuStreamWaitValue32(pub_stream, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
              CUDA_SUCCESS);
    ASSERT_TRUE(buffer.export_handle(pub_stream).has_value());

    // Drop the owner buffer while the publisher pool/context remain alive; only the deferred
    // ref-release host function should stay pending on pub_stream.
    buffer = holomq::Buffer{};

    ASSERT_EQ(cuStreamQuery(pub_stream), CUDA_ERROR_NOT_READY)
        << "deferred ref release must be pending on pub_stream after buffer drop";

    EXPECT_FALSE(pool.acquire().has_value())
        << "slot must stay leased until deferred release runs on pub_stream";

    gate.release();
    ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);
  }

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredRefReleaseSurvivesSubscriberTeardownBeforeStreamSync) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  holomq::BufferHandle wire_handle{};
  CUstream sub_stream{};

  auto pub_context_or =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(pub_context_or.has_value());
  holomq::Context pub_context = std::move(pub_context_or).value();

  constexpr std::uint32_t kOneSlot = 1U;
  constexpr std::size_t kLargeBufferBytes = 4U * 1024U * 1024U;
  constexpr std::uint32_t kSeed = 0xE1000000U;

  auto pool_or = pub_context.create_device_buffer_pool(kLargeBufferBytes, kOneSlot);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  {
    CUstream pub_stream{};
    ASSERT_EQ(cuStreamCreate(&pub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

    std::string error;
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value());
    holomq::Buffer buffer = std::move(buffer_or).value();
    ASSERT_TRUE(write_pattern_async(
        buffer_device_ptr(buffer), buffer.size_bytes(), kSeed, pub_stream, &error))
        << error;

    const auto exported_or = buffer.export_handle(pub_stream);
    ASSERT_TRUE(exported_or.has_value());
    wire_handle = exported_or.value();

    ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);
    buffer = holomq::Buffer{};
    ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  }

  auto sub_context_or = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  ASSERT_TRUE(sub_context_or.has_value());
  holomq::Context sub_context = std::move(sub_context_or).value();

  ASSERT_EQ(cuStreamCreate(&sub_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  auto imported_or = sub_context.import_buffer(wire_handle, sub_stream);
  ASSERT_TRUE(imported_or.has_value());
  holomq::ConstBuffer view = std::move(imported_or).value();

  CUdeviceptr device_scratch{};
  ASSERT_EQ(cuMemAlloc(&device_scratch, view.size_bytes()), CUDA_SUCCESS);
  struct ScratchCleanup {
    CUdeviceptr pointer;
    ~ScratchCleanup() { static_cast<void>(cuMemFree(pointer)); }
  } const scratch_cleanup{device_scratch};
  {
    // Hold sub_stream so the imported read, and the deferred release queued behind it, stay
    // pending until the gate opens; an ungated stream can drain before the query below. The gate
    // is declared after the scratch so a failed assertion opens it before the scratch is freed.
    ReleaseGate gate;
    ASSERT_TRUE(gate.create(sub_stream));
    ASSERT_EQ(cuStreamWaitValue32(sub_stream, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
              CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyAsync(device_scratch, buffer_device_ptr(view), view.size_bytes(), sub_stream),
              CUDA_SUCCESS);

    EXPECT_FALSE(pool.acquire().has_value())
        << "publisher slot must stay leased while subscriber holds imported view";

    // Drop the imported view while subscriber context/stream remain alive; the deferred
    // ref-release host function must stay pending on sub_stream behind the gated read.
    view = holomq::ConstBuffer{};

    ASSERT_EQ(cuStreamQuery(sub_stream), CUDA_ERROR_NOT_READY)
        << "deferred ref release must be pending on sub_stream after view drop";

    EXPECT_FALSE(pool.acquire().has_value())
        << "slot must stay leased until deferred subscriber release runs on sub_stream";

    gate.release();
    ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);
  }

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(sub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

namespace {
struct ConsumerStreams {
  CUstream first{};
  CUstream last{};
  ~ConsumerStreams() {
    for (const auto stream : {first, last}) {
      if (stream != nullptr) {
        static_cast<void>(cuStreamSynchronize(stream));
        static_cast<void>(cuStreamDestroy(stream));
      }
    }
  }
};
}  // namespace

class CudaBufferFanoutTest : public testing::TestWithParam<bool> {};

TEST_P(CudaBufferFanoutTest, ImportedLeaseWaitsForEveryConsumer) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  auto view = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(view.has_value());
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_TRUE(view->enqueue_ready_wait(streams.first));
  ASSERT_TRUE(view->enqueue_ready_wait(streams.last));
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(streams.last));
  ASSERT_EQ(cuStreamWaitValue32(streams.last, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
            CUDA_SUCCESS);
  // The second consumer is still using the slot after the first has drained.
  view->release();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  EXPECT_FALSE(pool->acquire().has_value());
  gate.release();
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
  EXPECT_TRUE(pool->acquire().has_value());
  cleanup_socket_path(socket_path);
}

namespace {
using StreamLifetime = holomq::cuda::StreamLifetime;
std::optional<holomq::Buffer> wait_for_released_slot(
    holomq::BufferPool& pool, const std::chrono::seconds timeout = std::chrono::seconds(2)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto buffer = pool.acquire();
    if (buffer) {
      return std::move(*buffer);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return std::nullopt;
}
}  // namespace

TEST_P(CudaBufferFanoutTest, FailedRegistrationCannotRetireThePreviousSlotAfterRebind) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 2U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 2U);
  ASSERT_TRUE(pool.has_value());
  auto first = pool->acquire();
  auto second = pool->acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto first_handle = first->export_handle_unsynchronized();
  auto second_handle = second->export_handle_unsynchronized();
  ASSERT_TRUE(first_handle.has_value());
  ASSERT_TRUE(second_handle.has_value());
  auto view = context->import_buffer_unsynchronized(*first_handle);
  ASSERT_TRUE(view.has_value());
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  holomq::EndpointStream::drain_retirements();  // initialize the worker before failing allocations
  bool accepted{};
  {
    // The old implementation published its group after these first two allocations,
    // then failed allocating the entry and kept the old slot captured across rebind.
    const holoscan::test::ScopedAllocationExhaustion allocations{2U};
    accepted = holomq::BufferStreamAccess::register_consumer(*view, streams.first, lifetime);
  }
  EXPECT_FALSE(accepted);
  view->release();
  EXPECT_FALSE(holomq::BufferStreamAccess::register_consumer(*view, streams.first, lifetime));
  EXPECT_FALSE(holomq::BufferStreamAccess::wait(*view, streams.first, lifetime));
  ASSERT_TRUE(context->rebind_buffer_unsynchronized(*second_handle, *view));
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.first, lifetime));
  view->release();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  holomq::EndpointStream::drain_retirements();
  // Both publisher owners are still live. Releasing the old slot twice would make it reusable.
  EXPECT_FALSE(pool->acquire().has_value());
  const auto* second_pointer = second->ptr();
  second = holomq::Buffer{};
  auto reused = wait_for_released_slot(*pool);
  ASSERT_TRUE(reused.has_value());
  EXPECT_EQ(reused->ptr(), second_pointer);
  cleanup_socket_path(socket_path);
}

TEST_P(CudaBufferFanoutTest, EscapedOwnerAndImportReleaseAfterNativeStreamsAreDestroyed) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto producer = std::make_shared<StreamLifetime>(streams.first, 0);
  auto consumer = std::make_shared<StreamLifetime>(streams.last, 0);
  auto handle = holomq::BufferStreamAccess::export_handle(*owner, streams.first, producer);
  ASSERT_TRUE(handle.has_value());
  auto view = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(view.has_value());
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.last, consumer));
  producer->close();
  consumer->close();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
  ASSERT_TRUE(producer->publish_retired());
  ASSERT_TRUE(consumer->publish_retired());
  ASSERT_EQ(cuStreamDestroy(streams.first), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(streams.last), CUDA_SUCCESS);
  streams.first = nullptr;
  streams.last = nullptr;
  owner = holomq::Buffer{};
  view->release();
  EXPECT_TRUE(pool->acquire().has_value()) << "Retired must release without querying dead handles";
  cleanup_socket_path(socket_path);
}

TEST_P(CudaBufferFanoutTest, ClosingOneConsumerDoesNotProveAllConsumersIdle) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  auto view = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(view.has_value());
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto first = std::make_shared<StreamLifetime>(streams.first, 0);
  auto last = std::make_shared<StreamLifetime>(streams.last, 0);
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.first, first));
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.last, last));
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(streams.last));
  ASSERT_EQ(cuStreamWaitValue32(streams.last, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
            CUDA_SUCCESS);
  first->close();
  last->close();
  view->release();  // no wait for the closing owner's GPU work
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(first->publish_retired());
  EXPECT_FALSE(pool->acquire().has_value());
  gate.release();
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
  ASSERT_TRUE(last->publish_retired());
  EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
  cleanup_socket_path(socket_path);
}

TEST_P(CudaBufferFanoutTest, ImportedPayloadOutlivesItsContextShell) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  std::string error;
  if (GetParam()) {
    auto* words = static_cast<std::uint32_t*>(owner->ptr());
    for (std::size_t i = 0; i < kBufferBytes / sizeof(*words); ++i) {
      words[i] = kPatternSeed + static_cast<std::uint32_t>(i);
    }
  } else {
    ASSERT_TRUE(write_pattern(buffer_device_ptr(*owner), kBufferBytes, kPatternSeed, &error))
        << error;
  }
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  holomq::ConstBuffer escaped;
  {
    auto importer = holomq::ContextBuilder().create();
    ASSERT_TRUE(importer.has_value());
    auto view = importer->import_buffer_unsynchronized(*handle);
    ASSERT_TRUE(view.has_value());
    escaped = std::move(*view);
  }
  // The payload and any split generation mapping have ownership separate from the cache shell.
  if (GetParam()) {
    EXPECT_EQ(static_cast<const std::uint32_t*>(escaped.ptr())[0], kPatternSeed);
  } else {
    EXPECT_TRUE(read_and_verify(buffer_device_ptr(escaped), kBufferBytes, kPatternSeed, &error))
        << error;
  }
  escaped.release();
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, QuarantinedReleaseRetainsItsMappingAndBackingOwner) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  auto mapping = holomq::ShmRefCountMapping::create(1U);
  ASSERT_TRUE(mapping.has_value());
  ASSERT_TRUE((*mapping)->table().try_acquire_slot(0U));
  auto payload = std::make_shared<int>(23);
  const std::weak_ptr<int> weak = payload;
  holomq::BufferStreamRelease releases;
  ASSERT_TRUE(releases.add(*mapping, 0U, streams.first, lifetime, {payload, {}}));
  payload.reset();
  lifetime->close();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(lifetime->publish_quarantined());
  releases.release();
  EXPECT_EQ((*mapping)->table().ref_count(0U), 1U);
  EXPECT_FALSE(weak.expired());
  // Intentional quarantine: the failed owner closure remains allocated through process exit.
}

TEST(CudaBufferImportSyncTest, ClosingQuarantineDeliveryDoesNotCallLogSinks) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  auto mapping = holomq::ShmRefCountMapping::create(1U);
  ASSERT_TRUE(mapping.has_value());
  ASSERT_TRUE((*mapping)->table().try_acquire_slot(0U));
  holomq::BufferStreamRelease releases;
  ASSERT_TRUE(releases.add(*mapping, 0U, streams.first, lifetime));
  lifetime->close();
  releases.release();
  holomq::EndpointStream::drain_retirements();  // ensure its Closing node has been registered
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  std::atomic_size_t log_calls{};
  struct RestoreLogger {
    holomq::LogLevel previous{holomq::log_level()};
    ~RestoreLogger() {
      holomq::reset_log_handler();
      holomq::set_log_level(previous);
    }
  } restore;
  holomq::set_log_level(holomq::LogLevel::Critical);
  holomq::set_log_handler(
      [](holomq::LogLevel,
         const char*,
         int,
         const char*,
         const char* message,
         void* data) noexcept {
        const std::string_view message_text{message};
        if (message_text == "buffer release completion unproven; retaining slot and mapping") {
          static_cast<std::atomic_size_t*>(data)->fetch_add(1U);
        }
      },
      &log_calls);
  EXPECT_TRUE(lifetime->publish_quarantined());
  EXPECT_EQ(log_calls.load(), 0U);
  EXPECT_EQ((*mapping)->table().ref_count(0U), 1U);
}

TEST_P(CudaBufferFanoutTest, LastSDKImportReleaseInsideCudaHostFunction) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  auto imported = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported.has_value());
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*imported, streams.first, lifetime));
  auto retained = std::make_unique<holomq::ConstBuffer>(std::move(*imported));
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(streams.first));
  ASSERT_EQ(cuStreamWaitValue32(streams.first, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
            CUDA_SUCCESS);
  ASSERT_EQ(cuLaunchHostFunc(
                streams.first,
                [](void* data) {
                  const std::unique_ptr<holomq::ConstBuffer> view{
                      static_cast<holomq::ConstBuffer*>(data)};
                },
                retained.get()),
            CUDA_SUCCESS);
  static_cast<void>(retained.release());
  lifetime->close();
  EXPECT_FALSE(pool->acquire().has_value());
  gate.release();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(lifetime->publish_retired());
  EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
  cleanup_socket_path(socket_path);
}

// An idle consumer stream already proves that every read of the released buffer finished, so the
// release returns the slot without waiting for CUDA to deliver a host function.
TEST_P(CudaBufferFanoutTest, IdleConsumerStreamReturnsTheSlotWithoutAHostFunction) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  auto imported = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported.has_value());
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*imported, streams.first, lifetime));
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);

  // Key setup: hold CUDA's host-function delivery with a callback that blocks on another stream
  // until the test releases it, so a release that needs a host function cannot complete.
  struct HostFunctionHold {
    std::atomic_bool started{};
    std::atomic_bool released{};
  } hold;
  ASSERT_EQ(cuLaunchHostFunc(
                streams.last,
                [](void* data) {
                  auto& held = *static_cast<HostFunctionHold*>(data);
                  held.started.store(true, std::memory_order_release);
                  while (!held.released.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                  }
                },
                &hold),
            CUDA_SUCCESS);
  struct ReleaseHold {
    HostFunctionHold& hold;
    ~ReleaseHold() { hold.released.store(true, std::memory_order_release); }
  } const release_hold{hold};
  const auto started_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!hold.started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < started_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(hold.started.load(std::memory_order_acquire));

  imported->release();
  auto reacquired = wait_for_released_slot(*pool);
  const bool returned_while_held = reacquired.has_value();
  hold.released.store(true, std::memory_order_release);
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);

  // Key assertion: the slot came back while every host function was still held.
  EXPECT_TRUE(returned_while_held);
  reacquired.reset();
  cleanup_socket_path(socket_path);
}

namespace {
// Owns a graph captured from a test stream and its executable instance.
struct CapturedGraph {
  CUgraph graph{};
  CUgraphExec exec{};
  CapturedGraph() = default;
  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  ~CapturedGraph() {
    if (exec != nullptr) {
      static_cast<void>(cuGraphExecDestroy(exec));
    }
    if (graph != nullptr) {
      static_cast<void>(cuGraphDestroy(graph));
    }
  }
};
}  // namespace

// A caller may capture its own stream once the reads of an import have finished. Releasing the
// import during that capture leaves the capture valid, and the slot returns when the graph runs.
TEST_P(CudaBufferFanoutTest, ReleaseDuringCallerStreamCaptureKeepsTheCaptureValid) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());
  auto imported = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported.has_value());
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_TRUE(imported->enqueue_ready_wait(streams.first));
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);

  // Key setup: begin an unrelated capture on the idle caller-owned stream, then release the import
  // on the capturing thread, which prepares a caller-owned stream's release itself.
  ASSERT_EQ(cuStreamBeginCapture(streams.first, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), CUDA_SUCCESS);
  imported->release();
  CUstreamCaptureStatus status{};
  const auto queried = cuStreamIsCapturing(streams.first, &status);
  CapturedGraph captured;
  const auto ended = cuStreamEndCapture(streams.first, &captured.graph);
  EXPECT_EQ(queried, CUDA_SUCCESS);
  EXPECT_EQ(status, CU_STREAM_CAPTURE_STATUS_ACTIVE);

  // Key assertion: the release kept the capture valid and recorded the slot's return in it.
  ASSERT_EQ(ended, CUDA_SUCCESS);
  EXPECT_FALSE(pool->acquire().has_value());
  ASSERT_EQ(cuGraphInstantiateWithFlags(&captured.exec, captured.graph, 0U), CUDA_SUCCESS);
  ASSERT_EQ(cuGraphLaunch(captured.exec, streams.first), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
  cleanup_socket_path(socket_path);
}

// A caller may also register its stream while capturing it, so the captured graph reads the import.
// Registration leaves the capture valid, records any producer-fence wait in the graph, and keeps
// the slot until the import is released after the graph has run.
TEST_P(CudaBufferFanoutTest, RegistrationDuringCallerStreamCaptureKeepsTheCaptureValid) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  for (const bool fenced : {false, true}) {
    SCOPED_TRACE(fenced ? "producer readiness fence" : "no producer readiness fence");
    auto owner = wait_for_released_slot(*pool);
    ASSERT_TRUE(owner.has_value());
    // Hold the producer's fence behind a gate, so only a graph that waits for it stays pending.
    ReleaseGate gate;
    ASSERT_TRUE(gate.create(streams.last));
    ASSERT_EQ(cuStreamWaitValue32(streams.last, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
              CUDA_SUCCESS);
    auto handle =
        fenced ? owner->export_handle(streams.last) : owner->export_handle_unsynchronized();
    ASSERT_TRUE(handle.has_value());
    auto imported = context->import_buffer_unsynchronized(*handle);
    ASSERT_TRUE(imported.has_value());
    EXPECT_EQ(imported->has_producer_readiness_fence(), fenced);
    owner.reset();

    // Key setup: register the caller-owned stream with the import while capturing that stream.
    ASSERT_EQ(cuStreamBeginCapture(streams.first, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
              CUDA_SUCCESS);
    const bool registered = imported->enqueue_ready_wait(streams.first);
    CUstreamCaptureStatus status{};
    const auto queried = cuStreamIsCapturing(streams.first, &status);
    CapturedGraph captured;
    const auto ended = cuStreamEndCapture(streams.first, &captured.graph);
    EXPECT_TRUE(registered);
    EXPECT_EQ(queried, CUDA_SUCCESS);
    EXPECT_EQ(status, CU_STREAM_CAPTURE_STATUS_ACTIVE);

    // Key assertion: the capture stayed valid, and the graph waits for the producer's fence.
    ASSERT_EQ(ended, CUDA_SUCCESS);
    ASSERT_EQ(cuGraphInstantiateWithFlags(&captured.exec, captured.graph, 0U), CUDA_SUCCESS);
    ASSERT_EQ(cuGraphLaunch(captured.exec, streams.first), CUDA_SUCCESS);
    if (fenced) {
      EXPECT_EQ(cuStreamQuery(streams.first), CUDA_ERROR_NOT_READY);
    }
    gate.release();
    ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
    EXPECT_FALSE(pool->acquire().has_value());
    imported->release();
  }
  EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
  cleanup_socket_path(socket_path);
}

// A wait captured in graph A orders neither ordinary work nor a second graph B on the same stream.
// Keep A unlaunched and the producer gated while checking both later scopes and readiness APIs.
TEST_P(CudaBufferFanoutTest, CapturedReadinessDoesNotOrderAnotherExecutionScope) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  CUdeviceptr device_readback{};
  ASSERT_EQ(cuMemAlloc(&device_readback, sizeof(std::uint32_t)), CUDA_SUCCESS);
  struct ReadbackCleanup {
    CUdeviceptr pointer;
    ~ReadbackCleanup() { static_cast<void>(cuMemFree(pointer)); }
  } const readback_cleanup{device_readback};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  for (const bool sdk_wait : {false, true}) {
    SCOPED_TRACE(sdk_wait ? "SDK readiness helper" : "public readiness call");
    auto lifetime = sdk_wait ? std::make_shared<StreamLifetime>(streams.first, 0) : nullptr;
    for (const bool capture_again : {false, true}) {
      SCOPED_TRACE(capture_again ? "capture A to capture B" : "capture to ordinary execution");
      CapturedGraph first_capture;
      CapturedGraph second_capture;
      auto owner = wait_for_released_slot(*pool);
      ASSERT_TRUE(owner.has_value());
      ASSERT_EQ(cuStreamWriteValue32(streams.last, buffer_device_ptr(*owner), 0U, 0U),
                CUDA_SUCCESS);
      ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
      ReleaseGate gate;
      ASSERT_TRUE(gate.create(streams.last));
      ASSERT_EQ(cuStreamWaitValue32(streams.last, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
                CUDA_SUCCESS);
      ASSERT_EQ(cuStreamWriteValue32(streams.last, buffer_device_ptr(*owner), kPatternSeed, 0U),
                CUDA_SUCCESS);
      auto handle = owner->export_handle(streams.last);
      ASSERT_TRUE(handle.has_value());
      auto imported = context->import_buffer_unsynchronized(*handle);
      ASSERT_TRUE(imported.has_value());
      ASSERT_TRUE(imported->has_producer_readiness_fence());
      owner.reset();
      const auto wait_ready = [&] {
        return sdk_wait ? holomq::BufferStreamAccess::wait(*imported, streams.first, lifetime)
                        : imported->enqueue_ready_wait(streams.first);
      };

      ASSERT_EQ(cuStreamBeginCapture(streams.first, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                CUDA_SUCCESS);
      const bool first_wait = wait_ready();
      const auto first_ended = cuStreamEndCapture(streams.first, &first_capture.graph);
      ASSERT_TRUE(first_wait);
      ASSERT_EQ(first_ended, CUDA_SUCCESS);

      if (capture_again) {
        ASSERT_EQ(cuStreamBeginCapture(streams.first, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                  CUDA_SUCCESS);
      }
      const bool second_wait = wait_ready();
      // A pinned-host payload copied directly to host can block this thread before it opens the
      // producer gate. Keep the gated read on the device and read it back after synchronization.
      const auto copied = cuMemcpyAsync(
          device_readback, buffer_device_ptr(*imported), sizeof(std::uint32_t), streams.first);
      if (capture_again) {
        const auto second_ended = cuStreamEndCapture(streams.first, &second_capture.graph);
        ASSERT_EQ(second_ended, CUDA_SUCCESS);
        ASSERT_EQ(cuGraphInstantiateWithFlags(&second_capture.exec, second_capture.graph, 0U),
                  CUDA_SUCCESS);
        ASSERT_EQ(cuGraphLaunch(second_capture.exec, streams.first), CUDA_SUCCESS);
      }
      ASSERT_TRUE(second_wait);
      ASSERT_EQ(copied, CUDA_SUCCESS);

      // Give an incorrectly unordered read time to finish; one immediate query could miss it.
      auto completion = CUDA_ERROR_NOT_READY;
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
      while (std::chrono::steady_clock::now() < deadline) {
        completion = cuStreamQuery(streams.first);
        if (completion != CUDA_ERROR_NOT_READY) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      EXPECT_EQ(completion, CUDA_ERROR_NOT_READY);
      EXPECT_FALSE(pool->acquire().has_value());
      gate.release();
      ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
      std::uint32_t readback{};
      ASSERT_EQ(cuMemcpyDtoH(&readback, device_readback, sizeof(readback)), CUDA_SUCCESS);
      EXPECT_EQ(readback, kPatternSeed);

      // Honor the lease contract even for retained graph A, then prove safe slot retirement.
      ASSERT_EQ(cuGraphInstantiateWithFlags(&first_capture.exec, first_capture.graph, 0U),
                CUDA_SUCCESS);
      ASSERT_EQ(cuGraphLaunch(first_capture.exec, streams.first), CUDA_SUCCESS);
      ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
      imported->release();
      EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
    }
  }
  cleanup_socket_path(socket_path);
}

// A producer may export a buffer from a stream it is capturing. The capture stays valid and records
// the fence write, so a consumer's stream waits until the captured graph runs.
TEST_P(CudaBufferFanoutTest, ExportDuringProducerStreamCaptureFencesWhenTheGraphRuns) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());

  // Key setup: export the buffer from the producer's stream while capturing that stream.
  ASSERT_EQ(cuStreamBeginCapture(streams.last, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), CUDA_SUCCESS);
  auto handle = owner->export_handle(streams.last);
  CUstreamCaptureStatus status{};
  const auto queried = cuStreamIsCapturing(streams.last, &status);
  CapturedGraph captured;
  const auto ended = cuStreamEndCapture(streams.last, &captured.graph);
  ASSERT_TRUE(handle.has_value());
  EXPECT_EQ(queried, CUDA_SUCCESS);
  EXPECT_EQ(status, CU_STREAM_CAPTURE_STATUS_ACTIVE);
  ASSERT_EQ(ended, CUDA_SUCCESS);
  auto imported = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported.has_value());
  EXPECT_TRUE(imported->has_producer_readiness_fence());
  ASSERT_TRUE(imported->enqueue_ready_wait(streams.first));

  // Key assertion: the consumer waits for the fence until the captured graph writes it.
  EXPECT_EQ(cuStreamQuery(streams.first), CUDA_ERROR_NOT_READY);
  ASSERT_EQ(cuGraphInstantiateWithFlags(&captured.exec, captured.graph, 0U), CUDA_SUCCESS);
  ASSERT_EQ(cuGraphLaunch(captured.exec, streams.last), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  owner = holomq::Buffer{};
  imported->release();
  EXPECT_TRUE(wait_for_released_slot(*pool).has_value());
  cleanup_socket_path(socket_path);
}

// One stream-ordered import release. Returns 0 when its slot comes back and the device's primary
// context is active afterwards; `waited` reports how long the slot took to return.
[[nodiscard]] int release_one_import(holomq::Context& context, holomq::BufferPool& pool,
                                     const CUdevice device, std::chrono::nanoseconds& waited) {
  auto owner = pool.acquire();
  if (!owner.has_value()) {
    return 4;
  }
  auto handle = owner->export_handle_unsynchronized();
  if (!handle.has_value()) {
    return 4;
  }
  auto imported = context.import_buffer_unsynchronized(*handle);
  if (!imported.has_value()) {
    return 4;
  }
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  if (cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    return 5;
  }
  const auto lifetime = std::make_shared<StreamLifetime>(streams.first, device);
  if (!holomq::BufferStreamAccess::wait(*imported, streams.first, lifetime)) {
    return 6;
  }
  const auto start = std::chrono::steady_clock::now();
  imported = holomq::ConstBuffer{};
  // The first release may create the primary context, so allow for a loaded host.
  const bool released = wait_for_released_slot(pool, std::chrono::seconds(10)).has_value();
  waited = std::chrono::steady_clock::now() - start;
  unsigned int flags{};
  int active{};
  const bool primary_active =
      cuDevicePrimaryCtxGetState(device, &flags, &active) == CUDA_SUCCESS && active != 0;
  lifetime->close();
  static_cast<void>(cuStreamSynchronize(streams.first));
  static_cast<void>(lifetime->publish_retired());
  if (!released) {
    return 7;
  }
  return primary_active ? 0 : 8;
}

// Runs in a fresh process whose only context is one it created itself. Returns 0 when two
// releases in a row return their slots, the primary context stays active, and the second release
// does not wait for a primary context to be created again.
[[nodiscard]] int cleanup_without_primary_context_probe() {
  auto cuda_ctx = make_cuda_context();
  CUdevice device{};
  if (!cuda_ctx || cuCtxSetCurrent(cuda_ctx.get()) != CUDA_SUCCESS ||
      cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return 2;
  }
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  if (!context.has_value()) {
    return 3;
  }
  auto pool = context->create_device_buffer_pool(kBufferBytes, 1U);
  if (!pool.has_value()) {
    return 3;
  }
  std::chrono::nanoseconds waited{};
  int result = release_one_import(*context, *pool, device, waited);
  if (result == 0) {
    result = release_one_import(*context, *pool, device, waited);
  }
  if (result == 0 && waited > std::chrono::milliseconds(250)) {
    result = 9;
  }
  cleanup_socket_path(socket_path);
  return result;
}

// A process that works only in its own CUDA context has no active primary context. Cleanup that
// retained and released the primary context per release created and destroyed it each time, and
// a release took 0.4-0.7 s. The probe runs in a fresh process so no other test holds the context.
TEST(CudaBufferImportSyncTest, CleanupKeepsThePrimaryContextItNeeds) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(cleanup_without_primary_context_probe()), testing::ExitedWithCode(0), "");
}

TEST(CudaBufferImportSyncTest, BlockedEndpointDoesNotBlockUnrelatedReleasesOrWorkerDrain) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  auto endpoint = holomq::EndpointStream::create();
  ASSERT_NE(endpoint, nullptr);
  const auto lifetime = endpoint->lifetime();
  CUcontext endpoint_context{};
  ASSERT_EQ(cuStreamGetCtx(endpoint->get(), &endpoint_context), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxPushCurrent(endpoint_context), CUDA_SUCCESS);
  struct RestoreContext {
    ~RestoreContext() {
      CUcontext previous{};
      static_cast<void>(cuCtxPopCurrent(&previous));
    }
  } restore;
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto independent = std::make_shared<StreamLifetime>(streams.first, 0);
  auto mapping = holomq::ShmRefCountMapping::create(1U);
  ASSERT_TRUE(mapping.has_value());
  ASSERT_TRUE((*mapping)->table().try_acquire_slot(0U));
  holomq::BufferStreamRelease releases;
  ASSERT_TRUE(releases.add(*mapping, 0U, streams.first, independent));
  // Reserve all CUDA resources before blocking the endpoint. CUDA allocation/context
  // initialization itself can synchronize, which would obscure shared-worker progress.
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(endpoint->get()));
  ASSERT_EQ(cuStreamWaitValue32(endpoint->get(), gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
            CUDA_SUCCESS);
  endpoint.reset();
  releases.release();
  std::atomic_bool drained{};
  std::thread drainer{[&] {
    holomq::EndpointStream::drain_retirements();
    drained.store(true, std::memory_order_release);
  }};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while ((!drained.load(std::memory_order_acquire) || (*mapping)->table().ref_count(0U) != 0U) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool drained_while_blocked = drained.load(std::memory_order_acquire);
  const bool independent_released = (*mapping)->table().ref_count(0U) == 0U;
  EXPECT_EQ(lifetime->state(), StreamLifetime::State::kClosing);
  // Always open the gate before joining, so the old blocking worker fails rather than hangs.
  gate.release();
  drainer.join();
  EXPECT_TRUE(drained_while_blocked);
  EXPECT_TRUE(independent_released);
  const auto retired_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (lifetime->state() != StreamLifetime::State::kRetired &&
         std::chrono::steady_clock::now() < retired_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(lifetime->state(), StreamLifetime::State::kRetired);
  gate.stream = nullptr;  // endpoint destruction owns this handle after its tail callback
  holomq::EndpointStream::drain_retirements();
}

TEST(CudaBufferImportSyncTest, EndpointCreationOnFreshThreadPreservesNullContext) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  std::shared_ptr<const StreamLifetime> lifetime;
  std::thread fresh_thread{[&] {
    CUcontext current{};
    ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
    ASSERT_EQ(current, nullptr);
    auto endpoint = holomq::EndpointStream::create();
    ASSERT_NE(endpoint, nullptr);
    lifetime = endpoint->lifetime();
    EXPECT_EQ(lifetime->device(), 0);
    ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
    EXPECT_EQ(current, nullptr);
  }};
  fresh_thread.join();
  ASSERT_NE(lifetime, nullptr);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (lifetime->state() != StreamLifetime::State::kRetired &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(lifetime->state(), StreamLifetime::State::kRetired);
  holomq::EndpointStream::drain_retirements();
}

TEST(CudaBufferImportSyncTest, EndpointOwnerRetiresOffTheCudaCallbackThread) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  auto owner = holomq::EndpointStream::create();
  ASSERT_TRUE(owner != nullptr);
  const auto lifetime = owner->lifetime();
  auto payload = std::make_unique<std::unique_ptr<holomq::EndpointStream>>(std::move(owner));
  ASSERT_EQ(cuLaunchHostFunc(
                lifetime->handle(),
                [](void* data) {
                  const std::unique_ptr<std::unique_ptr<holomq::EndpointStream>> released{
                      static_cast<std::unique_ptr<holomq::EndpointStream>*>(data)};
                },
                payload.get()),
            CUDA_SUCCESS);
  static_cast<void>(payload.release());
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (lifetime->state() != StreamLifetime::State::kRetired &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(lifetime->state(), StreamLifetime::State::kRetired);
  holomq::EndpointStream::drain_retirements();
}

TEST(CudaBufferImportSyncTest, RetiredBackingIsDestroyedOffTheCudaCallbackThread) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  auto mapping = holomq::ShmRefCountMapping::create(1U);
  ASSERT_TRUE(mapping.has_value());
  ASSERT_TRUE((*mapping)->table().try_acquire_slot(0U));
  std::atomic_bool destroyed{};
  std::atomic_bool destroyed_in_callback{};
  auto payload = std::shared_ptr<int>(new int{23}, [&](int* pointer) {
    destroyed_in_callback.store(in_release_callback);
    delete pointer;
    destroyed.store(true);
  });
  holomq::BufferStreamRelease releases;
  ASSERT_TRUE(releases.add(*mapping, 0U, streams.first, lifetime, {payload, {}}));
  payload.reset();
  lifetime->close();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(lifetime->publish_retired());
  ASSERT_EQ(cuLaunchHostFunc(
                streams.last,
                [](void* data) {
                  in_release_callback = true;
                  static_cast<holomq::BufferStreamRelease*>(data)->release();
                  in_release_callback = false;
                },
                &releases),
            CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
  holomq::EndpointStream::drain_retirements();
  EXPECT_TRUE(destroyed.load());
  EXPECT_FALSE(destroyed_in_callback.load());
  EXPECT_EQ((*mapping)->table().ref_count(0U), 0U);
}

TEST_P(CudaBufferFanoutTest, ImportBackingCleanupAfterCacheAndReleaseGroupDestruction) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto publisher =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(publisher);
  auto pool = GetParam() ? publisher->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : publisher->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool);
  auto owner = pool->acquire();
  ASSERT_TRUE(owner);
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  for (const bool release_lease_first : {false, true}) {
    std::unique_ptr<holomq::ConstBuffer> retained;
    CUdeviceptr address{};
    {
      auto importer = holomq::ContextBuilder().create();
      ASSERT_TRUE(importer);
      auto imported = importer->import_buffer_unsynchronized(*handle);
      ASSERT_TRUE(imported);
      address = buffer_device_ptr(*imported);
      retained = std::make_unique<holomq::ConstBuffer>(std::move(*imported));
      if (release_lease_first) {
        ASSERT_TRUE(holomq::BufferStreamAccess::wait(*retained, streams.first, lifetime));
        retained->release();
        // Complete both preparation and callback cleanup before dropping the cached impl.
        holomq::EndpointStream::drain_retirements();
        ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
        holomq::EndpointStream::drain_retirements();
      }
    }
    // The cache is gone. The active or inactive ConstBuffer impl is the final mapping owner.
    unsigned int memory_type{};
    ASSERT_EQ(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address),
              CUDA_SUCCESS);
    ASSERT_EQ(cuLaunchHostFunc(
                  streams.last,
                  [](void* data) {
                    const std::unique_ptr<holomq::ConstBuffer> final_owner{
                        static_cast<holomq::ConstBuffer*>(data)};
                  },
                  retained.get()),
              CUDA_SUCCESS);
    static_cast<void>(retained.release());
    ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
    holomq::EndpointStream::drain_retirements();
    EXPECT_NE(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address),
              CUDA_SUCCESS);
  }
  cleanup_socket_path(socket_path);
}

// REQ: TENSOR-STORAGE-017
TEST(CudaBufferImportSyncTest, DeferredBackingLastOwnerCleanupRunsOffCallback) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  std::atomic_bool destroyed{};
  std::atomic_bool destroyed_in_callback{};
  auto payload = std::shared_ptr<int>(new int{23}, [&](int* pointer) {
    destroyed_in_callback.store(in_release_callback);
    delete pointer;
    destroyed.store(true);
  });
  // The callback releases `owners` on a driver thread. Stream synchronization orders that release
  // before the test continues, but ThreadSanitizer cannot see it, so the callback also publishes
  // completion through an atomic.
  struct CallbackState {
    holomq::BufferStreamRelease::Owners owners;
    std::atomic_bool released{};
  } callback{.owners = {payload, {}}};
  ASSERT_TRUE(holomq::BufferStreamRelease::defer_cuda_owners(callback.owners, 0));
  payload.reset();
  ASSERT_EQ(cuLaunchHostFunc(
                streams.first,
                [](void* data) {
                  auto& state = *static_cast<CallbackState*>(data);
                  in_release_callback = true;
                  state.owners = {};
                  in_release_callback = false;
                  state.released.store(true, std::memory_order_release);
                },
                &callback),
            CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(callback.released.load(std::memory_order_acquire));
  holomq::EndpointStream::drain_retirements();
  EXPECT_TRUE(destroyed.load());
  EXPECT_FALSE(destroyed_in_callback.load());
}

TEST(CudaBufferImportSyncTest, EndpointOutlivesItsCreatingCudaContext) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto creating_context = cuda_ctx.get();
  auto owner = holomq::EndpointStream::create();
  ASSERT_TRUE(owner != nullptr);
  struct RetireBeforeContext {
    std::unique_ptr<holomq::EndpointStream>& owner;
    ~RetireBeforeContext() {
      owner.reset();
      holomq::EndpointStream::drain_retirements();
    }
  } cleanup{owner};
  CUcontext current{};
  ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
  EXPECT_EQ(current, creating_context) << "factory must restore caller context";
  CUcontext endpoint_context{};
  ASSERT_EQ(cuStreamGetCtx(owner->get(), &endpoint_context), CUDA_SUCCESS);
  ASSERT_NE(endpoint_context, creating_context)
      << "endpoint must not borrow the creating temporary context";
  const auto lifetime = owner->lifetime();
  cuda_ctx = holomq::cuda::Context{};
  ASSERT_EQ(cuCtxPushCurrent(endpoint_context), CUDA_SUCCESS)
      << "endpoint must retain an independent native context";
  EXPECT_EQ(cuStreamSynchronize(owner->get()), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxPopCurrent(&current), CUDA_SUCCESS);
  owner.reset();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (lifetime->state() != StreamLifetime::State::kRetired &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(lifetime->state(), StreamLifetime::State::kRetired);
  holomq::EndpointStream::drain_retirements();
}

TEST_P(CudaBufferFanoutTest, WarmSDKAdmissionDoesNotAllocateAndGrowthFailsClosed) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context.has_value());
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool.has_value());
  auto owner = pool->acquire();
  ASSERT_TRUE(owner.has_value());
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto first = std::make_shared<StreamLifetime>(streams.first, 0);
  auto last = std::make_shared<StreamLifetime>(streams.last, 0);
  ASSERT_TRUE(holomq::BufferStreamAccess::register_consumer(*owner, streams.first, first));
  std::size_t allocation_count{};
  bool warm_ok = true;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (unsigned index = 0U; index < 1000U; ++index) {
      warm_ok = owner->prepare_for_reuse() && warm_ok;
      warm_ok =
          holomq::BufferStreamAccess::register_consumer(*owner, streams.first, first) && warm_ok;
    }
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(warm_ok);
  EXPECT_EQ(allocation_count, 0U);
  bool growth_accepted{};
  bool existing_accepted{};
  {
    const holoscan::test::ScopedAllocationExhaustion allocations{0U};
    growth_accepted = holomq::BufferStreamAccess::register_consumer(*owner, streams.last, last);
    existing_accepted = holomq::BufferStreamAccess::register_consumer(*owner, streams.first, first);
  }
  EXPECT_FALSE(growth_accepted);
  EXPECT_TRUE(existing_accepted);
  first->close();
  last->close();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
  ASSERT_TRUE(first->publish_retired());
  ASSERT_TRUE(last->publish_retired());
  owner = holomq::Buffer{};
  holomq::EndpointStream::drain_retirements();
  EXPECT_TRUE(pool->acquire().has_value());
  cleanup_socket_path(socket_path);
}

// A retired SDK identity cannot stand in for a caller stream with the same native handle.
TEST_P(CudaBufferFanoutTest, RetiredIdentityDoesNotHideAReusedCallerHandle) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  auto cuda_ctx = make_cuda_context();
  ASSERT_TRUE(cuda_ctx);
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context);
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool);
  auto owner = pool->acquire();
  ASSERT_TRUE(owner);
  auto handle = owner->export_handle_unsynchronized();
  ASSERT_TRUE(handle);
  auto view = context->import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(view);
  owner = holomq::Buffer{};
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  auto lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.first, lifetime));
  lifetime->close();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  ASSERT_TRUE(lifetime->publish_retired());
  // Model native handle reuse without depending on a driver's handle allocator.
  ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.first, {}));
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(streams.first));
  ASSERT_EQ(cuStreamWaitValue32(streams.first, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
            CUDA_SUCCESS);
  view->release();
  std::optional<holomq::Buffer> early;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{30};
  while (!early && std::chrono::steady_clock::now() < deadline) {
    auto candidate = pool->acquire();
    if (candidate) {
      early.emplace(std::move(*candidate));
    }
    std::this_thread::yield();
  }
  EXPECT_FALSE(early) << "the Retired SDK registration hid pending caller-stream work";
  gate.release();
  ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
  early.reset();
  EXPECT_TRUE(wait_for_released_slot(*pool));
  cleanup_socket_path(socket_path);
}

// Fresh, replayable seeds mix the gated stream, lifetime closure and final-release thread.
// REQ: TENSOR-STORAGE-012
TEST_P(CudaBufferFanoutTest, RandomizedReleaseAndClosingNeverReuseAnUnreadSlot) {
  skip_unless_cuda_available();
  if (testing::Test::IsSkipped()) {
    return;
  }
  // Match SDK graphs and CuPy: use the primary context for readers and retirement.
  CUdevice device{};
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  CUcontext previous{};
  ASSERT_EQ(cuCtxGetCurrent(&previous), CUDA_SUCCESS);
  CUcontext primary{};
  ASSERT_EQ(cuDevicePrimaryCtxRetain(&primary, device), CUDA_SUCCESS);
  struct RestorePrimary {
    CUdevice device;
    CUcontext previous;
    ~RestorePrimary() {
      static_cast<void>(cuCtxSetCurrent(previous));
      static_cast<void>(cuDevicePrimaryCtxRelease(device));
    }
  } const restore{device, previous};
  ASSERT_EQ(cuCtxSetCurrent(primary), CUDA_SUCCESS);
  const auto socket_path = make_socket_path();
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  ASSERT_TRUE(context);
  auto pool = GetParam() ? context->create_host_pinned_buffer_pool(
                               kBufferBytes, 1U, CU_MEMHOSTREGISTER_DEVICEMAP)
                         : context->create_device_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool);
  ConsumerStreams streams;
  ASSERT_EQ(cuStreamCreate(&streams.first, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&streams.last, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  CUdeviceptr device_output{};
  ASSERT_EQ(cuMemAlloc(&device_output, kBufferBytes * 2U), CUDA_SUCCESS);
  void* output{};
  ASSERT_EQ(cuMemHostAlloc(&output, kBufferBytes * 2U, CU_MEMHOSTALLOC_PORTABLE), CUDA_SUCCESS);
  struct OutputCleanup {
    void* output;
    CUdeviceptr device_output;
    ~OutputCleanup() {
      static_cast<void>(cuMemFreeHost(output));
      static_cast<void>(cuMemFree(device_output));
    }
  } const output_cleanup{output, device_output};
  ReleaseGate gate;
  ASSERT_TRUE(gate.create(streams.first));
  // Failure-only stall report. Every step should finish within milliseconds (the released-slot
  // wait within two seconds). A step that makes no progress for ten seconds is named, with its
  // frame's choices and both streams' state, before the CTest timeout, and the gate is opened so a
  // parked stream cannot keep the test hung.
  struct Progress {
    std::atomic<const char*> step{"start"};
    std::atomic<std::uint64_t> ticks{};
    std::atomic<std::uint32_t> seed{};
    std::atomic<std::uint32_t> frame{};
    std::atomic<bool> gate_first{};
    std::atomic<bool> closing{};
    std::atomic<bool> closing_first{};
    std::atomic<bool> callback_release{};
    std::atomic<bool> coupled{};
    std::atomic<bool> stalled{};
    std::atomic<bool> gate_forced{};
    void at(const char* name) noexcept {
      step.store(name, std::memory_order_release);
      ticks.fetch_add(1U, std::memory_order_acq_rel);
    }
  } progress;
  // Each reader's copies are followed by an event, so the test can wait for the reads themselves
  // rather than for everything later queued on the reader's stream.
  struct ReadEvents {
    std::array<CUevent, 2U> done{};
    ~ReadEvents() {
      for (const auto event : done) {
        if (event != nullptr) {
          static_cast<void>(cuEventDestroy(event));
        }
      }
    }
  } reads;
  for (auto& event : reads.done) {
    ASSERT_EQ(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), CUDA_SUCCESS);
  }
  const auto completes_within = [](const auto& query, const std::chrono::milliseconds bound) {
    const auto deadline = std::chrono::steady_clock::now() + bound;
    CUresult status = query();
    while (status == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds{100});
      status = query();
    }
    return status;
  };
  const std::jthread watchdog([&](const std::stop_token& stop) {
    static_cast<void>(cuCtxSetCurrent(primary));
    constexpr auto kStall = std::chrono::seconds{10};
    auto ticks = progress.ticks.load(std::memory_order_acquire);
    auto since = std::chrono::steady_clock::now();
    bool reported{};
    while (!stop.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
      const auto now = std::chrono::steady_clock::now();
      if (const auto current = progress.ticks.load(std::memory_order_acquire); current != ticks) {
        ticks = current;
        since = now;
        reported = false;
        continue;
      }
      if (reported || now - since < kStall) {
        continue;
      }
      reported = true;
      progress.stalled.store(true, std::memory_order_release);
      const auto name_of = [](CUresult result) {
        const char* name = nullptr;
        static_cast<void>(cuGetErrorName(result, &name));
        return std::string{name != nullptr ? name : "?"};
      };
      std::cerr << "retirement stress stalled for " << kStall.count() << " s at step '"
                << progress.step.load(std::memory_order_acquire)
                << "' seed=" << progress.seed.load() << " frame=" << progress.frame.load()
                << " gate_first=" << progress.gate_first.load()
                << " closing=" << progress.closing.load()
                << " closing_first=" << progress.closing_first.load()
                << " callback_release=" << progress.callback_release.load()
                << " coupled=" << progress.coupled.load()
                << " first=" << name_of(cuStreamQuery(streams.first))
                << " first_reads=" << name_of(cuEventQuery(reads.done[0]))
                << " last=" << name_of(cuStreamQuery(streams.last))
                << " last_reads=" << name_of(cuEventQuery(reads.done[1]))
                << " gate_word=" << std::atomic_ref<std::uint32_t>(*gate.word).load()
                << "; opening the gate" << std::endl;
      progress.gate_forced.store(true, std::memory_order_release);
      gate.release();
    }
  });
  // The driver maps streams onto a limited set of hardware work queues (see
  // CUDA_DEVICE_MAX_CONNECTIONS), and work queued behind a parked stream wait cannot start until
  // the wait is released. If the two reader streams share a queue, the fast reader cannot finish
  // while the slow one is gated, so each frame then checks retention with both reads pending
  // instead of waiting for the fast reader. The mapping is fixed when the streams are created.
  progress.at("shared-queue probe");
  bool shared_queue{};
  {
    CUevent probe{};
    ASSERT_EQ(cuEventCreate(&probe, CU_EVENT_DISABLE_TIMING), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamWaitValue32(streams.first, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
              CUDA_SUCCESS);
    // Queue real GPU work: an event recorded on an idle stream can complete without the queue.
    ASSERT_EQ(cuMemsetD8Async(device_output, 0U, 1U, streams.last), CUDA_SUCCESS);
    ASSERT_EQ(cuEventRecord(probe, streams.last), CUDA_SUCCESS);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (cuEventQuery(probe) == CUDA_ERROR_NOT_READY &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    shared_queue = cuEventQuery(probe) == CUDA_ERROR_NOT_READY;
    gate.release();
    ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(streams.last), CUDA_SUCCESS);
    ASSERT_EQ(cuEventDestroy(probe), CUDA_SUCCESS);
  }
  if (shared_queue) {
    std::cout << "CUDA reader streams share a hardware queue; frames check retention with both "
                 "reads pending"
              << std::endl;
  }
  progress.coupled.store(shared_queue);
  // Frames that reached the state under test: the fast reader's stream complete while the slow
  // reader is still gated.
  std::uint32_t isolated_frames{};
  // Keep the imported mapping cached without holding a slot reference. This also
  // makes premature reuse a payload-poison failure instead of an unmapped GPU read.
  std::optional<holomq::ConstBuffer> cached_mapping;
  std::random_device entropy;
  std::uint32_t base_seed = entropy();
  if (const char* override_seed = std::getenv("HOLOMQ_CUDA_RETIREMENT_SEED")) {
    const std::string text{override_seed};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), base_seed);
    ASSERT_EQ(static_cast<int>(parsed.ec), 0);
    ASSERT_EQ(parsed.ptr, text.data() + text.size());
  }
  for (std::uint32_t run = 0U; run < 3U; ++run) {
    const auto seed = base_seed + run * 0x9e3779b9U;
    std::cout << "CUDA retirement seed: " << seed
              << " (replay: HOLOMQ_CUDA_RETIREMENT_SEED=" << seed << ")" << std::endl;
    SCOPED_TRACE("seed=" + std::to_string(seed));
    std::mt19937 random{seed};
    progress.seed.store(seed);
    for (std::uint32_t frame = 1U; frame <= 32U; ++frame) {
      SCOPED_TRACE(frame);
      progress.frame.store(frame);
      progress.at("frame fill");
      auto owner = pool->acquire();
      ASSERT_TRUE(owner);
      // Unsynchronized export requires initialization to finish before either non-blocking
      // reader starts. A default-stream device memset need not complete before returning.
      ASSERT_EQ(cuMemsetD8Async(buffer_device_ptr(*owner),
                                static_cast<unsigned char>(frame),
                                kBufferBytes,
                                streams.first),
                CUDA_SUCCESS)
          << describe_device_address(owner->ptr());
      ASSERT_EQ(cuStreamSynchronize(streams.first), CUDA_SUCCESS);
      progress.at("export and import");
      auto handle = owner->export_handle_unsynchronized();
      ASSERT_TRUE(handle);
      if (!cached_mapping) {
        auto cached = context->import_buffer_unsynchronized(*handle);
        ASSERT_TRUE(cached);
        cached_mapping.emplace(std::move(*cached));
        cached_mapping->release();
      }
      auto view = context->import_buffer_unsynchronized(*handle);
      ASSERT_TRUE(view);
      owner = holomq::Buffer{};
      const bool gate_first = (random() & 1U) != 0U;
      const auto slow = gate_first ? streams.first : streams.last;
      const auto fast = gate_first ? streams.last : streams.first;
      auto first_lifetime = std::make_shared<StreamLifetime>(streams.first, 0);
      auto last_lifetime = std::make_shared<StreamLifetime>(streams.last, 0);
      auto closing_lifetime = (random() & 1U) != 0U ? first_lifetime : last_lifetime;
      const bool closing = (random() & 1U) != 0U;
      const bool callback_release = (random() & 1U) != 0U;
      progress.gate_first.store(gate_first);
      progress.closing.store(closing);
      progress.closing_first.store(closing_lifetime == first_lifetime);
      progress.callback_release.store(callback_release);
      progress.at("reader registration");
      ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.first, first_lifetime));
      ASSERT_TRUE(holomq::BufferStreamAccess::wait(*view, streams.last, last_lifetime));
      progress.at("gate and reader copies");
      gate.stream = slow;
      std::atomic_ref<std::uint32_t>(*gate.word).store(0U, std::memory_order_release);
      ASSERT_EQ(cuStreamWaitValue32(slow, gate.device_word, 1U, CU_STREAM_WAIT_VALUE_EQ),
                CUDA_SUCCESS);
      for (const auto stream : {streams.first, streams.last}) {
        const auto offset = stream == streams.first ? 0U : kBufferBytes;
        ASSERT_EQ(
            cuMemcpyAsync(device_output + offset, buffer_device_ptr(*view), kBufferBytes, stream),
            CUDA_SUCCESS);
        ASSERT_EQ(cuMemcpyDtoHAsync(static_cast<std::uint8_t*>(output) + offset,
                                    device_output + offset,
                                    kBufferBytes,
                                    stream),
                  CUDA_SUCCESS);
        ASSERT_EQ(cuEventRecord(reads.done[stream == streams.first ? 0U : 1U], stream),
                  CUDA_SUCCESS);
      }
      progress.at(callback_release ? "callback release" : "finalizer release");
      if (callback_release) {
        auto* held = new holomq::ConstBuffer(std::move(*view));
        ASSERT_EQ(cuLaunchHostFunc(
                      fast,
                      [](void* pointer) { delete static_cast<holomq::ConstBuffer*>(pointer); },
                      held),
                  CUDA_SUCCESS);
        if (closing) {
          closing_lifetime->close();
        }
      } else {
        if (closing) {
          closing_lifetime->close();
        }
        std::jthread finalizer([lease = std::move(*view)]() mutable { lease.release(); });
      }
      progress.at("fast reader completion");
      if (!progress.coupled.load()) {
        // The state under test needs the fast reader's whole stream complete, including the
        // release work queued after its reads, while the slow reader stays gated. Poll the stream
        // rather than synchronize it: on some platforms cuStreamSynchronize can wait for the gated
        // stream even after this one has completed. If the stream cannot complete while the slow
        // reader is gated, the readers are coupled on this platform and the state is unreachable.
        const CUresult status =
            completes_within([&] { return cuStreamQuery(fast); }, std::chrono::seconds{2});
        if (status == CUDA_ERROR_NOT_READY) {
          progress.coupled.store(true);
          const bool reads_finished =
              cuEventQuery(reads.done[fast == streams.first ? 0U : 1U]) == CUDA_SUCCESS;
          std::cout << "the fast CUDA reader's stream cannot complete while the slow reader is "
                       "gated (its reads "
                    << (reads_finished ? "finished" : "did not finish")
                    << "); remaining frames check retention with both readers pending" << std::endl;
        } else {
          ASSERT_EQ(status, CUDA_SUCCESS);
          ++isolated_frames;
        }
      }
      progress.at("early-reuse probe");
      std::this_thread::sleep_for(std::chrono::microseconds{random() % 200U});
      // Poll throughout the pending-read window; poison a wrongly returned slot before its read.
      auto early = pool->acquire();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{3};
      while (!early && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        early = pool->acquire();
      }
      if (early) {
        ASSERT_EQ(cuMemsetD8Async(buffer_device_ptr(*early), 0xe5U, kBufferBytes, fast),
                  CUDA_SUCCESS);
        static_cast<void>(
            completes_within([&] { return cuStreamQuery(fast); }, std::chrono::seconds{2}));
      }
      progress.at("slow reader synchronize");
      gate.release();
      ASSERT_EQ(cuStreamSynchronize(slow), CUDA_SUCCESS);
      progress.at("reader synchronize after gate");
      ASSERT_EQ(cuStreamSynchronize(fast), CUDA_SUCCESS);
      if (closing) {
        ASSERT_TRUE(closing_lifetime->publish_retired());
      }
      // A gate the watchdog opened frees the slot legitimately; that stall is reported.
      if (!progress.gate_forced.exchange(false)) {
        EXPECT_FALSE(early) << "slot was returned while its CUDA reader was gated";
      }
      early = holomq::Buffer{};
      progress.at("released-slot wait");
      auto recycled = wait_for_released_slot(*pool);
      ASSERT_TRUE(recycled);
      for (std::size_t i = 0U; i < kBufferBytes * 2U; ++i) {
        ASSERT_EQ(static_cast<std::uint8_t*>(output)[i], frame);
      }
    }
  }
  EXPECT_FALSE(progress.stalled.load(std::memory_order_acquire))
      << "a step stalled; see the stall report above";
  cleanup_socket_path(socket_path);
  if (isolated_frames == 0U) {
    // Retention with both readers pending was checked, but not that every consumer must finish.
    GTEST_SKIP() << "no frame completed the fast reader while the slow reader was gated, so the "
                    "every-consumer retention case was not exercised on this platform";
  }
}

INSTANTIATE_TEST_SUITE_P(DeviceAndPinned, CudaBufferFanoutTest, testing::Bool());
