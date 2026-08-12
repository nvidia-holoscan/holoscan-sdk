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
#include "cuda/context.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::uint32_t kBufferCount = 4U;
constexpr std::size_t kBufferBytes = 256U;
constexpr std::uint32_t kPatternSeed = 0xC0FF0000U;

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

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::ConstBuffer& buffer) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(buffer.ptr()));
}

[[nodiscard]] CUdeviceptr buffer_device_ptr(const holomq::Buffer& buffer) noexcept {
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

  EXPECT_FALSE(buffer.enqueue_ready_set(nullptr));
  EXPECT_FALSE(buffer.export_handle(nullptr).has_value());
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredOwnerReleaseCanRemainGenerationFree) {
  skip_unless_cuda_available();

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
  ASSERT_TRUE(buffer.export_handle(pub_stream).has_value());

  // Drop the owner buffer without synchronizing pub_stream. Deferred ref release keeps the SHM
  // ref-count elevated until prior stream work and the host function complete.
  buffer = holomq::Buffer{};

  ASSERT_EQ(cuStreamQuery(pub_stream), CUDA_ERROR_NOT_READY)
      << "deferred ref release must be pending on pub_stream after buffer drop";

  EXPECT_FALSE(pool.acquire().has_value())
      << "slot must stay leased until deferred release runs on pub_stream";

  ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredRefReleaseSurvivesPublisherTeardownBeforeStreamSync) {
  skip_unless_cuda_available();

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

  ASSERT_TRUE(buffer.export_handle(pub_stream).has_value());

  // Drop the owner buffer while the publisher pool/context remain alive; only the deferred
  // ref-release host function should stay pending on pub_stream.
  buffer = holomq::Buffer{};

  ASSERT_EQ(cuStreamQuery(pub_stream), CUDA_ERROR_NOT_READY)
      << "deferred ref release must be pending on pub_stream after buffer drop";

  EXPECT_FALSE(pool.acquire().has_value())
      << "slot must stay leased until deferred release runs on pub_stream";

  ASSERT_EQ(cuStreamSynchronize(pub_stream), CUDA_SUCCESS);

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(pub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaBufferImportSyncTest, DeferredRefReleaseSurvivesSubscriberTeardownBeforeStreamSync) {
  skip_unless_cuda_available();

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

  std::vector<std::uint32_t> host_scratch(view.size_bytes() / sizeof(std::uint32_t));
  ASSERT_EQ(cuMemcpyDtoHAsync(
                host_scratch.data(), buffer_device_ptr(view), view.size_bytes(), sub_stream),
            CUDA_SUCCESS);

  EXPECT_FALSE(pool.acquire().has_value())
      << "publisher slot must stay leased while subscriber holds imported view";

  // Drop the imported view while subscriber context/stream remain alive; only the deferred
  // ref-release host function should stay pending on sub_stream.
  view = holomq::ConstBuffer{};

  ASSERT_EQ(cuStreamQuery(sub_stream), CUDA_ERROR_NOT_READY)
      << "deferred ref release must be pending on sub_stream after view drop";

  EXPECT_FALSE(pool.acquire().has_value())
      << "slot must stay leased until deferred subscriber release runs on sub_stream";

  ASSERT_EQ(cuStreamSynchronize(sub_stream), CUDA_SUCCESS);

  auto reacquire_or = pool.acquire();
  ASSERT_TRUE(reacquire_or.has_value()) << "slot should be reusable after stream drain";
  static_cast<void>(std::move(reacquire_or).value());

  ASSERT_EQ(cuStreamDestroy(sub_stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}
