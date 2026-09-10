/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/buffer.hpp>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/host_buffer_import.hpp>
#include "cuda/context.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kDefaultBufferCount = 4U;
constexpr std::size_t kBufferBytes = 256U;
constexpr std::uint32_t kPatternSeed = 0xA5A50000U;

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
  std::string path = "/tmp/holomq_host_" + std::to_string(static_cast<long>(::getpid())) + "_" +
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

[[nodiscard]] bool write_host_pattern(void* ptr, const std::size_t size_bytes,
                                      const std::uint32_t seed) {
  auto* words = static_cast<std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t i = 0; i < word_count; ++i) {
    words[i] = seed + static_cast<std::uint32_t>(i);
  }
  return true;
}

[[nodiscard]] bool verify_host_pattern(const void* ptr, const std::size_t size_bytes,
                                       const std::uint32_t seed) {
  const auto* words = static_cast<const std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t i = 0; i < word_count; ++i) {
    if (words[i] != seed + static_cast<std::uint32_t>(i)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] auto make_context(const std::string& socket_path, const bool vmm_fd_export) {
  holomq::ContextBuilder builder;
  if (vmm_fd_export) {
    builder.with_vmm_fd_export(true).with_fd_pass_socket(socket_path);
  }
  return builder.create();
}

constexpr unsigned int kDevicemapInterleavedHostRegisterFlags =
    CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
constexpr unsigned int kSplitGenerationVMMHostRegisterFlags = CU_MEMHOSTREGISTER_PORTABLE;

struct HostPoolCrossProcessConfig {
  bool vmm_fd_export{false};
  bool use_registered_host_pool{true};
  unsigned int host_register_flags{0U};
};

enum class InvalidHostPoolGeometry : std::uint8_t {
  kPayload,
  kGeneration,
};

void expect_host_import_rejects_out_of_range_slot(const InvalidHostPoolGeometry invalid_geometry) {
  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_buffer_pool(kBufferBytes, kDefaultBufferCount);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  constexpr std::uint32_t kOutOfRangeSlotIndex = 2U;
  std::vector<holomq::Buffer> buffers;
  for (std::uint32_t index = 0U; index <= kOutOfRangeSlotIndex; ++index) {
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value());
    buffers.emplace_back(std::move(buffer_or).value());
  }

  auto handle_or = buffers.back().export_handle_unsynchronized();
  ASSERT_TRUE(handle_or.has_value());
  holomq::BufferHandle malformed_handle = std::move(handle_or).value();
  ASSERT_EQ(holomq::detail::BufferHandleWireAccess::slot_index(malformed_handle),
            kOutOfRangeSlotIndex);

  auto& handle_data = const_cast<holomq::detail::host::BufferHandleData&>(
      holomq::detail::BufferHandleWireAccess::host_wire(malformed_handle));
  if (invalid_geometry == InvalidHostPoolGeometry::kPayload) {
    handle_data.pool_total_bytes =
        handle_data.slot_stride_bytes * static_cast<std::size_t>(kOutOfRangeSlotIndex);
  } else {
    handle_data.gen_pool_total_bytes =
        handle_data.gen_slot_stride_bytes * static_cast<std::size_t>(kOutOfRangeSlotIndex);
  }

  ASSERT_TRUE(holomq::detail::is_valid_host_buffer_wire(handle_data));
  EXPECT_FALSE(
      holomq::detail::is_valid_host_buffer_slot_geometry(handle_data, kOutOfRangeSlotIndex));
  EXPECT_FALSE(context.import_buffer_unsynchronized(malformed_handle).has_value());

  cleanup_socket_path(socket_path);
}

[[nodiscard]] bool read_exact(const int fd, void* const buffer, const std::size_t length) {
  auto* out = static_cast<std::uint8_t*>(buffer);
  std::size_t done = 0U;
  while (done < length) {
    const ssize_t n = ::read(fd, out + done, length - done);
    if (n <= 0) {
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

[[nodiscard]] bool write_exact(const int fd, const void* const buffer, const std::size_t length) {
  const auto* in = static_cast<const std::uint8_t*>(buffer);
  std::size_t done = 0U;
  while (done < length) {
    const ssize_t n = ::write(fd, in + done, length - done);
    if (n <= 0) {
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  return true;
}

#ifndef HOLOMQ_CUDA_HOST_IMPORT_CHILD_EXE
#define HOLOMQ_CUDA_HOST_IMPORT_CHILD_EXE ""
#endif

void run_host_pool_cross_process_import_test(const HostPoolCrossProcessConfig& config) {
  const char* const import_child_exe = HOLOMQ_CUDA_HOST_IMPORT_CHILD_EXE;
  ASSERT_NE(import_child_exe[0], '\0') << "HOLOMQ_CUDA_HOST_IMPORT_CHILD_EXE not configured";

  int handle_pipe[2]{};
  int result_pipe[2]{};
  ASSERT_EQ(::pipe(handle_pipe), 0);
  ASSERT_EQ(::pipe(result_pipe), 0);

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(handle_pipe[1]);
    ::close(result_pipe[0]);

    const std::string handle_fd_str = std::to_string(handle_pipe[0]);
    const std::string result_fd_str = std::to_string(result_pipe[1]);
    static_cast<void>(::setenv("HOLOMQ_IMPORT_HANDLE_FD", handle_fd_str.c_str(), 1));
    static_cast<void>(::setenv("HOLOMQ_IMPORT_RESULT_FD", result_fd_str.c_str(), 1));
    static_cast<void>(::setenv("HOLOMQ_IMPORT_VMM_FD_EXPORT", config.vmm_fd_export ? "1" : "0", 1));

    ::execl(import_child_exe, "holomq_cuda_host_import_child", static_cast<char*>(nullptr));
    _exit(127);
  }

  ::close(handle_pipe[0]);
  ::close(result_pipe[1]);

  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, config.vmm_fd_export);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = config.use_registered_host_pool
                     ? context.create_host_pinned_buffer_pool(
                           kBufferBytes, kDefaultBufferCount, config.host_register_flags)
                     : context.create_host_buffer_pool(kBufferBytes, kDefaultBufferCount);
  ASSERT_TRUE(pool_or.has_value());

  holomq::BufferPool pool = std::move(pool_or).value();
  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  EXPECT_EQ(buffer.memory_type(), pool.memory_type());
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kPatternSeed));

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  const auto exported_or = buffer.export_handle(stream);
  ASSERT_TRUE(exported_or.has_value());
  EXPECT_FALSE(context.import_buffer(exported_or.value(), nullptr).has_value());

  const holomq::BufferHandle handle = exported_or.value();
  EXPECT_EQ(handle.memory_type(), buffer.memory_type());
  ASSERT_TRUE(write_exact(handle_pipe[1], &handle, sizeof(handle)));
  ::close(handle_pipe[1]);

  std::uint8_t ok = 0U;
  ASSERT_TRUE(read_exact(result_pipe[0], &ok, 1U));
  ::close(result_pipe[0]);

  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  ASSERT_EQ(ok, 1U) << "child stage/error code " << static_cast<unsigned>(ok);

  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  buffer = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);

  if (config.vmm_fd_export) {
    cleanup_socket_path(socket_path);
  }
}

}  // namespace

TEST(CudaHostPoolTest, DevicemapInterleavedOwnerAcquireRelease) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, false);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_pinned_buffer_pool(
      kBufferBytes, kDefaultBufferCount, kDevicemapInterleavedHostRegisterFlags);
  ASSERT_TRUE(pool_or.has_value());

  holomq::BufferPool pool = std::move(pool_or).value();
  EXPECT_EQ(pool.memory_type(), holomq::BufferMemoryType::HostPinned);
  EXPECT_FALSE(pool.cuda_device_ordinal().has_value());

  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  EXPECT_EQ(buffer.memory_type(), holomq::BufferMemoryType::HostPinned);
  EXPECT_FALSE(buffer.cuda_device_ordinal().has_value());
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kPatternSeed));
  ASSERT_TRUE(verify_host_pattern(buffer.ptr(), buffer.size_bytes(), kPatternSeed));
}

TEST(CudaHostPoolTest, PageableHostPoolMemoryType) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, false);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_buffer_pool(kBufferBytes, kDefaultBufferCount);
  ASSERT_TRUE(pool_or.has_value());

  holomq::BufferPool pool = std::move(pool_or).value();
  EXPECT_EQ(pool.memory_type(), holomq::BufferMemoryType::HostPageable);
  EXPECT_FALSE(pool.cuda_device_ordinal().has_value());

  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  EXPECT_EQ(buffer.memory_type(), holomq::BufferMemoryType::HostPageable);
  EXPECT_FALSE(buffer.cuda_device_ordinal().has_value());
}

TEST(CudaHostPoolTest, DevicemapInterleavedCrossProcessImport) {
  run_host_pool_cross_process_import_test(
      HostPoolCrossProcessConfig{.vmm_fd_export = false,
                                 .use_registered_host_pool = true,
                                 .host_register_flags = kDevicemapInterleavedHostRegisterFlags});
}

TEST(CudaHostPoolTest, SplitGenerationVmmExportImportWithFdPass) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_pinned_buffer_pool(
      kBufferBytes, kDefaultBufferCount, kSplitGenerationVMMHostRegisterFlags);
  ASSERT_TRUE(pool_or.has_value()) << "host pool create failed";

  holomq::BufferPool pool = std::move(pool_or).value();
  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kPatternSeed));

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  const auto exported_or = buffer.export_handle(stream);
  ASSERT_TRUE(exported_or.has_value());
  EXPECT_FALSE(context.import_buffer(exported_or.value(), nullptr).has_value());

  auto view_or = context.import_buffer(exported_or.value(), stream);
  ASSERT_TRUE(view_or.has_value());
  holomq::ConstBuffer view = std::move(view_or).value();
  EXPECT_EQ(view.memory_type(), holomq::BufferMemoryType::HostPinned);
  EXPECT_FALSE(view.cuda_device_ordinal().has_value());
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_TRUE(verify_host_pattern(view.ptr(), view.size_bytes(), kPatternSeed));

  view = holomq::ConstBuffer{};
  buffer = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

TEST(CudaHostPoolTest, PageableSplitGenerationVmmExportImportWithFdPass) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_buffer_pool(kBufferBytes, kDefaultBufferCount);
  ASSERT_TRUE(pool_or.has_value()) << "pageable host pool create failed";

  holomq::BufferPool pool = std::move(pool_or).value();
  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), kPatternSeed));

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  const auto exported_or = buffer.export_handle(stream);
  ASSERT_TRUE(exported_or.has_value());
  EXPECT_FALSE(context.import_buffer(exported_or.value(), nullptr).has_value());

  auto view_or = context.import_buffer(exported_or.value(), stream);
  ASSERT_TRUE(view_or.has_value());
  holomq::ConstBuffer view = std::move(view_or).value();
  EXPECT_EQ(view.memory_type(), holomq::BufferMemoryType::HostPageable);
  EXPECT_FALSE(view.cuda_device_ordinal().has_value());
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_TRUE(verify_host_pattern(view.ptr(), view.size_bytes(), kPatternSeed));

  view = holomq::ConstBuffer{};
  buffer = holomq::Buffer{};
  ASSERT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
  cleanup_socket_path(socket_path);
}

// A reusable import drops each remote lease while retaining the exact slot mapping for the next
// producer generation.
TEST(CudaHostPoolTest, ReusablePageableImportReacquiresTheSamePoolSlot) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();
  holomq::HostBufferImport imported;

  for (std::uint32_t cycle = 0U; cycle < 3U; ++cycle) {
    auto buffer_or = pool.acquire();
    ASSERT_TRUE(buffer_or.has_value());
    holomq::Buffer buffer = std::move(buffer_or).value();
    const std::uint32_t seed = kPatternSeed + cycle * 0x100U;
    ASSERT_TRUE(write_host_pattern(buffer.ptr(), buffer.size_bytes(), seed));

    auto handle = buffer.export_handle_unsynchronized();
    ASSERT_TRUE(handle.has_value());
    ASSERT_TRUE(context.import_host_buffer_unsynchronized(*handle, imported).has_value());
    EXPECT_TRUE(static_cast<bool>(imported));
    EXPECT_EQ(imported.memory_type(), holomq::BufferMemoryType::HostPageable);
    EXPECT_EQ(imported.size_bytes(), kBufferBytes);
    EXPECT_TRUE(verify_host_pattern(imported.ptr(), imported.size_bytes(), seed));
    EXPECT_FALSE(context.import_host_buffer_unsynchronized(*handle, imported).has_value());

    imported.release();
    EXPECT_FALSE(static_cast<bool>(imported));
    EXPECT_EQ(imported.memory_type(), holomq::BufferMemoryType::None);
    EXPECT_EQ(imported.size_bytes(), 0U);
    EXPECT_EQ(imported.ptr(), nullptr);
  }

  cleanup_socket_path(socket_path);
}

// The table overload routes opaque handle slot identities without a runtime-side scan.
TEST(CudaHostPoolTest, ReusableImportRoutesIntoAPreallocatedSlotTable) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();

  auto pool_or = context.create_host_buffer_pool(kBufferBytes, 2U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  auto first_buffer_or = pool.acquire();
  auto second_buffer_or = pool.acquire();
  ASSERT_TRUE(first_buffer_or.has_value());
  ASSERT_TRUE(second_buffer_or.has_value());
  holomq::Buffer first_buffer = std::move(first_buffer_or).value();
  holomq::Buffer second_buffer = std::move(second_buffer_or).value();
  ASSERT_TRUE(write_host_pattern(first_buffer.ptr(), first_buffer.size_bytes(), kPatternSeed));
  ASSERT_TRUE(
      write_host_pattern(second_buffer.ptr(), second_buffer.size_bytes(), kPatternSeed + 1U));
  auto first_handle = first_buffer.export_handle_unsynchronized();
  auto second_handle = second_buffer.export_handle_unsynchronized();
  ASSERT_TRUE(first_handle.has_value());
  ASSERT_TRUE(second_handle.has_value());

  std::array<holomq::HostBufferImport, 2U> imports;
  auto first_index = context.import_host_buffer_unsynchronized(*first_handle, imports);
  auto second_index = context.import_host_buffer_unsynchronized(*second_handle, imports);
  ASSERT_TRUE(first_index.has_value());
  ASSERT_TRUE(second_index.has_value());
  EXPECT_NE(*first_index, *second_index);
  EXPECT_TRUE(static_cast<bool>(imports[*first_index]));
  EXPECT_TRUE(static_cast<bool>(imports[*second_index]));

  imports[*first_index].release();
  imports[*second_index].release();
  cleanup_socket_path(socket_path);
}

// A bounded borrow control can retarget among slots while retaining one validated pool mapping.
TEST(CudaHostPoolTest, ReusableImportRebindsWithinTheSamePool) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();
  auto pool_or = context.create_host_buffer_pool(kBufferBytes, 2U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  auto first_buffer_or = pool.acquire();
  auto second_buffer_or = pool.acquire();
  ASSERT_TRUE(first_buffer_or.has_value());
  ASSERT_TRUE(second_buffer_or.has_value());
  holomq::Buffer first_buffer = std::move(first_buffer_or).value();
  holomq::Buffer second_buffer = std::move(second_buffer_or).value();
  ASSERT_TRUE(write_host_pattern(first_buffer.ptr(), first_buffer.size_bytes(), kPatternSeed));
  ASSERT_TRUE(
      write_host_pattern(second_buffer.ptr(), second_buffer.size_bytes(), kPatternSeed + 1U));
  auto first_handle = first_buffer.export_handle_unsynchronized();
  auto second_handle = second_buffer.export_handle_unsynchronized();
  ASSERT_TRUE(first_handle.has_value());
  ASSERT_TRUE(second_handle.has_value());

  holomq::HostBufferImport imported;
  ASSERT_TRUE(context.import_host_buffer_unsynchronized(*first_handle, imported).has_value());
  EXPECT_TRUE(verify_host_pattern(imported.ptr(), imported.size_bytes(), kPatternSeed));
  imported.release();
  EXPECT_FALSE(context.import_host_buffer_unsynchronized(*second_handle, imported).has_value());
  ASSERT_TRUE(context.rebind_host_buffer_unsynchronized(*second_handle, imported).has_value());
  EXPECT_TRUE(verify_host_pattern(imported.ptr(), imported.size_bytes(), kPatternSeed + 1U));
  imported.release();

  cleanup_socket_path(socket_path);
}

// A reusable generic import releases its current lease while retaining the validated pool mapping.
TEST(CudaHostPoolTest, GenericImportRebindsWithinTheSamePool) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();
  auto pool_or = context.create_host_buffer_pool(kBufferBytes, 2U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();

  auto first_buffer_or = pool.acquire();
  auto second_buffer_or = pool.acquire();
  ASSERT_TRUE(first_buffer_or.has_value());
  ASSERT_TRUE(second_buffer_or.has_value());
  holomq::Buffer first_buffer = std::move(first_buffer_or).value();
  holomq::Buffer second_buffer = std::move(second_buffer_or).value();
  ASSERT_TRUE(write_host_pattern(first_buffer.ptr(), first_buffer.size_bytes(), kPatternSeed));
  ASSERT_TRUE(
      write_host_pattern(second_buffer.ptr(), second_buffer.size_bytes(), kPatternSeed + 1U));
  const auto first_handle = first_buffer.export_handle_unsynchronized();
  const auto second_handle = second_buffer.export_handle_unsynchronized();
  ASSERT_TRUE(first_handle.has_value());
  ASSERT_TRUE(second_handle.has_value());

  // Key setup: import the first slot, release its active lease, then reuse the same control.
  auto imported_or = context.import_buffer_unsynchronized(*first_handle);
  ASSERT_TRUE(imported_or.has_value());
  holomq::ConstBuffer imported = std::move(imported_or).value();
  EXPECT_TRUE(verify_host_pattern(imported.ptr(), imported.size_bytes(), kPatternSeed));
  imported.release();
  EXPECT_FALSE(static_cast<bool>(imported));
  EXPECT_EQ(imported.memory_type(), holomq::BufferMemoryType::None);
  EXPECT_EQ(imported.size_bytes(), 0U);
  EXPECT_EQ(imported.ptr(), nullptr);

  // Key assertion: rebinding selects another slot without reconstructing the import control.
  ASSERT_TRUE(context.rebind_buffer_unsynchronized(*second_handle, imported).has_value());
  EXPECT_TRUE(verify_host_pattern(imported.ptr(), imported.size_bytes(), kPatternSeed + 1U));
  EXPECT_FALSE(context.rebind_buffer_unsynchronized(*first_handle, imported).has_value());
  imported.release();

  cleanup_socket_path(socket_path);
}

// A cold-bound producer lease cannot begin a new generation while any imported view is live.
TEST(CudaHostPoolTest, ColdBoundOwnerWaitsForImportedViewsBeforeReuse) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  const std::string socket_path = make_socket_path();
  ASSERT_FALSE(socket_path.empty());

  auto context_or = make_context(socket_path, true);
  ASSERT_TRUE(context_or.has_value());
  holomq::Context& context = context_or.value();
  auto pool_or = context.create_host_buffer_pool(kBufferBytes, 1U);
  ASSERT_TRUE(pool_or.has_value());
  holomq::BufferPool pool = std::move(pool_or).value();
  auto buffer_or = pool.acquire();
  ASSERT_TRUE(buffer_or.has_value());
  holomq::Buffer buffer = std::move(buffer_or).value();
  const auto handle = buffer.export_handle_unsynchronized();
  ASSERT_TRUE(handle.has_value());

  // Key setup: retain one imported reference to the cold-bound producer slot.
  auto imported_or = context.import_buffer_unsynchronized(*handle);
  ASSERT_TRUE(imported_or.has_value());
  holomq::ConstBuffer imported = std::move(imported_or).value();

  // Key assertion: reuse is gated by the shared lease and succeeds after that lease retires.
  EXPECT_FALSE(buffer.prepare_for_reuse());
  imported.release();
  EXPECT_TRUE(buffer.prepare_for_reuse());
  EXPECT_TRUE(buffer.export_handle_unsynchronized().has_value());

  cleanup_socket_path(socket_path);
}

// A received handle must not construct a payload SlotRegion beyond its claimed host pool.
TEST(CudaHostPoolTest, RejectsPayloadSlotOutsideClaimedPool) {
  skip_unless_cuda_available();
  expect_host_import_rejects_out_of_range_slot(InvalidHostPoolGeometry::kPayload);
}

// Split layouts must also contain the selected slot in the imported generation VMM pool.
TEST(CudaHostPoolTest, RejectsGenerationSlotOutsideClaimedPool) {
  skip_unless_cuda_available();
  expect_host_import_rejects_out_of_range_slot(InvalidHostPoolGeometry::kGeneration);
}

TEST(CudaHostPoolTest, PageableSplitGenerationCrossProcessImport) {
  run_host_pool_cross_process_import_test(
      HostPoolCrossProcessConfig{.vmm_fd_export = true, .use_registered_host_pool = false});
}

TEST(CudaHostPoolTest, DefaultRegisterFlagsSplitGenerationCrossProcessImport) {
  run_host_pool_cross_process_import_test(HostPoolCrossProcessConfig{
      .vmm_fd_export = true, .use_registered_host_pool = true, .host_register_flags = 0U});
}
