/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone import helper for cross-process host buffer pool tests. Executed via exec after fork
 * so CUDA initializes in a process that never inherited a prior cuInit from gtest.
 */

#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/context_builder.hpp>
#include "cuda/context.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include <unistd.h>

namespace {

constexpr std::uint32_t kPatternSeed = 0xA5A50000U;

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

[[nodiscard]] std::string make_socket_path() {
  std::string path =
      "/tmp/holomq_host_pool_test_" + std::to_string(static_cast<long>(::getpid())) + ".sock";
  if (path.size() > holomq::fd_pass::endpoint_max_length()) {
    return {};
  }
  return path;
}

[[nodiscard]] holomq::cuda::Context make_cuda_context() {
  CUdevice device{};
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return {};
  }
  return holomq::cuda::Context{device};
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

void fail_with_code(const int result_fd, const std::uint8_t code) {
  static_cast<void>(write_exact(result_fd, &code, 1U));
  ::close(result_fd);
  _exit(1);
}

[[nodiscard]] int env_fd(const char* const name) {
  const char* const value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  return std::atoi(value);
}

}  // namespace

int main() {
  const int handle_fd = env_fd("HOLOMQ_IMPORT_HANDLE_FD");
  const int result_fd = env_fd("HOLOMQ_IMPORT_RESULT_FD");
  const char* const vmm_fd_export_env = std::getenv("HOLOMQ_IMPORT_VMM_FD_EXPORT");
  const bool vmm_fd_export = vmm_fd_export_env != nullptr && vmm_fd_export_env[0] == '1';

  if (handle_fd < 0 || result_fd < 0) {
    return 1;
  }

  holomq::BufferHandle handle{};
  if (!read_exact(handle_fd, &handle, sizeof(handle))) {
    fail_with_code(result_fd, 10U);
  }
  ::close(handle_fd);

  if (!holomq::is_host_buffer_memory_type(handle.memory_type())) {
    fail_with_code(result_fd, 19U);
  }

  if (cuInit(0) != CUDA_SUCCESS) {
    fail_with_code(result_fd, 11U);
  }

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  if (!cuda_ctx) {
    fail_with_code(result_fd, 12U);
  }
  if (cuCtxSetCurrent(cuda_ctx.get()) != CUDA_SUCCESS) {
    fail_with_code(result_fd, 13U);
  }

  CUstream stream{};
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    fail_with_code(result_fd, 14U);
  }

  const std::string socket_path = make_socket_path();
  if (socket_path.empty()) {
    fail_with_code(result_fd, 15U);
  }

  // Import uses the fd-pass path embedded in the handle (publisher process), not this path.
  auto context_or = make_context(socket_path, vmm_fd_export);
  if (!context_or.has_value()) {
    fail_with_code(result_fd, 16U);
  }

  auto view_or = context_or->import_buffer(handle, stream);
  if (!view_or.has_value()) {
    fail_with_code(result_fd, 17U);
  }
  if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
    fail_with_code(result_fd, 18U);
  }

  holomq::ConstBuffer view = std::move(view_or).value();
  if (view.memory_type() != handle.memory_type()) {
    fail_with_code(result_fd, 20U);
  }
  if (view.cuda_device_ordinal().has_value()) {
    fail_with_code(result_fd, 21U);
  }
  const std::uint8_t ok =
      verify_host_pattern(view.ptr(), view.size_bytes(), kPatternSeed) ? 1U : 2U;
  if (!write_exact(result_fd, &ok, 1U)) {
    ::close(result_fd);
    _exit(1);
  }
  ::close(result_fd);
  static_cast<void>(cuStreamDestroy(stream));
  _exit(static_cast<int>(ok == 1U ? 0 : 2));
}
