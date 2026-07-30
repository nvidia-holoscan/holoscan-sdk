/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_UTILS_CUDA_CUDA_RTC_HPP
#define HOLOSCAN_UTILS_CUDA_CUDA_RTC_HPP

#include <cuda.h>
#include <vector_types.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/utils/cuda/nullable_pointer.hpp"

namespace holoscan {

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS) {                                                           \
      const char* error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char* error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

/**
 * This class is used to compile the provided CUDA source code and call kernel functions
 * defined by that code.
 */
class CudaFunctionLauncher {
 public:
  /**
   * Construct a new CudaFunctionLauncher object
   *
   * @param source pointer to source code
   * @param functions list of functions defined by the source code.
   * @param options options to pass to the compiler
   */
  CudaFunctionLauncher(const char* source, const std::vector<std::string>& functions,
                       const std::vector<std::string>& options = std::vector<std::string>());
  ~CudaFunctionLauncher();

  /**
   * Launch a kernel on a grid.
   *
   * @param grid [in] grid size
   * @param stream [in] stream
   * @param args [in] kernel arguments (optional)
   */
  template <class... TYPES>
  void launch(const std::string& name, const dim3& grid, CUstream stream, TYPES... args) const {
    void* args_array[] = {reinterpret_cast<void*>(&args)...};
    launch_internal(name, grid, nullptr, stream, args_array);
  }

  /**
   * Launch a kernel with block size on a grid.
   *
   * @param grid [in] grid size
   * @param block [in] block size
   * @param stream [in] stream
   * @param args [in] kernel arguments (optional)
   */
  template <class... TYPES>
  void launch(const std::string& name, const dim3& grid, const dim3& block, CUstream stream,
              TYPES... args) const {
    void* args_array[] = {reinterpret_cast<void*>(&args)...};
    launch_internal(name, grid, &block, stream, args_array);
  }

 private:
  void launch_internal(const std::string& name, const dim3& grid, const dim3* block,
                       CUstream stream, void** args) const;

  CUmodule module_ = nullptr;
  struct LaunchParams {
    dim3 block_dim_;
    CUfunction function_;
  };
  std::unordered_map<std::string, LaunchParams> functions_;
};

/**
 * RAII type classes for CUDA allocations and objects, use like std::unique_ptr.
 */
using UniqueCUdeviceptr =
    std::unique_ptr<Nullable<CUdeviceptr>, Nullable<CUdeviceptr>::Deleter<CUresult, &cuMemFree>>;

/**
 * @brief Custom deleter for CUDA objects managed by unique_ptr
 */
template <typename T, CUresult func(T)>
struct Deleter {
  typedef T pointer;
  void operator()(T value) const { func(value); }
};

using UniqueCUhostptr = std::unique_ptr<void, Deleter<void*, &cuMemFreeHost>>;
using UniqueCUevent = std::unique_ptr<CUevent, Deleter<CUevent, &cuEventDestroy>>;
using UniqueCUstream = std::unique_ptr<CUstream, Deleter<CUstream, &cuStreamDestroy>>;

/**
 * RAII type class to push a CUDA context.
 */
class CudaContextScopedPush {
 public:
  explicit CudaContextScopedPush(CUcontext cuda_context);
  CudaContextScopedPush() = delete;
  ~CudaContextScopedPush();

 private:
  const CUcontext cuda_context_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_UTILS_CUDA_CUDA_RTC_HPP */
