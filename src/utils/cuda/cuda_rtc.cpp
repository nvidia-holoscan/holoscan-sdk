/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoscan/utils/cuda/cuda_rtc.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <holoscan/logger/logger.hpp>

/**
 * NvRTC API error check helper
 */
#define NvRTCCheck(FUNC)                                                              \
  {                                                                                   \
    const nvrtcResult result = FUNC;                                                  \
    if (result != NVRTC_SUCCESS) {                                                    \
      std::stringstream buf;                                                          \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] NvRTC error " << result << ": " \
          << nvrtcGetErrorString(result);                                             \
      throw std::runtime_error(buf.str().c_str());                                    \
    }                                                                                 \
  }

namespace holoscan {

namespace {

// Returns the compute-capability arch string for the device attached to the
// current CUDA context, e.g. "sm_90", "sm_100", "sm_120".  Used to target
// NVRTC CUBIN emission at a specific SM so we bypass the driver's PTX -> SASS
// JIT (required on platforms like R580.00 where PTX from newer NVRTC
// toolchains is rejected with CUDA_ERROR_UNSUPPORTED_PTX_VERSION).  Mirrors
// the helper of the same name in HSB's hololink::common::current_device_sm_arch.
std::string current_device_sm_arch() {
  CUdevice device;
  CudaCheck(cuCtxGetDevice(&device));
  int major = 0;
  int minor = 0;
  CudaCheck(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CudaCheck(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  std::stringstream buf;
  buf << "sm_" << major << minor;
  return buf.str();
}

bool has_arch_option(const std::vector<std::string>& options) {
  for (const auto& option : options) {
    if (option.find("--gpu-architecture=") == 0 || option.find("-arch=") == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

CudaFunctionLauncher::CudaFunctionLauncher(const char* source,
                                           const std::vector<std::string>& functions,
                                           const std::vector<std::string>& options) {
  nvrtcProgram prog;
  NvRTCCheck(nvrtcCreateProgram(&prog,   // prog
                                source,  // buffer
                                "",      // name
                                0,       // numHeaders
                                NULL,    // headers
                                NULL));  // includeNames
  // NVRTC must target a real SM (sm_<m><n>) to emit CUBIN.  We load CUBIN via
  // cuModuleLoadDataEx below to bypass the driver's PTX -> SASS JIT, which is
  // rejected on R580.00 (CUDA 13.0) when the NVRTC toolchain is newer than
  // the driver (CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
  // If the caller already supplied --gpu-architecture / -arch, respect it;
  // otherwise auto-derive sm_<m><n> from the live CUDA context's device.
  std::string arch_option;
  std::vector<const char*> compile_options;
  if (!has_arch_option(options)) {
    arch_option = "--gpu-architecture=" + current_device_sm_arch();
    compile_options.push_back(arch_option.c_str());
  }
  for (auto&& option : options) {
    compile_options.push_back(option.c_str());
  }
  compile_options.push_back("--include-path=/usr/local/cuda/include");
  compile_options.push_back("--include-path=/usr/local/cuda/include/cccl");
  if (nvrtcCompileProgram(prog, compile_options.size(), compile_options.data()) != NVRTC_SUCCESS) {
    // Obtain compilation log from the program.
    size_t logSize;
    NvRTCCheck(nvrtcGetProgramLogSize(prog, &logSize));
    std::unique_ptr<char[]> log(new char[logSize]);
    NvRTCCheck(nvrtcGetProgramLog(prog, log.get()));
    std::stringstream buf;
    buf << "Failed to compile: " << log.get();
    throw std::runtime_error(buf.str().c_str());
  }
  // Obtain CUBIN from the program (real-arch target required, set above).
  size_t cubinSize;
  NvRTCCheck(nvrtcGetCUBINSize(prog, &cubinSize));
  std::unique_ptr<char[]> cubin(new char[cubinSize]);
  NvRTCCheck(nvrtcGetCUBIN(prog, cubin.get()));
  // Destroy the program.
  NvRTCCheck(nvrtcDestroyProgram(&prog));

  // Load the generated CUBIN and get a handle to the kernels.
  CudaCheck(cuModuleLoadDataEx(&module_, cubin.get(), 0, 0, 0));
  for (auto&& function : functions) {
    LaunchParams launch_params;
    CudaCheck(cuModuleGetFunction(&launch_params.function_, module_, function.c_str()));

    // calculate the optimal block size for max occupancy
    int min_grid_size = 0;
    int optimal_block_size = 0;
    CudaCheck(cuOccupancyMaxPotentialBlockSize(
        &min_grid_size, &optimal_block_size, launch_params.function_, nullptr, 0, 0));

    // get a 2D block size from the optimal block size
    launch_params.block_dim_.x = 1;
    launch_params.block_dim_.y = 1;
    launch_params.block_dim_.z = 1;
    while (static_cast<int>(launch_params.block_dim_.x * launch_params.block_dim_.y * 2) <=
           optimal_block_size) {
      if (launch_params.block_dim_.x > launch_params.block_dim_.y) {
        launch_params.block_dim_.y *= 2;
      } else {
        launch_params.block_dim_.x *= 2;
      }
    }

    functions_[function] = launch_params;
  }
}

CudaFunctionLauncher::~CudaFunctionLauncher() {
  try {
    CudaCheck(cuModuleUnload(module_));
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("CudaFunctionLauncher destructor failed with {}", e.what());
  }
}

void CudaFunctionLauncher::launch_internal(const std::string& name, const dim3& grid,
                                           const dim3* block, CUstream stream, void** args) const {
  const LaunchParams& launch_params = functions_.at(name);
  dim3 cur_block;
  if (block) {
    cur_block = *block;
  } else {
    cur_block = launch_params.block_dim_;
  }

  // calculate the launch grid size
  dim3 launch_grid;
  launch_grid.x = (grid.x + (cur_block.x - 1)) / cur_block.x;
  launch_grid.y = (grid.y + (cur_block.y - 1)) / cur_block.y;
  launch_grid.z = (grid.z + (cur_block.z - 1)) / cur_block.z;
  CudaCheck(cuLaunchKernel(launch_params.function_,
                           launch_grid.x,
                           launch_grid.y,
                           launch_grid.z,
                           cur_block.x,
                           cur_block.y,
                           cur_block.z,
                           0,
                           stream,
                           args,
                           nullptr));
}

CudaContextScopedPush::CudaContextScopedPush(CUcontext cuda_context) : cuda_context_(cuda_context) {
  // might be called from a different thread than the thread
  // which constructed the context, therefore call cuInit()
  CudaCheck(cuInit(0));
  CudaCheck(cuCtxPushCurrent(cuda_context_));
}

CudaContextScopedPush::~CudaContextScopedPush() {
  try {
    CUcontext popped_context;
    CudaCheck(cuCtxPopCurrent(&popped_context));
    if (popped_context != cuda_context_) {
      HOLOSCAN_LOG_ERROR("Cuda: Unexpected context popped");
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("ScopedPush destructor failed with {}", e.what());
  }
}

}  // namespace holoscan
