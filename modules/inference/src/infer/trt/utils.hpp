/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <holoinfer/holoinfer_export.h>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Class to extend TensorRT logger
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // Initialize with a safe fallback so an unexpected Severity value never leaves
    // log_level indeterminate before the HOLOSCAN_LOG_CALL invocation below.
    LogLevel log_level = LogLevel::INFO;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        log_level = LogLevel::CRITICAL;
        break;
      case Severity::kERROR:
        log_level = LogLevel::ERROR;
        break;
      case Severity::kWARNING:
        log_level = LogLevel::WARN;
        break;
      case Severity::kINFO:
        log_level = LogLevel::INFO;
        break;
      case Severity::kVERBOSE:
        log_level = LogLevel::DEBUG;
        break;
      default:
        log_level = LogLevel::INFO;
        break;
    }
    try {  // ignore potential fmt::format_error exception
      HOLOSCAN_LOG_CALL(log_level, msg);
    } catch (const std::exception&) {  // NOLINT(bugprone-empty-catch)
      // TensorRT's logger callback is noexcept; formatting failures cannot escape it.
    }
  };
};

/**
 * @brief Parameters for Engine creation
 */
struct NetworkOptions {
  /// @brief Use FP16 in engine generation
  bool use_fp16 = true;

  /// @brief Batch sizes supported
  std::vector<std::vector<int32_t>> batch_sizes = {{1, 1, 1}};

  /// @brief Max batch size allowed
  int32_t max_batch_size = 256;

  /// @brief Maximum GPU memory allocated for model conversion
  size_t max_memory = 10000000000;

  /// @brief GPU device
  int device_index = 0;

  /// @brief The DLA core index to execute the engine on, starts at 0. Set to -1 (the default) to
  /// disable DLA.
  int32_t dla_core = -1;

  /// @brief If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is
  /// disabled, engine creation will fail if a layer cannot executed on DLA.
  bool dla_gpu_fallback = true;

  /// @brief CUDA context scoped to a green context SM partition for engine building.
  /// When non-null, TRT engine building runs with this context active so that tactic
  /// selection and kernel timing are constrained to the assigned SM partition.
  /// nullptr means use the default primary context (all SMs).
  CUcontext build_cuda_context = nullptr;

  /// @brief SM count of the green context partition used during engine building.
  /// When non-zero, embedded in the engine filename so a full-GPU engine is not
  /// reused for a partitioned build (or vice versa). 0 means full GPU.
  int32_t build_sm_count = 0;
};

/**
 * @brief Checks the validity of input file path
 * @param filepath Input file path
 */
HOLOINFER_EXPORT bool valid_file_path(const std::string& filepath);

/**
 * @brief Compute the canonical cached-engine filename for a given (`model_path`, `options`) pair
 * and write it into `engine_name`.
 *
 * The filename is constructed to uniquely identify every combination of build inputs that
 * changes the produced TensorRT engine, so a stale engine is never silently reused when the
 * build config changes:
 *
 *   `<model_path_without_ext>` `.` `<gpu_name>` `.` `<CC_major>` `.` `<CC_minor>` `.`
 *   `<sm_count>` `.trt.` `<TRT_major>` `.` `<TRT_minor>` `.` `<TRT_patch>` `.` `<TRT_build>`
 *   [ `.dla<core>` ] `.cfg<16-hex-chars>` `.engine.` `{fp16|fp32|typed}`
 *
 * The `.engine.<precision>` terminal suffix is intentionally last so the file reads as if
 * `.engine.fp16` / `.engine.fp32` were the file extension. TensorRT 11 and later always use
 * `.engine.typed`, since precision follows the model's tensor types.
 *
 * The `.cfg<16-hex-chars>` suffix is a 64-bit FNV-1a hash over every `NetworkOptions` field that
 * affects the built engine but is NOT already reflected in the human-readable portion of the
 * filename — currently `max_batch_size`, `max_memory`, `batch_sizes` (parsed
 * `trt_opt_profile`), and `dla_gpu_fallback`. Any new build-affecting parameter added to
 * `NetworkOptions` must be folded into `network_options_cfg_hash()` (see `utils.cpp`), or a
 * change to that parameter will not invalidate the cache.
 *
 * @param options The network options driving the engine build.
 * @param model_path Path to the source ONNX model.
 * @param engine_name Output: the canonical engine filename.
 * @return true on success, false if device properties could not be queried.
 */
HOLOINFER_EXPORT bool generate_engine_path(const NetworkOptions& options,
                                           const std::string& model_path, std::string& engine_name);

/**
 * @brief Publish one complete engine cache file at `engine_path`.
 *
 * The bytes are written to a uniquely named sibling file that is then renamed into place, so the
 * cache path never names a partial engine and a reader holding the previous file keeps a complete
 * copy. The sibling's name has a fixed length, so any destination name the filesystem accepts can
 * be published. A failed write or rename removes its temporary file.
 *
 * @param engine_path Destination cache path.
 * @param data Serialized engine bytes.
 * @param size Number of bytes at `data`.
 * @return true when `engine_path` names the complete engine.
 */
HOLOINFER_EXPORT bool publish_engine_file(const std::string& engine_path, const void* data,
                                          std::size_t size);

/**
 * @brief Build the (trt engine) network
 */
HOLOINFER_EXPORT bool build_engine(const std::string& onnxModelPath,
                                   const std::string& engine_name_,
                                   const NetworkOptions& network_options_, Logger& logger_);

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream != nullptr) {
    // Guard against destroying an uninitialized handle: makeCudaStream() may
    // hand the deleter a pointer whose stream was never successfully created.
    if (*pStream != nullptr) {
      cudaStreamDestroy(*pStream);
    }
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  // Value-initialize so the deleter's null-check on *pStream is meaningful even
  // if cudaStreamCreateWithFlags fails and leaves the handle untouched.
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t{}, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
    pStream.reset(nullptr);
  }

  return pStream;
}

}  // namespace inference
}  // namespace holoscan
