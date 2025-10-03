/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef HOLOINFER_TRT_UTILS_H
#define HOLOINFER_TRT_UTILS_H

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Class to extend TensorRT logger
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    LogLevel log_level;
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
    }
    try {  // ignore potential fmt::format_error exception
      HOLOSCAN_LOG_CALL(log_level, msg);
    } catch (std::exception& e) {
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
};

/**
 * @brief Checks the validity of input file path
 * @param filepath Input file path
 */
bool valid_file_path(const std::string& filepath);

/**
 * @brief Build the (trt engine) network
 */
bool generate_engine_path(const NetworkOptions& options, const std::string& model_path,
                          std::string& engine_name);

/**
 * @brief Build the (trt engine) network
 */
bool build_engine(const std::string& onnxModelPath, const std::string& engine_name_,
                  const NetworkOptions& network_options_, Logger& logger_);

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
    pStream.reset(nullptr);
  }

  return pStream;
}

}  // namespace inference
}  // namespace holoscan
#endif  // HOLOINFER_TRT_UTILS_H
