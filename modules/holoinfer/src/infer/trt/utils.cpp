/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils.hpp"

#include <filesystem>

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

namespace holoscan {
namespace inference {

cudaError_t check_cuda(cudaError_t result) {
  if (result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Cuda runtime error: {}", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

bool valid_file_path(const std::string& filepath) {
  return std::filesystem::exists(filepath);
}

bool generate_engine_path(const NetworkOptions& options, const std::string& onnx_model_path,
                               std::string& engine_path) {
  cudaDeviceProp device_prop;
  auto status = cudaGetDeviceProperties(&device_prop, options.device_index);
  if (status != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Error in getting device properties.");
    return false;
  }

  std::string gpu_name{device_prop.name};
  gpu_name.erase(remove(gpu_name.begin(), gpu_name.end(), ' '), gpu_name.end());

  engine_path.reserve(1024);
  engine_path = std::filesystem::path(onnx_model_path).replace_extension("").string() + "."
                + gpu_name + "."
                + std::to_string(device_prop.major) + "."
                + std::to_string(device_prop.minor) + "."
                + std::to_string(device_prop.multiProcessorCount) + ".trt.engine";
  if (options.use_fp16) {
    engine_path += ".fp16";
  } else {
    engine_path += ".fp32";
  }

  return true;
}

void set_dimensions_for_profile(const char* input_name, nvinfer1::IOptimizationProfile* profile,
                                const nvinfer1::Dims4& dims, const nvinfer1::Dims4& opt_dims,
                                const nvinfer1::Dims4& max_dims) {
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dims);
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims);
  return;
}

bool build_engine(const std::string& onnx_model_path, const std::string& engine_path,
                  const NetworkOptions& network_options, Logger& logger) {
  if (valid_file_path(engine_path)) {
    HOLOSCAN_LOG_INFO("Cached engine found: {}", engine_path);
    return true;
  }

  HOLOSCAN_LOG_INFO(
      "Engine file missing at {}. Starting generation process.\nNOTE: This could take a couple of "
      "minutes depending on your model size and GPU!",
      engine_path);

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) { return false; }

  builder->setMaxBatchSize(network_options.max_batch_size);

  auto explicit_batch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
      std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) { return false; }

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) { return false; }

  std::ifstream file(onnx_model_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) { throw std::runtime_error("Unable to read engine file"); }

  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed) { return false; }

  const auto input_name = network->getInput(0)->getName();
  const auto input_dims = network->getInput(0)->getDimensions();

  const auto dims = nvinfer1::Dims4(1, input_dims.d[1], input_dims.d[2], input_dims.d[3]);
  const auto max_batch_dim = nvinfer1::Dims4(
      network_options.max_batch_size, input_dims.d[1], input_dims.d[2], input_dims.d[3]);

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) { return false; }

  nvinfer1::IOptimizationProfile* default_profile = builder->createOptimizationProfile();
  set_dimensions_for_profile(input_name, default_profile, dims, dims, max_batch_dim);
  config->addOptimizationProfile(default_profile);

  for (const auto& batch_size : network_options.batch_sizes) {
    if (batch_size == 1) { continue; }

    if (batch_size > network_options.max_batch_size) {
      throw std::runtime_error("Batch Size cannot be greater than maxBatchSize!");
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    auto batch_dims = nvinfer1::Dims4(batch_size, dims.d[1], dims.d[2], dims.d[3]);
    set_dimensions_for_profile(input_name, profile, dims, batch_dims, max_batch_dim);
    config->addOptimizationProfile(profile);
  }

  config->setMaxWorkspaceSize(network_options.max_memory);

  if (network_options.use_fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }

  auto profileStream = makeCudaStream();
  if (!profileStream) { return false; }
  config->setProfileStream(*profileStream);

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) { return false; }

  std::ofstream outfile(engine_path, std::ofstream::binary);
  outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

  HOLOSCAN_LOG_INFO("Engine file generated, saved as: {}", engine_path);

  return true;
}

}  // namespace inference
}  // namespace holoscan
