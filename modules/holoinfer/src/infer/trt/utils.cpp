/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace holoscan {
namespace inference {

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
  engine_path =
      std::filesystem::path(onnx_model_path).replace_extension("").string() + "." + gpu_name + "." +
      std::to_string(device_prop.major) + "." + std::to_string(device_prop.minor) + "." +
      std::to_string(device_prop.multiProcessorCount) + ".trt." +
      std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
      std::to_string(NV_TENSORRT_PATCH) + "." + std::to_string(NV_TENSORRT_BUILD) + ".engine";

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

  if (network_options.batch_sizes.size() != 3) {
    HOLOSCAN_LOG_ERROR("Size of batches for optimization profile must be 3. Size provided: {}",
                       network_options.batch_sizes.size());
    return false;
  }

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) { return false; }

  auto explicit_batch = 1;
  auto network =
      std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) { return false; }

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) { return false; }

  if (!parser->parseFromFile(onnx_model_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    HOLOSCAN_LOG_ERROR("Failed to parse onnx file at: {}", onnx_model_path);
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) { return false; }

  // Create optimization profile
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

  for (int i = 0; i < network->getNbInputs(); i++) {
    nvinfer1::ITensor* input = network->getInput(i);
    const auto input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();
    std::vector<nvinfer1::Dims> profile_batch_dims = {dims, dims, dims};

    profile_batch_dims[0].d[0] = network_options.batch_sizes[0];
    profile_batch_dims[1].d[0] = network_options.batch_sizes[1];
    profile_batch_dims[2].d[0] = network_options.batch_sizes[2];

    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, profile_batch_dims[0]);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, profile_batch_dims[1]);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, profile_batch_dims[2]);
  }

  config->addOptimizationProfile(profile);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, network_options.max_memory);

  if (network_options.use_fp16) { config->setFlag(nvinfer1::BuilderFlag::kFP16); }

  auto profileStream = makeCudaStream();
  if (!profileStream) { return false; }
  config->setProfileStream(*profileStream);

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) { return false; }

  std::ofstream outfile(engine_path, std::ofstream::binary);
  if (!outfile) {
    HOLOSCAN_LOG_ERROR("Cannot write engine file as: {}", engine_path);
    return false;
  }

  outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
  if (outfile.fail()) {
    HOLOSCAN_LOG_ERROR("Cannot write engine file as: {}", engine_path);
    return false;
  }

  HOLOSCAN_LOG_INFO("Engine file generated, saved as: {}", engine_path);

  return true;
}

}  // namespace inference
}  // namespace holoscan
