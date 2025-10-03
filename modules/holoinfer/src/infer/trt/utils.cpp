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

#include "utils.hpp"

#include <NvInferPlugin.h>
#include <NvInferVersion.h>
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

  if (options.dla_core > -1) {
    engine_path += ".dla" + std::to_string(options.dla_core);
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

  for (auto opt_profile : network_options.batch_sizes) {
    if (opt_profile.size() % 3 != 0) {
      HOLOSCAN_LOG_ERROR(
          "Size of batches for optimization profile must be a multiple of 3. Size provided: {}",
          opt_profile.size());
      return false;
    }
  }

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return false;
  }

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  if (!network) {
    return false;
  }

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) {
    return false;
  }

  if (!parser->parseFromFile(onnx_model_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    HOLOSCAN_LOG_ERROR("Failed to parse onnx file at: {}", onnx_model_path);
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  // Create optimization profile
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

  for (int i = 0; i < network->getNbInputs(); i++) {
    nvinfer1::ITensor* input = network->getInput(i);
    const auto input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();

    std::vector<nvinfer1::Dims> profile_batch_dims = {dims, dims, dims};

    auto num_of_opt_profiles = network_options.batch_sizes.size();

    // if num_of_opt_profiles is 1 but number of dynamic inputs are more, same opt profile is
    // applied for all dynamic inputs. Otherwise num_of_opt_profiles must be equal to model input
    // size. If some intermediate input is not dynamic user must provide default profile for that
    // in the trt profile in the parameter set.

    if (num_of_opt_profiles != 1 && num_of_opt_profiles != network->getNbInputs()) {
      HOLOSCAN_LOG_ERROR(
          "Size of TRT optimization profile must either be 1 or equal to the number of inputs.");
      return false;
    }

    auto current_opt_profile_index = 0;
    if (num_of_opt_profiles > 1) {
      current_opt_profile_index = i;
    }
    auto opt_profile_size = network_options.batch_sizes[current_opt_profile_index].size() / 3;

    for (auto nd = 0; nd < opt_profile_size; ++nd) {
      profile_batch_dims[0].d[nd] = network_options.batch_sizes[current_opt_profile_index][3 * nd];
      profile_batch_dims[1].d[nd] =
          network_options.batch_sizes[current_opt_profile_index][3 * nd + 1];
      profile_batch_dims[2].d[nd] =
          network_options.batch_sizes[current_opt_profile_index][3 * nd + 2];

      if (nd == 0) {
        for (auto prof_d = 0; prof_d < 3; ++prof_d) {
          if (profile_batch_dims[prof_d].d[nd] > network_options.max_batch_size) {
            HOLOSCAN_LOG_INFO("Max batch size supported is {}", network_options.max_batch_size);
            HOLOSCAN_LOG_ERROR("Input batch size for tensor {} is {}",
                               input_name,
                               profile_batch_dims[prof_d].d[nd]);
            return false;
          }
        }
      }
    }

    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, profile_batch_dims[0]);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, profile_batch_dims[1]);
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, profile_batch_dims[2]);
  }

  config->addOptimizationProfile(profile);

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, network_options.max_memory);

  if (network_options.use_fp16) {
    // TensorRT 10.12 deprecated the BuilderFlag::kFP16 flag, but we want to keep the feature
    // until it is removed from the API. Therefore disabled the warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
  }
  if (network_options.dla_core > -1) {
    const int32_t available_dla_cores = builder->getNbDLACores();
    if (network_options.dla_core > available_dla_cores - 1) {
      HOLOSCAN_LOG_ERROR("DLA core '{}' is requested but max DLA core index is '{}'",
                         network_options.dla_core,
                         available_dla_cores - 1);
      return false;
    }
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    // TensorRT 10.12 deprecated the BuilderFlag::kPREFER_PRECISION_CONSTRAINTS flag, but we want
    // to keep the feature until it is removed from the API. Therefore disabled the warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#pragma GCC diagnostic pop
    config->setDLACore(network_options.dla_core);
    // if requested, enable GPU fallback. If this is not set and a layer is not supported by DLA,
    // building the engine will fail with an error.
    if (network_options.dla_gpu_fallback) {
      config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    } else {
      // Deprecated in TensorRT 10.7
#if (NV_TENSORRT_MAJOR * 10000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH) < 100700
      // Reformatting runs on GPU, so avoid I/O reformatting.
      config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
#endif
    }
  }

  auto profileStream = makeCudaStream();
  if (!profileStream) {
    return false;
  }
  config->setProfileStream(*profileStream);

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

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
