/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

namespace {

// Deterministic 64-bit FNV-1a hash over a byte range. Portable across compilers and runs,
// required for the engine-cache-invalidation tag appended to the engine filename.
std::uint64_t fnv1a(const void* data, std::size_t len,
                    std::uint64_t state = 0xcbf29ce484222325ULL) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < len; ++i) {
    state ^= bytes[i];
    state *= 0x100000001b3ULL;
  }
  return state;
}

// Hash every NetworkOptions field that affects the built engine but is NOT already reflected in
// the human-readable portion of the filename (GPU name, CC, SM count, TRT version, fp16/fp32,
// DLA core). Adding a new build-affecting param to NetworkOptions requires adding it here so
// stale engines are invalidated on config change.
std::uint64_t network_options_cfg_hash(const NetworkOptions& options) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  h = fnv1a(&options.max_batch_size, sizeof(options.max_batch_size), h);
  h = fnv1a(&options.max_memory, sizeof(options.max_memory), h);
  h = fnv1a(&options.dla_gpu_fallback, sizeof(options.dla_gpu_fallback), h);
  // Optimization profile: hash the flattened min/opt/max triples per input, with a byte separator
  // between input entries so [{1,2,3},{}] and [{1,2},{3}] hash distinctly.
  for (const auto& profile : options.batch_sizes) {
    for (const auto val : profile) {
      h = fnv1a(&val, sizeof(val), h);
    }
    const std::uint8_t sep = 0xFF;
    h = fnv1a(&sep, 1, h);
  }
  return h;
}

// Name a private file in the engine cache path's directory, unique across concurrent builders.
// The name does not derive from the destination's basename, which can already be close to the
// filesystem's component-length limit; a fixed-size name keeps every valid destination usable.
[[nodiscard]] std::string temporary_engine_path(const std::string& engine_path) {
  static std::atomic<std::uint64_t> ordinal{};
  std::random_device entropy;
  const std::uint64_t token = (static_cast<std::uint64_t>(entropy()) << 32U) ^ entropy();
  const std::string name = "holoinfer-engine.tmp." + std::to_string(token) + "." +
                           std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
  return (std::filesystem::path(engine_path).parent_path() / name).string();
}

}  // namespace

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

  // When building within a green context SM partition, embed the partition's SM count so
  // engines built with different SM budgets get distinct filenames and do not collide.
  const int sm_count =
      (options.build_sm_count > 0) ? options.build_sm_count : device_prop.multiProcessorCount;

  engine_path = fmt::format("{}.{}.{}.{}.{}.trt.{}.{}.{}.{}",
                            std::filesystem::path(onnx_model_path).replace_extension("").string(),
                            gpu_name,
                            device_prop.major,
                            device_prop.minor,
                            sm_count,
                            NV_TENSORRT_MAJOR,
                            NV_TENSORRT_MINOR,
                            NV_TENSORRT_PATCH,
                            NV_TENSORRT_BUILD);

  if (options.dla_core > -1) {
    engine_path += ".dla" + std::to_string(options.dla_core);
  }

  // Cache-invalidation tag for build-affecting NetworkOptions fields NOT already present in the
  // human-readable portion of the filename.
  engine_path += fmt::format(".cfg{:016x}", network_options_cfg_hash(options));

#if NV_TENSORRT_MAJOR < 11
  engine_path += options.use_fp16 ? ".engine.fp16" : ".engine.fp32";
#else
  engine_path += ".engine.typed";
#endif

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

namespace {

bool configure_optimization_profile(nvinfer1::INetworkDefinition& network,
                                    nvinfer1::IOptimizationProfile& profile,
                                    const NetworkOptions& network_options) {
  for (int i = 0; i < network.getNbInputs(); i++) {
    nvinfer1::ITensor* input = network.getInput(i);
    const auto input_name = input->getName();
    const nvinfer1::Dims dims = input->getDimensions();

    std::vector<nvinfer1::Dims> profile_batch_dims = {dims, dims, dims};

    auto num_of_opt_profiles = network_options.batch_sizes.size();

    // if num_of_opt_profiles is 1 but number of dynamic inputs are more, same opt profile is
    // applied for all dynamic inputs. Otherwise num_of_opt_profiles must be equal to model input
    // size. If some intermediate input is not dynamic user must provide default profile for that
    // in the trt profile in the parameter set.

    if (num_of_opt_profiles != 1 &&
        std::cmp_not_equal(num_of_opt_profiles, network.getNbInputs())) {
      HOLOSCAN_LOG_ERROR(
          "Size of TRT optimization profile must either be 1 or equal to the number of inputs.");
      return false;
    }

    auto current_opt_profile_index = 0;
    if (num_of_opt_profiles > 1) {
      current_opt_profile_index = i;
    }
    auto opt_profile_size = network_options.batch_sizes[current_opt_profile_index].size() / 3;

    // Array subscript access to TensorRT Dims.d[] is performance-critical for inference.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
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
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

    profile.setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, profile_batch_dims[0]);
    profile.setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, profile_batch_dims[1]);
    profile.setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, profile_batch_dims[2]);
  }

  return true;
}

}  // namespace

// Keep builder options, green-context scope, serialization, and cache publication in
// acquisition order so only a valid partition-specific plan is persisted.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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

  for (const auto& opt_profile : network_options.batch_sizes) {
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

  if (!configure_optimization_profile(*network, *profile, network_options)) {
    return false;
  }

  config->addOptimizationProfile(profile);

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, network_options.max_memory);

  if (network_options.use_fp16) {
#if NV_TENSORRT_MAJOR < 11
    // TensorRT 10.12 deprecated the BuilderFlag::kFP16 flag, but we want to keep the feature
    // until it is removed from the API. Therefore disabled the warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
#else
    // TensorRT 11 builds every network strongly typed: layer precision comes from the
    // model's tensor types and there is no builder flag to lower it.
    HOLOSCAN_LOG_WARN(
        "use_fp16 is ignored with TensorRT {}: precision follows the model's tensor types",
        NV_TENSORRT_MAJOR);
#endif
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
#if NV_TENSORRT_MAJOR < 11
    // TensorRT 10.12 deprecated the BuilderFlag::kPREFER_PRECISION_CONSTRAINTS flag, but we want
    // to keep the feature until it is removed from the API. Therefore disabled the warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#pragma GCC diagnostic pop
#endif
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

  // If a CUDA green context is provided, push it so TRT's tactic search and
  // kernel timing run against the partition's SM subset rather than the full GPU.
  CUcontext prev_context = nullptr;
  const bool using_green_ctx = (network_options.build_cuda_context != nullptr);
  bool push_succeeded = false;
  if (using_green_ctx) {
    CUresult cu_status = cuCtxPushCurrent(network_options.build_cuda_context);
    if (cu_status == CUDA_SUCCESS) {
      push_succeeded = true;
    } else {
      HOLOSCAN_LOG_WARN(
          "TRT engine build: failed to push green context (CUresult={}); "
          "falling back to default context — engine will use all GPU SMs.",
          static_cast<int>(cu_status));
    }
  }

  // RAII guard: ensures cuCtxPopCurrent runs on scope exit (including if
  // buildSerializedNetwork throws), but only when the push actually succeeded.
  struct CtxPopGuard {
    CUcontext* prev;
    bool active;
    ~CtxPopGuard() {
      if (active) {
        cuCtxPopCurrent(prev);
      }
    }
  };

  std::unique_ptr<nvinfer1::IHostMemory> plan;
  {
    const CtxPopGuard pop_guard{&prev_context, push_succeeded};

    auto profileStream = makeCudaStream();
    if (!profileStream) {
      return false;
    }
    config->setProfileStream(*profileStream);

    if (push_succeeded) {
      HOLOSCAN_LOG_INFO("TRT engine build: running within green context ({} SMs)",
                        network_options.build_sm_count);
    }

    plan.reset(builder->buildSerializedNetwork(*network, *config));
  }

  if (!plan) {
    return false;
  }

  if (using_green_ctx && !push_succeeded) {
    // `engine_path` encodes the green-context partition's SM count (see
    // generate_engine_path()), but this plan was actually built against the full GPU because the
    // green context push failed above. Do not persist it under the partition-specific path: a
    // later run for the same partition would find it and mistake it for a green-context engine.
    HOLOSCAN_LOG_WARN(
        "TRT engine build: skipping engine cache write to {} because the green context push "
        "failed; the built plan reflects a full-GPU build, not the requested partition.",
        engine_path);
    return false;
  }

  if (!publish_engine_file(engine_path, plan->data(), plan->size())) {
    return false;
  }

  HOLOSCAN_LOG_INFO("Engine file generated, saved as: {}", engine_path);

  return true;
}

bool publish_engine_file(const std::string& engine_path, const void* const data,
                         const std::size_t size) {
  // Concurrent builds of one model share this cache path, and a loader accepts any existing file.
  // Write a private file in the same directory and rename it into place, so no loader can observe
  // a partial engine. Concurrent builders each publish a complete engine; the last rename wins.
  const std::string temporary_path = temporary_engine_path(engine_path);
  std::error_code cleanup_error;
  {
    std::ofstream outfile(temporary_path, std::ofstream::binary);
    if (!outfile) {
      HOLOSCAN_LOG_ERROR("Cannot write engine file as: {}", temporary_path);
      return false;
    }
    outfile.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    outfile.close();
    if (outfile.fail()) {
      HOLOSCAN_LOG_ERROR("Cannot write engine file as: {}", temporary_path);
      std::filesystem::remove(temporary_path, cleanup_error);
      return false;
    }
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary_path, engine_path, rename_error);
  if (rename_error) {
    HOLOSCAN_LOG_ERROR(
        "Cannot publish engine file as: {} ({})", engine_path, rename_error.message());
    std::filesystem::remove(temporary_path, cleanup_error);
    return false;
  }
  return true;
}

}  // namespace inference
}  // namespace holoscan
