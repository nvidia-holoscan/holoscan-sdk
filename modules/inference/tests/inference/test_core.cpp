/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_core.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "support/host_unique_test_name.hpp"

namespace {

// CTest runs each HoloInferTests case in its own process, several at a time. Engine caches are
// written beside their ONNX model and cleanup_engines() deletes every engine in that folder, so a
// shared folder lets one test delete or overwrite the engine another test is loading. Stage the
// models once per process in a private directory that is removed at exit.
[[nodiscard]] std::filesystem::path private_model_path() {
  return std::filesystem::temp_directory_path() /
         holoscan::test::make_host_unique_test_name("holoinfer_models");
}

class StagedModelFolder {
 public:
  StagedModelFolder() : path_(private_model_path()) {
    const std::filesystem::path source{"modules/inference/tests/test_models"};
    std::filesystem::create_directories(path_);
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
      if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
        std::filesystem::copy_file(entry.path(),
                                   path_ / entry.path().filename(),
                                   std::filesystem::copy_options::overwrite_existing);
      }
    }
  }

  StagedModelFolder(const StagedModelFolder&) = delete;
  StagedModelFolder& operator=(const StagedModelFolder&) = delete;

  ~StagedModelFolder() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] std::string folder() const { return path_.string() + "/"; }

 private:
  std::filesystem::path path_;
};

}  // namespace

std::string HoloInferTests::staged_model_folder() {
  static const StagedModelFolder staged;
  return staged.folder();
}

void HoloInferTests::clear_specs() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
  for (auto stream : allocated_test_streams_) {
    cudaStreamDestroy(stream);
  }
  allocated_test_streams_.clear();
}

HoloInfer::InferStatus HoloInferTests::setup_inference() {
  holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();
  auto status = holoscan_infer_context_->set_inference_params(inference_specs_);
  return status;
}

void HoloInferTests::setup_specifications() {
  // Create inference specification structure
  std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream;
  stream_pool_calls_ = 0;
  if (fast_multi_model || provide_backend_stream_pool) {
    allocate_cuda_stream = [this](int32_t /*device_id*/) -> cudaStream_t {
      if (stream_pool_limit >= 0 && stream_pool_calls_ >= stream_pool_limit) {
        return nullptr;
      }
      cudaStream_t stream = nullptr;
      if (cudaStreamCreate(&stream) != cudaSuccess) {
        return nullptr;
      }
      ++stream_pool_calls_;
      allocated_test_streams_.push_back(stream);
      return stream;
    };
  }

  inference_specs_ = std::make_shared<HoloInfer::InferenceSpecs>(backend,
                                                                 backend_map,
                                                                 model_path_map,
                                                                 input_map,
                                                                 output_map,
                                                                 dla_core_map,
                                                                 batch_sizes,
                                                                 dynamic_inputs,
                                                                 is_engine_path,
                                                                 parallel_inference,
                                                                 enable_fp16,
                                                                 input_on_cuda,
                                                                 output_on_cuda,
                                                                 use_cuda_graphs,
                                                                 dla_core,
                                                                 dla_gpu_fallback,
                                                                 false,
                                                                 allocate_cuda_stream);
  inference_specs_->fast_multi_model_ = fast_multi_model;
}

HoloInfer::InferStatus HoloInferTests::create_specifications() {
  HoloInfer::InferStatus status = HoloInfer::InferStatus(HoloInfer::holoinfer_code::H_ERROR);

  try {
    setup_specifications();
    status = setup_inference();
  } catch (const std::runtime_error& rt) {
    return status;
  }
  return status;
}

HoloInfer::InferStatus HoloInferTests::call_parameter_check_inference() {
  return HoloInfer::inference_validity_check(
      model_path_map, input_map, output_map, in_tensor_names, out_tensor_names);
}

HoloInfer::InferStatus HoloInferTests::prepare_for_inference() {
  clear_specs();

  auto status = create_specifications();

  for (const auto& td : in_tensor_dimensions) {
    auto db = std::make_shared<HoloInfer::DataBuffer>();
    const size_t buffer_size =
        std::accumulate(td.second.begin(), td.second.end(), 1, std::multiplies<size_t>());

    db->device_buffer_->resize(buffer_size);
    db->host_buffer_->resize(buffer_size);
    inference_specs_->data_per_tensor_.insert({td.first, std::move(db)});
    auto dims = td.second;
    dims.insert(dims.begin(), 1);
    inference_specs_->dims_per_tensor_.insert({td.first, dims});
  }

  return status;
}

HoloInfer::InferStatus HoloInferTests::do_inference() {
  HoloInfer::InferStatus status = HoloInfer::InferStatus(HoloInfer::holoinfer_code::H_ERROR);

  try {
    if (!holoscan_infer_context_) {
      return status;
    }
    return holoscan_infer_context_->execute_inference(inference_specs_);
  } catch (...) {
    std::cout << "Exception occurred in inference.\n";
    return status;
  }
}

void HoloInferTests::cleanup_engines() {
  // cleaning engine files
  for (const auto& file : std::filesystem::directory_iterator(model_folder)) {
    if (file.is_regular_file()) {
      const auto filename = file.path().filename().string();
      if (filename.find(".engine.") != std::string::npos) {
        std::filesystem::remove(file.path());
        HOLOSCAN_LOG_INFO("Cleaning up engine file: {}", filename);
      }
    }
  }
}
