/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

namespace holoscan {
namespace inference {

/**
 * @brief Per-model configuration bundle owned by ManagerInfer: model path,
 * instance label, input/output tensor names, device residency, and device index.
 */
class Params {
 public:
  /// @brief Construct with defaults (empty paths, CPU, device 0).
  Params();
  /// @brief Construct with model path, instance name, CUDA flag, and device id.
  Params(std::string _model_path, std::string _infer_name, bool _usecuda, int _device_id = 0);
  /// @brief Return the configured model (ONNX or engine) file path.
  const std::string get_model_path() const;
  /// @brief Return the instance label used to disambiguate multiple models.
  const std::string get_instance_name() const;
  /// @brief Return the ordered list of input tensor names for this model.
  const std::vector<std::string> get_input_tensor_names() const;
  /// @brief Return the ordered list of output tensor names for this model.
  const std::vector<std::string> get_output_tensor_names() const;
  /// @brief Return whether the model runs with CUDA-resident buffers.
  bool get_cuda_flag() const;
  /// @brief Return the CUDA device index assigned to this model.
  int get_device_id() const;
  /// @brief Set the model file path.
  void set_model_path(const std::string&);
  /// @brief Set the CUDA device index.
  void set_device_id(int);
  /// @brief Set the instance label.
  void set_instance_name(const std::string&);
  /// @brief Set the CUDA residency flag.
  void set_cuda_flag(bool);
  /// @brief Populate input (true) or output (false) tensor-name list.
  void set_tensor_names(const std::vector<std::string>&, bool);

 private:
  bool use_cuda_;
  std::string model_file_path_;
  std::string instance_name_;
  int device_id_;
  std::vector<std::string> in_tensor_names_;
  std::vector<std::string> out_tensor_names_;
};

}  // namespace inference
}  // namespace holoscan
