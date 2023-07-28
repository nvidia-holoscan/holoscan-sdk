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
#include "infer_param.hpp"

#include <string>
#include <vector>

namespace holoscan {
namespace inference {

Params::Params(const std::string& _model_path, const std::string& _infer_name, bool _usecuda)
    : model_file_path_(_model_path), instance_name_(_infer_name), use_cuda_(_usecuda) {}

Params::Params() {
  model_file_path_ = "";
  instance_name_ = "";
  use_cuda_ = false;
}

bool Params::get_cuda_flag() const {
  return use_cuda_;
}

const std::string Params::get_model_path() const {
  return model_file_path_;
}

const std::string Params::get_instance_name() const {
  return instance_name_;
}

void Params::set_instance_name(const std::string& _inst_name) {
  instance_name_ = _inst_name;
}

void Params::set_model_path(const std::string& _model_path) {
  model_file_path_ = _model_path;
}

void Params::set_cuda_flag(bool _usecuda) {
  use_cuda_ = _usecuda;
}

void Params::set_device_id(int device_id) {
  device_id_ = device_id;
}

int Params::get_device_id() const {
  return device_id_;
}

void Params::set_tensor_names(const std::vector<std::string>& _tensor_names, bool type) {
  if (type) {
    in_tensor_names_.assign(_tensor_names.begin(), _tensor_names.end());
  } else {
    out_tensor_names_.assign(_tensor_names.begin(), _tensor_names.end());
  }
}

const std::vector<std::string> Params::get_input_tensor_names() const {
  return in_tensor_names_;
}

const std::vector<std::string> Params::get_output_tensor_names() const {
  return out_tensor_names_;
}

}  // namespace inference
}  // namespace holoscan
