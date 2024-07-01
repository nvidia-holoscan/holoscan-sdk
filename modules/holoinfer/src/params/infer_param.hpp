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

#ifndef _HOLOSCAN_INFER_PARAM_H
#define _HOLOSCAN_INFER_PARAM_H

#include <iostream>
#include <string>
#include <vector>

namespace holoscan {
namespace inference {

class Params {
 public:
  Params();
  Params(const std::string&, const std::string&, bool, int device_id_ = 0);
  const std::string get_model_path() const;
  const std::string get_instance_name() const;
  const std::vector<std::string> get_input_tensor_names() const;
  const std::vector<std::string> get_output_tensor_names() const;
  bool get_cuda_flag() const;
  int get_device_id() const;
  unsigned int get_temporal_id() const;
  void set_model_path(const std::string&);
  void set_device_id(int);
  void set_temporal_id(unsigned int&);
  void set_instance_name(const std::string&);
  void set_cuda_flag(bool);
  void set_tensor_names(const std::vector<std::string>&, bool);

 private:
  bool use_cuda_;
  std::string model_file_path_;
  std::string instance_name_;
  int device_id_;
  unsigned int temporal_id_ = 0;
  std::vector<std::string> in_tensor_names_;
  std::vector<std::string> out_tensor_names_;
};

}  // namespace inference
}  // namespace holoscan

#endif
