/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan {
namespace inference {

class Params {
 public:
  Params();
  Params(const std::string&, const std::string&, bool);
  const std::string get_model_path() const;
  const std::string get_instance_name() const;
  bool get_cuda_flag() const;
  void set_model_path(const std::string&);
  void set_instance_name(const std::string&);
  void set_cuda_flag(bool);

 private:
  bool use_cuda_;
  std::string model_file_path_;
  std::string instance_name_;
};

}  // namespace inference
}  // namespace holoscan

#endif
