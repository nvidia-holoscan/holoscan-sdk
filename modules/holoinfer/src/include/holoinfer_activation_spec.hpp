/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_ACTIVATION_SPEC_HPP
#define MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_ACTIVATION_SPEC_HPP

#include <string>

namespace holoscan {
namespace inference {

/**
 * @brief Activation specification struct, used along with activation_map to select a subset of
 * models at runtime.
 */
struct ActivationSpec {
  ActivationSpec() = default;

  /**
   * @brief Construct a new Activation Spec object.
   * @param model_name Name of model which is defined in the model_path_map parameter.
   * @param active Active model flag (true or false), default true.
   */
  explicit ActivationSpec(const std::string& model_name, bool active = true)
      : model_name_(model_name), active_(active) {}

  bool is_active() const { return active_; }
  std::string model() const { return model_name_; }
  void set_active(bool value = true) { active_ = value; }
  std::string model_name_;
  bool active_;
};

}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_ACTIVATION_SPEC_HPP */
