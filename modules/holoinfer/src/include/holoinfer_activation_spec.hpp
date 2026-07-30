/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
