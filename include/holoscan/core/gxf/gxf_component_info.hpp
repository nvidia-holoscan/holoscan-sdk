/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_COMPONENT_INFO_HPP
#define HOLOSCAN_CORE_GXF_GXF_COMPONENT_INFO_HPP

#include <gxf/core/gxf.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/core/arg.hpp"

namespace holoscan::gxf {

/**
 * @brief A class that encapsulates the information about a GXF component.
 *
 * This class provides methods to access various properties of a GXF component,
 * such as its receiver and transmitter TIDs, parameter keys, parameter infos, etc.
 */
class ComponentInfo {
 public:
  /// Maximum number of parameters a component can have.
  static constexpr int MAX_PARAM_COUNT = 512;

  /**
   * @brief Construct a new component info object.
   *
   * @param context The GXF context.
   * @param tid The TID of the component.
   */
  ComponentInfo(gxf_context_t context, gxf_tid_t tid);

  /**
   * @brief Destroy the component info object.
   */
  ~ComponentInfo();

  /**
   * @brief Get the arg type object
   *
   * Returns the Holoscan argument type for the given GXF parameter info.
   *
   * @param param_info The GXF parameter info.
   * @return The argument type of the parameter.
   */
  static ArgType get_arg_type(const gxf_parameter_info_t& param_info);

  /**
   * @brief Get the receiver TID of the component.
   *
   * @return The receiver TID.
   */
  gxf_tid_t receiver_tid() const;

  /**
   * @brief Get the transmitter TID of the component.
   *
   * @return The transmitter TID.
   */
  gxf_tid_t transmitter_tid() const;

  /**
   * @brief Get the component info.
   *
   * @return The component info.
   */
  const gxf_component_info_t& component_info() const;

  /**
   * @brief Get the parameter keys of the component.
   *
   * @return The parameter keys.
   */
  const std::vector<const char*>& parameter_keys() const;

  /**
   * @brief Get the parameter infos of the component.
   *
   * @return The parameter infos.
   */
  const std::vector<gxf_parameter_info_t>& parameter_infos() const;

  /**
   * @brief Get the parameter info map of the component.
   *
   * @return The parameter info map.
   */
  const std::unordered_map<std::string, gxf_parameter_info_t>& parameter_info_map() const;

  /**
   * @brief Get the receiver parameters of the component.
   *
   * @return The receiver parameters.
   */
  const std::vector<const char*>& receiver_parameters() const;

  /**
   * @brief Get the transmitter parameters of the component.
   *
   * @return The transmitter parameters.
   */
  const std::vector<const char*>& transmitter_parameters() const;

  /**
   * @brief Get the normal parameters of the component.
   *
   * @return The normal parameters.
   */
  const std::vector<const char*>& normal_parameters() const;

 private:
  gxf_context_t gxf_context_ = nullptr;                ///< The GXF context.
  gxf_tid_t component_tid_ = GxfTidNull();             ///< The TID of the component.
  gxf_component_info_t component_info_{};              ///< The component info.
  std::vector<const char*> parameter_keys_;            ///< The parameter keys.
  std::vector<gxf_parameter_info_t> parameter_infos_;  ///< The parameter infos.
  /// The parameter info map.
  std::unordered_map<std::string, gxf_parameter_info_t> parameter_info_map_;
  std::vector<const char*> receiver_parameters_;     ///< The receiver parameters.
  std::vector<const char*> transmitter_parameters_;  ///< The transmitter parameters.
  std::vector<const char*> normal_parameters_;       ///< The normal parameters.
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_COMPONENT_INFO_HPP */
