/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GXF_HOLOSCAN_WRAPPER_RESOURCE_WRAPPER_HPP
#define GXF_HOLOSCAN_WRAPPER_RESOURCE_WRAPPER_HPP

#include <list>
#include <memory>

#include "fragment_wrapper.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"

#include "gxf/core/component.hpp"
#include "gxf/core/parameter_parser_std.hpp"

namespace holoscan::gxf {

// Forward declaration
struct CommonGXFParameter;

/**
 * @brief Class to wrap a Resource to interface with the GXF framework.
 */
class ResourceWrapper : public nvidia::gxf::Component {
 public:
  ResourceWrapper();
  virtual ~ResourceWrapper() = default;

  /// Get the type name of the Resource.
  virtual const char* holoscan_typename() const = 0;

  /// Get the Holoscan Resource.
  std::shared_ptr<Resource> resource() const;

  /// Create and initialize the Resource.
  gxf_result_t initialize() override;
  /// Destroy the Resource and free resources.
  gxf_result_t deinitialize() override;

  /// Register the Resource's parameters with the GXF framework.
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;

 protected:
  std::shared_ptr<Resource> res_;  ///< The Resource to wrap.
  FragmentWrapper fragment_;       ///< The fragment for the Resource.
  std::list<std::shared_ptr<CommonGXFParameter>>
      parameters_;  ///< The parameters for the GXF Component.
};

}  // namespace holoscan::gxf

#endif /* GXF_HOLOSCAN_WRAPPER_RESOURCE_WRAPPER_HPP */
