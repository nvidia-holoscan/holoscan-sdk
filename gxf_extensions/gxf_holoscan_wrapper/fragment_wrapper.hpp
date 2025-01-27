/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GXF_HOLOSCAN_WRAPPER_FRAGMENT_WRAPPER_HPP
#define GXF_HOLOSCAN_WRAPPER_FRAGMENT_WRAPPER_HPP

#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to wrap an Operator/Resource's Fragment to interface with the GXF framework.
 *
 * This class is used to create Operator/Resource instances for OperatorWrapper/ResourceWrapper
 * objects.
 */
class FragmentWrapper : public holoscan::Fragment {
 public:
  FragmentWrapper();

  GXFExecutor& gxf_executor() { return static_cast<GXFExecutor&>(executor()); }
};

}  // namespace holoscan::gxf

#endif /* GXF_HOLOSCAN_WRAPPER_FRAGMENT_WRAPPER_HPP */
