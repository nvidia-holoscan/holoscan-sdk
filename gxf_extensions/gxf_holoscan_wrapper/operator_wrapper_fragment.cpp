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

#include "operator_wrapper_fragment.hpp"

#include "holoscan/core/executors/gxf/gxf_executor.hpp"

namespace holoscan::gxf {

OperatorWrapperFragment::OperatorWrapperFragment() : Fragment() {
  // Create a GXFExecutor without creating a GXF Context
  // This is because the GXF Context is already created by the GXF executor (gxe).
  executor_ = make_executor<gxf::GXFExecutor>(this, false);
}

}  // namespace holoscan::gxf
