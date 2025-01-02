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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP

#include <memory>
#include <string>

#include <gxf/cuda/cuda_scheduling_terms.hpp>

#include "../../component_spec.hpp"
#include "../../gxf/gxf_condition.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/*TODO(Greg): Can only use CudaBufferAvailableCondition when the message Entity contains a
 * CudaBuffer component. Would need to update HoloscanSDK to support a way to use this type of
 * object via Holoscan APIs in a way that is convenient to combine with existing TensorMap
 * functionality
 *
 * See, for example gxf/cuda/tests/test_cuda_helper.hpp's CudaAsyncBufferGenerator
 */

/**
 * @brief Condition based on data availability in a cuda buffer.
 *
 * A component which specifies the availability of data at the receiver based on the CudaBuffer
 * present in incoming messages.
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * **Note:** The nvidia::gxf::CudaBuffer class is currently unused by Holoscan SDK. This condition
 * is intended exclusively for interoperation with wrapped GXF Codelets that use GXF's CudaBuffer type.
 *
 * ==Parameters==
 *
 * - **receiver** (std::string): The receiver to check for a CudaBuffer. This should be specified
 * by the name of the Operator's input port the condition will apply to. The Holoscan SDK will then
 * automatically replace the port name with the actual receiver object at application run time.
 */
class CudaBufferAvailableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(CudaBufferAvailableCondition, GXFCondition)

  CudaBufferAvailableCondition() = default;
  CudaBufferAvailableCondition(const std::string& name,
                               nvidia::gxf::CudaBufferAvailableSchedulingTerm* term);

  const char* gxf_typename() const override {
    return "nvidia::gxf::CudaBufferAvailableSchedulingTerm";
  }
  void setup(ComponentSpec& spec) override;

  void receiver(std::shared_ptr<gxf::GXFResource> receiver) { receiver_ = receiver; }
  std::shared_ptr<gxf::GXFResource> receiver() { return receiver_.get(); }

  nvidia::gxf::CudaBufferAvailableSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<gxf::GXFResource>> receiver_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP */
