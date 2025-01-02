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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP

#include <memory>
#include <string>

#include <gxf/cuda/cuda_scheduling_terms.hpp>

#include "../../component_spec.hpp"
#include "../../gxf/gxf_condition.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * TODO(Greg): This condition requires there be a CudaStreamId component in the message Entity.
 *
 * e.g., it calls
 *     auto stream_id = message->get<CudaStreamId>();
 *
 * Need to check if this works as-is with the existing Holoscan CudaStreamHandler utility
 */

/**
 * @brief Condition class to indicate data availability on CUDA stream completion.
 *
 * This condition will register a call back function which will be called once the work on the
 * specified CUDA stream completes indicating that the data is available for consumption
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * ==Parameters==
 *
 * - **receiver** (std::string): The receiver to check for a CudaStreamId. This should be specified
 * by the name of the Operator's input port the condition will apply to. The Holoscan SDK will then
 * automatically replace the port name with the actual receiver object at application run time.
 */
class CudaStreamCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(CudaStreamCondition, GXFCondition)

  CudaStreamCondition() = default;
  CudaStreamCondition(const std::string& name, nvidia::gxf::CudaStreamSchedulingTerm* term);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaStreamSchedulingTerm"; }
  void setup(ComponentSpec& spec) override;

  void receiver(std::shared_ptr<gxf::GXFResource> receiver) { receiver_ = receiver; }
  std::shared_ptr<gxf::GXFResource> receiver() { return receiver_.get(); }

  nvidia::gxf::CudaStreamSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<gxf::GXFResource>> receiver_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP */
