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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_EVENT_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_EVENT_HPP

#include <memory>
#include <string>

#include <gxf/cuda/cuda_scheduling_terms.hpp>

#include "../../component_spec.hpp"
#include "../../gxf/gxf_condition.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * TODO(Greg): This condition requires there be a CudaEvent component in the message Entity.
 *
 * e.g. it calls
 *    auto maybe_event = message->get<CudaEvent>(event_name_.get().c_str());
 *
 * See StreamBasedOps codelet in gxf/cuda/tests/test_cuda_helper.hpp
 *    specifically the methods `addNewEvent` and `initOpsEvent` and how they are used in the
 *    operators that inherit from StreamBasedOps
 *
 * We have not yet exposed CudaEvent object from Holoscan. Need to provide a convenient way to use
 * it.
 */

/**
 * @brief Condition class to indicate data availability on CUDA stream completion via an event.
 *
 * A condition which specifies the availability of data at the receiver on completion of the
 * work on the provided cuda stream with the help of cuda event. This condition will keep polling
 * on the event provided to check for data availability for consumption.
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * **Note:** The nvidia::gxf::CudaEvent class is currently unused by Holoscan SDK. This condition is
 * intended exclusively for interoperation with wrapped GXF Codelets that use GXF's CudaEvent type.
 *
 * ==Parameters==
 *
 * - **receiver** (std::string): The receiver to check for a CudaEvent. This should be specified
 * by the name of the Operator's input port the condition will apply to. The Holoscan SDK will then
 * automatically replace the port name with the actual receiver object at application run time.
 * - **event_name** (std::string): The name of the CUDA event to wait on.
 */
class CudaEventCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(CudaEventCondition, GXFCondition)

  CudaEventCondition() = default;
  CudaEventCondition(const std::string& name, nvidia::gxf::CudaEventSchedulingTerm* term);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaEventSchedulingTerm"; }
  void setup(ComponentSpec& spec) override;

  void receiver(std::shared_ptr<gxf::GXFResource> receiver) { receiver_ = receiver; }
  std::shared_ptr<gxf::GXFResource> receiver() { return receiver_.get(); }

  nvidia::gxf::CudaEventSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<gxf::GXFResource>> receiver_;
  Parameter<std::string> event_name_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_EVENT_HPP */
