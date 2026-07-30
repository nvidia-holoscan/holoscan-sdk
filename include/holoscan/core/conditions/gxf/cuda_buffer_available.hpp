/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP

#include <memory>
#include <string>
#include <utility>

#include <gxf/cuda/cuda_scheduling_terms.hpp>

#include "../../component_spec.hpp"
#include "../../gxf/gxf_condition.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Condition based on data availability in a cuda buffer.
 *
 * A component which specifies the availability of data at the receiver based on the CudaBuffer
 * present in incoming messages.
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * **Note:** The `nvidia::gxf::CudaBuffer` class is currently unused by Holoscan SDK. This
 * condition is intended exclusively for interoperation with wrapped GXF Codelets that use GXF's
 * `CudaBuffer` type.
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

  void receiver(std::shared_ptr<Receiver> receiver) { receiver_ = std::move(receiver); }
  std::shared_ptr<Receiver> receiver() { return receiver_.get(); }

  nvidia::gxf::CudaBufferAvailableSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<Receiver>> receiver_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_BUFFER_AVAILABLE_HPP */
