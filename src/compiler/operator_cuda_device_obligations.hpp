/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <vector>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// One compiler-derived reason that an Operator requires a CUDA compute stream on a device.
struct OperatorCudaDeviceObligation {
  VertexId operator_vertex{};
  PortId source_port{};
  DeviceId device{};
};

/// Borrow every physical-plan record family that can make an Operator CUDA-active.
struct OperatorCudaDeviceObligationView {
  std::span<const PortPlanRecord> ports;
  std::span<const InterfacePortBindingPlanRecord> interface_bindings;
  std::span<const TensorAllocationPlanRecord> tensor_allocations;
  std::span<const TensorFieldAllocationPlanRecord> tensor_field_allocations;
  std::span<const ProcessLocalChannelServicePlanRecord> process_local_services;
  std::span<const ProcessLocalChannelSubscriberPlanRecord> process_local_subscribers;
  std::span<const CrossProcessChannelServicePlanRecord> cross_process_services;
  std::span<const CrossProcessChannelSubscriberPlanRecord> cross_process_subscribers;
  std::span<const TensorInterfacePoolPlanRecord> tensor_interface_pools;
};

/// Resolve the shared Tensor/service/interface obligation walk used by lowering and validation.
[[nodiscard]] std::vector<OperatorCudaDeviceObligation> collect_operator_cuda_device_obligations(
    const OperatorCudaDeviceObligationView& view);

}  // namespace holoscan::detail
