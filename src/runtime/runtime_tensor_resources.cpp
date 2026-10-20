/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_tensor_resources.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "../compiler/execution_plan_state.hpp"
#include "runtime_tensor_output_pool.hpp"
#include "runtime_tensor_resource_materialization.hpp"
#include "runtime_work_inventory.hpp"

namespace holoscan::detail {

class RuntimeTensorResources::Impl final {
 public:
  std::vector<RuntimeTensorDeviceResource> devices;
  std::unique_ptr<RuntimeTensorOutputResources> outputs;
};

RuntimeTensorResources::RuntimeTensorResources(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeTensorResources::~RuntimeTensorResources() = default;

expected<std::unique_ptr<RuntimeTensorResources>, Error> RuntimeTensorResources::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, ProcessGroupId process_group,
    const RuntimeCudaExecutionResources& cuda_resources,
    RuntimeRateMaterializationView materialization) {
  if (plan.plan_generation == 0U || run_generation == 0U ||
      (!process_group && has_structural_runtime_topology(plan)) ||
      !std::is_sorted(plan.tensor_device_resources.begin(),
                      plan.tensor_device_resources.end(),
                      canonical_tensor_device_resource_order)) {
    return make_unexpected(tensor_resource_error(
        "tensor resource inventory invalid", "plan_generation", plan.plan_generation));
  }

  auto output_plan = plan_runtime_tensor_output_materialization(
      plan, run_generation, process_group, materialization);
  if (!output_plan) {
    return make_unexpected(std::move(output_plan.error()));
  }

  auto impl = std::make_unique<Impl>();
  auto device_resources = materialize_tensor_device_resources(plan, process_group, impl->devices);
  if (!device_resources) {
    return make_unexpected(std::move(device_resources.error()));
  }
  auto operator_streams = bind_tensor_operator_streams(plan, operators, cuda_resources);
  if (!operator_streams) {
    return make_unexpected(std::move(operator_streams.error()));
  }
  auto outputs =
      RuntimeTensorOutputResources::create(plan,
                                           operators,
                                           cuda_resources,
                                           impl->devices,
                                           run_generation,
                                           process_group,
                                           plan.executable_factories.test_tensor_output_control,
                                           materialization,
                                           std::move(*output_plan));
  if (!outputs) {
    return make_unexpected(std::move(outputs.error()));
  }
  impl->outputs = std::move(*outputs);
  return std::unique_ptr<RuntimeTensorResources>{new RuntimeTensorResources{std::move(impl)}};
}

expected<RuntimeFixedTensorAddress, Error> RuntimeTensorResources::fixed_output_address(
    const PortId output) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("Tensor resources are unavailable"));
  }
  return impl_->outputs->fixed_output_address(output);
}

expected<bool, Error> RuntimeTensorResources::fixed_output_available(const PortId output) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("Tensor resources are unavailable"));
  }
  return impl_->outputs->fixed_output_available(output);
}

bool RuntimeTensorResources::restore_output_binding(const PortId output) noexcept {
  return impl_ != nullptr && impl_->outputs != nullptr &&
         impl_->outputs->restore_output_binding(output);
}

void RuntimeTensorResources::close() noexcept {
  if (impl_ != nullptr && impl_->outputs != nullptr) {
    impl_->outputs->close();
  }
}

expected<bool, Error> RuntimeTensorResources::preflight_owner_terminal_close(
    const OperatorInstanceId owner) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return impl_->outputs->preflight_owner_terminal_close(owner);
}

expected<bool, Error> RuntimeTensorResources::advance_owner_terminal_retirement(
    const OperatorInstanceId owner) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return impl_->outputs->advance_owner_terminal_retirement(owner);
}

void RuntimeTensorResources::detach_ports() noexcept {
  if (impl_ != nullptr && impl_->outputs != nullptr) {
    impl_->outputs->detach_ports();
  }
}

expected<void, Error> RuntimeTensorResources::prepare_rate_candidate_startup(
    RuntimeRateMaterializationView materialization) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("startup rate candidate state invalid"));
  }
  return impl_->outputs->prepare_rate_candidate_startup(materialization);
}

expected<void, Error> RuntimeTensorResources::begin_restart(
    std::span<const OperatorInstanceId> members, LifecycleStage from_stage) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("restart request state invalid"));
  }
  return impl_->outputs->begin_restart(members, from_stage);
}

expected<void, Error> RuntimeTensorResources::prepare_rate_candidate_restart(
    RuntimeRateMaterializationView materialization) noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("restart rate candidate state invalid"));
  }
  return impl_->outputs->prepare_rate_candidate_restart(materialization);
}

expected<void, Error> RuntimeTensorResources::publish_restart() noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("restart publication state invalid"));
  }
  return impl_->outputs->publish_restart();
}

expected<void, Error> RuntimeTensorResources::preflight_restart_publication() noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("restart publication state invalid"));
  }
  return impl_->outputs->preflight_restart_publication();
}

expected<void, Error> RuntimeTensorResources::rollback_restart() noexcept {
  if (impl_ == nullptr || impl_->outputs == nullptr) {
    return make_unexpected(tensor_resource_error("restart rollback state invalid"));
  }
  return impl_->outputs->rollback_restart();
}

std::size_t RuntimeTensorResources::device_resource_count() const noexcept {
  return impl_ != nullptr ? impl_->devices.size() : 0U;
}

std::size_t RuntimeTensorResources::output_pool_count() const noexcept {
  return impl_ != nullptr && impl_->outputs != nullptr ? impl_->outputs->output_pool_count() : 0U;
}

}  // namespace holoscan::detail
