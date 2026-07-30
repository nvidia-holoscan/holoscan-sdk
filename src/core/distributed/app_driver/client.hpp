/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CORE_DISTRIBUTED_APP_DRIVER_CLIENT_HPP
#define CORE_DISTRIBUTED_APP_DRIVER_CLIENT_HPP

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/flow_graphs/flow_graph.hpp>
#include <holoscan/core/system/cpu_info.hpp>
#include <holoscan/core/system/gpu_info.hpp>
#include "../generated/app_driver.grpc.pb.h"

namespace holoscan {

// Forward declarations
enum class AppWorkerTerminationCode;
namespace distributed {

class AppDriverClient {
 public:
  AppDriverClient(const std::string& driver_address, const std::shared_ptr<grpc::Channel>& channel);

  bool fragment_allocation(const std::string& worker_ip, const std::string& worker_port,
                           const std::vector<FragmentNodeType>& target_fragments,
                           const CPUInfo& cpuinfo, const std::vector<GPUInfo>& gpuinfo);

  bool worker_execution_finished(const std::string& worker_ip, const std::string& worker_port,
                                 AppWorkerTerminationCode code);

  // Request the AppDriver to initiate a clean shutdown
  bool initiate_shutdown(const std::string& fragment_name);

 private:
  std::string driver_address_;
  std::unique_ptr<holoscan::distributed::AppDriverService::Stub> stub_;
};
}  // namespace distributed
}  // namespace holoscan

#endif /* CORE_DISTRIBUTED_APP_DRIVER_CLIENT_HPP */
