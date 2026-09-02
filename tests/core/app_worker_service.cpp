/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <holoscan/core/system/network_utils.hpp>

#include "../../src/core/distributed/app_worker/service_impl.hpp"

namespace holoscan::distributed {

TEST(AppWorkerService, RejectsUnboundedAvailablePortRequest) {
  AppWorkerServiceImpl service(nullptr);
  grpc::ServerContext context;
  AvailablePortsRequest request;
  AvailablePortsResponse response;
  request.set_number_of_ports(std::numeric_limits<uint32_t>::max());
  request.set_min_port(10000);
  request.set_max_port(32767);

  const auto status = service.GetAvailablePorts(&context, &request, &response);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(response.unused_ports_size(), 0);
}

TEST(AppWorkerService, RejectsInvalidPortMetadata) {
  AppWorkerServiceImpl service(nullptr);
  grpc::ServerContext context;
  AvailablePortsRequest request;
  AvailablePortsResponse response;
  request.set_number_of_ports(1);
  request.set_min_port(20000);
  request.set_max_port(10000);

  auto status = service.GetAvailablePorts(&context, &request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  request.set_min_port(10000);
  request.set_max_port(32767);
  request.add_used_ports(65536);
  status = service.GetAvailablePorts(&context, &request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(NetworkUtils, CapsReservationForImpossiblePortCount) {
  constexpr int kOnlyCandidate = 12345;
  const auto ports = get_unused_network_ports(
      std::numeric_limits<uint32_t>::max(), kOnlyCandidate, kOnlyCandidate, {kOnlyCandidate});

  EXPECT_TRUE(ports.empty());
}

}  // namespace holoscan::distributed
