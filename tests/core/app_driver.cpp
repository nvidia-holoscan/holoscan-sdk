/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <stdlib.h>  // POSIX setenv

#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "holoscan/core/app_driver.hpp"
#include <holoscan/core/system/system_resource_manager.hpp>

namespace holoscan {

TEST(AppDriver, TestSetUcxToExcludeCudaIpc) {
  const char* env_orig = std::getenv("UCX_TLS");

  // if unset, will be set to ^cuda_ipc
  if (env_orig) {
    unsetenv("UCX_TLS");
  }
  AppDriver::set_ucx_to_exclude_cuda_ipc();
  const char* env_var = std::getenv("UCX_TLS");
  if (env_var) {
    EXPECT_EQ(std::string{env_var}, std::string("^cuda_ipc"));
  } else {
    FAIL() << "UCX_TLS environment variable not set";
  }

  // does not clobber pre-existing UCX_TLS value
  const char* new_env_var = "tcp,cuda_copy";
  setenv("UCX_TLS", new_env_var, 1);
  AppDriver::set_ucx_to_exclude_cuda_ipc();
  env_var = std::getenv("UCX_TLS");
  if (env_var) {
    EXPECT_EQ(std::string{env_var}, std::string(new_env_var));
  } else {
    FAIL() << "UCX_TLS environment variable not set";
  }

  // warns if allow list contains cuda_ipc
  testing::internal::CaptureStderr();
  new_env_var = "tcp,cuda_ipc";
  setenv("UCX_TLS", new_env_var, 1);
  AppDriver::set_ucx_to_exclude_cuda_ipc();
  std::string log_output = testing::internal::GetCapturedStderr();
  env_var = std::getenv("UCX_TLS");
  if (env_var) {
    EXPECT_EQ(std::string{env_var}, std::string(new_env_var));
  } else {
    FAIL() << "UCX_TLS environment variable not set";
  }
  EXPECT_TRUE(log_output.find("warn") != std::string::npos) << "=== LOG ===\n"
                                                            << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("UCX_TLS is set") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // restore the original environment variable
  if (env_orig) {
    setenv("UCX_TLS", env_orig, 1);
  } else {
    unsetenv("UCX_TLS");
  }
}

TEST(AppDriver, TestExcludeCudaIpcTransportOnIgpu) {
  const char* env_orig = std::getenv("UCX_TLS");

  bool is_integrated = false;
  // Ensure that GPUResourceMonitor is instantiated in a separate scope to avoid nested nvmlInit()
  // calls (AppDriver::exclude_cuda_ipc_transport_on_igpu() creates a GPUResourceMonitor instance).
  {
    // Check if we are running on an integrated GPU.
    holoscan::GPUResourceMonitor gpu_resource_monitor;
    gpu_resource_monitor.update();
    is_integrated =
        (gpu_resource_monitor.num_gpus() > 0) && gpu_resource_monitor.is_integrated_gpu(0);
  }

  // if unset and on iGPU, will be set to ^cuda_ipc
  if (env_orig) {
    unsetenv("UCX_TLS");
  }
  AppDriver::exclude_cuda_ipc_transport_on_igpu();
  const char* env_var = std::getenv("UCX_TLS");
  if (is_integrated) {
    if (env_var) {
      EXPECT_EQ(std::string{env_var}, std::string("^cuda_ipc"));
    } else {
      FAIL() << "UCX_TLS environment variable not set";
    }
  } else {
    // environment variable is not set when not in iGPU environment
    EXPECT_EQ(env_var, nullptr);
  }

  // does not clobber pre-existing UCX_TLS value
  const char* new_env_var = "tcp,cuda_copy";
  setenv("UCX_TLS", new_env_var, 1);
  AppDriver::exclude_cuda_ipc_transport_on_igpu();
  env_var = std::getenv("UCX_TLS");
  if (env_var) {
    EXPECT_EQ(std::string{env_var}, std::string(new_env_var));
  } else {
    FAIL() << "UCX_TLS environment variable not set";
  }

  // warns if on iGPU and allow list contains cuda_ipc
  testing::internal::CaptureStderr();
  new_env_var = "tcp,cuda_ipc";
  setenv("UCX_TLS", new_env_var, 1);
  AppDriver::exclude_cuda_ipc_transport_on_igpu();
  std::string log_output = testing::internal::GetCapturedStderr();
  env_var = std::getenv("UCX_TLS");
  if (env_var) {
    EXPECT_EQ(std::string{env_var}, std::string(new_env_var));
  } else {
    FAIL() << "UCX_TLS environment variable not set";
  }
  if (is_integrated) {
    EXPECT_TRUE(log_output.find("warn") != std::string::npos) << "=== LOG ===\n"
                                                              << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find("UCX_TLS is set") != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  } else {
    EXPECT_TRUE(log_output.find("UCX_TLS is set") == std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // restore the original environment variable
  if (env_orig) {
    setenv("UCX_TLS", env_orig, 1);
  } else {
    unsetenv("UCX_TLS");
  }
}

TEST(AppDriver, TestGetBoolEnvVar) {
  // Test cases for get_bool_env_var (issue 4616525)
  std::vector<std::pair<std::string, bool>> test_cases{
      {"tRue", true},
      {"False", false},
      {"1", true},
      {"0", false},
      {"On", true},
      {"Off", false},
      {"", true},
      {"invalid", true},
  };

  const char* env_name = "HOLOSCAN_TEST_BOOL_ENV_VAR";
  for (const auto& [env_value, expected_result] : test_cases) {
    setenv(env_name, env_value.c_str(), 1);
    bool result = AppDriver::get_bool_env_var(env_name, true);
    EXPECT_EQ(result, expected_result);
    unsetenv(env_name);
  }
}

}  // namespace holoscan
