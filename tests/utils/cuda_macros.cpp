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

#include <gtest/gtest.h>

#include <string>

#include <holoscan/utils/cuda_macros.hpp>

TEST(CudaMacros, CudaCall) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  cudaError_t expected_error = cudaErrorInvalidValue;
  int line = __LINE__ + 1;
  cudaError_t error = HOLOSCAN_CUDA_CALL(expected_error);
  EXPECT_EQ(error, expected_error);
  EXPECT_EQ(
      testing::internal::GetCapturedStderr(),
      fmt::format("[error] [cuda_macros.cpp:{}] CUDA Runtime call expected_error in line {} of "
                  "file {} failed with 'invalid argument' ({}).\n",
                  line,
                  line,
                  __FILE__,
                  (int)expected_error));
}

TEST(CudaMacros, CudaCallWarn) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  cudaError_t expected_error = cudaErrorInvalidValue;
  int line = __LINE__ + 1;
  cudaError_t error = HOLOSCAN_CUDA_CALL_WARN(expected_error);
  EXPECT_EQ(error, expected_error);
  EXPECT_EQ(
      testing::internal::GetCapturedStderr(),
      fmt::format("[warning] [cuda_macros.cpp:{}] CUDA Runtime call expected_error in line {} of "
                  "file {} failed with 'invalid argument' ({}).\n",
                  line,
                  line,
                  __FILE__,
                  (int)expected_error));
}

TEST(CudaMacros, CudaCallErrMsg) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  std::string message("message");
  cudaError_t expected_error = cudaErrorInvalidValue;
  int line = __LINE__ + 1;
  cudaError_t error = HOLOSCAN_CUDA_CALL_ERR_MSG(expected_error, message);
  EXPECT_EQ(error, expected_error);
  EXPECT_EQ(
      testing::internal::GetCapturedStderr(),
      fmt::format("[error] [cuda_macros.cpp:{}] CUDA Runtime call expected_error in line {} of "
                  "file {} failed with 'invalid argument' ({}).\n[error] [cuda_macros.cpp:{}] {}\n",
                  line,
                  line,
                  __FILE__,
                  (int)expected_error,
                  line,
                  message));
}

TEST(CudaMacros, CudaCallWarnMsg) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  std::string message("message");
  cudaError_t expected_error = cudaErrorInvalidValue;
  int line = __LINE__ + 1;
  cudaError_t error = HOLOSCAN_CUDA_CALL_WARN_MSG(expected_error, message);
  EXPECT_EQ(error, expected_error);
  EXPECT_EQ(testing::internal::GetCapturedStderr(),
            fmt::format(
                "[warning] [cuda_macros.cpp:{}] CUDA Runtime call expected_error in line {} of "
                "file {} failed with 'invalid argument' ({}).\n[warning] [cuda_macros.cpp:{}] {}\n",
                line,
                line,
                __FILE__,
                (int)expected_error,
                line,
                message));
}

TEST(CudaMacros, CudaCallThrowError) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  std::string what("exception message");
  cudaError_t expected_error = cudaErrorInvalidValue;
  int line;
  EXPECT_THROW(
      {
        try {
          line = __LINE__ + 1;
          HOLOSCAN_CUDA_CALL_THROW_ERROR(expected_error, what);
        } catch (const std::runtime_error& e) {
          EXPECT_EQ(e.what(), what);
          throw;
        }
      },
      std::runtime_error);

  EXPECT_EQ(
      testing::internal::GetCapturedStderr(),
      fmt::format("[error] [cuda_macros.cpp:{}] CUDA Runtime call expected_error in line {} of "
                  "file {} failed with 'invalid argument' ({}).\n",
                  line,
                  line,
                  __FILE__,
                  (int)expected_error));
}
