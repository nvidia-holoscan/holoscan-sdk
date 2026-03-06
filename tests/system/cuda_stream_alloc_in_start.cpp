/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Tests that ExecutionContext::allocate_cuda_stream works from an operator's start() method,
 * regardless of how the CudaStreamPool was provided:
 *   1. Named Arg("cuda_stream_pool", pool)
 *   2. Positionally via make_operator (stored as a resource, not a parameter)
 *   3. No explicit pool (CudaObjectHandler creates a default pool)
 *
 * This validates the fix for the issue where positionally-passed CudaStreamPool resources
 * were silently ignored when InferenceOp used cuda_stream_pool_.try_get() directly.
 */

#include <gtest/gtest.h>

#include <string>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"

namespace holoscan {
namespace ops {

/**
 * @brief Operator that allocates a CUDA stream in start() via ExecutionContext.
 *
 * Mimics the InferenceOp pattern: has an optional cuda_stream_pool parameter and allocates
 * streams in start() rather than compute(). The operator logs success/failure so tests can
 * verify behavior via captured log output.
 */
class StreamAllocInStartOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamAllocInStartOp)

  StreamAllocInStartOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CUDA Stream Pool",
               "Optional CUDA stream pool.",
               ParameterFlag::kOptional);
  }

  void start() override {
    auto exec_ctx = execution_context();
    if (!exec_ctx) {
      HOLOSCAN_LOG_ERROR("{}: ExecutionContext is null in start()", name());
      return;
    }

    auto maybe_stream = exec_ctx->allocate_cuda_stream("test_stream");
    if (!maybe_stream) {
      HOLOSCAN_LOG_ERROR("{}: Failed to allocate stream in start()", name());
      return;
    }
    allocated_stream_ = maybe_stream.value();

    if (allocated_stream_ != cudaStreamDefault) {
      HOLOSCAN_LOG_INFO(
          "{}: allocated non-default stream in start: {}", name(), fmt::ptr(allocated_stream_));
    } else {
      HOLOSCAN_LOG_ERROR("{}: allocated stream is the default stream", name());
    }

    // Verify device_from_stream works on the allocated stream
    auto maybe_dev = exec_ctx->device_from_stream(allocated_stream_);
    if (maybe_dev) {
      HOLOSCAN_LOG_INFO("{}: stream is on device {}", name(), maybe_dev.value());
    } else {
      HOLOSCAN_LOG_ERROR("{}: device_from_stream failed", name());
    }

    // Verify idempotency: same name returns same stream
    auto maybe_stream2 = exec_ctx->allocate_cuda_stream("test_stream");
    if (maybe_stream2 && maybe_stream2.value() == allocated_stream_) {
      HOLOSCAN_LOG_INFO("{}: same name returned same stream (idempotent)", name());
    } else {
      HOLOSCAN_LOG_ERROR("{}: same name returned different stream (not idempotent)", name());
    }
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_value = op_input.receive<int>("in");
    if (allocated_stream_ != nullptr && allocated_stream_ != cudaStreamDefault) {
      HOLOSCAN_LOG_INFO("{}: stream valid in compute", name());
    }
  }

 private:
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};
  cudaStream_t allocated_stream_ = nullptr;
};

}  // namespace ops

// ============================================================================
// Test applications
// ============================================================================

/**
 * @brief App that passes CudaStreamPool as a named Arg("cuda_stream_pool", pool).
 *
 * This is the standard approach that has always worked.
 */
class StreamAllocStartNamedArgApp : public holoscan::Application {
  void compose() override {
    using namespace holoscan;

    auto pool = make_resource<CudaStreamPool>("pool", 0, 0, 0, 1, 5);

    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(3));
    auto rx = make_operator<ops::StreamAllocInStartOp>("rx", Arg("cuda_stream_pool", pool));

    add_flow(tx, rx, {{"out", "in"}});
  }
};

/**
 * @brief App that passes CudaStreamPool positionally (as a resource, not a named parameter).
 *
 * This is the pattern used in the cuda_green_context example. Previously, InferenceOp's
 * cuda_stream_pool_.try_get() would fail for this case because positional resources go to
 * the operator's resources_ map, not to the typed Parameter. CudaObjectHandler's resource
 * scan handles this correctly.
 */
class StreamAllocStartPositionalApp : public holoscan::Application {
  void compose() override {
    using namespace holoscan;

    auto pool = make_resource<CudaStreamPool>("pool", 0, 0, 0, 1, 5);

    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(3));
    auto rx = make_operator<ops::StreamAllocInStartOp>("rx", pool);

    add_flow(tx, rx, {{"out", "in"}});
  }
};

/**
 * @brief App that provides no explicit CudaStreamPool.
 *
 * CudaObjectHandler creates a default pool (capacity 1, unlimited max) automatically.
 */
class StreamAllocStartDefaultPoolApp : public holoscan::Application {
  void compose() override {
    using namespace holoscan;

    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(3));
    auto rx = make_operator<ops::StreamAllocInStartOp>("rx");

    add_flow(tx, rx, {{"out", "in"}});
  }
};

// ============================================================================
// Helper to run an app and check for expected log messages
// ============================================================================

static void run_and_verify_stream_alloc(holoscan::Application& app) {
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app.run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify stream was allocated successfully in start()
  std::string alloc_msg = "rx: allocated non-default stream in start";
  EXPECT_TRUE(log_output.find(alloc_msg) != std::string::npos)
      << "Expected stream allocation success message in log output.\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify device_from_stream worked
  std::string device_msg = "rx: stream is on device";
  EXPECT_TRUE(log_output.find(device_msg) != std::string::npos)
      << "Expected device_from_stream success message in log output.\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify idempotency (same name returns same stream)
  std::string idempotent_msg = "rx: same name returned same stream (idempotent)";
  EXPECT_TRUE(log_output.find(idempotent_msg) != std::string::npos)
      << "Expected idempotency message in log output.\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify stream remains valid during compute()
  std::string compute_msg = "rx: stream valid in compute";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected stream valid in compute message in log output.\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no error messages
  EXPECT_EQ(log_output.find("Failed to allocate stream"), std::string::npos)
      << "Unexpected allocation failure.\n=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_EQ(log_output.find("ExecutionContext is null"), std::string::npos)
      << "Unexpected null ExecutionContext.\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

// ============================================================================
// Tests
// ============================================================================

TEST(CudaStreamAllocInStart, NamedArg) {
  auto app = holoscan::make_application<StreamAllocStartNamedArgApp>();
  run_and_verify_stream_alloc(*app);
}

TEST(CudaStreamAllocInStart, PositionalResource) {
  auto app = holoscan::make_application<StreamAllocStartPositionalApp>();
  run_and_verify_stream_alloc(*app);
}

TEST(CudaStreamAllocInStart, DefaultPool) {
  auto app = holoscan::make_application<StreamAllocStartDefaultPoolApp>();
  run_and_verify_stream_alloc(*app);
}

}  // namespace holoscan
