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

#include <memory>
#include <string>
#include <utility>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

#include "../env_wrapper.hpp"
#include "distributed_app_fixture.hpp"
#include "utility_apps.hpp"

class DistributedDemosaicOpApp : public DistributedApp {};

class GenerateAndDemosaicFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    int32_t rows = 3840;
    int32_t columns = 2160;
    int32_t channels = 1;
    std::string tensor_name{"signal"};
    auto tx = make_operator<ops::PingTensorTxOp>("tx",
                                                 Arg("rows", rows),
                                                 Arg("columns", columns),
                                                 Arg("channels", channels),
                                                 Arg("tensor_name", tensor_name),
                                                 make_condition<CountCondition>(3));

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);
    int32_t out_channels = 3;
    ArgList demosaic_arglist = ArgList{
        Arg("in_tensor_name", tensor_name),
        Arg("out_tensor_name", tensor_name),
        Arg("generate_alpha", false),
        Arg("bayer_grid_pos", 2),
        Arg("interpolation_mode", 0),
        // The pool size is set to 10 to prevent memory allocation errors during testing.
        // Additional memory pool may be required as UcxTransmitter sends data asynchronously
        // without checking the receiver's queue.
        Arg("pool", make_resource<BlockMemoryPool>("pool", 1, rows * columns * out_channels, 10)),
        Arg("cuda_stream_pool", cuda_stream_pool)};

    std::shared_ptr<Operator> bayer_demosaic;
    // native Operator version
    bayer_demosaic = make_operator<ops::BayerDemosaicOp>("bayer_demosaic", demosaic_arglist);
    add_flow(tx, bayer_demosaic);
  }
};

class RxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<ops::PingTensorRxOp>("rx");
    add_operator(rx);
  }
};

/**
 * @brief Distributed version of DummyDemosaicApp.
 *
 * The purpose of this app is to test transmission of entities containing a stream ID
 * (nvidia::gxf::CudaStreamId) across fragments.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: signal
 *   - Operator (bayer_demosaic)
 *     - Input port: signal
 *     - Output port: signal
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: signal
 *
 * The following connections are established:
 *
 * - fragment1.bayer_demosaic -> fragment2.rx
 *
 */
class DistributedDummyDemosaicApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<GenerateAndDemosaicFragment>("fragment1");
    auto fragment2 = make_fragment<RxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"bayer_demosaic", "rx"}});
  }
};

TEST_F(DistributedDemosaicOpApp, TestDistributedDummyDemosaicApp) {
  using namespace holoscan;

  auto app = make_application<DistributedDummyDemosaicApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) { HOLOSCAN_LOG_ERROR("Exception: {}", e.what()); }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // Currently expect a warning that the CUDA stream ID object will not be serialized
  // over the distributed connection.
  std::string serializer_warning = "No serializer found for component 'cuda_stream_id_'";
  EXPECT_TRUE(log_output.find(serializer_warning) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
