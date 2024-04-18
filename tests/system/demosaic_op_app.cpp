/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../config.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"

#include "ping_tensor_rx_op.hpp"
#include "ping_tensor_tx_op.hpp"

static HoloscanTestConfig test_config;

class DummyDemosaicApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    int32_t rows = 3840;
    int32_t columns = 2160;
    int32_t channels = 3;
    std::string tensor_name{"signal"};
    auto tx = make_operator<ops::PingTensorTxOp>("tx",
                                                 Arg("rows", rows),
                                                 Arg("columns", columns),
                                                 Arg("channels", channels),
                                                 Arg("tensor_name", tensor_name),
                                                 make_condition<CountCondition>(3));

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);
    if (explicit_stream_pool_init_) { cuda_stream_pool->initialize(); }
    ArgList demosaic_arglist = ArgList{
        Arg("in_tensor_name", tensor_name),
        Arg("out_tensor_name", tensor_name),
        Arg("generate_alpha", false),
        Arg("bayer_grid_pos", 2),
        Arg("interpolation_mode", 0),
        Arg("pool", make_resource<BlockMemoryPool>("pool", 1, rows * columns * channels, 2)),
        Arg("cuda_stream_pool", cuda_stream_pool)};

    auto bayer_demosaic = make_operator<ops::BayerDemosaicOp>("bayer_demosaic", demosaic_arglist);

    auto rx = make_operator<ops::PingTensorRxOp>("rx", Arg("tensor_name", tensor_name));
    add_flow(tx, bayer_demosaic);
    add_flow(bayer_demosaic, rx);
  }

  void set_explicit_stream_pool_init(bool value) { explicit_stream_pool_init_ = value; }

 private:
  bool explicit_stream_pool_init_ = false;
};

TEST(DemosaicOpApp, TestDummyDemosaicApp) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos);

  // verify that there are now no warnings about GPUDevice not being found
  std::string resource_warning = "cannot find Resource of type: nvidia::gxf::GPUDevice";
  EXPECT_TRUE(log_output.find(resource_warning) == std::string::npos);

  // Verify that BlockMemoryPool and CudaStreamPool did not get initialized on a separate entity
  // from DummyDemosaicApp. (check for absence of warning from GXFResource::initialize).
  std::string graph_entity_warning = "initialized independent of a parent entity";
  EXPECT_TRUE(log_output.find(graph_entity_warning) == std::string::npos);
}

TEST(DemosaicOpApp, TestDummyDemosaicAppWithExplicitInit) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_explicit_stream_pool_init(true);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos);

  // verify that there are now no warnings about GPUDevice not being found
  std::string resource_warning = "cannot find Resource of type: nvidia::gxf::GPUDevice";
  EXPECT_TRUE(log_output.find(resource_warning) == std::string::npos);

  // Due to `set_explicit_stream_pool_init = true` we expect to see a warning from
  // GXFResource::initialize due to explicit initialization of a resource to its own entity.
  std::string graph_entity_warning = "initialized independent of a parent entity";
  EXPECT_TRUE(log_output.find(graph_entity_warning) != std::string::npos);
}
