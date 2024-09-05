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
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

static HoloscanTestConfig test_config;

class DummyDemosaicApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    std::string tensor_name{"signal"};
    auto tx = make_operator<ops::PingTensorTxOp>("tx",
                                                 Arg("rows", rows_),
                                                 Arg("columns", columns_),
                                                 Arg("channels", channels_),
                                                 Arg("tensor_name", tensor_name),
                                                 Arg("storage_type", std::string("device")),
                                                 make_condition<CountCondition>(3));

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);
    if (explicit_stream_pool_init_) { cuda_stream_pool->initialize(); }
    bool generate_alpha = false;
    int32_t out_channels = generate_alpha ? 4 : 3;
    ArgList demosaic_arglist = ArgList{
        Arg("in_tensor_name", tensor_name),
        Arg("out_tensor_name", tensor_name),
        Arg("generate_alpha", generate_alpha),
        Arg("bayer_grid_pos", 2),
        Arg("interpolation_mode", 0),
        Arg("pool", make_resource<BlockMemoryPool>("pool", 1, rows_ * columns_ * out_channels, 2)),
        Arg("cuda_stream_pool", cuda_stream_pool)};

    auto bayer_demosaic = make_operator<ops::BayerDemosaicOp>("bayer_demosaic", demosaic_arglist);

    auto rx = make_operator<ops::PingTensorRxOp>("rx");
    add_flow(tx, bayer_demosaic);
    add_flow(bayer_demosaic, rx);
  }

  void set_explicit_stream_pool_init(bool value) { explicit_stream_pool_init_ = value; }

  void set_storage_type(const std::string& storage_type) { storage_type_ = storage_type; }

  void set_rows(int32_t rows) { rows_ = rows; }
  void set_columns(int32_t columns) { columns_ = columns; }
  void set_channels(int32_t channels) { channels_ = channels; }

 private:
  bool explicit_stream_pool_init_ = false;
  std::string storage_type_ = std::string("device");
  int32_t rows_ = 3840;
  int32_t columns_ = 3840;
  int32_t channels_ = 1;
};

class DemosaicStorageParameterizedTestFixture : public ::testing::TestWithParam<std::string> {};

INSTANTIATE_TEST_CASE_P(DemosaicOpAppTests, DemosaicStorageParameterizedTestFixture,
                        ::testing::Values(std::string("device"), std::string("host"),
                                          std::string("system")));

TEST_P(DemosaicStorageParameterizedTestFixture, TestDummyDemosaicApp) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  std::string storage_type = GetParam();
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_storage_type(storage_type);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // verify that there are now no warnings about GPUDevice not being found
  std::string resource_warning = "cannot find Resource of type: nvidia::gxf::GPUDevice";
  EXPECT_TRUE(log_output.find(resource_warning) == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify that BlockMemoryPool and CudaStreamPool did not get initialized on a separate entity
  // from DummyDemosaicApp. (check for absence of warning from GXFResource::initialize).
  std::string graph_entity_warning = "initialized independent of a parent entity";
  EXPECT_TRUE(log_output.find(graph_entity_warning) == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
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
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // verify that there are now no warnings about GPUDevice not being found
  std::string resource_warning = "cannot find Resource of type: nvidia::gxf::GPUDevice";
  EXPECT_TRUE(log_output.find(resource_warning) == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Due to `set_explicit_stream_pool_init = true` we expect to see a warning from
  // GXFResource::initialize due to explicit initialization of a resource to its own entity.
  std::string graph_entity_warning = "initialized independent of a parent entity";
  EXPECT_TRUE(log_output.find(graph_entity_warning) != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(DemosaicOpApp, TestDummyDemosaicAppWithOddRows) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_rows(1919);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Input tensor must have an even number of rows") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(DemosaicOpApp, TestDummyDemosaicAppWithOddColumns) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_columns(799);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Input tensor must have an even number of columns") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(DemosaicOpApp, TestDummyDemosaicAppWithMultipleChannels) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_channels(2);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  EXPECT_THROW(app->run(), std::runtime_error);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("For 3D HWC input, the number of channels, C, must be 1.") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(DemosaicOpApp, TestDummyDemosaicAppWith2DInput) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<DummyDemosaicApp>();
  app->set_channels(0);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Graph activation failed") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
