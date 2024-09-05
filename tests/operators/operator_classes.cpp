/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../utils.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "common/assert.hpp"

#ifdef HOLOSCAN_BUILD_AJA
#include "holoscan/operators/aja_source/aja_source.hpp"
#endif
#include "holoscan/operators/async_ping_rx/async_ping_rx.hpp"
#include "holoscan/operators/async_ping_tx/async_ping_tx.hpp"
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/inference/inference.hpp"
#include "holoscan/operators/inference_processor/inference_processor.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"
#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"
#include "holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp"
#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"
#include "holoscan/operators/video_stream_replayer/video_stream_replayer.hpp"

using namespace std::string_literals;

namespace holoscan {

using OperatorClassesWithGXFContext = TestWithGXFContext;

#ifdef HOLOSCAN_BUILD_AJA
TEST_F(OperatorClassesWithGXFContext, TestAJASourceOpChannelFromYAML) {
  const std::string name{"aja-source"};

  ArgList args{
      Arg{"device", "0"s},
      Arg{"channel", YAML::Load("NTV2_CHANNEL1"s)},
      Arg{"width", static_cast<uint32_t>(1920)},
      Arg{"height", static_cast<uint32_t>(1080)},
      Arg{"framerate", static_cast<uint32_t>(60)},
      Arg{"rdma", false},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::AJASourceOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::AJASourceOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(TestWithGXFContext, TestAJASourceOpChannelFromEnum) {
  const std::string name{"aja-source"};

  ArgList args{
      Arg{"device", "0"s},
      Arg{"channel", NTV2Channel::NTV2_CHANNEL1},
      Arg{"width", static_cast<uint32_t>(1920)},
      Arg{"height", static_cast<uint32_t>(1080)},
      Arg{"framerate", static_cast<uint32_t>(60)},
      Arg{"rdma", false},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::AJASourceOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::AJASourceOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}
#endif

TEST_F(OperatorClassesWithGXFContext, TestFormatConverterOp) {
  const std::string name{"format_converter"};

  ArgList pool_arglist{
      Arg{"storage_type", static_cast<int32_t>(1)},
      Arg{"block_size", static_cast<uint64_t>(1024 * 1024 * 16)},
      Arg{"num_blocks", static_cast<uint64_t>(1)},
  };

  ArgList args{
      Arg{"in_tensor_name", "in"s},
      Arg{"in_dtype", ""s},
      Arg{"out_tensor_name", "out"s},
      Arg{"out_dtype", "float32"s},
      Arg{"scale_min", 0.f},
      Arg{"scale_max", 1.f},
      Arg{"alpha_value", static_cast<uint8_t>(255)},
      Arg{"resize_width", 0},
      Arg{"resize_height", 0},
      Arg{"resize_mode", 0},
      Arg{"out_channel_order", std::vector<int>{{1, 2, 3}}},
      Arg{"pool", F.make_resource<BlockMemoryPool>(pool_arglist)},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::FormatConverterOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::FormatConverterOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestVideoStreamRecorderOp) {
  const std::string name{"recorder"};
  ArgList args{
      Arg{"directory", "/tmp"s},
      Arg{"basename", "video_out"s},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::VideoStreamRecorderOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::VideoStreamRecorderOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestVideoStreamReplayerOp) {
  const std::string name{"replayer"};
  auto in_path = std::getenv("HOLOSCAN_INPUT_PATH");
  if (!in_path) { GTEST_SKIP() << "Skipping test due to undefined HOLOSCAN_INPUT_PATH env var"; }
  const std::string sample_data_path = std::string(in_path);
  ArgList args{
      Arg{"directory", sample_data_path + "/racerx"s},
      Arg{"basename", "racerx"s},
      Arg{"batch_size", static_cast<size_t>(1UL)},
      Arg{"ignore_corrupted_entities", true},
      Arg{"frame_rate", 0.f},
      Arg{"realtime", true},
      Arg{"repeat", false},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::VideoStreamReplayerOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::VideoStreamReplayerOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestSegmentationPostprocessorOp) {
  const std::string name{"segmentation_postprocessor"};

  ArgList args{
      Arg{"in_tensor_name", "inference_output_tensor"s},
      Arg{"network_output_type", "softmax"s},
      Arg{"data_format", "nchw"s},
      Arg{"allocator", F.make_resource<UnboundedAllocator>("allocator")},
  };
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::SegmentationPostprocessorOp>(name, args);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::SegmentationPostprocessorOp>(args)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestHolovizOp) {
  const std::string name{"holoviz"};

  ArgList kwargs = F.from_config("holoviz");

  std::vector<std::vector<float>> color_lut = {
      {0.65f, 0.81f, 0.89f, 0.1f},
      {0.2f, 0.63f, 0.17f, 0.7f},
      {0.98f, 0.6f, 0.6f, 0.7f},
      {0.89f, 0.1f, 0.11f, 0.7f},
      {0.99f, 0.75f, 0.44f, 0.7f},
      {1.0f, 0.5f, 0.0f, 0.7f},
      {0.0f, 0.0f, 0.0f, 0.1f},
  };
  kwargs.add(Arg{"color_lut", color_lut});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::HolovizOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::HolovizOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestHolovizOpInputSpec) {
  ops::HolovizOp::InputSpec tensor{"video", ops::HolovizOp::InputType::COLOR};

  // serialize to YAML string
  std::string description = tensor.description();

  // deserialize the YAML description string into a new InputSpec
  ops::HolovizOp::InputSpec new_spec{description};
  EXPECT_EQ(new_spec.tensor_name_, "video");
  EXPECT_EQ(new_spec.opacity_, 1.0);
  EXPECT_EQ(new_spec.priority_, 0);
}

TEST_F(OperatorClassesWithGXFContext, TestInferenceOp) {
  const std::string name{"inference"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("inference");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::InferenceOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::InferenceOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestInferenceProcessorOp) {
  const std::string name{"processor"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("processor");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::InferenceProcessorOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::InferenceProcessorOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestBayerDemosaicOp) {
  const std::string name{"bayer_demosaic"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("demosaic");
  kwargs.add(Arg{"pool", F.make_resource<UnboundedAllocator>("pool")});
  kwargs.add(
      Arg{"cuda_stream_pool", F.make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5)});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::BayerDemosaicOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::BayerDemosaicOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestBayerDemosaicOpDefaultConstructor) {
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::BayerDemosaicOp>();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST(Operator, TestNativeOperatorWithoutFragment) {
  Operator op;
  EXPECT_EQ(op.name(), ""s);

  // test that calling initialize without setting a Fragment will warn
  testing::internal::CaptureStderr();

  op.initialize();
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Fragment is not set") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestPingRxOp) {
  const std::string name{"rx"};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::PingRxOp>(name);
  EXPECT_EQ(op->name(), name);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestOperatorMetadataAttributes) {
  const std::string name{"rx"};

  testing::internal::CaptureStderr();

  // defaults to disabled
  EXPECT_FALSE(F.is_metadata_enabled());

  // enable metadata on the fragment
  F.is_metadata_enabled(true);
  EXPECT_TRUE(F.is_metadata_enabled());

  // add an operator
  auto op = F.make_operator<ops::PingRxOp>(name);
  EXPECT_EQ(op->name(), name);

  // default metadata policy is kRaise
  EXPECT_EQ(op->metadata_policy(), MetadataPolicy::kRaise);

  // at construction metadata is disabled
  EXPECT_FALSE(op->is_metadata_enabled());

  // after initialize, operator metadata will be enabled if it was enabled on the Fragment
  op->initialize();
  EXPECT_TRUE(op->is_metadata_enabled());

  std::string log_output = testing::internal::GetCapturedStderr();
}

TEST_F(OperatorClassesWithGXFContext, TestPingTxOp) {
  const std::string name{"tx"};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::PingTxOp>(name);
  EXPECT_EQ(op->name(), name);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestPingTxWithStringName) {
  std::string name{"tx"};  // string is not a const

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::PingTxOp>(name);
  EXPECT_EQ(op->name(), name);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestAsyncPingRxOp) {
  const std::string name{"async_rx"};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::AsyncPingRxOp>(name, Arg("delay", 10L), Arg("count", 10UL));
  EXPECT_EQ(op->name(), name);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestAsyncPingTxOp) {
  const std::string name{"async_tx"};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::AsyncPingTxOp>(name, Arg("delay", 10L), Arg("count", 10UL));
  EXPECT_EQ(op->name(), name);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestV4L2VideoCaptureOp) {
  const std::string name{"video_capture"};
  uint32_t width = 1024;
  uint32_t height = 768;
  uint32_t exposure_time = 500;
  uint32_t gain = 100;

  ArgList kwargs{Arg{"device", std::string("/dev/video0")},
                 Arg{"pixel_format", std::string("auto")},
                 Arg{"pass_through", false},
                 Arg{"width", width},
                 Arg{"height", height},
                 Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")},
                 Arg{"exposure_time", exposure_time},
                 Arg{"gain", gain}};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::V4L2VideoCaptureOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::V4L2VideoCaptureOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestV4L2VideoCaptureOpYAMLConfig) {
  const std::string name{"video_capture"};

  ArgList kwargs = F.from_config("v4l2_video_capture");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});
  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::V4L2VideoCaptureOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::V4L2VideoCaptureOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestV4L2VideoCaptureOpDefaults) {
  const std::string name{"video_capture"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("demosaic");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::V4L2VideoCaptureOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::V4L2VideoCaptureOp>(kwargs)));
  EXPECT_TRUE(op->description().find("name: " + name) != std::string::npos);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST_F(OperatorClassesWithGXFContext, TestInvalidOperatorName) {
  EXPECT_THROW(
      {
        // "." character is not allowed in operator names
        auto op = F.make_operator<ops::PingRxOp>("rx.1");
      },
      std::invalid_argument);
}
}  // namespace holoscan
