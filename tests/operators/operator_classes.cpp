/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

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

#include "holoscan/operators/aja_source/aja_source.hpp"
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/multiai_inference/multiai_inference.hpp"
#include "holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp"
#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"
#include "holoscan/operators/tensor_rt/tensor_rt_inference.hpp"
#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"
#include "holoscan/operators/video_stream_replayer/video_stream_replayer.hpp"

using namespace std::string_literals;

namespace holoscan {

using OperatorClassesWithGXFContext = TestWithGXFContext;

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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'aja-source'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'aja-source'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
}

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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'format_converter'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
}

TEST_F(OperatorClassesWithGXFContext, TestTensorRTInferenceOp) {
  const std::string name{"segmentation_inference"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("segmentation_inference");
  kwargs.add(Arg{"pool", F.make_resource<UnboundedAllocator>("pool")});
  kwargs.add(
      Arg{"cuda_stream_pool", F.make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5)});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::TensorRTInferenceOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::TensorRTInferenceOp>(kwargs)));

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos);
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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'recorder'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
}

TEST_F(OperatorClassesWithGXFContext, TestVideoStreamReplayerOp) {
  const std::string name{"replayer"};
  const std::string sample_data_path = std::string(std::getenv("HOLOSCAN_SAMPLE_DATA_PATH"));
  ArgList args{
      Arg{"directory", sample_data_path + "/endoscopy/video"s},
      Arg{"basename", "surgical_video"s},
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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'replayer'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
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

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator
    //   'segmentation_postprocessor'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
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

  std::string log_output = testing::internal::GetCapturedStderr();
  // EXPECT_TRUE(log_output.find("[error]") == std::string::npos);
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'holoviz'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
}

TEST_F(OperatorClassesWithGXFContext, TestMultiAIInferenceOp) {
  const std::string name{"multiai_inference"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("multiai_inference");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::MultiAIInferenceOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::MultiAIInferenceOp>(kwargs)));

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator 'multiai_inference'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
}

TEST_F(OperatorClassesWithGXFContext, TestMultiAIPostprocessorOp) {
  const std::string name{"multiai_postprocessor"};

  // load most arguments from the YAML file
  ArgList kwargs = F.from_config("multiai_postprocessor");
  kwargs.add(Arg{"allocator", F.make_resource<UnboundedAllocator>("pool")});

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::MultiAIPostprocessorOp>(name, kwargs);
  EXPECT_EQ(op->name(), name);
  EXPECT_EQ(typeid(op), typeid(std::make_shared<ops::MultiAIPostprocessorOp>(kwargs)));

  std::string log_output = testing::internal::GetCapturedStderr();
  auto error_pos = log_output.find("[error]");
  if (error_pos != std::string::npos) {
    // Initializing a native operator outside the context of app.run() will result in the
    // following error being logged because the GXFWrapper will not yet have been created for
    // the operator:
    //   [error] [gxf_executor.cpp:452] Unable to get GXFWrapper for Operator
    //   'multiai_postprocessor'

    // GXFWrapper was mentioned and no additional error was logged
    EXPECT_TRUE(log_output.find("GXFWrapper", error_pos + 1) != std::string::npos);
    EXPECT_TRUE(log_output.find("[error]", error_pos + 1) == std::string::npos);
  }
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

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos);
}

TEST_F(OperatorClassesWithGXFContext, TestBayerDemosaicOpDefaultConstructor) {
  const std::string name{"bayer_demosaic"};

  testing::internal::CaptureStderr();

  auto op = F.make_operator<ops::BayerDemosaicOp>();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos);
}

TEST(Operator, TestNativeOperatorWithoutFragment) {
  Operator op;
  EXPECT_EQ(op.name(), ""s);

  // test that calling initialize without setting a Fragment will warn
  testing::internal::CaptureStderr();

  op.initialize();
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[warning]") != std::string::npos);
  EXPECT_TRUE(log_output.find("Fragment is not set") != std::string::npos);
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos);
}

}  // namespace holoscan
