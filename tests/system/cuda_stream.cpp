/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../config.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/operators/format_converter/format_converter.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"

namespace holoscan {

namespace ops {

// Operator with two input ports accepting TensorMaps. Streams are also received from each.
class PingTensorDualRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorDualRxOp)

  PingTensorDualRxOp() = default;

  void setup(OperatorSpec& spec) {
    spec.input<TensorMap>("in1");
    spec.input<TensorMap>("in2");
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) {
    const std::vector<std::string> in_port_names{"in1", "in2"};
    for (const auto& in_port_name : in_port_names) {
      auto maybe_in_message = op_input.receive<holoscan::TensorMap>(in_port_name.c_str());
      if (!maybe_in_message) {
        HOLOSCAN_LOG_ERROR("Failed to receive message from port '{}'", in_port_name);
        return;
      }
      cudaStream_t stream = op_input.receive_cuda_stream(in_port_name.c_str(), true);
      HOLOSCAN_LOG_INFO("{} received {}default CUDA stream from port '{}'",
                        name(),
                        stream == cudaStreamDefault ? "" : "non-",
                        in_port_name);

      auto maybe_device = context.device_from_stream(stream);
      if (maybe_device) {
        HOLOSCAN_LOG_INFO("CUDA stream from port '{}' corresponds to device {}",
                          in_port_name,
                          maybe_device.value());
      }

      auto streams = op_input.receive_cuda_streams(in_port_name.c_str());
      if (streams.size() == 1) {
        bool non_default = streams[0].has_value() && (streams[0].value() != cudaStreamDefault);
        HOLOSCAN_LOG_INFO("{}: receive_cuda_streams found 1 {}default stream on port {}",
                          name(),
                          non_default ? "non-" : "",
                          in_port_name);
      }

      auto& in_message = maybe_in_message.value();
      // Loop over any tensors found, printing their names and shapes.
      for (auto& [key, tensor] : in_message) {
        HOLOSCAN_LOG_INFO("{} received message {} on input {}: Tensor key: '{}', shape: ({})",
                          name(),
                          count_,
                          in_port_name,
                          key,
                          fmt::join(tensor->shape(), ", "));
      }
    }
    count_++;
  }

 private:
  size_t count_ = 1;
};

// Operator with one input port accepting a std::vector<TensorMap> of size 5.
// Streams are received an it is printed how many of the five messages have non-default streams.
class PingTensorSizeFiveRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorSizeFiveRxOp)

  PingTensorSizeFiveRxOp() = default;

  void setup(OperatorSpec& spec) { spec.input<TensorMap>("in", IOSpec::IOSize(5)); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) {
    auto maybe_in_message = op_input.receive<std::vector<holoscan::TensorMap>>("in");
    if (!maybe_in_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
      return;
    }

    // Test InputContext::receive_cuda_stream
    cudaStream_t stream = op_input.receive_cuda_stream("in");
    HOLOSCAN_LOG_INFO("{}: Stream on port 'in' was a {}default CUDA stream",
                      name(),
                      stream == cudaStreamDefault ? "" : "non-");

    // Test InputContext::receive_cuda_streams
    auto stream_vec = op_input.receive_cuda_streams("in");
    size_t num_non_default = std::count_if(
        stream_vec.begin(), stream_vec.end(), [](const std::optional<cudaStream_t>& s) {
          return s.has_value();
        });
    HOLOSCAN_LOG_INFO("{}: {} of {} elements contained non-default streams ",
                      name(),
                      num_non_default,
                      stream_vec.size());

    // Print info about the tensors that were received
    auto& in_message = maybe_in_message.value();
    HOLOSCAN_LOG_INFO("Received vector of {} TensorMaps", in_message.size());
    // Loop over any tensors found, printing their names and shapes.
    int map_count = 1;
    for (const auto& tensormap : in_message) {
      for (auto& [key, tensor] : tensormap) {
        HOLOSCAN_LOG_INFO("{} received message {} queue item {}: Tensor key: '{}', shape: ({})",
                          name(),
                          count_,
                          map_count,
                          key,
                          fmt::join(tensor->shape(), ", "));
      }
      map_count++;
    }
    count_++;
  }

 private:
  size_t count_ = 1;
};

// Operator with one multi-receiver input port (std::vector<TensorMap>)
// Streams are received and it is printed how many of the five messages have non-default streams.
class PingTensorMultiRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorMultiRxOp)

  PingTensorMultiRxOp() = default;

  void setup(OperatorSpec& spec) { spec.input<TensorMap>("in", IOSpec::kAnySize); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) {
    auto maybe_in_message = op_input.receive<std::vector<holoscan::TensorMap>>("in");
    if (!maybe_in_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
      return;
    }

    cudaStream_t stream = op_input.receive_cuda_stream("in");
    HOLOSCAN_LOG_INFO("{}: Stream on port 'in' was a {}default CUDA stream {}",
                      name(),
                      stream == cudaStreamDefault ? "" : "non-",
                      fmt::ptr(stream));

    auto stream_vec = op_input.receive_cuda_streams("in");
    size_t num_non_default = std::count_if(
        stream_vec.begin(), stream_vec.end(), [](const std::optional<cudaStream_t>& s) {
          return s.has_value();
        });
    HOLOSCAN_LOG_INFO("{} port 'in' with size {} contained {} non-default streams",
                      name(),
                      stream_vec.size(),
                      num_non_default);

    auto& in_message = maybe_in_message.value();
    HOLOSCAN_LOG_INFO("Received vector of {} TensorMaps", in_message.size());
    // Loop over any tensors found, printing their names and shapes.
    int map_count = 1;
    for (const auto& tensormap : in_message) {
      for (auto& [key, tensor] : tensormap) {
        HOLOSCAN_LOG_INFO("{} received message {} queue item {}: Tensor key: '{}', shape: ({})",
                          name(),
                          count_,
                          map_count,
                          key,
                          fmt::join(tensor->shape(), ", "));
      }
      map_count++;
    }
    count_++;
  }

 private:
  size_t count_ = 1;
};

}  // namespace ops

/**
 * @brief Application testing various aspects of CudaStreamHandling
 *
 * PingTensorTxOp tests
 *    ExecutionContext::allocate_cuda_stream
 *    OutputContext::set_cuda_stream
 *
 * FormatConverterOp tests
 *    InputContext::receive_cuda_stream
 *
 * PingTensorDualRxOp tests
 *    InputContext::receive_cuda_stream
 *    InputContext::receive_cuda_streams
 */
class StreamDualRxApp : public holoscan::Application {
 public:
  void compose() override {
    const int32_t width = 320;
    const int32_t height = 240;
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 10);

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("cuda_stream_pool", cuda_stream_pool),
        Arg("async_device_allocation", async_alloc_tx_),
    });
    auto source1 = make_operator<ops::PingTensorTxOp>(
        "ping_source1", make_condition<CountCondition>(4), tx_args);

    auto source2 = make_operator<ops::PingTensorTxOp>(
        "ping_source2", make_condition<CountCondition>(4), tx_args);

    ArgList format_converter_args{
        Arg("in_dtype", std::string("rgba8888")),
        Arg("out_dtype", std::string("rgb888")),
        Arg("pool", make_resource<holoscan::RMMAllocator>("rmm-allocator")),
        Arg("in_tensor_name", std::string("tensor")),
        Arg("out_tensor_name", std::string("rgb"))};
    if (converters_use_stream_pool_) {
      format_converter_args.add(Arg("cuda_stream_pool", cuda_stream_pool));
    }
    auto converter1 = make_operator<ops::FormatConverterOp>("converter1", format_converter_args);
    HOLOSCAN_LOG_INFO("converter1->description() = {}", converter1->description());
    auto converter2 = make_operator<ops::FormatConverterOp>("converter2", format_converter_args);

    ArgList rx_args{};
    auto dual_rx = make_operator<ops::PingTensorDualRxOp>("dual_rx", rx_args);
    if (rx_use_stream_pool_) { dual_rx->add_arg(cuda_stream_pool); }

    add_flow(source1, converter1, {{"out", "source_video"}});
    add_flow(source2, converter2, {{"out", "source_video"}});
    add_flow(converter1, dual_rx, {{"tensor", "in1"}});
    add_flow(converter2, dual_rx, {{"tensor", "in2"}});
  }

  void async_alloc_tx(bool value) { async_alloc_tx_ = value; }
  void converters_use_stream_pool(bool value) { converters_use_stream_pool_ = value; }
  void rx_use_stream_pool(bool value) { rx_use_stream_pool_ = value; }

 private:
  bool async_alloc_tx_ = false;
  bool converters_use_stream_pool_ = false;
  bool rx_use_stream_pool_ = false;
};

/**
 * @brief Application testing various aspects of CudaStreamHandling
 *
 * PingTensorTxOp tests
 *    ExecutionContext::allocate_cuda_stream
 *    OutputContext::set_cuda_stream
 *
 * PingTensorMultiRxOp tests
 *    InputContext::receive_cuda_stream with a multi-receiver
 *    InputContext::receive_cuda_streams with a multi-receiver
 */
class StreamMultiRxApp : public holoscan::Application {
  void compose() override {
    const int32_t width = 320;
    const int32_t height = 240;
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 10);

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("cuda_stream_pool", cuda_stream_pool),
        Arg("async_device_allocation", true),
    });
    auto source1 = make_operator<ops::PingTensorTxOp>(
        "ping_source1", make_condition<CountCondition>(10), tx_args);

    auto source2 = make_operator<ops::PingTensorTxOp>(
        "ping_source2", make_condition<CountCondition>(10), tx_args);

    auto source3 = make_operator<ops::PingTensorTxOp>(
        "ping_source3", make_condition<CountCondition>(10), tx_args);

    auto multi_rx = make_operator<ops::PingTensorMultiRxOp>("multi_rx");

    add_flow(source1, multi_rx, {{"out", "in"}});
    add_flow(source2, multi_rx, {{"out", "in"}});
    add_flow(source3, multi_rx, {{"out", "in"}});
  }
};

/**
 * @brief Application testing various aspects of CudaStreamHandling
 *
 * PingTensorTxOp tests
 *    ExecutionContext::allocate_cuda_stream
 *    OutputContext::set_cuda_stream
 *
 * PingTensorSizeFiveRxOp tests
 *    InputContext::receive_cuda_stream with a size 5 receiver where only 3 elements having a stream
 *    InputContext::receive_cuda_streams with a size 5 receiver where only 3 elements having a
 * stream
 */
class StreamSizeFiveRxApp : public holoscan::Application {
  void compose() override {
    const int32_t width = 320;
    const int32_t height = 240;
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 10);

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("cuda_stream_pool", cuda_stream_pool),
    });
    auto async_arg = Arg("async_device_allocation", true);
    auto no_async_arg = Arg("async_device_allocation", false);
    // Make 5 source operators, three of which use a non-default stream for async memory allocation
    auto source1 = make_operator<ops::PingTensorTxOp>(
        "ping_source1", make_condition<CountCondition>(10), tx_args, no_async_arg);
    auto source2 = make_operator<ops::PingTensorTxOp>(
        "ping_source2", make_condition<CountCondition>(10), tx_args, async_arg);
    auto source3 = make_operator<ops::PingTensorTxOp>(
        "ping_source3", make_condition<CountCondition>(10), tx_args, async_arg);
    auto source4 = make_operator<ops::PingTensorTxOp>(
        "ping_source4", make_condition<CountCondition>(10), tx_args, no_async_arg);
    auto source5 = make_operator<ops::PingTensorTxOp>(
        "ping_source5", make_condition<CountCondition>(10), tx_args, async_arg);

    auto five_rx = make_operator<ops::PingTensorSizeFiveRxOp>("five_rx");

    add_flow(source1, five_rx);
    add_flow(source2, five_rx);
    add_flow(source3, five_rx);
    add_flow(source4, five_rx);
    add_flow(source5, five_rx);
  }
};

/**
 * @brief Application testing various aspects of CudaStreamHandling
 */
class StreamSingleSourceTwoSinks : public holoscan::Application {
  void compose() override {
    const int32_t width = 320;
    const int32_t height = 240;
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 10);

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("cuda_stream_pool", cuda_stream_pool),
        Arg("async_device_allocation", true),
    });
    // Make one source operator and two sinks
    auto source = make_operator<ops::PingTensorTxOp>(
        "ping_source", make_condition<CountCondition>(10), tx_args);

    auto sink1 = make_operator<ops::PingTensorMultiRxOp>("ping_sink1", cuda_stream_pool);
    auto sink2 = make_operator<ops::PingTensorMultiRxOp>("ping_sink2", cuda_stream_pool);

    add_flow(source, sink1, {{"out", "in"}});
    add_flow(source, sink2, {{"out", "in"}});
  }
};

}  // namespace holoscan

class CudaStreamParameterizedTestFixture
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

INSTANTIATE_TEST_CASE_P(
    CudaStreamAppTests, CudaStreamParameterizedTestFixture,
    ::testing::Values(std::make_tuple(false, false, false), std::make_tuple(true, false, false),
                      std::make_tuple(false, true, false), std::make_tuple(true, true, false),
                      std::make_tuple(false, false, true), std::make_tuple(true, false, true),
                      std::make_tuple(false, true, true), std::make_tuple(true, true, true)));

TEST_P(CudaStreamParameterizedTestFixture, TestStreamDualRxApp) {
  using namespace holoscan;

  auto& [tx_stream_allocation, format_converter_stream_allocation, rx_stream_allocation] =
      GetParam();
  HOLOSCAN_LOG_INFO("tx_stream_allocation = {}", tx_stream_allocation);
  HOLOSCAN_LOG_INFO("format_converter_stream_allocation = {}", format_converter_stream_allocation);
  HOLOSCAN_LOG_INFO("rx_stream_allocation = {}", rx_stream_allocation);

  auto app = make_application<StreamDualRxApp>();
  app->async_alloc_tx(tx_stream_allocation);
  app->converters_use_stream_pool(format_converter_stream_allocation);
  app->rx_use_stream_pool(rx_stream_allocation);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  std::vector<std::string> port_names{"in1", "in2"};
  bool receivers_all_default = !(format_converter_stream_allocation || tx_stream_allocation);
  // either:
  //   1.) FormatConverterOp sent it internal non-default stream
  //   2.) a stream created by tx was automatically passed through format converter to rx
  //   3.) rx allocated its own internal stream
  bool receive_default =
      !(rx_stream_allocation || format_converter_stream_allocation || tx_stream_allocation);
  for (const auto& port_name : port_names) {
    std::string log_msg = fmt::format("dual_rx received {}default CUDA stream from port '{}'",
                                      receive_default ? "" : "non-",
                                      port_name);
    EXPECT_TRUE(log_output.find(log_msg) != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";

    log_msg = fmt::format("receive_cuda_streams found 1 {}default stream",
                          receivers_all_default ? "" : "non-");
    EXPECT_TRUE(log_output.find(log_msg) != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";

    log_msg = fmt::format("CUDA stream from port '{}' corresponds to device 0", port_name);
    if (receive_default) {
      // will not find the message in this case
      EXPECT_TRUE(log_output.find(log_msg) == std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
    } else {
      EXPECT_TRUE(log_output.find(log_msg) != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
    }
  }
}

TEST(CudaStreamApps, TestStreamMultiRxApp) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<StreamMultiRxApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that 3 streams were received
  std::string stream_msg = "multi_rx port 'in' with size 3 contained 3 non-default streams";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";

  // Check that 3rd item in queue was a TensorMap
  stream_msg =
      "multi_rx received message 10 queue item 3: Tensor key: 'tensor', shape: (240, 320, 4)";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

TEST(CudaStreamApps, TestStreamSizeFiveRxApp) {
  // Test fix for issue 4313690 (failure to initialize graph when using BayerDemosaicOp)
  using namespace holoscan;

  auto app = make_application<StreamSizeFiveRxApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Check that 3 non-default streams were received
  std::string stream_msg = "five_rx: 3 of 5 elements contained non-default streams";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";

  // Check that 3rd item in queue was a TensorMap
  stream_msg =
      "five_rx received message 10 queue item 5: Tensor key: 'tensor', shape: (240, 320, 4)";
  EXPECT_TRUE(log_output.find(stream_msg) != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
}

TEST(CudaStreamApps, TestStreamSingleSourceTwoSinks) {
  using namespace holoscan;

  auto app = make_application<StreamSingleSourceTwoSinks>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  std::string sink1_msg = "ping_sink1: Stream on port 'in' was a non-default CUDA stream ";
  auto sink1_msg_pos = log_output.find(sink1_msg);
  EXPECT_NE(sink1_msg_pos, std::string::npos);
  auto stream1 = std::stoull(log_output.substr(sink1_msg_pos + sink1_msg.length()), nullptr, 16);

  std::string sink2_msg = "ping_sink2: Stream on port 'in' was a non-default CUDA stream ";
  auto sink2_msg_pos = log_output.find(sink2_msg);
  EXPECT_NE(sink2_msg_pos, std::string::npos);
  auto stream2 = std::stoull(log_output.substr(sink2_msg_pos + sink2_msg.length()), nullptr, 16);

  // the streams of the two sinks should be different
  EXPECT_NE(stream1, stream2) << "=== LOG ===\n" << log_output << "\n===========\n";
}
