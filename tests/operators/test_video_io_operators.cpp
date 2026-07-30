/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "../utils.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/operators/video_io/video_acquisition_operator.hpp"
#include "holoscan/operators/video_io/video_io_capabilities.hpp"
#include "holoscan/operators/video_io/video_io_registry.hpp"
#include "holoscan/operators/video_io/video_transmission_operator.hpp"

using namespace std::string_literals;

namespace holoscan::ops {

class TestVideoAcquisitionOp : public VideoAcquisitionOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestVideoAcquisitionOp, VideoAcquisitionOperator)

  TestVideoAcquisitionOp() = default;

  explicit TestVideoAcquisitionOp(uint32_t num_streams) : VideoAcquisitionOperator(num_streams) {}

  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit TestVideoAcquisitionOp(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : VideoAcquisitionOperator(num_streams) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

class TestVideoTransmissionOp : public VideoTransmissionOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(TestVideoTransmissionOp, VideoTransmissionOperator)

  TestVideoTransmissionOp() = default;

  explicit TestVideoTransmissionOp(uint32_t num_streams) : VideoTransmissionOperator(num_streams) {}

  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit TestVideoTransmissionOp(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : VideoTransmissionOperator(num_streams) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}
};

}  // namespace holoscan::ops

namespace holoscan {

TEST(VideoIoRegistry, AcquisitionEnumeratorRuns) {
  using holoscan::ops::video_io::enumerate_video_acquisition_devices;
  using holoscan::ops::video_io::register_video_acquisition_enumerator;
  constexpr char kBackend[] = "holoscan.test.video_io";
  register_video_acquisition_enumerator(kBackend, [kBackend] {
    holoscan::ops::video_io::VideoCaptureCapabilities d;
    d.backend_id = kBackend;
    d.device_id = "unit_device";
    d.max_concurrent_inputs = 1;
    holoscan::ops::video_io::VideoCaptureChannelCapabilities ch{};
    ch.channel_index = 0;
    d.input_channels.push_back(ch);
    return std::vector<holoscan::ops::video_io::VideoCaptureCapabilities>{d};
  });
  auto caps = enumerate_video_acquisition_devices(kBackend);
  ASSERT_EQ(caps.size(), 1U);
  EXPECT_EQ(caps[0].device_id, "unit_device");
  ASSERT_EQ(caps[0].input_channels.size(), 1U);
}

TEST(VideoIoCapabilities, CaptureToGenericDeviceRoundTrip) {
  using holoscan::ops::video_io::to_video_device_capabilities;
  using holoscan::ops::video_io::VideoCaptureCapabilities;
  using holoscan::ops::video_io::VideoCaptureChannelCapabilities;
  VideoCaptureCapabilities cap;
  cap.backend_id = "b";
  cap.device_id = "d";
  cap.device_uri = "rtp://x";
  cap.max_concurrent_inputs = 2;
  cap.connection_uri_schemes.push_back("rtp");
  VideoCaptureChannelCapabilities ch0{};
  ch0.channel_index = 0;
  ch0.progressive_capture_supported = true;
  ch0.interlaced_capture_supported = true;
  cap.input_channels.push_back(ch0);
  VideoCaptureChannelCapabilities ch1{};
  ch1.channel_index = 1;
  cap.input_channels.push_back(ch1);

  auto g = to_video_device_capabilities(cap);
  EXPECT_EQ(g.max_concurrent_inputs, 2U);
  EXPECT_EQ(g.max_concurrent_outputs, 0U);
  ASSERT_EQ(g.channels.size(), 2U);
  EXPECT_EQ(g.channels[0].channel_index, 0U);
  EXPECT_TRUE(cap.input_channels[0].progressive_capture_supported);
  EXPECT_TRUE(cap.input_channels[0].interlaced_capture_supported);
}

using VideoIoOperatorTest = TestWithGXFContext;

TEST_F(VideoIoOperatorTest, VideoAcquisitionCapabilitiesFromParameters) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq",
                                                         Arg("uri", "rtp://127.0.0.1:5000"s),
                                                         Arg("transport", "ethernet"s),
                                                         Arg("width", 1920u),
                                                         Arg("height", 1080u),
                                                         Arg("frame_rate", 60.f),
                                                         Arg("pixel_format", "UYVY"s));
  op->set_parameters();
  auto cap = op->query_capture_capabilities();
  auto c = op->query_capabilities();

  EXPECT_EQ(cap.device_uri, "rtp://127.0.0.1:5000");
  ASSERT_EQ(cap.input_channels.size(), 1U);
  EXPECT_EQ(cap.input_channels[0].resolutions.size(), 1U);
  EXPECT_EQ(cap.input_channels[0].resolutions[0].width, 1920U);
  EXPECT_EQ(cap.input_channels[0].resolutions[0].height, 1080U);
  EXPECT_EQ(cap.input_channels[0].framerates.size(), 1U);
  EXPECT_EQ(cap.input_channels[0].pixel_formats.size(), 1U);
  EXPECT_TRUE(cap.input_channels[0].progressive_capture_supported);
  ASSERT_EQ(cap.connection_uri_schemes.size(), 1U);
  EXPECT_EQ(cap.connection_uri_schemes[0], "rtp");

  EXPECT_EQ(c.device_uri, cap.device_uri);
  ASSERT_EQ(c.channels.size(), 1U);
  EXPECT_EQ(c.channels[0].resolutions[0].width, 1920U);
  EXPECT_EQ(c.transports.size(), 1U);
}

TEST_F(VideoIoOperatorTest, VideoTransmissionCapabilitiesMultiChannel) {
  auto op = F.make_operator<ops::TestVideoTransmissionOp>(
      "vtx",
      Arg("uri", "sdi://0"s),
      Arg("transport", "sdi"s),
      Arg("channel_indices", std::vector<uint32_t>{0U, 1U}));
  op->set_parameters();
  auto tx = op->query_transmit_capabilities();
  auto c = op->query_capabilities();

  EXPECT_EQ(tx.max_concurrent_outputs, 2U);
  ASSERT_EQ(tx.output_channels.size(), 2U);
  EXPECT_EQ(tx.output_channels[0].channel_index, 0U);
  EXPECT_EQ(tx.output_channels[1].channel_index, 1U);
  EXPECT_TRUE(tx.output_channels[0].progressive_output_supported);
  ASSERT_EQ(tx.connection_uri_schemes.size(), 1U);
  EXPECT_EQ(tx.connection_uri_schemes[0], "sdi");

  EXPECT_EQ(c.max_concurrent_outputs, 2U);
  ASSERT_EQ(c.channels.size(), 2U);
  EXPECT_EQ(c.channels[0].channel_index, 0U);
  EXPECT_EQ(c.channels[1].channel_index, 1U);
}

TEST_F(VideoIoOperatorTest, VideoAcquisitionDefaultRegistersOnlySignal) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_default");
  ASSERT_NE(op->spec(), nullptr);
  auto& outs = op->spec()->outputs();
  EXPECT_NE(outs.find("signal"), outs.end());
  EXPECT_EQ(outs.find("signal_1"), outs.end());
  EXPECT_EQ(op->num_streams(), 1U);
}

TEST_F(VideoIoOperatorTest, VideoAcquisitionDynamicPortRegistersExactCount) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_4s", 4u);
  ASSERT_NE(op->spec(), nullptr);
  auto& outs = op->spec()->outputs();
  EXPECT_EQ(op->num_streams(), 4U);
  EXPECT_NE(outs.find("signal"), outs.end());
  EXPECT_NE(outs.find("signal_1"), outs.end());
  EXPECT_NE(outs.find("signal_2"), outs.end());
  EXPECT_NE(outs.find("signal_3"), outs.end());
  EXPECT_EQ(outs.find("signal_4"), outs.end());
}

TEST_F(VideoIoOperatorTest, VideoTransmissionDefaultRegistersOnlySignal) {
  auto op = F.make_operator<ops::TestVideoTransmissionOp>("vtx_default");
  ASSERT_NE(op->spec(), nullptr);
  auto& ins = op->spec()->inputs();
  EXPECT_NE(ins.find("signal"), ins.end());
  EXPECT_EQ(ins.find("signal_1"), ins.end());
  EXPECT_EQ(op->num_streams(), 1U);
}

TEST_F(VideoIoOperatorTest, VideoTransmissionDynamicPortRegistersExactCount) {
  auto op = F.make_operator<ops::TestVideoTransmissionOp>("vtx_3s", 3u);
  ASSERT_NE(op->spec(), nullptr);
  auto& ins = op->spec()->inputs();
  EXPECT_EQ(op->num_streams(), 3U);
  EXPECT_NE(ins.find("signal"), ins.end());
  EXPECT_NE(ins.find("signal_1"), ins.end());
  EXPECT_NE(ins.find("signal_2"), ins.end());
  EXPECT_EQ(ins.find("signal_3"), ins.end());
}

TEST_F(VideoIoOperatorTest, VideoAcquisitionCapabilitiesDualChannelIndices) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>(
      "vacq_dual_ch",
      Arg("channel_indices", std::vector<uint32_t>{0U, 1U}),
      Arg("transport", "sdi"s));
  op->set_parameters();
  auto cap = op->query_capture_capabilities();
  EXPECT_EQ(cap.max_concurrent_inputs, 2U);
  ASSERT_EQ(cap.input_channels.size(), 2U);
  EXPECT_EQ(cap.input_channels[0].channel_index, 0U);
  EXPECT_EQ(cap.input_channels[1].channel_index, 1U);
}

TEST_F(VideoIoOperatorTest, MultiInstanceAcquisitionSameDevice) {
  auto op0 = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_ch0",
                                                          Arg("channel_index", 0u),
                                                          Arg("backend_id", "vendor.test"s),
                                                          Arg("transport", "sdi"s));
  auto op1 = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_ch1",
                                                          Arg("channel_index", 1u),
                                                          Arg("backend_id", "vendor.test"s),
                                                          Arg("transport", "sdi"s));
  op0->set_parameters();
  op1->set_parameters();
  auto cap0 = op0->query_capture_capabilities();
  auto cap1 = op1->query_capture_capabilities();
  EXPECT_EQ(cap0.input_channels[0].channel_index, 0U);
  EXPECT_EQ(cap1.input_channels[0].channel_index, 1U);
}

TEST_F(VideoIoOperatorTest, MultiInstanceTransmissionSameDevice) {
  auto op0 = F.make_operator<ops::TestVideoTransmissionOp>("vtx_ch0",
                                                           Arg("channel_index", 0u),
                                                           Arg("backend_id", "vendor.test"s),
                                                           Arg("transport", "sdi"s));
  auto op1 = F.make_operator<ops::TestVideoTransmissionOp>("vtx_ch1",
                                                           Arg("channel_index", 1u),
                                                           Arg("backend_id", "vendor.test"s),
                                                           Arg("transport", "sdi"s));
  op0->set_parameters();
  op1->set_parameters();
  auto cap0 = op0->query_transmit_capabilities();
  auto cap1 = op1->query_transmit_capabilities();
  EXPECT_EQ(cap0.output_channels[0].channel_index, 0U);
  EXPECT_EQ(cap1.output_channels[0].channel_index, 1U);
}

TEST(VideoIoRegistry, TransmissionEnumeratorRuns) {
  using holoscan::ops::video_io::enumerate_video_transmission_devices;
  using holoscan::ops::video_io::register_video_transmission_enumerator;
  constexpr char kBackend[] = "holoscan.test.video_io.tx";
  register_video_transmission_enumerator(kBackend, [kBackend] {
    holoscan::ops::video_io::VideoTransmitCapabilities d;
    d.backend_id = kBackend;
    d.device_id = "unit_tx_device";
    d.max_concurrent_outputs = 2;
    holoscan::ops::video_io::VideoTransmitChannelCapabilities ch{};
    ch.channel_index = 0;
    d.output_channels.push_back(ch);
    ch.channel_index = 1;
    d.output_channels.push_back(ch);
    return std::vector<holoscan::ops::video_io::VideoTransmitCapabilities>{d};
  });
  auto caps = enumerate_video_transmission_devices(kBackend);
  ASSERT_EQ(caps.size(), 1U);
  EXPECT_EQ(caps[0].device_id, "unit_tx_device");
  EXPECT_EQ(caps[0].max_concurrent_outputs, 2U);
  ASSERT_EQ(caps[0].output_channels.size(), 2U);
}

TEST_F(VideoIoOperatorTest, AcquisitionStreamEnabledMatchesNumStreams) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_enabled_check", 3u);
  EXPECT_TRUE(op->is_capture_stream_enabled(0));
  EXPECT_TRUE(op->is_capture_stream_enabled(1));
  EXPECT_TRUE(op->is_capture_stream_enabled(2));
  EXPECT_FALSE(op->is_capture_stream_enabled(3));
  EXPECT_FALSE(op->is_capture_stream_enabled(ops::video_io::kVideoIoMaxStreams));
}

TEST_F(VideoIoOperatorTest, AcquisitionDefaultIsOneStream) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_default_enabled");
  EXPECT_TRUE(op->is_capture_stream_enabled(0));
  EXPECT_FALSE(op->is_capture_stream_enabled(1));
  EXPECT_EQ(op->num_streams(), 1U);
}

TEST_F(VideoIoOperatorTest, TransmissionStreamEnabledMatchesNumStreams) {
  auto op = F.make_operator<ops::TestVideoTransmissionOp>("vtx_enabled_check", 2u);
  EXPECT_TRUE(op->is_transmit_stream_enabled(0));
  EXPECT_TRUE(op->is_transmit_stream_enabled(1));
  EXPECT_FALSE(op->is_transmit_stream_enabled(2));
  EXPECT_EQ(op->num_streams(), 2U);
}

TEST_F(VideoIoOperatorTest, TransmissionDefaultIsOneStream) {
  auto op = F.make_operator<ops::TestVideoTransmissionOp>("vtx_default_enabled");
  EXPECT_TRUE(op->is_transmit_stream_enabled(0));
  EXPECT_FALSE(op->is_transmit_stream_enabled(1));
  EXPECT_EQ(op->num_streams(), 1U);
}

TEST_F(VideoIoOperatorTest, NumStreamsClampedToMax) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_clamp", 200u);
  EXPECT_EQ(op->num_streams(), ops::video_io::kVideoIoMaxStreams);
}

TEST_F(VideoIoOperatorTest, NumStreamsClampedToMin) {
  auto op = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_clamp_min", 0u);
  EXPECT_EQ(op->num_streams(), 1U);
}

TEST_F(VideoIoOperatorTest, DynamicPortsWithArgs) {
  auto op =
      F.make_operator<ops::TestVideoAcquisitionOp>("vacq_dyn_args", 2u, Arg("transport", "sdi"s));
  EXPECT_EQ(op->num_streams(), 2U);
  auto& outs = op->spec()->outputs();
  EXPECT_NE(outs.find("signal"), outs.end());
  EXPECT_NE(outs.find("signal_1"), outs.end());
  EXPECT_EQ(outs.find("signal_2"), outs.end());
}

TEST_F(VideoIoOperatorTest, DroppedFrameCountersStartAtZero) {
  auto acq = F.make_operator<ops::TestVideoAcquisitionOp>("vacq_ctr");
  EXPECT_EQ(acq->dropped_frame_count(), 0U);
  EXPECT_EQ(acq->acquired_frame_count(), 0U);

  auto tx = F.make_operator<ops::TestVideoTransmissionOp>("vtx_ctr");
  EXPECT_EQ(tx->dropped_frame_count(), 0U);
  EXPECT_EQ(tx->transmitted_frame_count(), 0U);
}

}  // namespace holoscan
