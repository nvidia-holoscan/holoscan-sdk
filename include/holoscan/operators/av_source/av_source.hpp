#ifndef INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP
#define INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP

#include "holoscan/core/gxf/gxf_operator.hpp"

#include <chrono>

struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;

namespace holoscan::ops {
/**
 * @brief Operator class to get the video stream from AV library.
 */
class AVSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AVSourceOp)

  AVSourceOp() = default;
  ~AVSourceOp() override;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<holoscan::IOSpec*> transmitter_;
  Parameter<std::string> filename_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  uint32_t video_width_;
  uint32_t video_height_;
  float framerate_;

  std::chrono::_V2::system_clock::time_point last_frame_time_;
  bool playback_started_ = false;

  AVFormatContext* format_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  int video_stream_index_ = -1;

 private:
  std::shared_ptr<AVFrame> read_frame();
};

}  // namespace holoscan::ops

#endif /* INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP */
