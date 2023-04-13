#ifndef INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP
#define INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP

#include "holoscan/core/gxf/gxf_operator.hpp"

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

  uint32_t framerate_;
};

}  // namespace holoscan::ops

#endif /* INCLUDE_HOLOSCAN_OPERATORS_AV_SOURCE_AV_SOURCE_HPP */
