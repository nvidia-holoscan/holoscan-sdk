#include "holoscan/operators/av_source/av_source.hpp"

namespace holoscan::ops {
void AVSourceOp::setup(OperatorSpec& spec) {
  auto& transmitter = spec.output<gxf::Entity>("transmitter");

  spec.param(transmitter_,
             "transmitter",
             "Entity transmitter",
             "Transmitter channel for playing files",
             &transmitter);

  spec.param(filename_, "filename", "Filename", "Filename", std::string(""));
}

void AVSourceOp::initialize() {
  Operator::initialize();
}

AVSourceOp::~AVSourceOp() {}

void AVSourceOp::start() {}

void AVSourceOp::stop() {}

void AVSourceOp::compute(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context) {
  // avoid warning about unused variable
  (void)op_input;
  (void)op_output;
  (void)context;
}

}  // namespace holoscan::ops