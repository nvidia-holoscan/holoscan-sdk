/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan {

class MinimalNativeResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MinimalNativeResource, Resource)

  MinimalNativeResource() = default;

  void setup(ComponentSpec& spec) override {
    spec.param(string_param_, "str_value", "String parameter", "String parameter for test", {});
  }
  std::string string_param() { return string_param_.get(); }

 private:
  Parameter<std::string> string_param_;
};

namespace ops {

class MinimalNativeResourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MinimalNativeResourceOp)

  MinimalNativeResourceOp() = default;

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto res = resource<MinimalNativeResource>("string_native_resource");
    if (res) {
      HOLOSCAN_LOG_INFO("MinimalNativeResource - string_native_resource.string_param: {}",
                        res->string_param());
    }
    auto res2 = resource<MinimalNativeResource>("hardcoded_native_resource");
    if (res2) {
      HOLOSCAN_LOG_INFO("MinimalNativeResource - hardcoded_native_resource.string_param: {}",
                        res2->string_param());
    }
    auto res3 = resource<MinimalNativeResource>("empty_native_resource");
    if (res3) {
      HOLOSCAN_LOG_INFO("MinimalNativeResource - empty_native_resource.string_param: '{}'",
                        res3->string_param());
    }
  };
};

}  // namespace ops

class MinimalNativeResourceApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto res = make_resource<MinimalNativeResource>("string_native_resource",
                                                    Arg("str_value", std::string("test_string")));
    auto res2 = make_resource<MinimalNativeResource>(
        "hardcoded_native_resource", Arg("str_value") = std::string("hardcoded_string"));
    auto res3 = make_resource<MinimalNativeResource>("empty_native_resource");

    auto op = make_operator<ops::MinimalNativeResourceOp>(
        "min_op", make_condition<CountCondition>(1), res, res2, res3);
    add_operator(op);
  }
};

}  // namespace holoscan

int main() {
  auto app = holoscan::make_application<holoscan::MinimalNativeResourceApp>();
  app->run();

  return 0;
}
