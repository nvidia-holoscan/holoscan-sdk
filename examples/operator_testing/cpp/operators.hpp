/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <utility>

namespace operator_testing_example {
struct ScaleParams {
  std::int32_t factor{1};
};

HOLOSCAN_PARAM_SCHEMA(ScaleParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.ScaleParams", 1),
                      holoscan::field<&ScaleParams::factor>("factor"));

class ScaleOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    value->data *= params().factor;
    return out.emit(std::move(*value));
  }

  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;
};

class AlignedSumOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(sum, "sum");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(left, right),
                                     .window = std::chrono::milliseconds{100}})
        .input(left, holoscan::Exact{})
        .input(right, holoscan::Exact{});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    return sum.emit(a->data + b->data);
  }
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> sum;
};

class ControlledPairOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(control, "control").optional();
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(out, "out");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnAligned{.ports = holoscan::ports(left, right),
                                  .window = std::chrono::nanoseconds{0}})
        .input(left, holoscan::Exact{})
        .input(right, holoscan::Exact{})
        .input(control, holoscan::Exact{})
        .on_missing(control, holoscan::SafeDefault{});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    auto factor = params().factor;
    if (control.selection().state != holoscan::InputState::kSafeDefault) {
      auto value = control.receive();
      if (!value)
        return holoscan::make_unexpected(std::move(value).error());
      factor = value->data;
    }
    return out.emit((a->data + b->data) * factor);
  }
  holoscan::Input<std::int32_t> control;
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> out;
};

}  // namespace operator_testing_example
