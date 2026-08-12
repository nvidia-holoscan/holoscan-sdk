/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>

// REQ: LIFECYCLE-002
class HiddenLifecycleHook final : public holoscan::Operator<> {
 public:
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

 private:
  void start() override {}
};

int main() {
  holoscan::Graph graph;
  (void)graph.op<HiddenLifecycleHook>("hidden");
}
