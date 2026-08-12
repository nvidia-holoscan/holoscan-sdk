/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>

// REQ: LIFECYCLE-002
class IndependentLifecycleHook {
 public:
  void start() {}
};

class AmbiguousLifecycleHook final : public holoscan::Operator<>, public IndependentLifecycleHook {
 public:
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

int main() {
  holoscan::Graph graph;
  (void)graph.op<AmbiguousLifecycleHook>("ambiguous");
}
