/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>

class PointerCaptureOp final : public holoscan::Operator<> {
 public:
  explicit PointerCaptureOp(int*) {}
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

// REQ: GRAPH-003
int main() {
  holoscan::Graph graph;
  int borrowed = 0;
  (void)graph.op<PointerCaptureOp>("pointer", &borrowed);
}
