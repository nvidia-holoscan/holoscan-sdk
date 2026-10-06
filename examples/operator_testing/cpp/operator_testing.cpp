/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "operators.hpp"

using namespace operator_testing_example;
using namespace std::chrono_literals;
using holoscan::testing::create_operator_test;

int main() {
  try {
    auto h = create_operator_test<ScaleOp>();
    h.set_params(ScaleParams{.factor = 2});
    h.push(&ScaleOp::in, std::int32_t{21});
    h.capture<std::int32_t>("out");
    const auto scaled = h.run();
    if (!scaled.ok())
      throw std::runtime_error{scaled.describe_failure()};
    if (scaled.value<std::int32_t>("out") != 42)
      throw std::runtime_error{"scaling failed"};

    auto aligned = create_operator_test<AlignedSumOp>();
    aligned.push_samples<std::int32_t>("left",
                                       {holoscan::testing::sample_at(std::int32_t{10}, 1s),
                                        holoscan::testing::sample_at(std::int32_t{20}, 2s)});
    aligned.push_samples<std::int32_t>("right",
                                       {holoscan::testing::sample_at(std::int32_t{1}, 1100ms),
                                        holoscan::testing::sample_at(std::int32_t{2}, 2100ms)});
    aligned.capture<std::int32_t>("sum");
    const auto pairs = aligned.run();
    if (!pairs.ok())
      throw std::runtime_error{pairs.describe_failure()};
    if (pairs.values<std::int32_t>("sum") != std::vector<std::int32_t>{11, 22}) {
      throw std::runtime_error{"alignment failed"};
    }

    auto grouped = create_operator_test<ControlledPairOp>();
    grouped.set_params(ScaleParams{.factor = 3});
    grouped.push_frame(
        holoscan::testing::frame_at(1s,
                                    holoscan::testing::input("control", std::int32_t{2}),
                                    holoscan::testing::input("left", std::int32_t{4}),
                                    holoscan::testing::input("right", std::int32_t{5})));
    grouped.capture<std::int32_t>("out");
    const auto frame = grouped.run();
    if (!frame.ok())
      throw std::runtime_error{frame.describe_failure()};
    if (frame.value<std::int32_t>("out") != 18)
      throw std::runtime_error{"grouped input failed"};

    // Configuration uses ordinary Graph authoring handles on each fresh attempt.
    h.input_queue_depth("in", 4U);
    h.input_flow("in", holoscan::ConnectionOptions{.queue_depth = 4U});
    h.configure_graph([](auto& f) {
      auto sources = f.graph().partition("sources");
      f.add_sources_to(sources);
      f.graph().partition("operator").add(f.tested_operator());
      auto collectors = f.graph().partition("collectors");
      f.add_collectors_to(collectors);
    });
    h.push_many<std::int32_t>("in", {1, 2, 3});
    const auto configured = h.run();
    if (!configured.ok())
      throw std::runtime_error{configured.describe_failure()};
    if (configured.values<std::int32_t>("out") != std::vector<std::int32_t>{2, 4, 6}) {
      throw std::runtime_error{"configured run failed"};
    }
    std::cout << "operator_testing example complete: parameters, alignment, frames, and partitions "
                 "verified\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
