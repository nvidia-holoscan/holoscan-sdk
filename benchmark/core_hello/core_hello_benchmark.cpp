/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/holoscan.hpp>

#include <benchmark/benchmark.h>

#include <iostream>
#include <sstream>

namespace {

void BM_HoloscanHello(benchmark::State& state) {
  // Keep this sample quiet: holoscan::hello() writes to stdout, but benchmark
  // output should stay focused on timing results. Real benchmarks should avoid
  // logging inside the measured loop unless logging is the behavior being measured.
  std::ostringstream output;
  auto* previous_buffer = std::cout.rdbuf(output.rdbuf());

  // Google Benchmark controls the iteration count. Put only the code being
  // measured inside this loop; setup that should not be timed should happen
  // before the loop, or inside state.PauseTiming()/state.ResumeTiming().
  for (auto _ : state) {
    output.str("");
    output.clear();
    holoscan::hello();

    // Prevent the compiler from treating memory effects as irrelevant and
    // optimizing the tiny sample too aggressively. For values returned from the
    // measured code, prefer benchmark::DoNotOptimize(value). Use ClobberMemory()
    // when the measured code mutates memory or performs side effects that need
    // to remain visible to the optimizer.
    benchmark::ClobberMemory();
  }

  std::cout.rdbuf(previous_buffer);
}

}  // namespace

// Register the benchmark function. Additional Google Benchmark options can be
// chained here. See the Google Benchmark user guide for fixtures, arguments,
// counters, and timing controls:
// https://google.github.io/benchmark/user_guide.html
//
// Example:
//
//   BENCHMARK(BM_MyFeature)->Arg(1024)->Unit(benchmark::kMicrosecond);
//
// For benchmarks with shared setup/teardown, use BENCHMARK_F with a fixture.
// Keep benchmark names stable because ./run bench derives friendly aliases from
// the CTest registration and benchmark reports are easier to compare over time.
BENCHMARK(BM_HoloscanHello);
