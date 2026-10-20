# Core Hello Benchmark

This benchmark is the minimal Google Benchmark example for the `holoscan::core`
target. It benchmarks the current `holoscan::hello()` scaffold and shows the
expected build, link, and CTest registration pattern for future benchmark
subdirectories.

For Google Benchmark features such as fixtures, arguments, counters, and timing
controls, see the [Google Benchmark user guide](https://google.github.io/benchmark/user_guide.html).

The commands below assume they are run from the repository root that contains the
top-level `benchmark` directory.

## Run With `./run`

For the full command reference and commented examples, run:

```sh
./run bench --help
```

List registered benchmarks:

```sh
./run bench --list
```

The list output shows `core_hello` as the short alias for the CTest name
`BENCHMARK_CORE_HELLO`. Both forms are accepted by `--name`.

Build missing benchmark targets automatically and run this benchmark:

```sh
./run bench --name core_hello
```

This command executes inside the Holoscan build container and uses a Release
build by default. It runs the benchmark binary directly, so Google Benchmark
output keeps its normal formatting without CTest line prefixes. It also rebuilds
benchmark targets first so changed CMake metadata and benchmark sources are
refreshed. Use `--no-build` only when intentionally running an existing build
tree unchanged.

The full CTest name is equivalent:

```sh
./run bench --name BENCHMARK_CORE_HELLO
```

Wildcard filters are accepted too:

```sh
./run bench --name 'core_*'
```

Export Google Benchmark JSON output:

```sh
./run bench --name core_hello --output-json core_hello_benchmark.json
```

The normal terminal timing table is still printed when JSON output is enabled.

Pass normal Google Benchmark arguments as needed:

```sh
./run bench --name core_hello \
  --benchmark-args "--benchmark_min_time=0.1s --benchmark_repetitions=5"
```

Use longer runtimes and aggregate-only reporting when comparing repeated runs:

```sh
./run bench --stable-system --name core_hello \
  --benchmark-args "--benchmark_min_time=1s --benchmark_repetitions=10 --benchmark_report_aggregates_only=true"
```

Run through CTest when you want CI-style aggregation:

```sh
./run bench --ctest --name core_hello
```

Check host settings that can add benchmark noise:

```sh
./run bench --check-system
```

For a lower-noise local run, temporarily disable ASLR and set CPU governors to
`performance` while this benchmark runs:

```sh
./run bench --stable-system --name core_hello
```

This is mostly host-side configuration, not Docker-local configuration. The
command saves the current host settings, runs the benchmark inside Docker with a
less restrictive seccomp profile so `setarch -R` can disable ASLR for the
benchmark process, and restores the saved settings when it exits.
`--stable-system` requires cached sudo credentials; run `sudo -v` on the host
first if needed.

`--stable-system` is intended to reduce known benchmark noise sources and make
the run more reproducible. It does not guarantee lower latency or higher
throughput. A normal run can still report a faster result if the processor
boosts higher for that run, and disabling ASLR can change code/data layout in a
way that is neutral, better, or worse for a small benchmark.

Use a non-release build only when debugging benchmark code:

```sh
./run bench --type debug --name core_hello
```

## Direct CMake Fallback

Configure with benchmark support enabled:

```sh
cmake -S . -B build -G Ninja \
  -DHOLOSCAN_BUILD_BENCHMARKS=ON \
  -DHOLOSCAN_USE_SCCACHE=OFF
```

Build the benchmark target:

```sh
cmake --build build --target core_hello_benchmark
```

## Run Through CTest

Run only this benchmark test:

```sh
ctest --test-dir build -R BENCHMARK_CORE_HELLO --output-on-failure
```

Run all benchmark-labeled tests:

```sh
ctest --test-dir build -L benchmark --output-on-failure
```

## Run Directly

Run the benchmark binary directly when tuning locally:

```sh
./build/benchmark/core_hello/core_hello_benchmark
```

Pass normal Google Benchmark arguments as needed:

```sh
./build/benchmark/core_hello/core_hello_benchmark \
  --benchmark_min_time=0.1s \
  --benchmark_repetitions=5
```

## Troubleshooting

If configure fails with `cannot find sccache executable`, keep
`-DHOLOSCAN_USE_SCCACHE=OFF` in the configure command or install `sccache` and
configure with `-DHOLOSCAN_USE_SCCACHE=ON`.

If the `build` directory was already configured with sccache enabled, rerun the
configure command with `-DHOLOSCAN_USE_SCCACHE=OFF` or start from a fresh build
directory.
