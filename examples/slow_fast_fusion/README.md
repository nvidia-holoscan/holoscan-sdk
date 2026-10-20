# baseline slow/fast fusion

> **User guide:** [Slow/Fast Fusion](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/slow-fast-fusion) —
> this reference application demonstrates ordinary multi-rate graph composition.

This reference application demonstrates the ordinary HSDK 5 application root:
one `holoscan::Graph`, free `compile()` and `run_async()` functions, typed
Operators, two periodic rates, `OnEach`, `LatestWithin`, `Skip`, and whole-sample
metadata forwarding. Its two independent periodic roots are assigned to an
explicit two-partition cover; each partition remains serial. It intentionally
contains no `Application` wrapper.

The fast sample drives fusion. The most recent causal slow sample is admitted
only while it is within the declared age; otherwise that activation is skipped.
The sink requests immediate whole-run completion after its sixth sample, and
the main routine verifies `kRequestedCompleted` before reporting the
compiled-plan trigger and selection counts.

`request_completion()` is not an end-of-stream or drain protocol: it closes
new callback admission for the whole run, while already admitted callbacks may
retire. It does not guarantee downstream drain or delivery of queued samples.

## Run

Run the C++ variant with:

```sh
./slow_fast_fusion
```

Use `./slow_fast_fusion --plan-json` to print only the admitted canonical plan
JSON to stdout without starting the runtime. This mode is mutually exclusive
with `--emit-artifacts` and its artifact options.

The Python variant implements every Operator in Python while preserving the
same temporal contracts, two-partition structure, and requested-completion
behavior:

```sh
PYTHONPATH=<build>/python/lib python3 python/slow_fast_fusion.py
```

From an examples-enabled SDK install, configure and test the Python directory
as a standalone project:

```sh
cmake -B python/build -S python -Dholoscan_DIR=<prefix>/lib/cmake/holoscan
cmake --build python/build --target python_slow_fast_fusion
ctest --test-dir python/build -R EXAMPLE_PYTHON_SLOW_FAST_FUSION_TEST
```

The same composition path also supports cold artifact publication without
starting operator lifecycle:

```sh
cmake --build build --target slow_fast_fusion_graph_artifacts
```

This is an approachable reference path, not the exhaustive trigger, profile,
placement, or fault-injection qualification matrix.

The cold four-file deployment bundle is currently a C++ tooling path. Python
artifact packaging is left to later wheel and packaging work.
