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

```sh
./slow_fast_fusion
```

The same composition path also supports cold artifact publication without
starting operator lifecycle:

```sh
cmake --build build --target slow_fast_fusion_graph_artifacts
```

This is an approachable reference path, not the exhaustive trigger, profile,
placement, or fault-injection qualification matrix.
