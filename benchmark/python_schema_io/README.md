# Python Schema I/O Benchmark

Compare `receive().metadata` with `selected_sample().metadata`, `receive_data()` with
`selected_sample().data`, and `forward(receive())` with `forward(selected_sample())`.
The last pair publishes unchanged ordinary generated schemas; Tensor descriptors require
separate output loans and are excluded.

The workload uses the SDK's prepared `Imu` schema, including optional covariance tables,
a nested timestamp and an 8-byte or 128-byte frame name. Read cases use source-to-consumer
flows; forwarding cases add a validating sink. All Operators share one partition with
queue depth one. The source paces publications by completed consumer/sink operations and
retries only unaccepted backpressured source emits outside measurement. Every admitted payload
and forwarded frame ID is validated. Graph setup,
source emission, sink validation and callback dispatch are excluded from each timed operation.
The benchmark rotates mode order between repeats and reports the median of their mean costs.

`selected_sample().data` still materializes an owning Python value. Metadata-only selection
and borrowed forwarding avoid that projection. Selection facades, Python call machinery,
encoding and transport can still allocate; no row promises allocation-free Python execution.

Memory is measured in a separate pass with `tracemalloc`, reporting the transient peak above
the pre-operation baseline. This observes traced Python memory, including buffers registered
with Python's allocator. It does **not** count allocations, retained growth, C++ `new`, or
transport allocations. Timing passes run without tracing. Compare matched pairs on the same
build and machine; CI checks completion and data correctness rather than a speed threshold.

Run from the SDK source root after building the Python bindings:

```bash
./run bench --name python_schema_io
```

For longer measurements in the SDK development container, start from the same source root:

```bash
./run launch build-cu13-x86_64 --run-cmd \
  "python3 ../benchmark/python_schema_io/schema_io_benchmark.py \
    --iterations 5000 --warmup 500 --repeats 7 --memory-iterations 200 \
    --configuration Release-cu13 --benchmark_out=python-schema-io.json \
    --benchmark_out_format=json"
```

The JSON records Python, platform, UTC timestamp, measurement boundaries and the supplied build
configuration label. Record the source revision alongside it when retaining performance evidence.

## Native byte-vector producer

`byte_vector_benchmark.py` measures byte-vector projection from the native C++ producer used
by `test_schema_bytes.py`. This isolates Python receive costs from Python source conversion
and includes a metadata-only control. It reports per-operation latency samples and percentiles;
run `--memory` separately for traced Python peaks. The existing payload-conversion benchmark
also covers numeric vectors and POD fields.

From the SDK source root, with tests and benchmarks enabled:

```bash
./run launch build-cu13-x86_64 --run-cmd \
  'cmake --build . --target python_schema_byte_vector_benchmark && \
   PYTHONPATH="$PYTHONPATH:$PWD/python/tests/core/generated/holoscan/schemas" \
   python3 ../benchmark/python_schema_io/byte_vector_benchmark.py \
   --iterations 100 --warmup 20 --output byte-vector-timing.json'
```

Add `--memory` and choose another output path for the allocation pass. Run the same harness,
native fixture and schema on each revision. JSON records the harness hash, module paths,
CPU affinity, Python version, payload digest, size, operation and sample values.
