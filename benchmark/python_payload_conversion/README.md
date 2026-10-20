# Python payload conversion

Measure `emit()`, `receive().data`, `receive_data()`, and `selected_sample().data` for:

- FlatBuffers byte vectors with 8 bytes and a 407 KiB camera statistics blob;
- FlatBuffers double vectors with 3 and 4096 elements;
- the built-in `Imu` schema with nested tables and fixed arrays;
- native POD `std::uint8_t` arrays with 8 and 16384 elements;
- native POD double arrays with 3 and 4096 elements;
- native POD `std::byte` arrays with 8 and 16384 elements.

The benchmark requires SDK support for `std::byte` POD fields and always registers and runs
the opaque-byte fixtures. For measurements before the conversion improvements, check out the
original benchmark-only commit (`b18564e7d1`), which conditionally registers these fixtures
and records unsupported cases as skipped in JSON. Do not substitute numeric uint8 fixtures for unsupported
opaque-byte fixtures when comparing representations.

Each case uses two Python Operators in one partition with a queue depth of one. The source
admits a record only after the previous record is consumed. Timings exclude graph setup,
callback dispatch, source pacing, and validation. The selected-data case obtains the selection
before timing its `.data` projection. The receive case includes both receive and projection;
`receive().data` already owns its projected Python value. Every sample's contents are validated
outside timing, including both list and bytes projections of schema byte vectors. Numeric POD
arrays must remain tuples. The JSON records the received representation.

Timing passes run without tracing, rotate operation order between repeats, and report the
median of repeat means. Separate `tracemalloc` passes measure transient Python allocations;
they exclude C++ and transport allocations. Byte read cases that actually return bytes also
assert that traced peak memory stays below four times the blob size. Add
`--require-byte-vectors` to require schema bytes output after the FlatBuffers improvement;
there is no machine-specific latency assertion.

Schema byte input defaults to a list, which works both before and after the improvements.
Use the same `--schema-byte-input` on both revisions for matched comparisons. After bulk byte
input is supported, use `--schema-byte-input bytes` for a separate optimized-input measurement.
Record it separately because it changes the Python emit input representation.

Run from the SDK source root:

```bash
./run bench --name python_payload_conversion
```

For matched measurements after building `python_payload_conversion_benchmark`, run the same
command at the benchmark-only revision and at the completed implementation revision, supplying
their respective Git hashes and output names:

```bash
./run launch build-cu13-x86_64 --run-cmd \
  'PYTHONPATH="$PYTHONPATH:$PWD/benchmark/python_payload_conversion" \
   python3 ../benchmark/python_payload_conversion/payload_conversion_benchmark.py \
   --iterations 128 --warmup 16 --repeats 5 --memory-iterations 16 \
   --schema-byte-input list --configuration Release-cu13 --revision <git-hash> \
   --benchmark_out=python-payload-conversion.json --benchmark_out_format=json'
```

Replace `<git-hash>` with the measured revision before running. Compare results on the same
machine and build configuration. JSON includes the timestamp, configuration, revision, input
representation, repeat means, and received representation. The benchmark does not require
camera hardware or the robot-control benchmark branch.

On SDKs with compact schema conversion, the bridge owns a native vector before copying its
contents into Python bytes. This removes per-element tagged values and Python integers; it
does not remove native receive, codec, or transport copies. An SDK admitting opaque-byte POD
arrays can copy those arrays directly from its native object bytes into Python bytes while
numeric uint8 arrays retain their tuple representation.
