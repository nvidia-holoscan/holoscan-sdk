# Operator testing

Run one operator through ordinary Graph compilation and Runtime execution, then
assert owning results. The examples use application-neutral arithmetic and
schema payloads:

- `operator_testing`: typed member inputs, Params, timestamped sequences,
  alignment, grouped optional control, native queue settings, and partitions.
- `operator_testing_clock`: controlled clock freshness boundaries, with Runtime
  evidence distinguishing suppressed candidates from retired callbacks.
- `operator_testing_tensor`: independent snapshots and capacity-proved native
  retention from a finite Tensor producer.
- `operator_testing_image`: `schema::ImageT` metadata plus its Tensor content.
- `operator_testing_schema`: a generated two-Tensor schema, independent nested
  metadata, timestamped records and field-by-field snapshots.
- `operator_testing_lifecycle`: zero or multiple outputs per input, fresh runs,
  and a no-output sink with an explicit observer read after joined cleanup.
- `operator_testing_device`: two aligned device inputs, device snapshots, manual
  partitions and optional Green Context deployment in a shared or separate partition.
- `operator_testing_restart`: manual input phases, self and external restart, restricted
  native handles, accepted versus terminal receipts and joined result ownership.
- `operator_testing_gtest`: optional GoogleTest counterpart to the scalar test.

Build against an installed SDK:

```sh
cmake -S cpp -B build -DCMAKE_PREFIX_PATH=/path/to/holoscan
cmake --build build
ctest --test-dir build --output-on-failure
```

Add `-DOPERATOR_TESTING_WITH_GTEST=ON` when a GoogleTest CMake package is
available. Only the example test executable depends on GoogleTest; the SDK's
`holoscan::testing` target does not.

The scalar, Tensor, schema and lifecycle examples use host memory. The device example requires
an explicit device ID; for example, after building:

```sh
./build/operator_testing_device 1
./build/operator_testing_device 1 --green-context
./build/operator_testing_device 1 --green-context-shared-partition
```

Set `-DOPERATOR_TESTING_DEVICE_ID=1` during CMake configuration to use that device
in CTest. The device checks return the CTest skip code 77 if the requested hardware
or Green Context capability is unavailable. Admission or execution failures on an
available target remain failures. All source and operator device outputs have
explicit placement bindings, including both ports on the grouped source.

Deployment uses ordinary `CompileOptions`. The advanced `configure_graph` hook
receives fresh Graph authoring handles. Ordinary deployment and `--green-context`
place sources, operator and collectors in three complete manual partitions.
The latter binds the operator partition to one structured Green Context group and
requires another group for the helpers' residual GPU work.

`--green-context-shared-partition` places all three roles in one partition and binds
it to one structured Green Context group, with no residual GPU work. This variant
requires only one group from the driver's structured split query. That count is
specific to the split policy, not a limit on every Green Context configuration a
device can support. Selection uses queried capabilities, not device names.
CTest registers both Green Context variants separately; only the
cross-partition variant requires at least two groups. Both verify the accepted
device and SM count, the partition count, and actual copied output. The
ordinary thread-placement value can be replaced with native affinity or scheduling
settings supported by the execution target.

Both Green Context invocations print the device name, compute capability, CUDA
driver API and runtime versions, and the queried SM split limits before checking
capacity. Use `ctest --test-dir build -V -R 'EXAMPLE_CPP_OPERATOR_TESTING_.*GREEN_CONTEXT'`
to retain those diagnostics even for successful or skipped tests. The driver API
version describes CUDA compatibility; collect `nvidia-smi` separately for the
installed driver release.

A capture count is a storage limit, not permission to truncate execution.
Timestamped samples do not advance clock time. A grouped frame emits in the
listed order but does not replace cross-partition observation barriers. Inspect
`result.ok()` and native diagnostics before using result values.

Native retention is qualified for a single finite source/output/collector in one
partition, including managed restart with fixed active pool capacity. It derives
the whole-run emission bound from the operator contract and checks every native
pool's capacity. It requests stop only
after collecting that complete bound, then verifies successful source progress.
The result preserves `kStopped`; `ok()` reports the verified boundary. A source
that emits fewer records fails this policy. Input-driven and variable-output tests
use independent snapshots. Lifecycle-bound queue sizing is not qualified for retained
captures. Deep restart uses the native bounded predecessor-pool inventory; it does
not reset the whole-run emission bound.

The controlled-clock example uses one partition and one scalar sample per required
input. It waits for native input availability, the requested tick's final outcome,
and output collection before requesting stop. A real startup tick at zero remains
part of execution. Observation overflow or a watchdog expiry fails the test; an
empty collector alone never proves suppression. Tensor, optional-input and
cross-partition controlled-clock tests are outside this initial profile.

## Python

Install `holoscan-core-cu13`, which includes `holoscan.testing`. For a build
published as `5.0.0` (replace it with your build's exact version), install pytest
and run from this example directory:

```bash
python -m pip install "holoscan-core-cu13==5.0.0" pytest
python -m pytest python
```

- `python/test_operator_testing.py`: the basic scalar Params workflow and
  explicit timestamps, including output timestamp assertions.
- `python/test_tensor_and_schema.py`: host Tensor inputs, generated schema
  value copies and owning Tensor-bearing schema snapshots.
- `python/test_advanced_testing.py`: typed capture-failure inspection and a
  complete managed restart with finite manual frames and a terminal receipt.

The Python examples use the same native fixture as the C++ examples and require
no GPU Python array package. Native SDK library dependencies still apply. Results
remain usable after the builder closes. The testing package checks its private
Core bridge at import; matching public version numbers alone do not guarantee
compatibility between independently rebuilt SDKs. The Core wheel keeps both
components together. Custom builds with `HOLOSCAN_BUILD_TESTING_LIBRARY=OFF`
omit testing.

See the [Python operator-testing guide](https://docs.nvidia.com/holoscan/sdk-user-guide/debugging-and-testing/testing#python-operator-tests)
for registered C++ provider construction, managed execution, restart permissions,
configuration and capture ownership.
