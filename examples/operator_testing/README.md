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
- `operator_testing_device`: two aligned device inputs, device snapshots, three
  manual partitions and optional Green Context deployment.
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
```

Set `-DOPERATOR_TESTING_DEVICE_ID=1` during CMake configuration to use that device
in CTest. The device checks return the CTest skip code 77 if the requested hardware
or Green Context capability is unavailable. Admission or execution failures on an
available target remain failures. All source and operator device outputs have
explicit placement bindings, including both ports on the grouped source.

Deployment uses ordinary `CompileOptions`. The advanced `configure_graph` hook
receives fresh Graph authoring handles and places sources, operator and collectors
in three complete manual partitions. Green Context is an explicit operator-partition
binding; the result verifies its accepted device and actual copied output. The
ordinary thread-placement value can be replaced with native affinity or scheduling
settings supported by the execution target.

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
