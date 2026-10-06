# Parameter-derived Tensor pool capacity

> **User guide:** [Tensors in depth](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth#derive-slot-bytes-from-deployment-parameters) — see how compile-time expressions size runtime-owned Tensor pools.

This example provides equivalent C++ and Python authoring for one graph with four scalar capacity
derivations and one memory-kind derivation. The C++ executable additionally runs one rate-aware
lifecycle graph:

- `ConfiguredResize` reduces a bounded HWC shape Parameter to the exact contiguous byte capacity
  for each output slot and independently maps an `OutputStorage` Parameter to pageable or pinned
  host memory.
- `PackedGrayscale` divides its producer's finite byte capacity by four for a packed RGBA-to-gray
  transform, demonstrating checked scalar expression composition.
- `ConservativeGrayscale` uses the producer's complete finite byte capacity directly, avoiding any
  packed-layout assumption at the cost of reserving more memory.
- `ProjectedBatchOutput` calculates `width * height * max_batch_size * 4 * sizeof(float)`. C++ uses
  a named compile-only projector with all three Parameters declared as dependencies; Python uses
  the equivalent bounded closed-expression tree. Runtime receives only the resolved per-slot byte
  span. Its output is connected to `ProjectedBatchSink`, so the projected capacity participates in
  an ordinary producer-to-consumer Tensor pipeline.
- C++ `CameraSource` and Python `CapacityCameraSource` declare a fixed 60 Hz output;
  `ConfiguredResize` carries a 60 Hz input
  expectation and declares `follows(input, RateRatio{1, 4})`, and a one-second clocked consumer
  causes the resulting 15 Hz output to receive the default 30-entry automatic recommendation.
- A second, direct-host Tensor source derives 6 MiB per slot from a Parameter, selects 60 Hz at
  `kDiscover` under a 120 Hz maximum, and restarts at 30 Hz while reusing the larger allocation.

The Python graph implements each Operator natively. Its `setup()` methods author the same input
contracts, exact rates, checked capacity formulas, and memory-kind mapping as the C++ graph without
registering or loading C++ example Operators. Cross-language tests compare the resulting
language-neutral Tensor, rate, and allocation evidence.

The command-line shape must have exactly four channels. The C++ positional memory-kind argument and
Python `--output-storage` option accept `host` or `pinned`, so the installed example remains CPU-only.
The 4:1 ratio is valid because this
example's input capacity describes packed RGBA bytes. It is not generally valid when an input bound
includes padded rows or another non-packed layout. Such an Operator must use a conservative bound
or a different formula justified by its input contract.

Both grayscale consumers declare an exact input memory kind matching the selected resize output.
That constraint is load-bearing: a derived producer may select only a kind accepted by every exact
consumer, and compilation rejects a mismatch rather than inserting an implicit copy. Each grayscale
stage deliberately produces a new pageable-host output buffer; that is an explicit transform-output
contract, not an inferred transition from the input kind.

The connected fixed output is authoritative for the resize input. `expected_rate(60 Hz)` is weaker
fallback and diagnostic evidence; it does not override a connected producer. The resize output's
1/4 ratio resolves to 15 Hz, but `follows()` does not schedule `compute()`. The temporal contract
still decides when the Operator runs and which samples it consumes.

With a one-second accumulation horizon, the automatic edge has 15 nominal arrivals and the default
2/1 headroom produces a recommendation of 30. The sibling edge uses
`ConnectionOptions{.queue_depth = 4}` and therefore stays exactly depth four; the nominal
recommendation is evidence, not permission to resize an explicit queue.

The lifecycle graph makes the byte and entry dimensions visible separately. Its automatic edge is
8 entries at the selected 60 Hz and 16 entries over the admitted 120 Hz domain. Four retained
history entries, one producer loan, and one retirement claim make the complete pool 14 active
entries and 22 admitted entries. At 6 MiB per slot those logical products are 84 MiB and 132 MiB.
The sums are valid because the compiler records those as distinct simultaneously live storage; it
uses maximum, not sum, for alternative views of the same entries.

The program prints resolved per-slot bytes, final provider slots, aggregate capacity, rate-derived
and explicit queue depths, lifecycle generation evidence, plan artifact size, and compile time.
It also renders the restarted lifecycle generation through the public
`render_rate_queue_inspection()` helper in both review-oriented human text and deterministic
canonical JSON. The human form is intended for diagnosis and the JSON form for tooling; neither
is an `ExecutionPlan` artifact or a runtime mutation. The inspection makes the logical payload
capacity distinct from the provider's physical reservation (which may be unavailable) and emits a
paste-ready exact queue-depth pin only for an automatically sized edge. An already explicit queue
depth is reported as authoritative and never offered as a replacement pin.
The default 2/1 headroom is a release policy to validate on representative workloads, not a jitter,
burst, scheduling, or safety guarantee.

Changing `ResizeParams::output_shape` or `ResizeParams::output_storage` requires compiling a new
execution plan and starting a session from that plan. It does not require rebuilding or relinking
the application binary, and neither expression is evaluated while the plan runs. Memory kind is
separate from byte capacity, allocator/provider identity, and CUDA device placement. A device-backed
selection would still require explicit deployment authority; this example intentionally demonstrates
only pageable and pinned host pools.

This remains true in both languages: Python freezes its hosted Parameter snapshot before Core
compiles the graph, and no Python callback participates in expression evaluation.

The C++ `ProjectedBatchOutput` demonstrates the escape hatch for formulas that are clearer as a
checked projector than as a long closed expression. Its static function is non-throwing, returns
`expected<uint64_t, Error>`, and has a stable semantic ID independent of the C++ symbol address.
The compiler invokes it once against finalized generation-zero `BatchProjectionParams`, records
only the stable ID and canonical dependency identities, and then uses the same allocation path as
a literal bound. `max_batch_size` remains an ordinary recompile-required Parameter because it is an
independent capacity dimension; it is not inferred from one frame's width and height.

The `param_ref` list passed beside the projector is normative compiler input. Core cannot inspect
the C++ function body, so every Params field the projector reads must appear in that list; an
undeclared read is an authoring correctness bug even if the code compiles. The versioned semantic
ID must change whenever the algorithm can produce a different result for the same declared inputs.
Renaming or moving the C++ function without changing its result does not require an ID change.
The native Python Operator expresses this particular product with checked `multiplied_by()` nodes,
which are bounded, overflow-checked, and fully portable in the plan without a Python callable.

The lifecycle source binds only from its declared `kDiscover` callback. Repeating the same
normalized rate is idempotent; a different repeat conflicts. The global Discover barrier resolves
the value before Core's first allocation work and before the user `kAllocate` callback. A later
60-to-30 Hz restart preserves `RunGenerationId` and `AllocationGenerationId`, while publishing a
fresh `RateGenerationId` and `StorageBindingGenerationId`. Backing larger than the active logical
limit remains invisible to queue behavior.

## Build the C++ example

From the `cpp/` folder against an installed SDK:

```bash
cmake -B build -S . -Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
cmake --build build
```

## Run the C++ example

The default deployment uses an output shape of `1080 1920 4` and pageable host memory. Supply another
positive packed RGBA HWC shape and optionally select pinned host memory without changing or rebuilding
the source:

```bash
./build/derived_tensor_pool_capacity 720 1280 4
./build/derived_tensor_pool_capacity 720 1280 4 pinned
```

To emit only the canonical plan JSON for tooling or cross-language comparison:

```bash
./build/derived_tensor_pool_capacity --plan-json 720 1280 4
```

## Run the Python example

From an SDK development build or an environment containing the installed `holoscan` package:

```bash
python python/derived_tensor_pool_capacity.py
python python/derived_tensor_pool_capacity.py 720 1280 4
python python/derived_tensor_pool_capacity.py --output-storage pinned 720 1280 4
```

The Python example compiles its native Operators and reports each resolved byte span through the
public named derivation lookup. It does not parse the private plan schema. Use `--plan-json` to emit
the canonical Python-authored artifact for tooling and structural comparison with C++:

```bash
python python/derived_tensor_pool_capacity.py --plan-json 720 1280 4
```

## Expected output

Timings and artifact size are host-specific. The self-checked rate and capacity lines below are
deterministic for the documented `720 1280 4` invocation:

```text
output=resize.output per_slot_bytes=3686400 ...
output=packed_grayscale.output per_slot_bytes=921600 ...
output=conservative_grayscale.output per_slot_bytes=3686400 ...
output=projected_batch.output per_slot_bytes=29491200 ...
lifecycle_rate=60/1 max_rate=120/1 active_entries=14 active_bytes=88080384 admitted_entries=22 admitted_bytes=138412032 restart_rate=30/1 restart_entries=10 allocation=reused-larger binding=fresh
compile_us=... artifact_bytes=... resize_memory=host ...
projector=ProjectedBatchOutput.output-max-byte-span.v1 max_batch_size=2 projected_per_slot_bytes=29491200
fixed_rate=60/1 follows_ratio=1/4 resolved_rate=15/1 automatic_depth=30 explicit_depth=4
...
derived tensor pool capacity: configured and checked transform bounds verified
derived tensor pool capacity Python: resize_bytes=3686400 packed_grayscale_bytes=921600 conservative_grayscale_bytes=3686400 projected_batch_bytes=29491200 output_storage=host resize_memory=host native Python expressions verified
```

## What's next

Read [Tensors in depth](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth)
for byte expressions, exact queue-sizing arithmetic, hard-floor composition, and the distinction
between logical payload capacity and provider reservation. See
[Lifecycle callbacks](https://docs.nvidia.com/holoscan/sdk-user-guide/building-operators/lifecycle-callbacks)
for binding deadlines, stage-barrier visibility, and restart reuse.
