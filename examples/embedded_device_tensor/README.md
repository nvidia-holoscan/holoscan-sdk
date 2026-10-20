# Embedded device Tensor

This example turns a flat device-Tensor pipeline into a reusable Graph without
changing its physical work:

```text
flat
  source -- Tensor<CUDA device 0> --> sink

embedded
  [device-stage/source -> exported frame] -- Tensor<CUDA device 0> --> sink
```

It teaches four boundaries that matter when a real application embeds a
device-producing component:

1. the child declares a named clock import and selects it as its default;
2. the immediate parent binds that exact import to a concrete root clock;
3. deployment binds the child output to device 0; and
4. one accepted `ExecutionPlan` starts two fresh `RunSession`s.

No application callback uses a private Runtime object, backend endpoint, test
accessor, implicit device zero, or build-tree-only dependency.

## Build and run

From an installed SDK prefix:

```bash
cmake -S cpp -B build
cmake --build build -j
./build/embedded_device_tensor
```

Expected output ends with:

```text
[embedded_device_tensor] replay=1 frames=4 termination=stopped
[embedded_device_tensor] replay=2 frames=4 termination=stopped
embedded_device_tensor PASS: flat-equivalent reusable plan replayed twice device=0
```

The normal replay path requires a CUDA-capable device. On a lane without a
usable device, it prints a skip reason and returns 77 so CTest records
unavailable hardware rather than weakening the device-path claim. The
artifact-only mode is a cold, GPU-independent check.

## Learning chain

### 1. Make the reusable boundary explicit

`DeviceStageGraph::compose()` owns `DeviceSource` and exports only its typed
`Output<Tensor>`. The parent sees the exported `frame` port, not the child's
internal Operator. The source and sink classes are also used by the flat
version, and the program checks that both accepted plans contain one payload
edge and one device resource. The embedded plan additionally retains one
interface-binding record; that alias is Graph identity, not extra device work.
Detailed plan-record equivalence belongs to the SDK's compiler and runtime
conformance tests rather than application code.

### 2. Import and bind the child's clock

The child calls:

```cpp
clock_import_ = import_clock("timeline");
set_default_clock(clock_import_);
```

The parent then resolves that exact instance-qualified dependency:

```cpp
graph.bind_clock(stage, stage->clock_import(), runtime_clock);
```

Matching names do not bind clocks. Every reusable boundary must export the
dependency and every immediate parent must bind it. Omitting the bind is a
compile-time error, not a Runtime fallback to another clock.

### 3. Freeze device placement

`CompileOptions::deployment` carries the cold device decision:

```cpp
options.deployment.bind_tensor_output_device({
    .operator_path = "device-stage/source",
    .output_port = "frame",
    .device = DeviceId{0},
});
```

The SDK never silently selects device zero, and one Operator cannot resolve
conflicting devices under the current one-compute-stream callback model.

### 4. Compile once, run twice

The source Graph can retire after `compile()`. The immutable accepted plan owns
the executable reconstruction state. Each `run_async(plan)` creates fresh
Operators, source counters, readiness state, clock bindings, and Tensor-pool
claims. The example stops each session cooperatively after four frames and
requires an empty structured diagnostic vector.

## Publish cold artifacts

Artifact publication is explicit and does not start lifecycle or Operator
callbacks. Compilation may perform read-only device-capability probes:

```bash
./build/embedded_device_tensor --emit-artifacts ./artifacts
```

The accepted plan is published beneath:

```text
artifacts/<plan-digest>/embedded-device-tensor/
```

`emit_artifacts()` writes canonical plan/catalog files and commits the bundle
with its manifest last. Re-running with an identical plan preserves canonical
content. Keep deployment artifacts with the build and configuration that
produced them; replay means constructing a fresh `RunSession` from an accepted
in-memory plan, not deserializing arbitrary JSON as executable authority.

## Limitations

- Device placement is explicit per Tensor-producing output in baseline; typed
  subtree and root defaults are later work.
- The demonstration uses one CUDA device.
- The sink validates Tensor metadata and the Runtime readiness-stream contract.
  It deliberately performs no host copy or synchronization on the steady-state
  path.

## Continue learning

- [`reusable_graphs`](../reusable_graphs/README.md) — interface ports, nested
  instances, clock imports, regions, and handle lifetime.
- [`tensors_in_depth`](../tensors_in_depth/README.md) — stream-ordered writes,
  device work, fanout retirement, and explicit host inspection.
- [`compile_and_check`](../compile_and_check/README.md) — accepted versus
  rejected plans, canonical JSON, and digests.
- [`experimental/plan_artifacts`](../experimental/plan_artifacts/README.md) —
  complete artifact publication and deterministic content identity.
