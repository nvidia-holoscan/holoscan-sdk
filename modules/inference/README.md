<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HoloInfer

HoloInfer is the TensorRT inference toolkit backing `holoscan::ops::InferenceOp`. It owns the
per-model TensorRT contexts (`TrtInfer`), the multi-model dispatch manager (`ManagerInfer`) and
the fast dispatch paths that `InferenceOp` selects automatically at `start()`.

## Python authoring package

The optional module package exposes the operator from its canonical provider namespace:

```python
import holoscan as hs
from holoscan_inference.operators import InferenceOp

graph = hs.Graph("inference")
inference = InferenceOp(
    graph,
    "inference",
    model_path_map={"detector": "/models/detector.onnx"},
    input_map={"detector": ["frames"]},
    output_map={"detector": ["detections"]},
)
```

The separately distributed `holoscan-inference-cu13` wheel owns the canonical
`holoscan_inference` provider package, HoloInfer, and the operator library. Its
entry point also makes `holoscan.inference` available as an identity-preserving
alias when distribution metadata is installed. The wheel exact-pins Core while
TensorRT 10 and CUDA 13 remain deployment prerequisites.

The module uses a dedicated stable-ABI extension and links the Inference DSO; importing ordinary
`holoscan` does not load TensorRT or register `InferenceOp`. Loading the Inference DSO registers the
fully qualified `holoscan.inference.InferenceOp` catalog row for C++ and language-binding consumers.
`holoscan::ops::register_inference_operator_catalog()` returns the retained status for that row when
an explicit diagnostic checkpoint is useful.

The row has pass-through introspection because Core's generic parameter ABI does not represent the
map-shaped fields. Consequently, the typed module facade above is the only short import spelling;
the weaker `holoscan.operators.InferenceOp` generic alias is deliberately absent. The facade accepts
either `config_path` by itself or the three direct maps. A simple `required_parameters` set cannot
truthfully express that alternative, so discovery leaves it empty and the facade enforces the mode.

Relative model paths loaded through `config_path` resolve from the YAML file's directory. Direct
`model_path_map` strings remain deployment-target paths and resolve from the runtime process's
working directory. Authoring validates map structure and nonempty paths, but deliberately does not
require files to exist on the machine compiling the graph.

Native archive installs keep the binding payload in the
`holoscan-inference-python_libs` component; the native `holoscan-inference`
component does not install Python files. Wheel assembly uses the independent
`holoscan-wheel-inference` component.

## Dispatch kinds

`InferenceOp::start()` picks one of seven dispatch kinds
(`include/holoscan/operators/inference/fast_path_eligibility.hpp`) from the operator's existing
configuration. There is no user-facing opt-in.

## Installed CMake package

New consumers should use the dedicated package, pointing CMake at the SDK's shared package
directory:

```cmake
find_package(holoscan-inference CONFIG REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::infer holoscan::op_inference)
```

Configure with `-Dholoscan-inference_DIR=<prefix>/lib/cmake/holoscan`. The required compatibility
form is `find_package(holoscan COMPONENTS inference REQUIRED)`.

| Kind          | Models | Shapes  | Concurrency | Notes                                            |
| ------------- | ------ | ------- | ----------- | ------------------------------------------------ |
| `kFast`       | 1      | static  | –           | cached pointers, event handshake                 |
| `kSeqFast`    | N      | static  | sequential  | all models on the caller's stream                |
| `kParFast`    | N      | static  | parallel    | one CUDA stream per model, inline fan-out/fan-in  |
| `kDynFast`    | 1      | dynamic | –           | re-binds shapes each `compute()`                  |
| `kDynSeqFast` | N      | dynamic | sequential  | –                                                |
| `kDynParFast` | N      | dynamic | parallel    | –                                                |
| `kStandard`   | N      | any     | work queue  | canonical `execute_inference()`; always available |

Fast paths require **every** model to use the `trt` backend, device-resident inputs, and no
chained models.

## InferenceOp parameters

Every field of `InferenceOpParams` (declared in
`include/holoscan/operators/inference/inference.hpp`), with type, default, and semantics:

### Required

| Field | Type | What it does |
|---|---|---|
| `model_path_map` | `map<string, string>` | Model name → path to `.onnx` or pre-built `.trt` engine file. |
| `input_map` | `map<string, vector<string>>` | Model name → ordered list of input tensor names. |
| `output_map` | `map<string, vector<string>>` | Model name → ordered list of output tensor names. |

### Backend + engine

| Field | Type | Default | What it does |
|---|---|---|---|
| `backend` | `string` | `"trt"` | Only `"trt"` is supported by this build. |
| `backend_map` | `map<string, string>` | `{}` | Per-model override (each value must be `"trt"`). |
| `is_engine_path` | `bool` | `false` | If true, `model_path_map` entries are pre-built `.trt` engines rather than `.onnx` sources. |
| `enable_fp16` | `bool` | `false` | Build TensorRT engines in FP16 (halves memory + latency where the model tolerates it). |
| `dla_core` | `int32_t` | `-1` | DLA core index; `-1` disables DLA. |
| `dla_gpu_fallback` | `bool` | `true` | When DLA is enabled, allow layers that can't run on DLA to fall back to GPU. |

#### Cached engine filenames

The first time HoloInfer sees an `.onnx` source it builds a TensorRT engine and writes it next to
the ONNX file. Subsequent runs pick up the cached engine directly. The filename encodes every
input that affects the produced engine so a stale cache is never silently reused when the config
changes:

```
<model>.<gpu_name>.<CC_major>.<CC_minor>.<sm_count>.trt.<TRT_M>.<TRT_m>.<TRT_p>.<TRT_b>[.dla<core>].cfg<16-hex>.engine.<fp16|fp32>
```

- The human-readable middle covers the device (GPU name, compute capability, SM count), the
  TensorRT toolchain version, and (when set) the DLA core.
- `.cfg<16-hex-chars>` is a 64-bit FNV-1a hash over every `NetworkOptions` field that changes the
  built engine but is *not* already in the human-readable portion — currently `max_batch_size`,
  `max_memory`, `trt_opt_profile`, and `dla_gpu_fallback`. Changing any of these produces a new
  filename and triggers a rebuild.
- `.engine.<precision>` is kept last so the file reads as if `.engine.fp16` / `.engine.fp32` were
  the extension.

See `generate_engine_path()` in `modules/inference/src/infer/trt/utils.hpp` for the canonical
format definition and the maintenance rule when adding new build-affecting parameters.

### Data residency

| Field | Type | Default | What it does |
|---|---|---|---|
| `input_on_cuda` | `bool` | `true` | Received input tensors are device-resident. If `false`, HoloInfer stages an H→D copy. |
| `output_on_cuda` | `bool` | `true` | HoloInfer's internal output buffers are device-resident. If `false`, a D→H copy is staged. |
| `transmit_on_cuda` | `bool` | `true` | Emitted output tensors are on device. |

### Dispatch + performance

| Field | Type | Default | What it does |
|---|---|---|---|
| `parallel_inference` | `bool` | `true` | Dispatch multi-model workloads in parallel where possible. |
| `dynamic_input_dims` | `bool` | `false` | Input tensor dimensions may change between compute() calls. Enables dyn-fast dispatch variants. |
| `output_dynamic_dims` | `bool` | `false` | Reserved for future use (output dims are already re-queried per tick when `dynamic_input_dims` is set). |
| `enable_cuda_graphs` | `bool` | `true` | Enable CUDA Graph capture in the TensorRT backend. Auto-fallback if capture cannot be protected — see the fallback ladder. |

### Resources

| Field | Type | Default | What it does |
|---|---|---|---|
| `allocator` | `shared_ptr<Allocator>` | `nullptr` (defaults to per-instance `UnboundedAllocator`) | Allocator for materialized output tensors. |
| `cuda_stream_pool` | `shared_ptr<CudaStreamPool>` | `nullptr` | Dedicated stream pool. Consumed by kFast (1 stream) and kParFast/kDynParFast (N streams, one per model). |
| `cuda_green_context_pool` | `shared_ptr<CudaGreenContextPool>` | `nullptr` | Provides SM partitioning + true concurrent-capture isolation for multi-model + CUDA Graphs. |

### From YAML

Two loader entry points, sharing the same schema:

- **`InferenceOpParams::from_yaml(std::string_view path)`** — reads the file's **root** as the
  entire aggregate. Use this when a YAML file configures one InferenceOp only (the "single-op
  YAML" convention used by every example config in `examples/configs/`).

- **`InferenceOpParams::from_yaml_node(const YAML::Node& node, const std::filesystem::path&
  config_dir)`** — reads any already-parsed YAML node as the aggregate. Use this when a single
  YAML file configures **multiple operators**, with each operator's parameters living under a
  named key at the file's root. `config_dir` is the directory that relative `model_path_map`
  paths resolve against (typically the parent of the YAML file that `node` was loaded from).

`from_yaml(path)` is a thin wrapper: it loads the file, uses the file's parent as `config_dir`,
and calls `from_yaml_node(root, parent)`. The two entry points are always in sync.

Schema is 1:1 with the fields above (map keys must match field names). The `cuda_stream_pool`
and `cuda_green_context_pool` sub-maps construct real resource objects
(`holoscan::CudaStreamPool` / `holoscan::CudaGreenContextPool`); when absent, the corresponding
`shared_ptr` member stays null. The `allocator` field cannot be populated from YAML — assign
it in code after loading if a non-default allocator is required.

#### Multi-operator composition

For applications with more than one operator, keep each operator's parameters under a distinct
top-level key:

```yaml
# multi_op_app.yaml — three operators in one file
primary_inference:                     # first InferenceOp's parameters
  model_path_map:
    model_a: models/model_a.onnx
  input_map:
    model_a: [tensor_in]
  output_map:
    model_a: [logits]
  enable_cuda_graphs: true

secondary_inference:                   # second InferenceOp's parameters
  model_path_map:
    model_b: models/model_b.onnx
  input_map:
    model_b: [logits]
  output_map:
    model_b: [prediction]
  parallel_inference: false

some_other_operator:                   # any other operator with its own from_yaml_node()
  ...
```

Then the app driver loads each subtree explicitly:

```cpp
#include <yaml-cpp/yaml.h>
#include <holoscan/operators/inference/inference.hpp>

YAML::Node root = YAML::LoadFile(config_path);
std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();

auto primary_params =
    holoscan::ops::InferenceOpParams::from_yaml_node(root["primary_inference"], config_dir);
auto secondary_params =
    holoscan::ops::InferenceOpParams::from_yaml_node(root["secondary_inference"], config_dir);
```

Each operator instance is constructed with its own aggregate — no shared or global state.
The subtree schema beneath each named key is identical to the single-op root schema; the same
required + optional keys apply, and the same defaults are used for absent scalar/bool keys.

`from_yaml(path)` throws `std::runtime_error` if the file cannot be read or parsed;
`from_yaml_node()` throws if a required key (`model_path_map`, `input_map`, or `output_map`) is
missing from the node.

### Dispatch selection

`InferenceOp::start()` uses `resolve_dispatch_with_fallback()` (see
`include/holoscan/operators/inference/fallback_ladder.hpp`) to pick the fastest safe
dispatch path for your configuration, and to emit a WARN if a fallback rung fires. The
seven possible dispatch modes are documented in
`include/holoscan/operators/inference/fast_path_eligibility.hpp`; the fallback rungs and
their triggers are documented in the "Fallback ladder" section of this file.

## Fallback ladder

InferenceOp automatically degrades to a safe dispatch mode when a config is
unsafe. Configuration-time rungs are resolved at start(); runtime rungs fire
during compute() if a specific fault surfaces. Every degrade emits a WARN (or INFO, per the
table) log in the standard format:

    fallback triggered — <REASON_KEY>
        requested: <primary_dispatch> + <protection>
        using:     <fallback_dispatch> + <fallback_protection>
        reason:    <one-line explanation>
        fix:       <user-side suggestion>

### Configuration-time rungs (start())

| Reason | Trigger | Fallback |
|---|---|---|
| StreamPoolInsufficient | `cuda_stream_pool.max_size < N_models` for a parallel path | Demote to sequential |
| DynParFastGraphsNoGreenContext | `kDynParFast` + graphs + no `CudaGreenContextPool` | Serialize into `kDynSeqFast` |
| GreenContextPartitionsInsufficient | GC pool present but `num_partitions < N_models` | Mutex net (static shapes) or serialize (dynamic shapes) |
| ChainedModels | Overlapping `input_map`/`output_map` values (chained pipeline) | `kStandard` |
| HostInputs | `input_on_cuda: false` | `kStandard` |
| NonTrtBackend | Any non-TRT model in the map | `kStandard` |
| ParallelRequestedSerialSelected | `parallel_inference: true` but a blocker forced sequential | (informational summary) |

### Runtime rungs (compute())

| Reason | Trigger | Fallback |
|---|---|---|
| BlockingStreamSwapped | Stream from pool is blocking-mode but CUDA Graphs need non-blocking | Auto-swap to a non-blocking stream |

### Worked example

Request `kParFast` + CUDA Graphs for 3 static-shape TRT models, with a `cuda_stream_pool` of
`max_size = 4` and **no** `cuda_green_context_pool`:

- Green-context isolation is unavailable — no pool.
- `StreamPoolInsufficient` does not fire: 4 ≥ 3 distinct streams are available, so the dispatch
  kind stays `kParFast`.
- `DynParFastGraphsNoGreenContext` does not apply: shapes are static, so a durable
  capture-serialization path is available.
- `decide_capture_protection()` therefore lands on **runtime mutex-based capture serialization**
  (`kMutexProtected`): a process-wide mutex guards concurrent CUDA-Graph capture across every
  `InferenceOp` in the process for the first few `compute()` ticks.
- Result: safe, correct, fully parallel steady state, with a small one-time setup cost and a
  mutex held for the first three ticks. No warning is emitted — this is a supported
  configuration, and an INFO line records the chosen protection.

See `INTERNAL_NOTES.md` for how the setup-time capture serialization is actually implemented.

Add a `cuda_green_context_pool` with 3 partitions and the same configuration moves up to
green-context isolation, where the mutex is skipped entirely.
