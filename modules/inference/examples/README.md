<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan Inference Operator Example

This example demonstrates `InferenceOp` with 8 configuration variants, from a single model up to
concurrent multi-model dispatch with CUDA Graphs and context isolation. Build the SDK via
`./run build`, then run:

```sh
./run launch --run-cmd './build-x86_64/modules/inference/examples/holoinfer_example --config modules/inference/examples/configs/01_single_model.yaml'
```

Try any config in `configs/` for a different scenario — see the index below.

## What is InferenceOp

`holoscan::ops::InferenceOp` is a TensorRT-only inference operator. Give it one or more model
paths and the input/output tensor names each model consumes and produces, and it builds (or loads
a cached) TensorRT engine per model, declares one Holoscan port per configured tensor, and runs
inference automatically on the fastest dispatch strategy safe for your configuration — including
automatically protecting concurrent CUDA-Graph capture across models when needed. This example
shows how to configure `InferenceOp` from a YAML file and run it as a small standalone pipeline.

## Prerequisites

Build the SDK first:

```sh
./run build --standalone --cudaarchs all
```

This builds `holoinfer_example` as part of the `examples` target. If you are running inside
Docker, pass `--gpus all` (or the SDK's `--runtime=nvidia` container flags) so the container can
see a GPU — `InferenceOp` requires CUDA and TensorRT.

## How to run

Run in a compatible runtime environment:

```sh
./holoinfer_example --config configs/<NN_config>.yaml [--ticks NUMBER_OF_TICKS]
```

`--config` is required; `--ticks` defaults to 10. Model paths in `configs/*.yaml` are resolved
relative to the YAML file itself, so the command above also works with an absolute `--config` path
run from any working directory.

### YAML shape used in this example

Every config in `configs/` is a **single-InferenceOp** YAML — the file's root IS the operator's
parameter block. The example driver calls `InferenceOpParams::from_yaml(path)` which reads the
root as the aggregate.

If your application configures more than one operator, use the multi-operator variant instead —
put each operator's parameters under a distinct named key at the file's root, and load each
sub-node with `InferenceOpParams::from_yaml_node(node, config_dir)`. See the "Multi-operator
composition" section of `../../modules/inference/README.md` for the schema, a worked YAML
example, and the C++ loading snippet. The two loaders share the same schema — the required
`model_path_map` / `input_map` / `output_map` and the optional scalar/bool/pool keys — so a
YAML block written for one InferenceOp instance works identically whether it sits at the file
root or under a named sub-key.

## Use-case index

| Config | Description |
| ------ | ----------- |
| [`01_single_model.yaml`](configs/01_single_model.yaml) | One TensorRT model on GPU-resident tensors. |
| [`02_multi_model_independent.yaml`](configs/02_multi_model_independent.yaml) | Two independent models with no data flow between them, run sequentially. |
| [`03_multi_model_chained.yaml`](configs/03_multi_model_chained.yaml) | One model's output tensor feeds another model's input. |
| [`04_multi_model_concurrent.yaml`](configs/04_multi_model_concurrent.yaml) | The two independent models from 02, run concurrently on dedicated streams. |
| [`05_multi_model_concurrent_graphs.yaml`](configs/05_multi_model_concurrent_graphs.yaml) | Same as 04, with CUDA Graphs enabled for lower per-tick launch overhead. |
| [`06_multi_model_concurrent_isolated.yaml`](configs/06_multi_model_concurrent_isolated.yaml) | Same as 05, with a green-context pool for the best concurrent throughput. |
| [`07_dynamic_shape_single.yaml`](configs/07_dynamic_shape_single.yaml) | One model whose input dimensions may change between ticks. |
| [`08_dynamic_shape_concurrent_isolated.yaml`](configs/08_dynamic_shape_concurrent_isolated.yaml) | Two dynamic-shape models, run concurrently with CUDA Graphs and context isolation. |

## Per-scenario walkthrough

### 01 — Single model

**Goal:** the simplest possible deployment — one model, one input, one output, all on
the GPU. **When to use it:** a single-model inference stage in a larger pipeline. Sample output:

```
Config: 01_single_model.yaml
  tick 1: ...
  tick 2: ...
Summary: 10 ticks in ...
```

### 02 — Multiple independent models

**Goal:** run two unrelated models (e.g. two different classifiers over two different
camera feeds) from a single operator instance. **When to use it:** you want one `InferenceOp` to
own multiple models but don't need them to run in parallel.

### 03 — Chained models

**Goal:** model A's output tensor feeds model B's input inside a single `InferenceOp` — a
multi-stage pipeline expressed via overlapping `input_map` / `output_map` tensor names.
`InferenceOp` derives the execution DAG from the tensor names automatically and runs each stage
in dependency order. **When to use it:** whenever you have a tightly-coupled multi-stage inference
pipeline that shares one allocator, one stream pool, and one lifecycle — encoder → decoder,
vision-language + denoiser, tokenizer → LLM → post-decode, etc. Chaining inside one operator
avoids per-stage queue/dispatch overhead and lets the module amortize TRT setup across every
stage at `start()`. Use separate `InferenceOp` instances only when a stage needs distinct
resources (its own stream pool, its own protection strategy) or when non-inference operators sit
between stages.

### 04 — Concurrent independent models

**Goal:** the same two independent models as scenario 02, but dispatched concurrently on
dedicated CUDA streams for lower end-to-end latency. **When to use it:** independent models with
enough GPU headroom to run side by side.

### 05 — Concurrent models with CUDA Graphs

**Goal:** same as 04, with CUDA Graphs enabled to cut per-tick CPU launch overhead once
the graphs are captured. **When to use it:** the same concurrent multi-model case as 04, once
launch overhead (not GPU compute) is your bottleneck.

### 06 — Concurrent models with context isolation

**Goal:** the highest-throughput concurrent configuration — each model gets its own SM
partition via a green-context pool, so CUDA Graph capture for one model can never interfere with
another's. **When to use it:** production concurrent multi-model deployments where you can afford
to reserve a green-context pool.

### 07 — Dynamic input shape

**Goal:** a single model whose input tensor dimensions may legitimately change between
ticks (e.g. variable-length sequences, variable image tiling). **When to use it:** whenever the
model was built with a TensorRT dynamic optimization profile.

### 08 — Dynamic shapes, concurrent, isolated

**Goal:** the full-featured configuration — two dynamic-shape models, concurrent
dispatch, CUDA Graphs, and a green-context pool for capture isolation. **When to use it:** as a
reference for combining every capability at once.

## Defaults you don't have to set

Three flags govern where tensors live at each stage of the pipeline. **All three default to
`true` (device-resident)**, so a YAML that omits them is equivalent to setting them all to `true`:

| Field | What it controls | Default |
|---|---|---|
| `input_on_cuda` | Whether the input tensor arriving at `InferenceOp` is device-resident. If `false`, HoloInfer stages a host→device copy before TensorRT dispatch. | `true` |
| `output_on_cuda` | Whether HoloInfer's internal output buffer is device-resident. If `false`, HoloInfer stages a device→host copy after TensorRT dispatch. | `true` |
| `transmit_on_cuda` | Whether the emitted output tensor is device-resident on the outbound port. | `true` |

None of the example configs sets any of these to `false`, because the default (all device-resident,
no staging) is what most GPU inference pipelines want. Set them to `false` only when your source
operator emits host tensors, your consumer needs host tensors, or you want an explicit staging
copy inside the inference stage.

## Advanced: concurrent-capture safety

When multiple TensorRT models run in parallel with CUDA Graphs enabled, `InferenceOp` engages
concurrent-capture protection automatically — you never need to request it explicitly. Supplying a
`cuda_green_context_pool` (see scenarios 06 and 08) gives the best concurrent throughput, because
each model captures its graph inside its own isolated SM partition; without one, `InferenceOp`
falls back to runtime mutex-based capture serialization, which is safe and correct but adds a
small amount of setup-time and steady-state synchronization overhead compared to context
isolation.

## Troubleshooting

- **TensorRT engine cache location:** the first run of a config builds a `.engine` file next to
  the source `.onnx` model (in `models/`) and reuses it on subsequent runs with the same model,
  GPU, and TensorRT version. Delete the `.engine` file to force a rebuild.
- **Model path resolution:** `model_path_map` entries in `configs/*.yaml` are resolved relative to
  the YAML file's own directory, not the process's current working directory — so
  `../models/identity_model.onnx` in `configs/01_single_model.yaml` always resolves to
  `models/identity_model.onnx` next to this README, regardless of where you invoke
  `holoinfer_example` from.
- **Common config errors:** `model_path_map`, `input_map`, and `output_map` are required keys —
  `InferenceOpParams::from_yaml()` throws `std::runtime_error` naming the missing key. A model
  name that appears in `input_map`/`output_map` but not in `model_path_map` (or vice versa) is
  rejected by `InferenceOp::start()` with a descriptive validation error.
