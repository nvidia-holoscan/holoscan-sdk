# GPU Resident Inference Example

This example demonstrates the use of the GPU-resident inference operator. Currently, we only support TensorRT backend.

## Overview

The example creates a simple pipeline in a fragment that uses the GPU-resident inference operator. The fragment includes three operators:

- A Source operator, that only has an output port and does not do anything in compute.
- An instance of the `GPUResidentInferenceOp` operator that inherits from `GPUResidentOperator`. It ingests the data from the source operator. The GPU Resident Inference operator ingests a YAML configuration file consisting of the infernece parameters and the tensor information.
- A Destination operator, that only has an input port and does nothing in the compute. It ingests the output from the inference operator.

## Configuration

The GPU Resident Infernece operator ingests a YAML configuration file that contains inference parameters and the tensor information.

### Inference Parameters

Three mandatory parameters must be present under the key `inference_parameters`.

- `model_path_map`: Map with key as model and value as path to the model.
- `pre_processor_map`: Map with key as model and value as the input tensor to the model.
- `inference_map`: Map with key as model and value as the output tensor from the model.

In the current release, only a single model with a single input and a single output is supported.

### Tensor information

With the GPU Resident infernece operator, it is mandatory to provide the dimensions and data type of the incoming and outgoing tensors to the model as shown below.

All data types supported by the Holoscan Inference module are supported. It includes float16, float32, uint8, int8, int32, int64 and bool.

```YAML
tensors:
  input_1:
    dim: "512,512,3"
    dtype: kFloat32
  output_1:
    dim: "512,512,3"
    dtype: kFloat32
```

## Running the example

```bash
cd <holoscan_sdk_root>
./<build-dir>/examples/gpu_resident_inference_example
```

## Expected Output

```
[info] [gpu_resident_inference_example.cpp:197] Iteration 10 - Preparing data
...
All results are correct
Tearing down the GPU-resident fragment
...
```

## Platform support and CUDA driver compatibility

GPU-resident execution relies on CUDA graph capture APIs (for example, `cudaStreamBeginCaptureToGraph`) that require CUDA Runtime 12.3+ support from the installed NVIDIA driver.

- Verify driver/runtime compatibility with `nvidia-smi` (the reported CUDA version must be at least 12.3).
- IGX Orin dGPU systems running IGX OS 1.x typically use R535 drivers (CUDA 12.2 compatibility), so this example is not supported on that default host configuration.
- On unsupported systems, startup can fail with:
  - `API call is not supported in the installed CUDA driver (36)`
- An optional workaround on IGX OS 1.x is to install [CUDA forward-compat components from the CUDA repository](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). This can make the example run, but it diverges from the default IGX OS 1.x package set and is not fully validated for all deployments.
