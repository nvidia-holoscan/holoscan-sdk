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