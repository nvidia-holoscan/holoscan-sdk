# MatX Allocator Example

This example demonstrates how to use `holoscan::MatXAllocator` to create MatX tensors backed by a Holoscan memory pool. Unlike the `matx_basic` example which uses MatX's default CUDA allocator, this example uses `RMMAllocator` for pooled GPU memory management.

## Overview

The application consists of two operators:

1. A transmitter (`MatXAllocTxOp`) that creates a MatX tensor using `MatXAllocator` with an `RMMAllocator`, populates it with data, and sends it downstream via DLPack.
2. A receiver (`MatXAllocRxOp`) that receives the tensor, demonstrates two ways to wrap it as a MatX tensor view (raw pointer and DLPack import), and performs a GPU-accelerated computation (`tensor * 2 + 1`).

## C++ API

- `MatXAllocTxOp`: Creates a `MatXAllocator` wrapping the `RMMAllocator`, uses it with `matx::make_tensor<float>()` to allocate a pooled tensor, and emits it as a `holoscan::TensorMap`.
- `MatXAllocRxOp`: Receives the `holoscan::TensorMap` and demonstrates two ways to create a MatX tensor view: (1) raw pointer via `tensor->data()` and (2) DLPack via `tensor->to_dlpack()` with MatX's native `make_tensor(TensorType&, const DLManagedTensor)` overload. Runs a MatX expression on the GPU.
- `MatXAllocatorApp`: Configures an `RMMAllocator` and wires the pipeline.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:

```bash
./examples/matx/matx_allocator/cpp/matx_allocator
```

## Key API Features Demonstrated

- `holoscan::MatXAllocator`: Wraps any Holoscan allocator for use with MatX's custom allocator interface.
- `holoscan::RMMAllocator`: Provides pooled GPU memory management backed by RMM.
- `matx::make_tensor<T>(shape, allocator)`: Creates a MatX tensor backed by a custom allocator.
- `matx::tensor::ToDlPack()`: Enables zero-copy MatX→Holoscan sharing via DLPack (used in TxOp).
- `matx::make_tensor(TensorType&, const DLManagedTensor)`: Type-safe zero-copy Holoscan→MatX import via DLPack (used in RxOp).
- `holoscan::Tensor::to_dlpack()`: Returns a `DLManagedTensor*` for DLPack-based tensor exchange.
- `holoscan::Tensor`: Represents tensor data within Holoscan, created from a DLPack object.

## Expected Output

The application will print log messages from both operators, showing the tensor creation and computation. The output will look similar to this:

C++:

```text
Created MatX tensor with 10 elements using pooled allocator
Received tensor 'tensor': 40 bytes
Input tensor (raw pointer):
tensor_1_f32: Tensor{float} Rank: 1, Sizes:[10], Strides:[1]
000000:  1.0000e+00
...
000009:  1.0000e+01
Input tensor (DLPack):
tensor_1_f32: Tensor{float} Rank: 1, Sizes:[10], Strides:[1]
000000:  1.0000e+00
...
000009:  1.0000e+01
Result of 'tensor * 2 + 1':
tensor_1_f32: Tensor{float} Rank: 1, Sizes:[10], Strides:[1]
000000:  3.0000e+00
000001:  5.0000e+00
000002:  7.0000e+00
000003:  9.0000e+00
000004:  1.1000e+01
000005:  1.3000e+01
000006:  1.5000e+01
000007:  1.7000e+01
000008:  1.9000e+01
000009:  2.1000e+01
```

The application will terminate after printing the output.
