# MatX Basic Example

This example demonstrates how to use the [MatX](https://github.com/NVIDIA/MatX) library within a Holoscan SDK application. It shows how to pass a MatX tensor between Holoscan operators using zero-copy conversion to a `holoscan::Tensor`.

## Overview

The application consists of two operators:
1.  A transmitter (`MatXTensorTxOp`) that creates a MatX tensor, populates it with data, and sends it downstream.
2.  A receiver (`MatXTensorRxOp`) that receives the tensor, performs a simple computation on it using MatX, and prints the result.

The conversion between the MatX tensor and `holoscan::Tensor` is achieved using the DLPack standard, which allows for zero-copy data sharing between the two libraries.

## C++ API

The C++ application defines a `holoscan::Application` that connects the transmitter and receiver operators.

- `MatXTensorTxOp`: Creates a `matx::tensor` on the GPU, converts it to a `holoscan::Tensor` via `ToDlPack()`, and emits it as a `holoscan::TensorMap`.
- `MatXTensorRxOp`: Receives the `holoscan::TensorMap`, wraps the `holoscan::Tensor` data into a `matx::tensor` view without copying, and performs a GPU-accelerated operation (`tensor * 2 + 1`).
- `MatxBasicApp`: A `holoscan::Application` that connects the `tx` and `rx` operators.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/matx/matx_basic/cpp/matx_basic
```

## Key API Features Demonstrated

- `holoscan::Application`: To define the application workflow.
- `holoscan::Operator`: To create custom operators for the pipeline.
- `matx::make_tensor`: To create a new MatX tensor.
- `matx::tensor::ToDlPack()`: To enable zero-copy sharing with other frameworks.
- `holoscan::Tensor`: To represent tensor data within Holoscan and create from a DLPack object.
- `matx::print()`: To print a tensor to the console for debugging.

## Expected Output

The application will print log messages from the receiver operator, showing the received tensor and the result of the computation. The output will look similar to this:

C++:
```text
tensor name: tensor
tensor nbytes: 40
MatX tensor:
tensor_1_f32: Tensor{float} Rank: 1, Sizes:[10], Strides:[1]
000000:  1.0000e+00
000001:  2.0000e+00
000002:  3.0000e+00
000003:  4.0000e+00
000004:  5.0000e+00
000005:  6.0000e+00
000006:  7.0000e+00
000007:  8.0000e+00
000008:  9.0000e+00
000009:  1.0000e+01
Result of 'matx_tensor * 2 + 1':
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
