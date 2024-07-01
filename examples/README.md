# Holoscan SDK Examples

This directory contains examples to help users learn how to use the Holoscan SDK for development.
See [HoloHub](https://nvidia-holoscan.github.io/holohub) to find additional reference applications.

## Build instructions

- **From source**: See the [building guide](../DEVELOP.md)
- **Python wheels**: Download the python examples from GitHub, no building necessary.
- **NGC container or debian package**: the python examples and pre-built C++ examples are already included under `/opt/nvidia/holoscan/examples`. You can rebuild the C++ examples like so:

   ```sh
   export src_dir="/opt/nvidia/holoscan/examples/" # Add "<example_of_your_choice>/cpp" to build a specific example
   export build_dir="/opt/nvidia/holoscan/examples/build" # Or the path of your choice
   cmake -S $src_dir -B $build_dir -D Holoscan_ROOT="/opt/nvidia/holoscan"
   cmake --build $build_dir -j
   ```

## Run instructions

See the README of each example for specific run instructions based on your installation type.

## Test instructions

- **From source**: See the [building guide](../DEVELOP.md#testing)
- **Python wheels**: not available
- **NGC container or debian package**:
  - Running the following command will run the examples and compare the results with expected baselines.

    ```sh
    ctest --test-dir $build_dir
    ```

  - To group building and testing:

    ```sh
    /opt/nvidia/holoscan/examples/testing/run_example_tests
    ```

## Example list

### Core

The following examples demonstrate the basics of the Holoscan core API, and are ideal for new users starting with the SDK:

1. [**Hello World**](hello_world): the simplest holoscan application running a single operator
2. **Ping**: simple application workflows with basic source/sink (Tx/Rx)
   1. [**ping_simple**](ping_simple): connecting existing operators
   2. [**ping_custom_op**](ping_custom_op): creating and connecting your own operator
   3. [**ping_multi_port**](ping_multi_port): connecting operators with multiple IO ports
   4. [**ping_distributed**](ping_distributed): transmit tensors from one fragment to another in a
      distributed application
   5. [**ping_cycle**](ping_cycle): connecting operators in a cyclic path
3. [**Video Replayer**](video_replayer): switch the source/sink from Tx/Rx to loading a video from disk and displaying its frames
4. [**Distributed Video Replayer**](video_replayer_distributed): switch the source/sink from Tx/Rx
   to loading a video from disk and displaying its frames, with a distributed application
5. [**Flow Tracker**](flow_tracker): simple application demonstrating data flow tracking for latency analysis

## Core: additional configurations

The following examples illustrate the use of specific **schedulers** to define when operators are run:

* [**Multithread or Event-Based Schedulers**](multithread): run operators in parallel
* [**Multi-Rate Pipeline**](multi_branch_pipeline): Demonstrates how to override default operator port properties to allow parallel downstream branches of a pipeline to operate at different frame rates

The following examples illustrate the use of specific **conditions** to modify the behavior of operators:

* [**PeriodicCondition**](conditions/periodic): trigger an operator at a user-defined time interval
* [**AsynchronousCondition**](conditions/asynchronous): allow operators to run asynchronously (C++ API only)

The following examples illustrate the use of specific resource classes that can be passed to operators or schedulers:

* [**Clock**](resources/clock): demonstrate assignment of a user-configured clock to the Holoscan SDK scheduler and how its runtime methods can be accessed from an operator's compute method.

## Decorator-based Python API

* [**Python Functions as Operators**](python_decorator): demonstrates how to use a decorator to convert a Python function into an Operator.

## Visualization
* [**Holoviz**](holoviz): display overlays of various geometric primitives

## Inference
* [**Bring-Your-Own-Model**](bring_your_own_model): create a simple inference pipeline for ML applications

### Working with third-party frameworks

The following examples demonstrate how to seamlessly leverage third-party frameworks in holoscan applications:

* [**NumPy native**](numpy_native): signal processing on the CPU using numpy arrays
* [**CuPy native**](cupy_native): basic computation on the GPU using cupy arrays

### Sensors

The following examples demonstrate how sensors can be used as input streams to your holoscan applications:

* [**v4l2 camera**](v4l2_camera): for USB and HDMI input, such as USB cameras or HDMI output of laptop
* [**AJA capture**](aja_capture): for AJA capture cards

### GXF and Holoscan

* [**Tensor interop**](tensor_interop): use the `Entity` message to pass tensors to/from Holoscan operators wrapping GXF codelets in Holoscan applications
* [**Import GXF Components**](import_gxf_components): import the existing GXF Codelets and Components into Holoscan applications
* [**Wrap operator as GXF extension**](wrap_operator_as_gxf_extension): wrap Holoscan native operators as GXF codelets to use in GXF applications
