# Holoscan SDK Examples

This directory contains examples to help users learn how to use the Holoscan SDK for development.
See [HoloHub](https://github.com/nvidia-holoscan/holohub) to find additional reference applications.

## Build instructions

- **From source**: See the [building guide](../DEVELOP.md)
- **Python wheels**: Download the python examples from GitHub, no building necessary.
- **NGC container**: the python examples and pre-built C++ examples are already included under `/opt/nvidia/holoscan/examples`. You can rebuild the C++ examples like so:

   ```sh
   export src_dir="/opt/nvidia/holoscan/examples/" # Add "<example_of_your_choice>/cpp" to build a specific example
   export build_dir="/opt/nvidia/holoscan/examples/build" # Or the path of your choice
   cmake -S $src_dir -B $build_dir -D Holoscan_ROOT="/opt/nvidia/holoscan"
   cmake --build $build_dir -j
   ```
- **Debian package**: Run pre-built examples under `/opt/nvidia/holoscan/examples` or rebuild the C++ examples as shown above. Debian packages
support C++ examples only.

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
   6. [**ping_simple_async_buffer**](ping_simple_async_buffer): connecting two operators with an async buffer
   7. [**ping_periodic_async_buffer**](ping_periodic_async_buffer): demonstrating async buffer connection between two operators that are running with periodic conditions
3. [**Video Replayer**](video_replayer): switch the source/sink from Tx/Rx to loading a video from disk and displaying its frames
4. [**Distributed Video Replayer**](video_replayer_distributed): switch the source/sink from Tx/Rx
   to loading a video from disk and displaying its frames, with a distributed application
5. [**Flow Tracker**](flow_tracker): simple application demonstrating data flow tracking for latency analysis
6. [**Custom CUDA kernel 1d**](custom_cuda_kernel_1d_sample): application demonstrating ingestion of 1D custom CUDA kernel in Holoscan SDK
7. [**Custom CUDA kernel multi sample**](custom_cuda_kernel_multi_sample): application demonstrating ingestion of multiple custom CUDA kernels of multi dimension in Holoscan SDK
8. [**Flow Control**](flow_control): demonstrate how to control dynamic operator execution flow in a pipeline (e.g. condition flow paths and loops)

## Core: additional configurations

The following examples illustrate the use of specific **schedulers** to define when operators are run:

* [**Multithread or Event-Based Schedulers**](multithread): run operators in parallel
* [**Round-Robin Broadcast and Gather**](round_robin_parallel): Illustrates processing of multiple sequential frames in parallel to allow removing a bottleneck. Multiple copies of a "slow" operator are launched in parallel for subsequent frames and then results are gathered back into a common pipeline for further processing/display. This app makes use of the event-based scheduler for this purpose.
* [**Multi-Rate Pipeline**](multi_branch_pipeline): Demonstrates how to override default operator port properties to allow parallel downstream branches of a pipeline to operate at different frame rates

The following examples illustrate the use of specific **conditions** to modify the behavior of operators:

* [**PeriodicCondition**](conditions/periodic): trigger an operator at a user-defined time interval
* [**AsynchronousCondition**](conditions/asynchronous): allow operators to run asynchronously (C++ API only)
* [**ExpiringMessageAvailableCondition**](conditions/expiring_message): allow operators to run when a certain number of messages have arrived or after a specified time interval has elapsed.
* [**MultiMessageAvailableCondition**, **MultiMessageAvailableTimeoutCondition**](conditions/multi_message): allow operators to run only once a certain number of messages have arrived across multiple associated input ports (optionally with a timeout on the interval to wait for messages).
* [**native Conditions**](conditions/native): demonstrates how a Holoscan native Condition (not wrapping an underlying GXF SchedulingTerm) can be created and used.
* [**OrConditionCombiner**](conditions/or_combiner): This example shows the use of the `or_combine_port_conditions` method to add a `OrConditionCombiner` resource that notifies the scheduler to use an OR combination instead of AND combination for the conditions present on the specified input ports.

The following examples illustrate the use of specific resource classes that can be passed to operators or schedulers:

* [**Clock**](resources/clock): demonstrate assignment of a user-configured clock to the Holoscan SDK scheduler and how its runtime methods can be accessed from an operator's compute method.

* [**ThreadPool**](resources/thread_pool): demonstrates pinning of operators to specific CPU threads in a thread pool.

* [**native Resources**](resources/native): demonstrates how a Holoscan native Resource (not wrapping an underlying GXF Component) can be created and used. 

* [**CudaStreamPool** and **CudaStreamCondition**](resources/cuda_stream_pool): demonstrates how Python apps can make use of a `CudaStreamPool` resource and `CudaStreamCondition` condition. Also demonstrates how to use stream-related API from native Python operator `compute` methods and have CuPy calls within `compute` use the desired stream.

## Decorator-based Python API

* [**Python Functions as Operators**](python_decorator): demonstrates how to use a decorator to convert a Python function into an Operator.

## Visualization
* [**Holoviz**](holoviz): display overlays of various geometric primitives

## Inference
* [**Activation-map**](activation_map): A simple inference pipeline demonstrates selecting a subset of models.
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
* [**Wrap Holoscan as GXF Extension**](wrap_holoscan_as_gxf_extension): wrap Holoscan native operators and resources as GXF codelets and components within a single GXF extension for use in GXF applications
