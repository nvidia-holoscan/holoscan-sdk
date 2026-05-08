# Holoscan SDK GPU-Resident Graphs

## Overview

Holoscan SDK's GPU-resident graphs enable deterministic, real-time and low-latency execution of Holoscan applications by keeping the (CUDA) compute pipeline on the GPU for the lifetime of an application. GPU-resident graph execution mode allows an application to be operated entirely on the GPU, without (or with minimal) involvement from the CPU. This mode eliminates scheduling, synchronization, orchestration and systems overheads on the CPU, making GPU applications more predictable. Unlike traditional CPU-driven scheduling and execution, GPU-resident graphs leverage CUDA Graphs to capture and replay an entire Holoscan application directly on the GPU, eliminating costly CPU-GPU coordination and synchronization overhead.

For sensor inputs and actuation outputs, Holoscan SDK GPU-resident graphs are combined with devices and mechanisms using GPU-direct RDMA technologies such as [Holoscan Sensor Bridge](https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/), [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca-gpunetio/index.html). Holoscan SDK GPU-resident graphs also support low-latency, highly responsive and predictable visualization outputs, especially on [NVIDIA G-SYNC](https://developer.nvidia.com/g-sync) supported monitors.

The GPU-resident graphs do not follow the traditional Holoscan SDK execution
workflow with GXF backend and cannot be interconnected with a GXF-backend
fragment and operator. It is a standalone and unique execution model, supported
by a separate Holoscan SDK executor backend.

:::{note}
GPU-resident graphs are only supported in C++. Python support is planned for the future.
:::

### Key Benefits

- **Deterministic Performance**: CUDA Graphs provide consistent and predictable execution time and end-to-end latency by eliminating scheduling, coordination and operating system overheads on the CPU.
- **Reduced Latency**: Eliminates CPU-GPU synchronization and CPU-driven GPU workload scheduling overheads by executing entirely on the GPU
- **Fast and Low-latency I/O**: GPU-direct RDMA combined with GPU-resident graphs in Holoscan SDK eliminates costly and unpredictable I/O interrupt and management overheads on the CPU, enabling fast and low-latency I/O for the GPU-resident applications.
- **Scalability**: CPU-dependent workloads in embedded systems are limited by the small number of CPU cores available to scale applications for a multitude of sensors and actuators for robotics, medical, and other real-time and embedded applications. In contrast, the exponentially higher number of CUDA cores on the GPU allows embedded applications to easily scale. GPU-resident graphs take advantage of this scalability of the GPUs to support a huge numbers of sensors and actuators.
- **Efficient Resource Usage**: Pre-allocated device memory and optimized data flow

### When to Use GPU-Resident Graphs

GPU-resident graphs are ideal for:

- Real-time applications where predictable and consistent execution timing is critical
- Applications with CUDA-only operators
- Applications that either have no CPU-based process dependencies or where a CPU-based Holoscan fragment can work asynchronously with a GPU-resident (CUDA-only) Holoscan fragment

## Key Concepts for Developers

### GPU-Resident Operators

GPU-resident operators inherit from `holoscan::GPUResidentOperator` instead of the standard `holoscan::Operator` class. These operators:

- Execute entirely on the GPU using CUDA kernels
- Use device memory for input/output ports
- Are captured into CUDA Graphs

### GPU-Resident Fragments

A GPU-resident Fragment is created by composing GPU-resident operators. The framework automatically detects that a `holoscan::Fragment` should use GPU-resident graphs when all operators in the fragment inherit from `holoscan::GPUResidentOperator`. GPU-resident execution supports acyclic operator graphs with a single source operator. During initialization, the framework flattens the graph in topological order before connecting device memory and capturing `compute()` calls into CUDA Graphs.

To create a GPU-resident Fragment:

1. Create operators that inherit from `holoscan::GPUResidentOperator`
2. Compose them in a standard Fragment using `make_operator<>()` and `add_flow()`
3. The framework will automatically enable GPU-resident graphs during initialization

```cpp
class MyGpuResidentFragment : public holoscan::Fragment {
 public:
  void compose() override {
    // Create GPU-resident operators
    auto source = make_operator<SourceGpuOp>("source");
    auto compute = make_operator<ComputeGpuOp>("compute");
    auto sink = make_operator<SinkGpuOp>("sink");

    // Connect them with standard dataflow edges
    add_flow(source, compute);
    add_flow(compute, sink);

    // For operators with multiple ports, specify the port mapping explicitly
    // add_flow(source, compute, {{"out0", "in0"}, {"out1", "in1"}});
    // add_flow(compute, sink, {{"sum", "in_sum"}, {"diff", "in_diff"}});
  }
};

// Holoscan SDK will automatically detect that MyGpuResidentFragment should use GPU-resident graphs
```

Access GPU-resident controls via `fragment->gpu_resident()` after calling `run_async()`. More details on the [API reference](#api-reference) section.

There can only be one GPU-resident fragment per Holoscan SDK application. Other traditional Holoscan SDK fragments can co-exist with a GPU-resident fragment in the same application. However, connections between GPU-resident and traditional Holoscan SDK fragments and operators (with GXF backend) are not supported.

#### Main Workload Processing GPU-resident Fragment

The GPU-resident fragment is designated as the main workload processing fragment. The main data processing such as image processing, AI model inference happens in this main fragment. There is also an optional [data ready handler](#data-ready-handler-optional) fragment that can be used to augment the main fragment to automatically handle external sensor data inputs.

### Data Ready Handler (Optional)

A data ready handler is an optional feature that allows a GPU-resident fragment to handle external data sources to be handled on the GPU itself. By leveraging this feature, a CUDA kernel, used as a data ready handler, can check whether new data is available for processing, and a GPU-resident data processing pipeline can subsequently be triggered.

A data ready handler is created with a separate GPU-resident fragment that is registered with the main GPU-resident fragment. This handler runs at the beginning of each iteration. It can determine if data is ready for processing. It is important to note that the data ready handler fragment is not a separately executable GPU-resident fragment in a Holoscan SDK application. This fragment is only used as a part of the main workload processing GPU-resident fragment. **Holoscan SDK currently only allows one main workload processing GPU-resident fragment per application.**

The data ready handler allows for:

- Custom data availability checking logic
- Conditional execution of the main workload
- Integration with external sensor data sources by leveraging GPU-direct technologies such as [Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/index.html).

This feature allows integration of GPU-direct technologies into the Holoscan SDK GPU-resident graph execution mode. By leveraging this feature, developers build sensor data-driven GPU-resident pipelines. As sensor data arrives in the GPU, without any involvement from the CPU, the data ready handler will detect it and trigger the main GPU-resident (data processing) workload.

### CUDA Graph Backend

When a GPU-resident fragment is initialized, the framework:

1. **Allocates device memory** for all inter-operator connections based on port specifications
2. **Captures operator execution** into CUDA Graphs by recording the `compute()` method
3. **Creates a GPU-resident graph execution** (that optionally includes control flow for data ready checking)

The GPU-resident CUDA graph is launched once from the host CPU process. The graph keeps running on the GPU until the application terminates. For every new (sensor) data inputs, the GPU-resident CUDA graph processes the data without ever leaving the GPU. In the absence of any intervention from the host CPU process, the GPU-resident graphs maintain deterministic execution timing for sensor data processing.

### Graph Execution Flow

1. Application calls `run_async()` to start GPU-resident graphs
2. GPU-resident CUDA graph is launched asynchronously
3. For each iteration:
   - Data ready handler (if present) checks if input data is available
   - If the data is ready, main workload operators execute in sequence
   - Result ready signal is set when processing completes

#### Host CPU-driven GPU-resident Graphs

Optionally, the host CPU can also control the GPU-resident graphs by the following steps:

- Write input data to device memory
- Call `data_ready()` to trigger processing
- Check `result_ready()` to know when results are available
- Read output data from device memory until the `data_ready()` signal is set again

This is useful for debugging, development, testing and cases where GPU-direct technologies are not available/yet integrated.

When using host CPU-driven graphs with functions like `cudaMemcpy` to read back results between iterations, enable `sync_with_host()` before launching the graph to guarantee that all device memory writes are visible to the host before `result_ready()` returns `true`. See [sync_with_host](#configuration-methods) for details.

## API Reference

### GPUResidentOperator

The base class for GPU-resident operators is `holoscan::GPUResidentOperator`. Inherit from this class to create operators that execute in GPU-resident mode.

#### Port Declaration

Ports can be declared in two ways: by **memory block size** (the executor allocates shared device memory) or by **device pointer** (the operator provides its own pre-allocated device memory).

**Memory block size (executor-allocated)**

```cpp
class MyGpuOp : public holoscan::GPUResidentOperator {
  void setup(OperatorSpec& spec) override {
    // Declare device input port with memory size (in bytes)
    spec.device_input("in", sizeof(float) * num_elements);

    // Declare device output port with memory size (in bytes)
    spec.device_output("out", sizeof(float) * num_elements);
  }
};
```

Use `device_input()` and `device_output()` with a `size_t` or integer literal to declare ports with executor-allocated device memory. The executor will allocate a shared device buffer for each connection. Connected ports map to the same device memory address. Operators may declare multiple input and/or output ports; each port is connected independently via the port map in `add_flow()`.

**Device pointer (operator-managed)**

```cpp
class MyCustomAllocGpuOp : public holoscan::GPUResidentOperator {
  void setup(OperatorSpec& spec) override {
    // Allocate device memory externally
    cudaMalloc(&my_input_ptr_, sizeof(float) * num_elements);
    cudaMalloc(&my_output_ptr_, sizeof(float) * num_elements);

    // Declare ports with externally managed device pointers
    spec.device_input("in", reinterpret_cast<CUdeviceptr>(my_input_ptr_));
    spec.device_output("out", reinterpret_cast<CUdeviceptr>(my_output_ptr_));
  }

 private:
  void* my_input_ptr_ = nullptr;
  void* my_output_ptr_ = nullptr;
};
```

Use `device_input()` and `device_output()` with a `CUdeviceptr` or `void*` argument to supply an externally allocated device pointer. The executor will use this pointer directly instead of allocating its own buffer. The connected port on the other operator will also map to this pointer.

:::{note}
Integer literals (e.g. `0`) always resolve to the memory block size overload, not the device pointer overload. The device pointer overload is only selected when the argument type is explicitly `CUdeviceptr` or `void*`.
:::

#### Connection Strategy

When two operators are connected, the executor decides how to set up shared device memory for each port pair independently based on what each port declares:

1. **Both ports declare a memory block size** -- the executor allocates a shared buffer (sizes must match).
2. **One port declares a device pointer** -- the executor uses that pointer for both ports. If the other port has a memory block size, it is ignored (with a warning).
3. **Both ports declare a device pointer** -- the source's pointer is used (an error is logged).
4. **Only one port declares a memory block size** (the other has neither) -- the executor allocates using the available size (with a warning).
5. **Neither port declares anything** -- a runtime error is thrown.

#### Important Helper Methods

**`cuda_stream()`**

```cpp
std::shared_ptr<cudaStream_t> cuda_stream();
```

Returns the CUDA stream for launching kernels in the operator's `compute()` method.

**`device_memory(port_name)`**

```cpp
void* device_memory(const std::string& port_name);
```

Returns the device memory address for a given input or output port. Use this to access pre-allocated buffers for kernel launches.

**`data_ready_handler_cuda_stream()`**

```cpp
std::shared_ptr<cudaStream_t> data_ready_handler_cuda_stream();
```

Returns the CUDA stream for data ready handler operations.

**`data_ready_device_address()`**

```cpp
void* data_ready_device_address();
```

Returns the device memory pointer for the data ready signal. This address can be used in data ready handler's CUDA kernels to signal that data is ready for processing. See `holoscan/core/executors/gpu_resident/gpu_resident_dev.cuh` for CUDA device functions like `gpu_resident_mark_data_ready_dev()` and `gpu_resident_mark_data_not_ready_dev()` where this address can be used.

#### Example Operator Implementation

```cpp
class ComputeGpuOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ComputeGpuOp, holoscan::GPUResidentOperator)

  void setup(OperatorSpec& spec) override {
    spec.device_input("in", sizeof(float) * 1024);
    spec.device_output("out", sizeof(float) * 1024);
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // Get device memory addresses
    auto* input_addr = device_memory("in");
    auto* output_addr = device_memory("out");

    // Get CUDA stream for kernel launch
    auto stream_ptr = cuda_stream();
    cudaStream_t stream = *stream_ptr;

    // Launch CUDA kernel
    my_kernel<<<grid, block, 0, stream>>>(
        static_cast<float*>(input_addr),
        static_cast<float*>(output_addr),
        1024);
  }
};
```

:::{note}
When the size of an input or output port is not known at `setup()` time, the size can be set to zero (a warning will be logged). Later, `initialize()` method can be used to set the final size of the port.
:::

### Fragment GPU-Resident API

Access GPU-resident functionality through the `Fragment::gpu_resident()` accessor.

```cpp
auto fragment = make_fragment<MyGpuResidentFragment>();

// Configure before launching (optional)
fragment->gpu_resident().timeout_ms(5000);  // Set timeout to 5 seconds
fragment->gpu_resident().data_not_ready_sleep_interval_us(250);  // Sleep 250us when waiting for data

// Launch the fragment
auto future = fragment->run_async();

// Control during graph execution
fragment->gpu_resident().data_ready();  // Signal data is ready
fragment->gpu_resident().tear_down();  // Tear down GPU-resident fragment (terminates the GPU-resident CUDA graph)
```

#### Control and Status Checking Methods from Host (CPU) Process

**`tear_down()`**

```cpp
void tear_down();
```

Sends a tear down signal to stop GPU-resident graph. It can take some time to tear down the GPU-resident CUDA graph. Check with `is_launched()` function to know if the graph has been torn down. **Note:** If `timeout_ms` is set to non-zero value, then the application will automatically be torn down after the timeout duration.

**`is_launched()`**

```cpp
bool is_launched();
```

Returns `true` if the GPU-resident CUDA graph has been launched and is running, `false` otherwise. Use this to wait for initialization to complete before sending data. If the graph has been torn down, this function will return `false` in that case.

**`data_ready()`**

```cpp
void data_ready();
```

Signals that input data is ready for processing. Call this after writing data to the application's input device memory. This could be the device memory allocated to the source operator of an application pipeline.

**`result_ready()`**

```cpp
bool result_ready();
```

Returns `true` if the current iteration's results are ready for consumption, `false` otherwise. Poll this after calling `data_ready()` to know when to read output data.

:::{note}
The `data_ready`, `result_ready` and other such CPU-side control methods can affect the deterministic performance of the GPU-resident CUDA graph and should be used with caution.
:::

#### Configuration Methods

**`timeout_ms(timeout)`**

```cpp
void timeout_ms(unsigned long long timeout_ms);
```

Sets the timeout for GPU-resident graph in milliseconds. GPU-resident graph will be torn down after the timeout duration. If nothing is set or set to 0, then the graph will run indefinitely until `tear_down()` is called.

**`data_not_ready_sleep_interval_us(sleep_interval_us)`**

```cpp
void data_not_ready_sleep_interval_us(unsigned int sleep_interval_us = 500);
```

Sets the sleep interval on the GPU device when data is not ready. The GPU-resident graph loop will sleep for this duration (in microseconds) before checking the data ready signal again. This helps reduce unnecessary GPU polling and power consumption when waiting for new data. Default is 500 microseconds. Lower values provide faster response to data ready signals but increase GPU and power usage, while higher values reduce GPU usage but may introduce increased latency.

**Important:** This setting must be configured before calling `run_async()` as it cannot be changed after the CUDA graph has been launched.

**`sync_with_host(enable)`**

```cpp
void sync_with_host(bool enable = true);
```

Enables or disables a system-wide memory fence at the end of each GPU-resident iteration. When enabled, the GPU issues a system-wide fence (`__threadfence_system()`) after the workload completes and before signaling result-ready. This ensures that all device memory writes are globally visible to the host before the result-ready flag is observed.

This option is intended for scenarios where the host controls the GPU-resident graph loop and reads back results between iterations (e.g., via `cudaMemcpy`). It is recommended for debugging, development, and testing purposes.

:::{note}
Enabling `sync_with_host` adds latency to each iteration and is not recommended for performance-critical workloads. When the GPU-resident pipeline is driven entirely by GPU-side data ready handlers (no host-side readback between iterations), this option is **not recommended and not required**.
:::

**Important:** This setting must be configured before calling `run_async()` as it cannot be changed after the CUDA graph has been launched.

**`register_data_ready_handler(fragment)`**

```cpp
void register_data_ready_handler(std::shared_ptr<Fragment> data_ready_handler_fragment);
```

Registers a data ready handler fragment that executes at the beginning of each iteration to determine if data is ready.

## How GPU-Resident Graphs Work

### Initialization Phase

A GPU-resident fragment is initialized in the following steps:

1. **Graph Topology Verification**: The framework verifies that the operator graph is a supported topology: a DAG with exactly one source operator.

2. **Device Memory Setup**: For each connection between operators:
   - If both ports specify a memory block size, the executor allocates a shared device buffer
   - If a port specifies a device pointer, the executor uses that pointer directly (no allocation)
   - Memory addresses are mapped to operator ports
   - Connected operator ports map to same device memory addresses

3. **CUDA Graph Capture**:
   - A CUDA stream is created for graph capture
   - Each operator's `compute()` method is executed during capture
   - CUDA operations (kernel launches, memcpy, etc.) are recorded into a graph
   - For data ready handlers, a separate graph is captured

4. **GPU-Resident Graph Construction**:
   - A conditional node checks the data ready signal
   - If data is ready, the main workload graph executes
   - A result ready signal is set upon completion
   - A tear down check determines if graph execution should continue

### Graph Execution Phase

In the execution phase, there is no CPU-driven graph execution unless explicitly requested by the host CPU process. During asynchronous execution:

1. **Graph Launch**: The GPU-resident CUDA graph is launched on a dedicated stream
2. **Iteration Loop** (on the GPU):
   - The graph polls the data ready signal and tear down signal
   - If data is not ready, the GPU sleeps for the configured interval (default 500 microseconds) before checking again
   - When data ready signal is set, executes all operators
   - Sets result ready signal when processing is complete
   - When tear down signal is set, the graph is torn down

### Memory Management

- **Pre-allocated Buffers**: All inter-operator buffers are pre-allocated during initialization, either by the executor (memory block size) or by the operator itself (device pointer)
- **Zero-Copy Access**: Operators access device memory directly via `device_memory()`
- **Persistent Addresses**: Memory addresses remain constant throughout execution
- **No Dynamic Allocation**: Our graph execution mandates static memory layouts
- **Operator-managed Memory**: Operators can supply their own device pointers via `device_input()`/`device_output()` with a `CUdeviceptr` or `void*`, giving full control over allocation strategy while still participating in the GPU-resident data flow

### Topological Ordering

Operators are executed in topological order based on the dataflow graph. The framework:

- Determines the correct execution sequence
- Ensures dependencies are satisfied before operator execution
- Captures operators in the correct order into the CUDA graph

## Current Limitations

- **Single-Source DAG Only**: GPU-resident execution supports DAGs with exactly one source operator; multiple disconnected roots are not supported
- **Single Upstream Per Input Port**: Each GPU-resident input port may be connected to only one upstream output port
- **Static Memory**: All memory must be pre-allocated; dynamic memory allocation not supported
- **Single Device**: Multi-GPU execution not yet supported. `CUDA Device 0` is used for GPU-resident graph execution by default.
- **No Scheduler**: Cannot use standard Holoscan schedulers to connect with GPU-resident graph execution

Some of the topology limitations are temporary and will be relaxed in future releases.

## Examples

Fully working examples demonstrating GPU-resident graph execution are available at:

**`public/examples/gpu_resident_example/gpu_resident_example.cpp`**
**`public/examples/gpu_resident_input/gpu_resident_input.cpp`**
**`public/examples/gpu_resident_multi_io/gpu_resident_multi_io.cpp`** — operators with multiple input/output ports

## Best Practices

### Operator Design

- **Keep compute() lightweight**: Only launch CUDA kernels; avoid CPU work as CPU calls won't be repeated in the graph execution phase.
- **Operator granularity**: GPU-resident graph execution captures the workflow into a CUDA Graph and replays it on the GPU. This typically has much lower per-operator scheduling overhead (~0.5-2 microseconds kernel transition latency) than CPU-driven graph execution, making finer-grained operator decomposition more practical. See {ref}`performance_considerations` for comparison with CPU-based scheduling.
- **Use provided streams**: Always use `cuda_stream()` for kernel launches in the main workload and `data_ready_handler_cuda_stream()` for kernel launches in the data ready handler.
- **Pre-calculate sizes**: Determine buffer sizes at setup time, not runtime
- **Avoid CPU-driven GPU Controls**: Avoid explicit CUDA synchronization and other CPU-driven GPU controls to get the most deterministic performance.

### Performance Tuning

- **Tune sleep interval**: Adjust `data_not_ready_sleep_interval_us()` based on your application needs:
  - **Low latency applications** (e.g., high-speed sensors): Use shorter intervals (e.g., 100-250 microseconds) for faster response to new data
  - **Power-constrained applications**: Use longer intervals (e.g., 500-1000 microseconds) to reduce GPU polling overhead
  - **Default (500 microseconds)**: Provides a balanced trade-off between latency and GPU utilization
  - **Compute Requirements**: Depending on the rest of the application pipeline's compute requirements, the sleep interval must be adjusted to ensure expected Quality-of-service and power usage trade-off.
- **Monitor GPU utilization**: Use NVIDIA tools (nvidia-smi, Nsight Systems) to verify GPU usage patterns match expectations

### Error Handling

- **Check is_launched()**: Ensure graph is ready before sending data
- **Handle timeouts**: Implement timeout logic when waiting for results
- **Clean up properly**: Always call `tear_down()` before application exit

## Troubleshooting

### Graph Not Launching

If `is_launched()` never returns `true`:

- Check for initialization errors in operator `compute()` methods
- Verify all operators inherit from `GPUResidentOperator`

### Result Never Ready

If `result_ready()` never returns `true`:

- Verify `data_ready()` was called after writing input data
- Check for kernel errors
- Ensure operators are launching kernels correctly

### Memory Access Errors

If encountering CUDA memory errors:

- Verify buffer sizes match between port declaration and usage
- Check that device memory addresses are not null before use
- Ensure no out-of-bounds access in kernels

### Performance Issues

If execution is slower than expected:

- Profile with NVIDIA Nsight Systems to identify bottlenecks
- Check for unnecessary CUDA synchronization
- Verify kernels are launched with optimal grid/block dimensions
