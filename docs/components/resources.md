# Resources

Resource classes represent resources such as a allocators, clocks, transmitters, or receivers that may be used as a parameter for operators or schedulers. The resource classes that are likely to be directly used by application authors are documented here.

:::{note}
There are a number of other resources classes used internally which are not documented here, but appear in the API Documentation ({ref}`C++ <api/holoscan_cpp_api:resources>`/{py:mod}`Python <holoscan.resources>`).
:::

## Allocator

### UnboundedAllocator

An allocator that uses dynamic host or device memory allocation without an upper bound. This allocator does not take any user-specified parameters. This memory pool is easy to use and is recommended for initial prototyping. Once an application is working, switching to a `BlockMemoryPool` instead may help provide additional performance.

### BlockMemoryPool

This is a memory pool which provides a user-specified number of equally sized blocks of memory. Using this memory pool provides a way to allocate memory blocks once and reuse the blocks on each subsequent call to an Operator's `compute` method. This saves overhead relative to allocating memory again each time `compute` is called. For the built-in operators which accept a memory pool parameer, there is a section in it's API docstrings titled "Device Memory Requirements" which provides guidance on the `num_blocks` and `block_size` needed for use with this memory pool.

- The `storage_type` parameter can be set to determine the memory storage type used by the operator. This can be 0 for page-locked host memory (allocated with `cudaMallocHost`), 1 for device memory (allocated with `cudaMalloc`) or 2 for system memory (allocated with C++ `new`).
- The `block_size` parameter determines the size of a single block in the memory pool in bytes. Any allocation requests made of this allocator must fit into this block size.
- The `num_blocks` parameter controls the total number of blocks that are allocated in the memory pool.
- The `dev_id` parameter is an optional parameter that can be used to specify the CUDA ID of the device on which the memory pool will be created.

### RMMAllocator

This allocator provides a pair of memory pools (one is a CUDA device memory pool and the other corresponds to pinned host memory). The underlying implementation is based on the [RAPIDS memory manager](https://github.com/rapidsai/rmm) (RMM) and uses a pair of `rmm::mr::pool_memory_resource` resource types (The device memory pool is a `rmm::mr::cuda_memory_resource` and the host pool is a `rmm::mr::pinned_memory_resource`) . Unlike `BlockMemoryPool`, this allocator can be used with operators like `VideoStreamReplayerOp` that require an allocator capable of allocating both host and device memory. Rather than fixed block sizes, it uses just an initial memory size to allocate and a maximum size that the pool can expand to.

- The `device_memory_initial_size` parameter specifies the initial size of the device (GPU) memory pool. This is an optional parameter that defaults to 8 MB on aarch64 and 16 MB on x86_64. See note below on the format used to specify the value.
- The `device_memory_max_size` parameter specifies the maximum size of the device (GPU) memory pool in MiB. This is an optional parameter that defaults to twice the value of `device_memory_initial_size`. See note below on the format used to specify the value.
- The `host_memory_initial_size` parameter specifies the initial size of the device (GPU) memory pool in MiB. This is an optional parameter that defaults to 8 MB on aarch64 and 16 MB on x86_64. See note below on the format used to specify the value.
- The `host_memory_max_size` parameter  specifies the maximum size of the device (GPU) memory pool in MiB. This is an optional parameter that defaults to twice the value of `host_memory_initial_size`. See note below on the format used to specify the value.
- The `dev_id` parameter is an optional parameter that can be used to specify the GPU device ID (as an integer) on which the memory pool will be created.

:::{note}
The values for the memory parameters, such as `device_memory_initial_size` must be specified in the form of a string containing a non-negative integer value followed by a suffix representing the units. Supported units are B, KB, MB, GB and TB where the values are powers of 1024 bytes
(e.g. MB = 1024 * 1024 bytes). Examples of valid units are "512MB", "256 KB", "1 GB". If a floating point number is specified that decimal portion will be truncated (i.e. the value is rounded down to the nearest integer).
:::

### CudaStreamPool

This allocator creates a pool of CUDA streams.

- The `stream_flags` parameter specifies the flags sent to [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) when creating the streams in the pool.
- The `stream_priority` parameter specifies the priority sent to [cudaStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) when creating the streams in the pool. Lower values have a higher priority.
- The `reserved_size` parameter specifies the initial number of CUDA streams created in the pool upon initialization.
- The `max_size` parameter is an optional parameter that can be used to specify a maximum number of CUDA streams that can be present in the pool. The default value of 0 means that the size of the pool is unlimited.
- The `dev_id` parameter is an optional parameter that can be used to specify the CUDA ID of the device on which the stream pool will be created.

## Clock

Clock classes can be provided via a `clock` parameter to the `Scheduler` classes to manage the flow of time.

All clock classes provide a common set of methods that can be used at runtime in user applications.

- The {cpp:func}`time()<Clock::time>` method returns the current time in seconds (floating point).
- The {cpp:func}`timestamp()<Clock::timestamp>` method returns the current time as an integer number of nanoseconds.
- The {cpp:func}`sleep_for()<Clock::sleep_for>` method sleeps for a specified duration in ns. An overloaded version of this method allows specifying the duration using a `std::chrono::duration<Rep, Period>` from the C++ API or a [datetime.timedelta](https://docs.python.org/3/library/datetime.html#datetime.timedelta) from the Python API.
- The {cpp:func}`sleep_until()<Clock::sleep_until>` method sleeps until a specified target time in ns.

### Realtime Clock

The `RealtimeClock` respects the true duration of conditions such as `PeriodicCondition`. It is the default clock type and the one that would likely be used in user applications.

In addition to the general clock methods documented above:

- This class has a {cpp:func}`set_time_scale()<Clock::set_time_scale>` method which can be used to dynamically change the time scale used by the clock.
- The parameter `initial_time_offset` can be used to set an initial offset in the time at initialization.
- The parameter `initial_time_scale` can be used to modify the scale of time. For instance, a scale of 2.0 would cause time to run twice as fast.
- The parameter `use_time_since_epoch` makes times relative to the [POSIX epoch](https://en.wikipedia.org/wiki/Epoch_(computing)) (`initial_time_offset` becomes an offset from epoch).

### Manual Clock

The `ManualClock` compresses time intervals (e.g., `PeriodicCondition` proceeds immediately rather than waiting for the specified period). It is provided mainly for use during testing/development.

The parameter `initial_timestamp` controls the initial timestamp on the clock in ns.

## Transmitter (advanced)

Typically users don't need to explicitly assign transmitter or receiver classes to the IOSpec ports of Holoscan SDK operators. For connections between operators a `DoubleBufferTransmitter` will automatically be used, while for connections between fragments in a distributed application, a `UcxTransmitter` will be used. When data frame flow tracking is enabled any `DoubleBufferTransmitter` will be replaced by an `AnnotatedDoubleBufferTransmitter` which also records the timestamps needed for that feature.

### DoubleBufferTransmitter

This is the transmitter class used by output ports of operators within a fragment.

### UcxTransmitter

This is the transmitter class used by output ports of operators that connect fragments in a distributed applications. It takes care of sending UCX active messages and serializing their contents.

## Receiver (advanced)

Typically users don't need to explicitly assign transmitter or receiver classes to the IOSpec ports of Holoscan SDK operators. For connections between operators, a `DoubleBufferReceiver` will be used, while for connections between fragments in a distributed application, the `UcxReceiver` will be used. When data frame flow tracking is enabled, any `DoubleBufferReceiver` will be replaced by an `AnnotatedDoubleBufferReceiver` which also records the timestamps needed for that feature.

### DoubleBufferReceiver

This is the receiver class used by input ports of operators within a fragment.

### UcxReceiver

This is the receiver class used by input ports of operators that connect fragments in a distributed applications. It takes care of receiving UCX active messages and deserializing their contents.

## Condition Combiners

The default behavior for Holoscan's schedulers is AND combination of any conditions on an operator when determining if it should execute. It is possible to assign conditions to a different `ConditionCombiner` class which will combine the conditions using the rules of this combiner and omit those conditions from consideration by the default AND combiner.

### OrConditionCombiner

The `OrConditionCombiner` applies an OR condition to the conditions passed to its "terms" argument. For example, assume an operator had a `CountCondition` as well as a `MessageAvailableCondition` for port "in1" and a `MessageAvailableCondition` for port "in2". If an `OrConditionCombiner` was added to the operator with the two message-available conditions passed to its "terms" argument, then the scheduling logic for the operator would be:

- (CountCondition satisfied) AND ((message available on port "in1") OR (message available on port "in2"))

In other words, any condition like the `CountCondition` in this example that is not otherwise assigned to a custom `ConditionCombiner` will use the default AND combiner.

Holoscan provides a `IOSpec::or_combine_port_conditions` method which can be called from `Operator::setup` to enable OR combination of conditions that apply to specific input (or output) ports.

## System Resources

The components in this "system resources" section are related to system resources such as CPU Threads that can be used by operators. 

### ThreadPool

This resource represents a thread pool that can be used to pin operators to run using specific CPU threads. This functionality is not supported by the `GreedyScheduler` because it is single-threaded, but it is supported by both the `EventBasedScheduler` and `MultiThreadScheduler`. Unlike other resource types, a ThreadPool should **not** be created via `make_resource` ({cpp:func}`C++ <holoscan::Fragment::make_resource>`/{py:func}`Python <holoscan.core.Fragment.make_resource>`), but should instead use the dedicated `make_thread_pool` ({cpp:func}`C++ <holoscan::Fragment::make_resource>`/{py:func}`Python <holoscan.core.Fragment.make_resource>`) method. This dedicated method is necessary as the thread pool requires some additional initialization logic that is not required by the other resource types. See the section on {ref}`configuring thread pools <configuring-app-thread-pools>` in the user guide for usage.

- The parameter `initial_size` indicates the number of threads to initialize the thread pool with.
