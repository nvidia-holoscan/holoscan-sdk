(holoscan-cuda-stream-handling)=

# CUDA Stream Handling in Holoscan Applications

CUDA provides the concept of streams to allow for asynchronous concurrent execution on the GPU. Each stream is a sequence of commands that execute in order, but work launched on separate streams can potentially operate concurrently. Examples are running multiple kernels in separate streams or overlapping data transfers and kernel execution. See the [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) section of the CUDA programming guide.

The `CudaStreamPool` class ({cpp:class}`C++ <holoscan::CudaStreamPool>`/{py:class}`Python <holoscan.resources.CudaStreamPool>`) is a resource that provides a mechanism for allocating CUDA streams from a pool of streams whose lifetime is managed by Holoscan. As of Holoscan v2.8, new APIs are provided to make use of dedicated CUDA streams easier for application authors. These APIs are intended as a replacement of the legacy `CudaStreamHandler` utility described in the note below.

:::{note}
There is a legacy `CudaStreamHandler` utility class (provided via `#include "holoscan/utils/cuda_stream_handler.hpp"`) that made it possible to write a C++ operator that could make use of a `CudaStreamPool`. This class had some limitations:
- It required receiving messages as type `holoscan::gxf::Entity`.
- It required using `nvidia::gxf::Entity` and `nvidia::gxf::Handle` methods from the underlying GXF library.
- It was not available for native Python operators.

This existing utility is still provided for backwards compatibility and operators using it can continue to interoperate with those using the new APIs. However, we encourage operator authors to migrate to using the new APIs going forward.
:::

(configuring-a-cuda-stream-pool)=

## Configuring a CUDA stream pool for an operator's internal use

:::{note}
Starting from Holoscan v2.9, a default `CudaStreamPool` is added to all operators if the user did not otherwise provide one. This means that in most cases, it will not be necessary for the user to explicitly add a strea pool. The default stream pool has unbounded size, no flags set and a priority value of 0. In cases when the user wants to allocate streams with different flags or priority, the section below can be followed to add a customized stream pool to the operator.

The only case when a default stream pool would not be added is if the application (fragment) is running on a node without any CUDA-capable devices. In that case, since use of CUDA is not possible a default stream pool would not be added.
:::

To enable an operator to allocate a CUDA stream, the user can pass a `CudaStreamPool` as in the following examples. The general pattern used for stream handling in Holoscan SDK is to have each Operator that wants to use a non-default stream have a `CudaStreamPool` assigned. That operator will then reserve a dedicated stream from the stream pool for use by any kernels launched by it. Multiple operators are allowed to use the same stream pool, with "max_size" of the shared pool equal to at least the number of Operators that are sharing it.

Note that the `CudaStreamPool` will manage the lifetimes of any CUDA streams used by the SDK. The user does not need typically need to explicitly call any CUDA APIs to create or destroy streams. Note that all streams from a single `CudaStreamPool` are on a single device (with CUDA id as passed to the "dev_id" argument). If the workflow involves operators that run on separate CUDA devices, those operators must use separate stream pools configured for the corresponding device.

`````{tab-set}
````{tab-item} C++
```cpp
// The code below would appear within `Application::compose` (or `Fragment::compose`)

// Create a stream pool with a 5 streams capacity (5 operators could share the same pool)
const auto cuda_stream_pool = make_resource<CudaStreamPool>("stream_pool",
                                                            Arg("dev_id", 0),
                                                            Arg("stream_flags", 0u),
                                                            Arg("stream_priority", 0),
                                                            Arg("reserved_size", 1u),
                                                            Arg("max_size", 5u));

auto my_op = make_operator<MyOperator>("my_op", cuda_stream_pool, arg_list);

// Alternatively, the argument can be added via `add_arg` after operator construction
// auto my_op = make_operator<MyOperator>("my_op", arg_list);
// my_op->add_arg(cuda_stream_pool);
```

Note that the the legacy `CudaStreamHandler` utility did not support passing the stream pool in
this way, but instead required that the user explicitly add a parameter to the operator's private
data members.

```cpp
private:
  // The legacy CudaStreamHandler required a "cuda_stream_pool" parameter.
  // The spec.param call in the Operator's `setup` method would use the name "cuda_stream_pool"
  // for it
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_{};
```
For backwards compatibility with prior releases, the built-in operators that were previously using the `CudaStreamHandler` utility class still offer this explicitly defined "cuda_stream_pool" parameter. It is not necessary for the user to add it to their own operators unless they prefer to explicitly use an `Arg` named "cuda_stream_pool" parameter when initializing the operator.

```cpp
auto visualizer = make_operator<HolovizOp>(
    "visualizer",
    from_config("holoviz"),
    Arg("cuda_stream_pool", make_resource<CudaStreamPool>(0, 0, 0, 1, 5)));
```
````
````{tab-item} Python
```python
# The code below would appear within `Application.compose` (or `Fragment.compose`)

# Create a stream pool with a 5 streams capacity (5 operators could share the same pool)
cuda_stream_pool = CudaStreamPool(
    self,
    name="stream_pool",
    dev_id=0,
    stream_flags=0,
    stream_priority=0,
    reserved_size=1,
    max_size=5,
)
my_op = MyOperator(self, cuda_stream_pool, name="my_op", **my_kwargs)

# Alternatively, the argument can be added via `add_arg` after operator construction
# auto my_op = MyOperator(self, name="my_op", **my_kwargs)
# my_op.add_arg(cuda_stream_pool)
```

The above is the recommended way for user-defined operators to add a `CudaStreamPool`. For purposes of backwards compatibility, the built-in operators of the SDK that already had a keyword-based `cuda_stream_pool` parameter continue to also allow passing the stream pool as in the following example:

```python
visualizer = HolovizOp(
  self,
  name="holoviz",
  cuda_stream_pool=CudaStreamPool(self, 0, 0, 0, 1, 5),
  **self.kwargs("holoviz"))
```
````
`````

## Sending stream information between operators

Because CUDA kernels are launched asynchronously by the host (CPU), it is possible for the `compute` method to return before the underlying computation on the GPU is complete (see a related warning regarding benchmarking in this scenario below). In this scenario, information about the stream that was used must be sent along with the data so that a downstream operator can handle any stream synchronization that is needed. For example, if an upstream kernel emitted a `Tensor` object immediately after launching a CUDA kernel, the downstream operator needs to be sure the kernel has completed before accessing the tensor's data.

The `CudaStreamPool` ({cpp:class}`C++ <holoscan::CudaStreamPool>`/{py:class}`Python <holoscan.resources.CudaStreamPool>`) class allocates `nvidia::gxf::CudaStream` objects behind the scenes. These stream objects exist as components in the entity-component system of the underlying GXF library. GXF defines an `nvidia::gxf::CudaStreamId` struct which contains the "component ID" corresponding to the stream. It is this `CudaStreamId` struct that actually gets transmitted along with each message emitted from an output port. The Holoscan application author is not expected to need to interact with either the `CudaStream` or `CudaStreamId` classes directly, but instead use the standard CUDA Runtime API `cudaStream_t` type that is returned by Holoscan's public stream handling methods described in the sections below. Methods like `receive_cuda_stream` ({cpp:func}`C++ <holoscan::InputContext::receive_cuda_stream>`/{py:func}`Python <holoscan.core.InputContext.receive_cuda_stream>`) or `allocate_cuda_stream` ({cpp:func}`C++ <holoscan::ExecutionContext::allocate_cuda_stream>`/{py:func}`Python <holoscan.core.ExecutionContext.allocate_cuda_stream>`) return a `cudaStream_t` that corresponds to an underlying `CudaStream` object. Similarly methods like `set_cuda_stream` ({cpp:func}`C++ <holoscan::OutputContext::set_cuda_stream>`/{py:func}`Python <holoscan.core.OutputContext.set_cuda_stream>`) and `device_from_stream` ({cpp:func}`C++ <holoscan::ExecutionContext::device_from_stream>`/{py:func}`Python <holoscan.core.ExecutionContext.device_from_stream>`) take a `cudaStream_t` as input, but only accept a `cudaStream_t` that corresponds to underlying `CudaStream` objects whose lifetime can be managed by the SDK.

The SDK provides several publicly accessible methods for working with streams that can be called from the `compute` method of an operator. These are described in detail below.

## Simplified CUDA streams handling via `receive_cuda_stream`

In many cases, users will only need to use the `receive_cuda_stream` ({cpp:func}`C++ <holoscan::InputContext::receive_cuda_stream>`/{py:func}`Python <holoscan.core.InputContext.receive_cuda_stream>`) method provided by `InputContext` in their `compute` method. This is because the method automatically manages multiple aspects of stream handling:
1. It automatically synchronizes any streams found on the named input port to the operator's internal CUDA stream
  - The first time `compute` is called, an operator's internal CUDA stream would be allocated from the assigned `CudaStreamPool`. The same stream is then reused on all subsequent `compute` calls.
  - There is a boolean flag which can also force synchronization to the default stream (false by default)
2. It returns the `cudaStream_t` corresponding to the operator's internal stream.
  - The user should use this returned stream for any kernels or memory copy operations to be run on a non-default stream.
3. It sets the CUDA device corresponding to the stream returned in step 2 as the active CUDA device
4. This method automatically configures all output ports to emit the stream returned by step 2 as a component in each message sent.
  - This ID will allow downstream operators to know what stream was used for any data received in this message.

:::{attention}
Please insure that, for a given input port, `receive` is always called **before** `receive_cuda_stream`. This is necessary because the `receive` call is what actually receives the messages and allows the operator to know about any stream IDs found in messages on the input port. That `receive` method only records information internally about any streams that were found. The subsequent `receive_cuda_stream` call is needed to perform synchronization and return the `cudaStream_t` to which any input streams were synchronized.
:::

Here is an example of the typical usage of this method from the built-in `BayerDemosaicOp`

`````{tab-set}
````{tab-item} C++
```cpp
// The code below would appear within `Operator::compute`

// Process input message
auto maybe_message = op_input.receive<gxf::Entity>("receiver");
if (!maybe_message || maybe_message.value().is_null()) {
  throw std::runtime_error("No message available");
}
auto in_message = maybe_message.value();

// Get the CUDA stream from the input message if present, otherwise generate one.
// This stream will also be transmitted on the "tensor" output port.
cudaStream_t cuda_stream = op_input.receive_cuda_stream("receiver", // input port name
                                                        true,       // allocate
                                                        false);     // sync_to_default

// assign the CUDA stream to the NPP stream context
npp_stream_ctx_.hStream = cuda_stream;
```
````
````{tab-item} Python

Note that `BayerDemosaicOp` is implemented in C++ using code shown in the C++ tab, but this shows how the equivalent code would look in the Python API.

```python
# The code below would appear within `Operator.compute`

# Process input message
in_message = op_input.receive("receiver")
if in_message is None:
  raise RuntimeError("No message available")

# Get the CUDA stream from the input message if present, otherwise generate one.
# This stream will also be transmitted on the "tensor" output port.
cuda_stream_ptr = op_input.receive_cuda_stream("receiver", allocate=True, sync_to_default=False)

# can then use cuda_stream_ptr to create a `cupy.cuda.ExternalStream` context, for example
```
````
`````

It can be seen that the call to `receive` occurs prior to the call to `receive_cuda_stream` for the "receiver" input port as required. Also note that unlike for the legacy `CudaStreamHandler` utility class, it is not required to use `gxf::Entity` in the "receive" call. That type is use by some built-in operators like `BayerDemosaicOp` as a way to support both the `nvidia::gxf::VideoBuffer` type and the usual `Tensor` type as inputs. If only `Tensor` was supported we could have used `receive<std::shared_ptr<Tensor>>` or `receive<TensorMap>` instead.

The second boolean argument to `receive_cuda_stream` defaults to true and indicates that the operator should allocate its own internal stream. This could be set to false to not allow the operator to allocate its own internal stream from the stream pool. See the note below on the details of how `receive_cuda_stream` behaves in that case.

There is also an optional third argument to `receive_cuda_stream` which is a boolean specifying whether synchronization of the input streams (and internal stream) to CUDA's default stream should also be performed. This option is `false` by default.

The above description of `receive_cuda_stream` is accurate when a `CudaStreamPool` has been passed to the operator in one of the ways {ref}`described above<configuring-a-cuda-stream-pool>`. See the note below for additional detail on how this method operates if the operator is unable to allocate an internal stream because a `CudaStreamPool` was unavailable.

#### Avoiding additional synchronization from Python's CUDA Array Interface

Python applications converting between Holoscan's Tensor and 3rd party tensor objects often use the {ref}`CUDA Array Interface<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`. This interface by default performs its own explicit synchronization (described {ref}`here<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#synchronization-in-numba>`). This may be unnecessary when using `receive_cuda_stream` which already synchronizes streams found on the input with the operator's internal stream. The environment variable `CUPY_CUDA_ARARAY_INTERFACE_SYNC` can be set to 0 to disable an additional synchronization by CuPy when creating a CUDA array from a holoscan Tensor via the array interface. Similarly, `HOLOSCAN_CUDA_ARRAY_INTERFACE_SYNC` can be set to 0 to disable synchronization by the array interface on the Holoscan side when creating a Holoscan tensor from a 3rd party tensor.

### Using `receive_cuda_stream` without a stream pool available

This section describes the behavior of `receive_cuda_stream` in the case where no streams are available in the operator's `CudaStreamPool` (or the `allocate` argument of `receive_cuda_stream` was set to false). In this case, `receive_cuda_stream` will not be able to allocate a dedicated internal stream for the operator's own use. Instead, the `cudaStream_t` corresponding to the **first** stream found on the named input port will be returned and any additional streams on that input port would be synchronized to it. If a subsequent `receive_cuda_stream` call was made for another input port, any streams found on that second port are synchronized to the `cudaStream_t` that was returned by the first `receive_cuda_stream` call and the stream returned is that same `cudaStream_t`. In other words, the first stream found on the initial call to `receive_cuda_stream` will be repurposed as the operator's internal stream to which any other input streams are synchronized. This same stream will also be the one automatically emitted on the output ports.

In the case that there is no `CudaStreamPool` and there is no stream found for the input port (or by any prior `receive_cuda_stream` calls for another port), then `receive_cuda_stream` will return the default stream (`cudaStreamDefault`). No stream would be emitted on the output ports in this case.

## InputContext: additional stream handling methods

The `receive_cuda_streams` ({cpp:func}`C++ <holoscan::InputContext::receive_cuda_streams>`/{py:func}`Python <holoscan.core.InputContext.receive_cuda_streams>`) method is designed for advanced use cases where the application author needs to manually manage all aspects of stream synchronization, allocation, and emission of CUDA streams. Unlike `receive_cuda_stream`, this method does not perform synchronization, does not automatically allocate an internal CUDA stream, does not update the active CUDA device, and does not configure any stream to be emitted on output ports. Instead, it simply returns a `std::vector<std::optional<cudaStream_t>>`, which is a vector of size equal to the number of messages on the input port. Each value in the vector corresponds to the `cudaStream_t` specified by the message (or `std::nullopt` if no stream ID is found).

Note that as for `receive_cuda_stream`, it is important that any `receive_cuda_streams` call for a port is **after** the corresponding `receive` call for that same port. An example is given below


`````{tab-set}
````{tab-item} C++
```cpp
// The code below would appear within `Operator::compute`

// Process a "receivers" port (e.g. one having IOSpec::kAnySize) that may
// have an arbitrary number of connections, each of which may have sent a
// TensorMap.
auto maybe_tensors = op_input.receive<std::vector<Tensor>>("receivers");
if (!maybe_tensors) { throw std::runtime_error("No message available"); }
auto tensormaps = maybe_tensors.value();

// Get a length two vector of std::option<CudaStream_t> containing any streams
// found by the any of the above receive calls.
auto cuda_streams = op_input.receive_cuda_streams("receivers");
```
````
````{tab-item} Python
```python
# The code below would appear within `Operator.compute`

auto tensors = op_input.receive("receivers")
if tensors is None:
    raise RuntimeError("No message available on 'receivers' input")

cuda_stream_ptrs = op_input.receive_cuda_streams("receivers")
```
````
`````

(execution-context-stream-methods)=

## ExecutionContext: additional stream handling methods

The `allocate_cuda_stream` ({cpp:func}`C++ <holoscan::ExecutionContext::allocate_cuda_stream>`/{py:func}`Python <holoscan.core.ExecutionContext.allocate_cuda_stream>`) method can be used to allocate additional CUDA streams from the Operator's `CudaStreamPool`. An `unexpected` (or `None` in Python) will be returned if there is no stream pool associated with the operator or if all streams in the stream pool were already in used. A user-provided stream name is given for the allocation so that for a given name, a new stream is only allocated the first time the method is called. The same stream is then reused on on any subsequent calls using the same name. Streams allocated in this way are not automatically emitted on the output ports. If this is needed, the user must specifically emit the stream IDs by calling `set_cuda_stream` for the output port **prior** to the call to `emit` for that port.

`````{tab-set}
````{tab-item} C++
```cpp
// The code below would appear within `Operator::compute`

cudaStream_t my_stream = context.allocate_cuda_stream("my_stream");

// some custom code using the CUDA stream here

// emit the allocated stream on the "out" port
op_output.set_cuda_stream(my_stream, "out");
```
````
````{tab-item} Python
```python
# The code below would appear within `Operator.compute`

my_stream_ptr = context.allocate_cuda_stream("my_stream")

# some custom code using the CUDA stream here

# emit the allocated stream on the "out" port
op_output.set_cuda_stream(my_stream, "out")
```
````
`````

The `synchronize_streams` ({cpp:func}`C++ <holoscan::ExecutionContext::synchronize_streams>`/{py:func}`Python <holoscan.core.ExecutionContext.synchronize_streams>`) method takes a vector of (optional) `cudaStream_t` values and synchronizes all of these streams to the specified `target_cuda_stream`. It is okay for the target stream to also appear in the vector of streams to synchronize (synchronization will be skipped for any element in the vector that is the same as the target stream). If the application author is using the `receive_cuda_stream` API described above, that will typically take care of any needed synchronization and this method does not need to be called. It is provided for manual stream handling use cases.

The `device_from_stream` ({cpp:func}`C++ <holoscan::ExecutionContext::device_from_stream>`/{py:func}`Python <holoscan.core.ExecutionContext.device_from_stream>`) method takes a `cudaStream_t` value and returns the integer CUDA device id corresponding to that stream. This method only supports querying the device in this way for streams managed by Holoscan SDK (i.e. it only supports streams that were returned by `receive_cuda_stream`, `receive_cuda_streams` or `allocate_cuda_stream`).

## OutputContext: stream handling methods

The `set_cuda_stream` ({cpp:func}`C++ <holoscan::OutputContext::set_cuda_stream>`/{py:func}`Python <holoscan.core.OutputContext.set_cuda_stream>`) method is used to indicate that the stream ID corresponding to a specific `cudaStream_t` should be emitted on the specified CUDA output port. This typically does not need to be explicitly called when using `receive_cuda_stream` as that method would have already configured the stream ID returned to be output on all ports. It is needed for cases where the user has allocated some additional stream via `allocate_cuda_stream` or is doing manual stream handling with `receive_cuda_streams`. An example of usage was given in the {ref}`section above <execution-context-stream-methods>` on `allocate_cuda_stream`.

## Using CudaStreamCondition to require stream work to complete before an operator executes

It is mentioned above that `receive_cuda_stream` automatically handles synchronization of streams found on an input port. If work on the stream was not already complete and the `compute` method is going to perform an operation which requires synchronization such as device->host memory copy, then some time will be spent waiting for work launched on an input stream by an upstream operator to complete. It may be beneficial to explicitly specify that work on the stream found on a given input port must be complete **before** the scheduler would execute the operator (call its `compute` method).

To require work on an input stream to complete before an operator is ready to schedule, a `CudaStreamCondition` ({cpp:class}`C++ <holoscan::CudaStreamCondition>`/{py:class}`Python <holoscan.conditions.CudaStreamCondition>`) can be added to the operator.
When a message is sent to the port to which a `CudaStreamCondition` has been assigned, this condition sets an internal host callback function on the CUDA stream found on this input port. The callback function will set the operator's status to READY once other work on the stream has completed. This will then allow the scheduler to execute the operator.

One limitation of `CudaStreamCondition` is that it only looks for a stream on the first message in the input port's queue. It does not currently support handling ports with multiple different input stream components within the same message (entity) or across multiple messages in the queue. The behavior of `CudaStreamCondition` is sufficient for Holoscan's default queue size of one and for use with `receive_cuda_stream` which places just a single CUDA stream component in an upstream operator's outgoing messages. Cases where it is not appropriate are:
  - The input port's {ref}`queue size was explicitly set <configuring-queue-size>` with capacity greater than one and it is not known that all messages in the queue correspond to the same CUDA stream.
  - The input port is a multi-receiver port (i.e. `IOSpec::kAnySize`) that any number of upstream operators could connect to.

In cases where no stream is found in the input message, this condition will allow execution of the operator.

Example usage is as follows

`````{tab-set}
````{tab-item} C++
```cpp
// The code below would appear within `Application::compose` (or `Fragment::compose`)

// assuming the Operator has a port named "in", we can create the condition
auto stream_cond = make_condition<CudaStreamCondition>(name="stream_sync", receiver="in")

// it can then be passed as an argument to `make_operator`
auto my_op = make_operator<ops::MyOperator>("my_op",
                                            stream_cond,
                                            from_config("my_operator"));
)
```
````
````{tab-item} Python
```python
# The code below would appear within `Application.compose` (or `Fragment.compose`)

# assuming the Operator has a port named "in", we can create the condition
stream_cond = CudaStreamCondition(self, receiver="in", name="stream_sync")

# the condition is then passed as a positional argument to an Operator's constructor
visualizer = MyOperator(
    self,
    stream_cond,
    **my_kwargs,
    name="my_op",
)
```
````
`````

## Sharp edges related to Operators launching asynchronous work

This section describes a couple of scenarios where application authors may encounter surprising behavior when using operators that launch kernels asynchronously. As mentioned above, once a CUDA kernel has launched, control immediately returns to the host and the `compute` method may exit before all work on the GPU has completed. This is desirable for application performance, but raises some additional considerations that application authors should be aware of.

:::{tip}
Tools like the built-in `{ref}Data Flow Tracking<holoscan-flow-tracking>` or `{ref}GXF JobStatistics<gxf-job-satistics>` measures report the times spent in the `compute` method for operators. This can be misleadingly short when the actual GPU kernels complete at some later time after the `compute` call has ended. A concrete example is when an upstream operator launches a CUDA kernel asynchronously and then a downstream operator needs to do a device->host transfer (which requires synchronization). In that scenario the downstream operator will need to wait for the kernel launched by the upstream operator to complete, so the time for that upstream kernel would be reflected in the downstream operator's `compute` duration (assuming no `CudaStreamCondition` was used to force the upstream kernel to have completed before the downstream `compute` method was called).

In such scenarios it is recommended to perform profiling with {ref}`Nsight Systems <nsight-profiling>` to get a more detailed view of the application timing. The Nsight Systems UI will have per-stream traces of CUDA calls as well as separate traces for any scheduler worker threads that show the durations of Operator `compute` calls.
:::

:::{tip}
When an operator uses an `Allocator` (e.g. `UnboundedAllocator`, `BlockMemoryPool`, `RMMAllocator` or `StreamOrderedAllocator`) to dynamically allocate memory on each `compute` call, it is possible that more memory will be required than initially estimated. For example, if a kernel is launched but `compute` returns while computation is still being done on a tensor, an upstream operator is then free to be scheduled again. If that upstream operator was using an `Allocator`, the memory from the prior compute call would still be in use. Thus the operator needs space to allocate a second tensor on top of the original one. This means the author has to set a larger number of required bytes (or blocks) than they would have otherwise estimated (e.g. 2x as many).
:::
