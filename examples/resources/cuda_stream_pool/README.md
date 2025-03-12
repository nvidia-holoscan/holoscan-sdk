# Stream Handling Example

This folder contains an example of using streams managed by the CudaStreamPool resource in a Python application involving native Python operators (and one wrapped C++ operator, `PingTensorRxOp`). The application demonstrates how to use stream-related APIs from native Python `compute` methods and how to have CuPy use the Holoscan-managed streams when running CuPy kernels.

If the app is run with `use_default_stream=False` then `CuPySourceOp` and `CuPyProcessOp` will use dedicated per-operator streams from the stream pool for any kernels launched. Otherwise, with `use_default_stream=True`, all operators just use the default stream (no stream handling APIs are called and the provided `CudaStreamPool` is unused).

The `CudaStreamPool` resource uses a pool of `nvidia::gxf::CudaStream` components internally. The `CudaStream` class manages the lifetime of the CUDA streams and makes sure, for example, streams don't go out of scope when the call to an operator's `compute` method ends. Each stream exists as a component in GXF's entity-component system and it is just the `nvidia::gxf::CudaStreamId` containing a unique integer ID of the component that is actually transmitted between operators on `emit`/`receive`. Holoscan SDK programmers do not need to know any details of the underlying GXF APIs to use the public stream-related methods available on Holoscan's `ExecutionContext`, `InputContext` and `OutputContext` classes. These public APIs all use the CUDA Runtime API's standard `cudaStream_t` type in C++. For the corresponding Python API, the `stream` values are a Python `int` that represents the memory address of the underlying C++ `cudaStream_t`. The lifetime of these streams is managed automatically by Holoscan and its underlying GXF framework.

Note that the kernels launched by the operators in this example are just arbitrary choices to have some kernels running on the various streams and demonstrate how the native Python APIs are called. The specific computations performed do not serve any real-world purpose other than to act as an example. For real-world applications of stream handling using operators written in C++ from a Python or C++ application, see Holohub's [Multi-AI Ultrasound](https://github.com/nvidia-holoscan/holohub/tree/holoscan-sdk-2.7.0/applications/multiai_ultrasound), [Endoscopy Tool Tracking](https://github.com/nvidia-holoscan/holohub/tree/holoscan-sdk-2.7.0/applications/endoscopy_tool_tracking) and [Ultrasound Bone Scoliosis Segmentation](https://github.com/nvidia-holoscan/holohub/tree/holoscan-sdk-2.7.0/applications/ultrasound_segmentation) examples.

## Key operators and components involved:

A common `CudaStreamPool` resource with a capacity of 5 streams is defined. For this application, only 3 non-default streams are used (one for each of the two `CudaSourceOp` operators and one for the `CudaProcessOp`). If the app is run with `use_default_stream=True` the stream pool is not utilized and all operators just use the default stream.

`CuPySourceOp`: This root operator generates a tensor using an internally allocated stream. This stream is allocated via `context.allocate_cuda_stream` and placed on the output port via `op_output.set_cuda_stream`. The name of this allocated stream can be anything. The operator will have cached this stream and it will be re-used across subsequent `compute` calls, so the actual allocation from the stream pool will only happen on the first call to `compute`. The call for `set_cuda_stream` is needed so that output ports transmit this stream on any emitted messages so downstream operators can access it. Two separate copies of this operator are used in the application, so each will run on its own CUDA stream.

`CuPyProcessOp`: This operator has two input ports (each receiving input from a separate `CuPySourceOp`) and one output port. Note that the call to `receive_cuda_stream` for a given port **must** occur after the corresponding call to `receive` so that any streams present on that input port would be found and synchronized to the operator's internal stream. The first call to `op_input.receive_cuda_stream("in1")` will allocate a stream from the stream pool for use by this operator and synchronize any input streams found on port "in1" to it. Any subsequent `receive_cuda_stream` calls within the same or subsequent `compute` call will return that same internal stream. The second `op_input.receive_cuda_stream("in2")` thus does not allocate any new stream, but is needed just to synchronize any input streams found on port "in2" to the operator's internal stream. Note that there is no need to call `context.allocate_cuda_stream` as we want to just use the internal stream already returned by `receive_cuda_stream`. The only reason to call `context.allocate_cuda_stream` would be if we needed an additional stream for use within this operator's compute method. There is no call to `op_output.set_cuda_stream` because any internal stream allocated by `op_input.receive_cuda_stream` is automatically set to be transmitted on the output ports. Explicit `set_cuda_stream` calls are only necessary if the user explicitly allocated additional streams via `context.allocate_cuda_stream` and wants to send that stream information on the output ports.

`PingTensorRxOp`: This is a wrapped C++ operator. It also uses `receive_cuda_stream` on its input port and then just explicitly calls `cudaStreamSynchronize(stream)` to wait for all computations on the stream to complete. It then prints information on the names and attributes of any tensors received.

This example also demonstrates use of a `CudaStreamCondition` on the "in" port of `PingTensorRxOp`. The purpose of this condition is to force work on the stream found on this input port to complete prior to the start of the `compute` call. This is not strictly necessary for this example, as there is an equivalent `cudaStreamSynchronize` within the operator's `compute` method. However, waiting on streams to complete outside of `compute` may help efficiency by allowing the scheduler to schedule other operators during the period that the kernels from upstream operators are finishing. Specifically, `CudaStreamCondition` adds a host callback function on the stream that only sets the operator as ready to execute once computation on that stream is complete. This avoids needing to spend any time within the `compute` method waiting for upstream kernels to finish.

## CuPy ExternalStream interoperability

CuPy provides a [cupy.cuda.ExternalStream](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.ExternalStream.html#cupy-cuda-externalstream) class that can be used as a context manager so that any CuPy kernels will be launched on the specified external stream. The use of this class is demonstrated by both `CuPySourceOp` and `CuPyProcessOp`. Note that while within this external stream context, the value of any CuPy array's `__cuda_array_interface__["stream"]` will report the value of that external stream (assuming the user did not override [CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION](https://docs.cupy.dev/en/stable/reference/environment.html#envvar-CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION)).

One other point is that by default, the `cupy.asarray` call uses the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) and synchronization to the default stream when converting the provided tensor to a CuPy array. Because we already handled any necessary synchronization to the operator's internal stream via the `receive_cuda_stream` call, we can disable this additional default stream synchronization by explicitly setting the tensor's `__cuda_array_interface__["stream"]`  to `None` prior to the `cupy.asarray` call.

## Interoperability with other third-party Python library Stream objects

Although CuPy interoperability is demonstrated in this application, it is possible to interoperate in a similar manner with the `Stream` objects of other libraries like `PyTorch`, `CV-CUDA`, or `DALI`.

For example, PyTorch's `torch.cuda.StreamContext` context manager could be used with a `torch.cuda.ExternalStream(stream)` (where `stream` is the integer stream address returned by `InputContext.receive_cuda_stream`).

CV-CUDA's `nvcv.cuda.as_stream` can take an integer `stream` as input to produce an `ExternalStream` object. This stream object can then be used as the stream argument to the provided computer vision operations.

For DALI, the `cuda_stream` argument used by some functions needs to be provided by indicating it has a C `void*` type. This can be done using Python's built-in `ctypes` module to set the argument via the integer `stream` using `cuda_stream=ctypes.c_void_p(stream)`.

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/cuda_stream.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/resources/cuda_stream_pool/python/cuda_stream.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/resources/cuda_stream_pool/python/cuda_stream.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/resources/cuda_stream_pool/python/cuda_stream.py
  ```
