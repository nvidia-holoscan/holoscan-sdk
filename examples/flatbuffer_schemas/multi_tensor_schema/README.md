# Multi-Tensor FlatBuffers schema

This example defines an application-owned `stereo.StereoFrame` FlatBuffers root with two
CUDA-device image Tensors, one host calibration Tensor, and ordinary frame metadata. The generated
traits make its owning object-API type available directly as
`Output<stereo::StereoFrameT>` and `Input<stereo::StereoFrameT>`.

The Source and Sink declare each Tensor field independently with generated selectors such as
`stereo::StereoFrameTensorFields::left`. Each field can have its own shape, dtype, memory kind, and
allocation bound. The two device fields share the output port's one device binding; that binding
does not change the calibration field from host memory to device memory.

The Source groups the two image selectors under one shared output contract and gives the host
calibration field its own contract. It calls `output.allocate_tensors()` with one shape per field;
the exact dtypes come from those admitted contracts. One `write_all()` call preflights and writes
the complete move-only bundle before `output.emit_tensors()` publishes the transaction. The
descriptor value supplies only `StereoFrame` metadata; the bundle's Tensors are authoritative.
The same application code uses each field's final provider when a graph is partitioned across
processes, so Tensor payload bytes are transported by buffer handle without a staging copy. The
bounded FlatBuffers and sample metadata are still encoded.

The Sink also demonstrates field-qualified `expects_tensor()` declarations. They are purely
optional consumer compatibility constraints: omitting them does not make a generated Tensor field
optional, and Runtime still validates the received descriptor metadata.

Application-owned Tensors can instead be assigned to the generated object and published with
ordinary `output.emit(frame)`. Runtime exports compatible HoloMQ backing directly. An application
that explicitly selects `kCopyIfNeeded` may also copy other storage once into each field's planned
staging pool. The policy permits a copy but does not require one. That path is useful for
interoperability, while the provider-backed bundle API remains direct-export-only and makes
zero-copy producer allocation explicit.

The example's current field contracts intentionally omit `kCopyIfNeeded`. To publish foreign
storage through the by-value form below, add that policy and its bounded staging span to each
affected field; directly exportable backing does not need the copy policy.

```cpp
stereo::StereoFrameT frame;
frame.left = application_left_tensor;
frame.right = application_right_tensor;
frame.calibration = application_calibration_tensor;
auto emitted = output.emit(std::move(frame));
```

The application uses only Operators, typed ports, `Graph::add_flow()`, `compile()`, and
`run_async()`. Application authors do not create channel Publishers or Subscribers.
