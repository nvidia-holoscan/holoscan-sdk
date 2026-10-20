# Multi-Tensor FlatBuffers schema

This example defines an application-owned `stereo.StereoFrame` FlatBuffers root with two
CUDA-device image Tensors, one host calibration Tensor, and ordinary frame metadata. The generated
traits make its owning object-API type available directly as
`Output<stereo::StereoFrameT>` and `Input<stereo::StereoFrameT>`.

The Source, Transform, and Sink declare each Tensor field independently with generated selectors such as
`stereo::StereoFrameTensorFields::left`. Each field can have its own shape, dtype, memory kind, and
allocation bound. The two device fields share the output port's one device binding; that binding
does not change the calibration field from host memory to device memory.

The Source groups the two image selectors under one shared output spec and gives the host
calibration field its own spec. It calls `output.allocate_tensors()` with one shape per field;
the exact dtypes come from those admitted specs. One `write_all()` call preflights and writes
the complete move-only bundle before `output.emit_tensors()` publishes the transaction. The
descriptor value supplies only `StereoFrame` metadata; the bundle's Tensors are authoritative.
The same application code uses each field's final provider when a graph is partitioned across
processes, so Tensor payload bytes are transported by buffer handle without a staging copy. The
bounded FlatBuffers and sample metadata are still encoded.

The Sink also demonstrates field-qualified `expects_tensor()` declarations. They are purely
optional consumer compatibility constraints: omitting them does not make a generated Tensor field
optional, and Runtime still validates the received descriptor metadata.

The schema opts into reusable generated metadata storage through `native_bounds.json`. Its
`header.source` string is capped at 64 bytes and the complete prepared native object is capped at
2,048 bytes. These native limits are independent of `MAX_SERIALIZED_SIZE`, which bounds the wire
metadata. CMake passes the policy to both the in-tree and installed-consumer builds:

```cmake
holoscan_add_flatbuffer_schema(
    TARGET multi_tensor_schema_generated
    SCHEMA stereo_frame.fbs
    MAX_SERIALIZED_SIZE 768
    NATIVE_BOUNDS_FILE native_bounds.json
)
```

The Transform demonstrates the complete generic descriptor path. It borrows a callback-local
sample with `selected_sample()`, copies ordinary fields into one cold-created
`DescriptorMetadataScratch<stereo::StereoFrameT>`, and edits that prepared metadata in place. It
allocates a fresh output bundle, registers both CUDA input reads on the operator stream, copies the
host calibration field, and publishes all three authoritative output Tensors with
`emit_tensors()`. Tensor members remain null in metadata scratch. `start()` creates scratch only
when it is absent, so an in-place lifecycle restart reuses the prepared capacity. The local
`derive_from()` helper mirrors `sensor_io::derive_from()` so this application-owned schema example
can retain its Core-only link dependency.

The SDK system suite also configures and builds this directory against the installed Holoscan
package. That fixture proves that third-party schema generation, the native bounds file, prepared
descriptor scratch, and the three-field transform remain available outside the source tree. A
machine without CUDA device 0 reports the example as skipped with exit code 77.

Application-owned Tensors can instead be assigned to the generated object and published with
ordinary `output.emit(frame)`. Runtime exports compatible HoloMQ backing directly. An application
that explicitly selects `kCopyIfNeeded` may also copy other storage once into each field's planned
staging pool. The policy permits a copy but does not require one. That path is useful for
interoperability, while the provider-backed bundle API remains direct-export-only and makes
zero-copy producer allocation explicit.

The example's current field specs intentionally omit `kCopyIfNeeded`. To publish foreign
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
