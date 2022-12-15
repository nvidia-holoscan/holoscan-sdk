# Tensor interoperability (GXF Tensor/DLPack/array interface)

## C++ API

This application demonstrates interoperability between a native operator (`ProcessTensorOp`) and two GXF Codelets (`SendTensor` and `ReceiveTensor`).
- The input and output ports are of type `holoscan::gxf::Entity` so that this operator can talk directly to the GXF codelets which send/receive GXF entities.
- The input/output of the entity has a tensor (`holoscan::Tensor` which is converted to `holoscan::gxf::Tensor` object inside the entity) which is used by the native operator to perform some computation and then the output tensor (in a new entity) is sent to the `ReceiveTensor` operator (codelet).
- The `ProcessTensorOp` operator uses the method in `holoscan::gxf::Tensor` to access the tensor data and perform some processing (multiplication by two) on the tensor data.
- The `ReceiveTensor` codelet gets the tensor from the entity and prints the tensor data to the terminal.

Notably, the two GXF codelets have not been wrapped as Holoscan operators, but are instead registered at runtime in the `compose` method of the application.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/tensor_interop/cpp/tensor_interop
```

## Python API

This application demonstrates interoperability between a native operator (`ImageProcessingOp`) and two operators (`VideoStreamReplayerOp` and `HolovizOp`) that wrap existing C++-based operators using GXF Tensors, through the Holoscan Tensor object (`holoscan.core.Tensor`).
- The Holoscan Tensor object is used to get the tensor data from the GXF Entity (`holoscan::gxf::Entity`) and perform some image processing (time-varying Gaussian blur) on the tensor data.
- The output tensor (in a new entity) is sent to the `HolovizOp` operator (codelet) which gets the tensor from the entity and displays the image in the GUI. The `VideoStreamReplayerOp` operator is used to replay the video stream from the sample data.
- The Holoscan Tensor object is interoperable with DLPack or array interfaces.

### Requirements

This example requires cupy which is included in the x86_64 development container. You'll need to build cupy for arm64 if you want to run this example on the developer kits.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then run: 
```bash
python3 ./examples/tensor_interop/python/tensor_interop.py
```

> ℹ️ Python apps can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).