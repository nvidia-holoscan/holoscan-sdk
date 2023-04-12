# GXF extensions

See the User Guide for details regarding the extensions in GXF and Holoscan SDK, and for instructions to build your own extensions

- `bayer_demosaic`: includes the `nvidia::holoscan::BayerDemosaic` codelet. It performs color filter array (CFA) interpolation for 1-channel inputs of 8 or 16-bit unsigned integer and outputs an RGB or RGBA image.
- `gxf_holoscan_wrapper`: includes the `holoscan::gxf::OperatorWrapper` codelet. It is used as a utility base class to wrap a holoscan operator to interface with the GXF framework.
- `opengl_renderer`: includes the `nvidia::holoscan::OpenGLRenderer` codelet. It displays a VideoBuffer, leveraging OpenGL/CUDA interop.
- `tensor_rt`: includes the `nvidia::holoscan::TensorRtInference` codelet. It takes input tensors and feeds them into TensorRT for inference.
- `stream_playback`: includes the `nvidia::holoscan::stream_playback::VideoStreamSerializer` entity serializer to/from a Tensor Object.
- `v4l2_source`: includes the `nvidia::holoscan::V4L2Source` codelet. It uses V4L2 to get image frames from a USB cameras. The output is a VideoBuffer object.
