# GXF extensions

See the User Guide for details regarding the extensions in GXF and Holoscan SDK, and for instructions to build your own extensions

- `aja`: support AJA capture card as source. It offers support for GPUDirect-RDMA on Quadro GPUs. The output is a VideoBuffer object.
- `bayer_demosaic`: calls to the NVIDIA CUDA based demosaicing algorithm to create RGB planes from Bayer format.
- `custom_lstm_inference`: provide LSTM (Long-Short Term Memory) stateful inference module using TensorRT
- `emergent`: support camera from Emergent Vision Technologies as the source. The datastream from this camera is transferred through Mellanox ConnectX NIC using Rivermax SDK.
- `format_converter`: provide common video or tensor operations in inference pipelines to change datatypes, resize images, reorder channels, and normalize and scale values.
- `holoviz`: Holoviz is a Vulkan API based visualizer to display video buffer and geometry from a tensor.
- `multiai_inference`: Multi AI inference supported by TRT and onnxruntime backend.
- `multiai_postprocessor`: Executes data processing on multiple tensors. Currently supports only one operation for icardio models.
- `visualizer_icardio`: Prepares results from plax chamber model (from icardio) for visualization.
- `opengl`: OpenGL Renderer(visualizer) to display a VideoBuffer, leveraging OpenGL/CUDA interop.
- `probe`: print tensor information
- `segmentation_postprocessor`: segmentation model postprocessing converting inference output to highest-probability class index, including support for sigmoid, softmax, and activations.
- `segmentation_visualizer`: OpenGL renderer that combines segmentation output overlaid on video input, using CUDA/OpenGL interop.
- `stream_playback`: provide video stream playback module to output video frames as a Tensor object.
- `tensor_rt` _(duplicate from GXF)_: Run inference with TensorRT
- `v4l2`: Video for Linux 2 source supporting USB cameras and other media inputs. The output is a VideoBuffer object.
- `visualizer_tool_tracking`: custom visualizer component that handles compositing, blending, and visualization of tool labels, tips, and masks given the output tensors of the custom_lstm_inference
- `mocks`: provide mock components such as VideoBufferMock.
- `sample`: provide sample ping_tx/ping_rx components.
