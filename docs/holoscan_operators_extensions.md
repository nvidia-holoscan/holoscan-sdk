# Built-in Operators and Extensions

The units of work of Holoscan applications are implemented within Operators, as described in the [core concepts](holoscan_core.md) of the SDK. The operators included in the SDK provide domain-agnostic functionalities such as IO, machine learning inference, processing, and visualization, optimized for AI streaming pipelines, relying on a set of [Core Technologies](relevant_technologies.md).

___

(holoscan-operators=)
## Operators

The operators below are defined under the `holoscan::ops` namespace for C++ and CMake, and under the `holoscan.operators` module in Python.

| Class    | CMake target/lib | Documentation                  |
|--------- |----------------- |------------------------------- |
| **BayerDemosaicOp** | `bayer_demosaic` | {cpp:class}`C++ <holoscan::ops::BayerDemosaicOp>`/{py:class}`Python <holoscan.operators.BayerDemosaicOp>` |
| **FormatConverterOp** | `format_converter` | {cpp:class}`C++ <holoscan::ops::FormatConverterOp>`/{py:class}`Python <holoscan.operators.FormatConverterOp>` |
| **HolovizOp** | `holoviz` | {cpp:class}`C++ <holoscan::ops::HolovizOp>`/{py:class}`Python <holoscan.operators.HolovizOp>` |
| **InferenceOp** | `inference` | {cpp:class}`C++ <holoscan::ops::InferenceOp>`/{py:class}`Python <holoscan.operators.InferenceOp>` |
| **InferenceProcessorOp** | `inference_processor` | {cpp:class}`C++ <holoscan::ops::InferenceProcessorOp>`/{py:class}`Python <holoscan.operators.InferenceProcessorOp>` |
| **PingRxOp** | `ping_rx` | {cpp:class}`C++ <holoscan::ops::PingRxOp>`/{py:class}`Python <holoscan.operators.PingRxOp>` |
| **PingTensorRxOp** | `ping_tensor_rx` | {cpp:class}`C++ <holoscan::ops::PingTensorRxOp>`/{py:class}`Python <holoscan.operators.PingTensorRxOp>` |
| **PingTensorTxOp** | `ping_tensor_tx` | {cpp:class}`C++ <holoscan::ops::PingTensorTxOp>`/{py:class}`Python <holoscan.operators.PingTensorTxOp>` |
| **PingTxOp** | `ping_tx` | {cpp:class}`C++ <holoscan::ops::PingTxOp>`/{py:class}`Python <holoscan.operators.PingTxOp>` |
| **SegmentationPostprocessorOp** | `segmentation_postprocessor` | {cpp:class}`C++ <holoscan::ops::SegmentationPostprocessorOp>`/{py:class}`Python <holoscan.operators.SegmentationPostprocessorOp>` |
| **VideoStreamRecorderOp** | `video_stream_recorder` | {cpp:class}`C++ <holoscan::ops::VideoStreamRecorderOp>`/{py:class}`Python <holoscan.operators.VideoStreamRecorderOp>` |
| **VideoStreamReplayerOp** | `video_stream_replayer` | {cpp:class}`C++ <holoscan::ops::VideoStreamReplayerOp>`/{py:class}`Python <holoscan.operators.VideoStreamReplayerOp>` |
| **V4L2VideoCaptureOp** | `v4l2` | {cpp:class}`C++ <holoscan::ops::V4L2VideoCaptureOp>`/{py:class}`Python <holoscan.operators.V4L2VideoCaptureOp>` |

Given an instance of an operator class, you can print a human-readable description of its specification to inspect the inputs, outputs, and parameters that can be configured on that operator class:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
std::cout << operator_object->spec()->description() << std::endl;
```
````
````{tab-item} Python
```{code-block} python
print(operator_object.spec)
```
````
`````

:::{note}
The Holoscan SDK uses meta-programming with templating and `std::any` to support arbitrary data types. Because of this, some type information (and therefore values) might not be retrievable by the `description` API. If more details are needed, we recommend inspecting the list of `Parameter` members in the operator [header](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/include/holoscan/operators) to identify their type.
:::
___

(sdk-extensions)=
## Extensions

The Holoscan SDK also includes some GXF extensions with GXF codelets, which are typically wrapped as operators, or present for legacy reasons. In addition to the core GXF extensions (std, cuda, serialization, multimedia) listed [here](gxf/doc/index.md), the Holoscan SDK includes the following GXF extensions:
- [gxf_holoscan_wrapper](#gxf-holoscan-wrapper)
- [ucx_holoscan](#ucx-holoscan)

### GXF Holoscan Wrapper

The `gxf_holoscan_wrapper` extension provides the `holoscan::gxf::OperatorWrapper` codelet and the `holoscan::gxf::ResourceWrapper` component. It serves as a utility base class for wrapping a Holoscan operator or resource as a GXF codelet or component, respectively. This extension allows Holoscan operators and resources to be integrated into GXF applications and GraphComposer workflows.

Learn more about it in the [Using Holoscan Operators in GXF Applications](gxf/gxf_wrap_holoscan.md) section.

(ucx-holoscan)=
### UCX (Holoscan)

The `ucx_holoscan` extension includes `nvidia::holoscan::UcxHoloscanComponentSerializer` which is a `nvidia::gxf::ComponentSerializer` that handles serialization of `holoscan::Message` and `holoscan::Tensor` types for transmission using the Unified Communication X (UCX) library. UCX is the library used by Holoscan SDK to enable communication of data between fragments in distributed applications.

:::{note}
The `UcxHoloscanComponentSerializer` is intended for use in combination with other UCX components defined in the GXF UCX extension. Specifically, it can be used by the `UcxEntitySerializer` where it can operate alongside the `UcxComponentSerializer` that serializes GXF-specific types (`nvidia::gxf::Tensor`, `nvidia::gxf::VideoBuffer`, etc.). This way both GXF and Holoscan types can be serialized by distributed applications.
:::

___

### HoloHub

Visit the [HoloHub repository](https://github.com/nvidia-holoscan/holohub) to find a collection of additional Holoscan operators and extensions.
