# Built-in Operators and Extensions

The units of work of Holoscan applications are implemented within Operators, as described in the [core concepts](holoscan_core.md) of the SDK. The operators included in the SDK provide domain-agnostic functionalities such as IO, machine learning inference, processing, and visualization, optimized for AI streaming pipelines, relying on a set of [Core Technologies](relevant_technologies.md).

___

(holoscan-operators)=
## Operators

The operators below are defined under the `holoscan::ops` namespace for C++ and CMake, and under the `holoscan.operators` module in Python.

If you are new to Holoscan, a common starting pipeline is:
`VideoStreamReplayerOp`/`V4L2VideoCaptureOp` -> `FormatConverterOp` -> `InferenceOp` -> `SegmentationPostprocessorOp` -> `HolovizOp`.

### Imaging and visualization

| Class                 | Typical use                                                                                                                       | API reference                                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **BayerDemosaicOp**   | Convert Bayer RAW camera input to RGB/BGR style images for downstream processing.                                                 | {cpp:class}`C++ <holoscan::ops::BayerDemosaicOp>`/{py:class}`Python <holoscan.operators.BayerDemosaicOp>`     |
| **FormatConverterOp** | Convert between tensor/image formats, memory layouts, and data types between operators. See {ref}`byom-example`.                  | {cpp:class}`C++ <holoscan::ops::FormatConverterOp>`/{py:class}`Python <holoscan.operators.FormatConverterOp>` |
| **HolovizOp**         | Visualize tensors, overlays, geometry, and results for debugging or runtime display. See {ref}`visualization`.                    | {cpp:class}`C++ <holoscan::ops::HolovizOp>`/{py:class}`Python <holoscan.operators.HolovizOp>`                 |

### Inference and post-processing

| Class                           | Typical use                                                                                                                        | API reference                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **InferenceOp**                 | Run one or more AI models (including dependency-ordered pipelines) in streaming applications. See {ref}`inference`.                | {cpp:class}`C++ <holoscan::ops::InferenceOp>`/{py:class}`Python <holoscan.operators.InferenceOp>`                                 |
| **InferenceProcessorOp**        | Apply model-specific preprocessing/postprocessing around inference outputs.                                                        | {cpp:class}`C++ <holoscan::ops::InferenceProcessorOp>`/{py:class}`Python <holoscan.operators.InferenceProcessorOp>`               |
| **SegmentationPostprocessorOp** | Convert raw segmentation model outputs into masks/labels suitable for display or downstream logic.                                 | {cpp:class}`C++ <holoscan::ops::SegmentationPostprocessorOp>`/{py:class}`Python <holoscan.operators.SegmentationPostprocessorOp>` |

### Source, replay, and recording

| Class                     | Typical use                                                                                                                            | API reference                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **V4L2VideoCaptureOp**    | Capture live video frames from V4L2 devices on Linux systems.                                                                          | {cpp:class}`C++ <holoscan::ops::V4L2VideoCaptureOp>`/{py:class}`Python <holoscan.operators.V4L2VideoCaptureOp>`       |
| **VideoStreamReplayerOp** | Replay recorded streams for deterministic development, benchmarking, and regression testing. See {ref}`video-replayer-example`.        | {cpp:class}`C++ <holoscan::ops::VideoStreamReplayerOp>`/{py:class}`Python <holoscan.operators.VideoStreamReplayerOp>` |
| **VideoStreamRecorderOp** | Record tensor streams to disk for offline replay and reproducible experiments.                                                         | {cpp:class}`C++ <holoscan::ops::VideoStreamRecorderOp>`/{py:class}`Python <holoscan.operators.VideoStreamRecorderOp>` |

### Basic networking

| Class              | Typical use                                                                                                                                                                                                    | API reference                                                                                           |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **PingTxOp**       | Emit synthetic scalar data to test graph connectivity and scheduling behavior. See {ref}`native operator ping example <ping-multi-port-cpp>` and {ref}`native operator ping example <ping-multi-port-python>`. | {cpp:class}`C++ <holoscan::ops::PingTxOp>`/{py:class}`Python <holoscan.operators.PingTxOp>`             |
| **PingRxOp**       | Receive and inspect synthetic scalar data in tutorial or debugging pipelines. See {ref}`native operator ping example <ping-multi-port-cpp>` and {ref}`native operator ping example <ping-multi-port-python>`.  | {cpp:class}`C++ <holoscan::ops::PingRxOp>`/{py:class}`Python <holoscan.operators.PingRxOp>`             |
| **PingTensorTxOp** | Emit synthetic tensor payloads to validate tensor paths and operator interoperability.                                                                                                                         | {cpp:class}`C++ <holoscan::ops::PingTensorTxOp>`/{py:class}`Python <holoscan.operators.PingTensorTxOp>` |
| **PingTensorRxOp** | Receive synthetic tensor payloads for debugging C++/Python tensor flow behavior.                                                                                                                               | {cpp:class}`C++ <holoscan::ops::PingTensorRxOp>`/{py:class}`Python <holoscan.operators.PingTensorRxOp>` |

For complete, end-to-end application examples that combine these operators, see the [Holoscan SDK examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) and the [Holohub repository](https://github.com/nvidia-holoscan/holohub).

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

(ucx-holoscan)=

### Holoscan UCX GXF Extension

The `ucx_holoscan` extension includes `nvidia::holoscan::UcxHoloscanComponentSerializer` which is a `nvidia::gxf::ComponentSerializer` that handles serialization of `holoscan::Message` and `holoscan::Tensor` types for transmission using the Unified Communication X (UCX) library. UCX is the library used by Holoscan SDK to enable communication of data between fragments in distributed applications. For more details see the {ref}`creating-holoscan-distributed-application` page.

:::{note}
The `UcxHoloscanComponentSerializer` is intended for use in combination with other UCX components defined in the GXF UCX extension. Specifically, it can be used by the `UcxEntitySerializer` where it can operate alongside the `UcxComponentSerializer` that serializes GXF-specific types (`nvidia::gxf::Tensor`, `nvidia::gxf::VideoBuffer`, etc.). This way both GXF and Holoscan types can be serialized by distributed applications.
:::

___

### Holohub

Holohub is an open collection of reusable Holoscan operators, applications, workflows, and supporting tools designed to accelerate Holoscan-based development. You can browse the full set of operators directly in the [Holohub repository](https://github.com/nvidia-holoscan/holohub/tree/main/operators) or explore them with detailed documentation and search features on the [Holohub documentation website](https://nvidia-holoscan.github.io/holohub/operators/).  
For a quick overview of Holohub, see the {ref}`Holohub Overview <holohub>`.
