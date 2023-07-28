# Holoscan Inference Module

Holoscan Inference module in Holoscan SDK is a framework that facilitates designing and executing inference applications. To start the process, user programs an inference application by initializing required operators in Holoscan SDK followed by specifying the flow between operators and configuring required parameters of the operators.

## Prerequisites

- Install Holoscan SDK

## Features

Key feature available with Holoscan Inference module:

- End-to-end CUDA based data buffer
    - With option to move data via host
- Multi backend support
    - Tensor RT:
        - CUDA based inference (supported on x86_64 and aarch64)
        - If Tensor RT engine file is automatically generated from an input model in onnx format, the generated engine file name is specific to the GPU type and Tensor RT version.
    - Torch:
        - Libtorch based inference
        - Libtorch version: 1.12.0
    - Onnx runtime:
        - Data flow via host
        - CUDA based inference (supported on x86_64)
        - CPU based inference (supported on x86_64 and aarch64)
- User configurable inference parameters
    - Single or Multi Inference
    - Multiple input and output for multiple models
    - Data flow related parameters
    - Parallel or Sequential execution of inferences
    - Generation of TRT engine files with FP16 option (supported for onnx based models)
- Datatype support
    - `float32`, `int32` and `int8` datatype support for input and output type

- Multi Receiver and Transmitter support for GXF based messages

### Multi GPU Inferencing

HoloInfer supports multi-GPU inferencing in a single node. Users need to populate `device_map` in the inference setting in the application yaml file. If `device_map` is not present, the multi-gpu inferencing feature is disabled and inferencing happens on the default GPU (GPU-0).

A sample `device_map` for a 3 model pipeline in a 2 GPU system may look like:

```yaml
inference:
    backend: "trt"
    model_path_map:
        "model_1_unique_identifier": "path_to_model_1"
        "model_2_unique_identifier": "path_to_model_2"
        "model_3_unique_identifier": "path_to_model_3"
    device_map:
        "model_1_unique_identifier": "0"
        "model_2_unique_identifier": "1"
        "model_3_unique_identifier": "0"
...
```

In the above setting, model_1 and model_3 infer on GPU with ID 0, and model_2 infers on GPU with ID 1.
GPU IDs used in the device map are derived from IDs as illustrated in `nvidia-smi -L`.

A sample GPU IDs output is shown below. Note that the IDs specified in device_map are the IDs followed by GPU keyword.

```sh
    $ nvidia-smi -L
    GPU 0: Quadro GV100 (UUID: GPU-873fc3d1-648c-de94-ff97-38afebf904cf)
    GPU 1: Quadro GV100 (UUID: GPU-f6fe8354-f724-3cde-bce2-b2ae29783b67)
```

Multi GPU Inferencing support comes with certain limitations in this version:

- Visualization
    - Multi-GPU inferencing supports visualization with Holoviz.
    - Holoviz runs on the first GPU connected to display (in a multi-GPU system). It is recommended to use multi-GPU inferencing (with visualization needs) with display connected to the default GPU (GPU-0)

- Data transfer
    - HoloInfer uses the concept of data transfer GPU (GPU-dt) to move data in and out of the inference operator.
    - GPU-dt is set to default GPU (GPU-0) and is not configurable in this release. It may be made configurable in future releases.

- Other operators
    - Other operators in the application (pre-processors, post-processors, etc.) are recommended to run on the default GPU (GPU-0). This is the default behavior if the user does not specify any specific GPU.
    - If other operators are running on a different GPU, operator must bring the data to default GPU (GPU-0) after processing.

- P2P access
    - GPUs specified in the multi-GPU configuration must have P2P (peer to peer) access and they must be connected to the same PCIE configuration. GPUs connected via different PCIE networks are not supported in multi-GPU inference.
### Data Processing

Module supports limited data processing capabilities.

### GXF Data Extraction and Transmission

- APIs to Receive and Transmit data via GXF Messages 
