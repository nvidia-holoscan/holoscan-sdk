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
        - CUDA based inference
    - Onnx runtime:
        - Data flow via host
        - CUDA or CPU based inference
- User configurable inference parameters
    - Single or Multi Inference
    - Single input for multiple models
    - Data flow related parameters
    - Parallel or Sequential execution of inferences
    - Generation of TRT engine files with FP16 option
- Multi Receiver and Transmitter support for GXF based messages

### Inference

Holoscan Inference Module supports:

- TRT and Onnxruntime based inference
- Models with 1 input node and 1 output node
- Floating point input and output type

### Data Processing

Module supports limited data processing capabilities.

### GXF Data Extraction and Transmission

- APIs to Receive and Transmit data via GXF Messages 
