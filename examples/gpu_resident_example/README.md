# GPU Resident Example

This example demonstrates how to use GPU-resident operators in Holoscan applications.

## Overview

The example creates a simple pipeline in a fragment with GPU-resident operators that execute directly on the GPU. It also creates another Holoscan SDK fragment that includes non-GPU-resident operators. The two fragments are not connected to each other. It includes:

- A custom `CustomGpuOp` operator that inherits from `GPUResidentOperator`
- Device memory allocation for input and output ports

This is WiP skeleton code for now. We will add more features to this example in the future.
