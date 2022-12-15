# Native Operator

These examples demonstrate how to use native operators (the operators that do not have an underlying, pre-compiled GXF Codelet):

## C++ API

This example shows the application using only native operators. There are three operators involved:
  1. a transmitter, set to transmit a sequence of even integers on port `out1` and odd integers on port `out2`.
  2. a middle operator that prints the received values, multiplies by a scalar and transmits the modified values
  3. a receiver that prints the received values to the terminal

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/native_operator/cpp/ping
```

## Python API

- `ping.py`: This example is similar to the C++ native operator example, using Python.
- `convolve.py`: This example demonstrates a simple 1D convolution-based signal processing application, to demonstrate passing NumPy arrays between operators as Python objects.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

```bash
python3 ./examples/native_operator/python/ping.py
python3 ./examples/native_operator/python/convolve.py
```

> ℹ️ Python apps can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).
