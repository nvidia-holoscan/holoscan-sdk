# Holoscan::PeriodicCondition

This example demonstrates how to use Holoscan::PeriodicCondition.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/conditions.html) to learn more about the Periodic Condition.*

## C++ API

This example has two operators involved:
  1. a transmitter, set to transmit a string message `Periodic ping...` on port `out`. This operator is configured to be executed 10 times each subsequent message is sent only after a period of 200 milliseconds has elapsed.
  2. a receiver that prints the received values to the terminal.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/conditions/periodic/cpp/ping_periodic
```

## Python API

This example demonstrates the use of PeriodicCondition using python API. This is a simple ping application with two operators connected using add_flow().

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-10 on its 'out' port and each subsequent message is sent only after a period of 200 milliseconds has elapsed.
  3. a receiver that prints the received values to the terminal

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the following command.

```bash
python3 ./examples/conditions/periodic/python/ping_periodic.py
```
