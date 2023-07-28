# Ping Simple Run Async

This example demonstrates a simple ping application 'asynchronously' with two operators connected using add_flow().

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-10 to it's 'out' port
  2. a receiver that prints the received values to the terminal

While the application is running, it prints the index value from PingTxOp to the terminal.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the following command.

```bash
# C++
./examples/ping_simple_run_async/cpp/ping_simple_run_async

# Python
python3 ./examples/ping_simple_run_async/python/ping_simple_run_async.py
```

> ℹ️ Python apps can run outside those folders if `HOLOSCAN_INPUT_PATH` is set in your environment (automatically done by `./run launch`).
