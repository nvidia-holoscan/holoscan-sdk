# Asynchronous Ping Example

This example demonstrates a simple ping application with two operators connected using add_flow().

It differs from `ping_simple` in that one can independently choose if the receive and/or transmit operators run asynchronously. One can also optionally choose between the single-threaded greedy scheduler or multi-threaded scheduler.

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-20 to it's 'out' port
  2. a receiver that prints the received values to the terminal

For the C++ application:
- The transmit operator will be asynchronous if `async_transmit: true` in `ping_async.yaml`.
- The receive operator will be asynchronous if `async_receive: true` in `ping_async.yaml`.
- The scheduler to be used can be set via the `scheduler` entry in `ping_async.yaml`. It defaults to `event_based` (an event-based multi-thread scheduler), but can also be set to either `multi_thread` (polling-based) or `greedy` (single thread).

For the Python application, configuration is via command line arguments as described below.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/conditions.html) to learn more about the Asynchronous Condition.*

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the following command.

```bash
# C++
./examples/conditions/asynchronous/cpp/ping_async
```

The three boolean values in the YAML file can be set as desired to control which operators run
asynchronously and whether the multi-threaded scheduler is used.

For example, to run with asynchronous receive and asynchronous transmit operators with the multi-threaded scheduler.

```bash
sed -i -e 's#^async_receive:.*#async_receive: true#' ./examples/conditions/asynchronous/cpp/ping_async.yaml
sed -i -e 's#^async_transmit:.*#async_transmit: true#' ./examples/conditions/asynchronous/cpp/ping_async.yaml
sed -i -e 's#^multithreaded:.*#multithreaded: true#' ./examples/conditions/asynchronous/cpp/ping_async.yaml
./examples/conditions/asynchronous/cpp/ping_async
```

# Python
```bash
python ./examples/conditions/asynchronous/python/ping_async.py
```

By default, both transmit and receive are asynchronous. To see the available options run the
application using `-h` or `--help`.

For example, to send 5 messages, waiting 500 ms between messages and use async transmit and
synchronous receive:

```bash
python ./examples/conditions/asynchronous/python/ping_async.py --delay=500 --count=5 --sync_rx
```

> ℹ️ Python apps can run outside those folders if `HOLOSCAN_INPUT_PATH` is set in your environment (automatically done by `./run launch`).
