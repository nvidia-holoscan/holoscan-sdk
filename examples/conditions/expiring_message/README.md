# Holoscan::ExpiringMessageAvailableCondition

This example demonstrates how to use Holoscan::ExpiringMessageAvailableCondition.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/conditions.html) to learn more about the ExpiringMessageAvailable Condition.*

## C++ API

This example has two operators involved:
  1. a transmitter that  a transmitter, set to transmit a string message `Periodic ping...` on port `out`. This operator is configured to be executed 8 times each subsequent message is sent only after a period of 10 milliseconds has elapsed.
  2. a receiver that waits for a 5 messages to be batched together to call compute. If 5 messages have not arrived by a specified interval of 1 second, compute will be called at that time.

Note that the `ExpiringMessageAvailableCondition` added to the input port of the receive operator requires that the message sent by the output port of the transmit operator attaches a timestamp. This timestamp is needed to be able to enforce the `max_delay_ns` timeout interval used by the condition.


### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/conditions/expiring_message/cpp/ping_expiring_message
```

## Python API

This example demonstrates the use of ExpiringMessageAvailableCondition using python API. This is a simple ping application with two operators connected using add_flow().

There are two operators involved in this example:
  1. a transmitter that on each tick, transmits an integer to the "out" port. This operator is configured to be executed 8 times each subsequent message is sent only after a period of 10 milliseconds has elapsed.
  3. a receiver that will wait for 5 messages from the "in" port before it will call compute. If 5 messages have not arrived by the specified interval of 1 second, compute will be called at that time.

Note that the `ExpiringMessageAvailableCondition` added to the input port of the receive operator requires that the message sent by the output port of the transmit operator attaches a timestamp. This timestamp is needed to be able to enforce the `max_delay_ns` timeout interval used by the condition.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the following command.

```bash
python3 ./examples/conditions/expiring_message/python/ping_expiring_message.py
```
