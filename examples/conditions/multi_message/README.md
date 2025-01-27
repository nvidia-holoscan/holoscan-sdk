# Holoscan::MultiMessageAvailableCondition

This example demonstrates how to use Holoscan::MultiMessageAvailableCondition.

This condition type is unique among the conditions that are currently provided by the SDK in that it is a condition that applies across **multiple** input ports in combination as compared to `MessageAvailableCondition` or `ExpiringMessageAvailableCondition` which apply only to a **single** input port. Since a holoscan::IOSpec object is associated with a single input (or output) port, the existing `IOSpec::condition` API cannot be used to support this multi-port condition type. Instead there is a dedicated `OperatorSpec::multi_port_condition` method for handling such conditions. The application here shows how the `OperatorSetup::method` can call this method, passing a list of the input port names to which the condition should apply.

The multi-message available condition has two modes of operation. 

  1. The "SumOfAll" mode will be satisfied when the **total** number of messages arriving across the associated input ports reaches a specified **min_sum**. In that case, it doesn't matter which of the inputs the messages arrive on, only that the specified number have arrived.
  2. In the "PerReceiver" mode, you instead specify a **min_sizes** vector of the minimum number of messages that must arrive on each individual port associated with the condition. The "PerReceiver" mode behaves equivalently to having a default `MessageAvailableCondition` with the corresponding `min_size` on each receiver individually.

When using such a multi-receiver condition, one should set a "None" condition on the individual input ports that are being passed as the "port_names" argument to `OperatorSpec::multi_port_condition`. Otherwise, the ports will **also** have Holoscan's default `MessageAvailable` condition, which is likely undesired in any scenario where the multi-receiver condition is being used.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/conditions.html) to learn more about the Periodic Condition.*

## App Description

The "multi_message_per_receiver" example in this folder has two types of operators involved:
  1. Multiple instances of `StringTransmitterOp`, each of which simply transmits a single, user-configurable message string.
  2. A single `PerReceiverRxOp` which has three input ports, one of which has a queue capacity of 2 while the others have the default capacity of 1. A `MultiMessageAvailableCondition` is used to set this operator to only call compute once the expected number of messages have arrived on each port.

The `StringTransmitterOp` operators use periodic conditions with different rates, so messages are emitted at different temporal frequencies by each. Each transmitter will have the default `DownstreamMessageAffordableCondition`, so each `StringTransmitterOp` will only call compute when there would be space in the corresponding receiver queue of the `PerReceiverRxOp`.

The "multi_message_sum_of_all" example is similar except the receiving operator is configured to use the "SumOfAll" mode with a variant of the multi-message available condition called `MultiMessageAvailableTimeoutCondition`. This condition operates exactly like `MultiMessageAvailableCondition` except that it has one additional required parameter named "execution_frequency". The operator will be ready to execute again after the time interval specified by "execution_frequency", **even if** the specified number of messages has not yet arrived.

The final, "single_message_timeout" example shows a simpler case where the default `MessageAvailableCondition` on a single input port receiver is instead replaced with the `MultiMessageAvailableTimeoutCondition`. In this case there is only one receiver (input port) associated with the multi-message condition, so it acts like the standard `MessageAvailableCondition` except with an additional "execution_frequency" parameter that will allow the operator to execute at that frequency even if not all messages have arrived. When there is only a single port associated with the condition, there is no difference in behavior between the "PerReceiver" and "SumOfAll" modes.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions (C++)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

To run the "multi_message_per_receiver" example with `MultiMessageAvailableCondition` in "PerReceiver" mode:
```bash
./examples/conditions/multi_message/cpp/multi_message_per_receiver
```

To run the "multi_message_sum_of_all" example with `MultiMessageAvailableTimeoutCondition` in "SumOfAll" mode:
```bash
./examples/conditions/multi_message/cpp/multi_message_sum_of_all
```

To run the "single_message_timeout" example with `MultiMessageAvailableTimeoutCondition` with only a single input port:
```bash
./examples/conditions/multi_message/cpp/single_message_timeout
```

### Run instructions (Python)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

To run the "multi_message_per_receiver" example with `MultiMessageAvailableCondition` in "PerReceiver" mode:
```bash
python ./examples/conditions/multi_message/python/multi_message_per_receiver.py
```

To run the "multi_message_sum_of_all" example with `MultiMessageAvailableTimeoutCondition` in "SumOfAll" mode:
```bash
python ./examples/conditions/multi_message/python/multi_message_sum_of_all.py
```

To run the "single_message_timeout" example with `MultiMessageAvailableTimeoutCondition` with only a single input port:
```bash
python ./examples/conditions/multi_message/python/single_message_timeout.py
```
