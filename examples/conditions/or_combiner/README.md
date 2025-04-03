# Holoscan::OrConditionCombiner

This example demonstrates how an operator can be configured to execute when a message arrives on either input of an operator with multiple receivers.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/conditions.html) to learn more about the OrConditionCombiner.*

## App Description

This is a very simple app that involves two independent transmitters (`PingTxOp`) that are each configured with a `PeriodicCondition` to emit integers at different intervals.

The two transmit operators connect to the "in1" and "in2" ports of the `MultiRxOrOp` receiver defined for this application. That receiver is using the `or_combine_port_conditions` method to allow the receiver to execute when a message arrives on either of its input ports. Without the call to `or_combine_port_conditions`, it would instead be required that a message arrived on both input ports before the operator would execute.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions (C++)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Run the example using
```bash
./examples/conditions/or_combiner/cpp/multi_port_or_combiner
```

### Run instructions (Python)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Run the example using
```bash
python ./examples/conditions/or_combiner/python/multi_port_or_combiner.py
```
