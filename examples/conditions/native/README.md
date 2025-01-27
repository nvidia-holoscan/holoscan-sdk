# Holoscan Native Conditions

This example demonstrates how to define and use a native `Holoscan::Condition` (C++) / `holoscan.core.Condition` (Python) (as opposed to one wrapping an underlying `nvidia::gxf::PeriodicSchedulingTerm`) . For purposes of demonstration, this folder contains two applications:

1. **ping_periodic_native**: demonstrates creation of a native version of `holoscan::PeriodicCondition` (C++) / `holoscan.conditions.PeriodicCondition` (Python).
2. **message_available_native**: demonstrates creation of a native version of `holoscan::MessageAvailableCondition` (C++) / `holoscan.conditions.MessageAvailableCondition` (Python) .

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_condition.html) to learn more about creating native Conditions.*

## Application Description: ping_periodic_native

This example has two operators involved:
  1. a transmitter, 'tx', that transmits a monotonically increasing integer value on port `out`. This operator is configured to be executed 5 times and each subsequent message is sent only after a period of 200 milliseconds has elapsed.
  2. a receiver, 'rx', that prints the received values to the terminal.

The 200 ms period between transmit calls is implemented via a native condition class called `NativePeriodicCondition`. This class works like the built-in `PeriodicCondition` (which wraps an underlying `nvidia::gxf::SchedulingTerm` class). The purpose of this demo is to illustrate how new conditions can be created purely via inheriting from Holoscan's `Condition` class without having to create and compile a separate GXF extension first.

## Application Description: message_available_native

Like the `ping_periodic_native` app, this example also has a single 'tx' operator and a single 'rx' operator. Note that the default `PingRxOp` would normally have a default condition (`MessageAvailableCondition` with `min_size=1`) automatically created for its "in" input port because no condition was specified in `PingRxOp::setup`. However, when the user-supplied `NativeMessageAvailableCondition` condition involving the "in" port is provided, it would replace this default condition. Thus, it is not necessary to edit the setup method to add a "none" condition in order to use the user-provided condition instead. Note that if a non-default condition had been explicitly assigned to the "in" port during `PingRxOp::setup` then the user provided condition would be in addition to that one (only the default condition is replaced by a user-provided one).

When creating the application, we pass our own `NativeMessageAvailableCondition` for use on port "in" of the operator instead. Note that for the case of an argument named "receiver" (or "transmitter") the user can just specify the name of the port whose `Receiver` (or `Transmitter`) object, the condition should apply to. The SDK will take care of replacing `Arg("receiver", "in")` (C++) / or kwarg `receiver="in"` (Python) with an argument containing the actual `Receiver` object that gets created by the SDK for that input port.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions (C++)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, for the `ping_periodic_native` app run:
```bash
./examples/conditions/native/cpp/ping_periodic_native
```
or for the `message_available_native` app run:
```bash
./examples/conditions/native/cpp/message_available_native
```

### Run instructions (Python)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:

Then, for the `ping_periodic_native` app run:
```bash
python ./examples/conditions/native/python/ping_periodic_native.py
```
or for the `message_available_native` app run:
```bash
python ./examples/conditions/native/python/message_available_native.py
```
