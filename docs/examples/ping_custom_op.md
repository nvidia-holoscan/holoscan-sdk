(ping-custom-op-example)=
# Ping Custom Op

In this section, we will modify the previous `ping_simple` example to add a custom operator into the workflow.  We've already seen a custom operator defined in the `hello_world` example but skipped over some of the details.

In this example we will cover:

- The details of creating your own custom operator class.
- How to add input and output ports to your operator.
- How to add parameters to your operator.
- The data type of the messages being passed between operators.

:::{note}
The example source code and run instructions can be found in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) directory on GitHub, or under `/opt/nvidia/holoscan/examples` in the NGC container and the Debian package, alongside their executables.
:::

## Operators and Workflow

Here is the diagram of the operators and workflow used in this example.

```{digraph} custom_op
:align: center
:caption: A linear workflow with new custom operator

    rankdir="LR"
    node [shape=record];

    tx [label="PingTxOp| |out(out) : int"];
    mx [label="PingMxOp| [in]in : int | out(out) : int "];
    rx [label="PingRxOp| [in]in : int | "];
    tx -> mx [label="out...in"]
    mx -> rx [label="out...in"]
```

Compared to the previous example, we are adding a new **PingMxOp** operator between the
**PingTxOp** and **PingRxOp** operators.  This new operator takes as input an integer, multiplies it by a constant factor, and then sends the new value to **PingRxOp**.  You can think of this custom operator as doing some data processing on an input stream before sending the result to downstream operators.

## Configuring Operator Input and Output Ports

Our custom operator needs 1 input and 1 output port and can be added by calling `spec.input()` and `spec.output()` methods within the operator's `setup()` method.
This requires providing the data type and name of the port as arguments (for C++ API), or just the port name (for Python API). We will see an example of this in the code snippet below. For more details, see  {ref}`specifying-operator-inputs-and-outputs-cpp` or {ref}`specifying-operator-inputs-and-outputs-python`.

## Configuring Operator Parameters

Operators can be made more reusable by customizing their parameters during initialization. The custom parameters can be provided either directly as arguments or accessed from the application's YAML configuration file.  We will show how to use the former in this example to customize the "multiplier" factor of our **PingMxOp** custom operator.
Configuring operators using a YAML configuration file will be shown in a subsequent {ref}`example<video-replayer-example>`. For more details, see {ref}`configuring-app-operator-parameters`.

The code snippet below shows how to define the **PingMxOp** class.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 9, 14-16, 20, 25, 27
:name: holoscan-one-operator-workflow-cpp

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace holoscan::ops {

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in");

    std::cout << "Middle message value: " << value << std::endl;

    // Multiply the value by the multiplier parameter
    value *= multiplier_;

    op_output.emit(value);
  };

 private:
  Parameter<int> multiplier_;
};

}  // namespace holoscan::ops
```
- The `PingMxOp` class inherits from the {cpp:class}`Operator <holoscan::ops::Operator>` base class (line `7`).
- The `HOLOSCAN_OPERATOR_FORWARD_ARGS` macro (line `9`) is syntactic sugar to help forward an operator's constructor arguments to the {cpp:class}`Operator <holoscan::ops::Operator>` base class, and is a convenient shorthand to avoid having to manually define constructors for your operator with the necessary parameters.
- Input/output ports with the names "in"/"out" are added to the operator spec on lines `14` and `15` respectively.  The port type of both ports are `int` as indicated by the template argument `<int>`.
- We add a "multiplier" parameter to the operator spec (line `16`) with a default value of 2.  This parameter is tied to the private "multiplier_" data member.
- In the `compute()` method, we receive the integer data from the operator's "in" port (line `20`), print its value, multiply its value by the multiplicative factor, and send the new value downstream (line `27`).
- On line `20`, note that the data being passed between the operators has the type `int`.
- The call to `op_output.emit(value)` on line `27` is equivalent to `op_output.emit(value, "out")` since this operator has only 1 output port.  If the operator has more than 1 output port, then the port name is required.
````
````{tab-item} Python
```{code-block} python
:linenos: true
:emphasize-lines: 5, 17-19, 22, 28
:name: holoscan-one-operator-workflow-python

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp

class PingMxOp(Operator):
    """Example of an operator modifying data.

    This operator has 1 input and 1 output port:
        input:  "in"
        output: "out"

    The data from the input is multiplied by the "multiplier" parameter

    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Middle message value: {value}")

        # Multiply the values by the multiplier parameter
        value *= self.multiplier

        op_output.emit(value, "out")
```
- The `PingMxOp` class inherits from the `Operator` base class (line `5`).
- Input/output ports with the names "in"/"out" are added to the operator spec on lines `17` and `18`, respectively.
- We add a "multiplier" parameter to the operator spec with a default value of 2 (line `19`).
- In the `compute()` method, we receive the integer data from the operator's "in" port (line `22`), print its value, multiply its value by the multiplicative factor, and send the new value downstream (line `28`).
````
`````
Now that the custom operator has been defined, we create the application, operators, and define the workflow.

<!-- Note that NVIDIA's public user guide doesn't seem to support the `lineno-start` tag, such as `:lineno-start: 35` in the code block, so we are removing it. -->

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 7, 11-12
:name: holoscan-ping-custom-op-app-cpp

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // Define the tx, mx, rx operators, allowing tx operator to execute 10 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto mx = make_operator<ops::PingMxOp>("mx", Arg("multiplier", 3));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Define the workflow:  tx -> mx -> rx
    add_flow(tx, mx);
    add_flow(mx, rx);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
```
- The tx, mx, and rx operators are created in the `compose()` method on lines `6-8`.
- The custom mx operator is created in exactly the same way with `make_operator()` (line `7`) as the built-in operators, and configured with a "multiplier" parameter initialized to 3 which overrides the parameter's default value of 2 (in the `setup()` method).
- The workflow is defined by connecting tx to mx, and mx to rx using `add_flow()` on lines `11-12`.
````
````{tab-item} Python
```{code-block} python
:linenos: true
:emphasize-lines: 5, 9-10
:name: holoscan-ping-custom-op-app-python

class MyPingApp(Application):
    def compose(self):
        # Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMxOp(self, name="mx", multiplier=3)
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> mx -> rx
        self.add_flow(tx, mx)
        self.add_flow(mx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
```
- The tx, mx, and rx operators are created in the `compose()` method on lines `4-6`.
- The custom mx operator is created in exactly the same way as the built-in operators (line `5`), and configured with a "multiplier" parameter initialized to 3 which overrides the parameter's default value of 2 (in the `setup()` method).
- The workflow is defined by connecting tx to mx, and mx to rx using `add_flow()` on lines `9-10`.
````
`````


## Message Data Types

For the C++ API, the messages that are passed between the operators are the objects of the data type at the inputs and outputs, so the `value` variable from lines 20 and 25 of the example above has the type `int`.  For the Python API, the messages passed between operators can be arbitrary Python objects so no special consideration is needed since it is not restricted to the stricter parameter typing used for C++ API operators.

Let's look at the code snippet for the built-in **PingTxOp** class and see if this helps to make it clearer.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 6, 11
:name: holoscan-one-operator-workflow-cpp

#include "holoscan/operators/ping_tx/ping_tx.hpp"

namespace holoscan::ops {

void PingTxOp::setup(OperatorSpec& spec) {
  spec.output<int>("out");
}

void PingTxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
  auto value = index_++;
  op_output.emit(value, "out");
}

}  // namespace holoscan::ops
```
- The "out" port of the **PingTxOp** has the type `int` (line `6`).
- An integer is published to the "out" port when calling `emit()` (line `11`).
- The message received by the downstream **PingMxOp** operator when it calls `op_input.receive<int>()` has the type `int`.
````

````{tab-item} Python
```{code-block} python
:linenos: true
:emphasize-lines: 14
:name: holoscan-one-operator-workflow-python

class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits an integer to the "out" port.
    """

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1
```
- No special consideration is necessary for the Python version, we simply call `emit()` and pass the integer object (line `14`).
````
`````

:::{attention}
For advanced use cases, e.g., when writing C++ applications where you need interoperability between C++ native and GXF operators, you will need to use the `holoscan::TensorMap` type instead. See {ref}`interoperability-with-gxf-operators-cpp` for more details. If you are writing a Python application which needs a mixture of Python wrapped C++ operators and native Python operators, see {ref}`interoperability-with-wrapped-operators-python`.
:::

## Running the Application

Running the application should give you the following output in your terminal:

```
Middle message value: 1
Rx message value: 3
Middle message value: 2
Rx message value: 6
Middle message value: 3
Rx message value: 9
Middle message value: 4
Rx message value: 12
Middle message value: 5
Rx message value: 15
Middle message value: 6
Rx message value: 18
Middle message value: 7
Rx message value: 21
Middle message value: 8
Rx message value: 24
Middle message value: 9
Rx message value: 27
Middle message value: 10
Rx message value: 30
```
