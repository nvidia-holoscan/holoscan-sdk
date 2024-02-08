(ping-multi-port-example)=
# Ping Multi Port

In this section, we look at how to create an application with a more complex workflow where
operators may have multiple input/output ports that send/receive a user-defined data type.

In this example we will cover:

- how to send/receive messages with a custom data type
- how to add a port that can receive any number of inputs

:::{note}
The example source code and run instructions can be found in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) directory on GitHub, or under `/opt/nvidia/holoscan/examples` in the NGC container and the debian package, alongside their executables.
:::

## Operators and Workflow

Here is the diagram of the operators and workflow used in this example.

```{mermaid}
:align: center
:caption: A workflow with multiple inputs and outputs

%%{init: {"theme": "base", "themeVariables": { "fontSize": "16px"}} }%%

classDiagram
    direction LR

    PingTxOp --|> PingMxOp : out1...in1
    PingTxOp --|> PingMxOp : out2...in2
    PingMxOp --|> PingRxOp : out1...receivers
    PingMxOp --|> PingRxOp : out2...receivers

    class PingTxOp {
        out1(out) ValueData
        out2(out) ValueData
    }
    class PingMxOp {
        [in]in1 : ValueData
        [in]in2 : ValueData
        out1(out) ValueData
        out2(out) ValueData
    }
    class PingRxOp {
        [in]receivers : ValueData
    }
```

In this example, `PingTxOp` sends a stream of odd integers to the `out1` port, and even integers to the `out2` port. `PingMxOp` receives these values using `in1` and `in2` ports, multiplies them by a constant factor, then forwards them to a single port - `receivers` - on `PingRxOp`.

## User Defined Data Types

In the previous `ping` examples, the port types for our operators were integers, but the Holoscan SDK can send any arbitrary data type. In this example, we'll see how to configure
operators for our user-defined `ValueData` class.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 6, 16
:name: holoscan-one-operator-workflow-cpp

#include "holoscan/holoscan.hpp"

class ValueData {
 public:
  ValueData() = default;
  explicit ValueData(int value) : data_(value) {
    HOLOSCAN_LOG_TRACE("ValueData::ValueData(): {}", data_);
  }
  ~ValueData() { HOLOSCAN_LOG_TRACE("ValueData::~ValueData(): {}", data_); }

  void data(int value) { data_ = value; }

  int data() const { return data_; }

 private:
  int data_;
};
```
The `ValueData` class wraps a simple integer (line `6`, `16`), but could have been arbitrarily complex.

:::{note}
The `HOLOSCAN_LOG_<LEVEL>()` macros can be used for logging with fmtlib syntax (lines `7`, `9` above) as demonstrated across this example. See the {ref}`Logging<holoscan-logging>` section for more details.
:::

````
````{tab-item} Python
```{code-block} python
:linenos: true
:name: holoscan-one-operator-workflow-python

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

class ValueData:
    """Example of a custom Python class"""

    def __init__(self, value):
        self.data = value

    def __repr__(self):
        return f"ValueData({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)
```
The `ValueData` class is a simple wrapper, but could have been arbitrarily complex.
````
`````

## Defining an Explicit Number of Inputs and Outputs

After defining our custom `ValueData` class, we configure our operators' ports to send/receive messages of this type, similarly to the [previous example](./ping_custom_op.md#configuring-operator-input-and-output-ports).

This is the first operator - `PingTxOp` - sending `ValueData` objects on two ports, `out1` and `out2`:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 18
:emphasize-lines: 10, 11, 16, 19
:name: holoscan-ping-custom-op-app-cpp

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = std::make_shared<ValueData>(index_++);
    op_output.emit(value1, "out1");

    auto value2 = std::make_shared<ValueData>(index_++);
    op_output.emit(value2, "out2");
  };
  int index_ = 1;
};
```
- We configure the output ports with the `ValueData` type on lines `27` and `28` using `spec.output<std::shared_ptr<ValueData>>()`. Therefore, the data type for the output ports is an object to a shared pointer to a `ValueData` object.
- The values are then sent out using `op_output.emit()` on lines `33` and `36`. The port name is required since there is more than one port on this operator.

:::{note}
Data types of the output ports are shared pointers (`std::shared_ptr`), hence the call to `std::make_shared<ValueData>(...)` on lines `32` and `35`.
:::

````

````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 19
:emphasize-lines: 17, 18, 23, 27
:name: holoscan-ping-custom-op-app-python

class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has:
        outputs: "out1", "out2"

    On each tick, it transmits a `ValueData` object at each port. The
    transmitted values are even on port1 and odd on port2 and increment with
    each call to compute.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        value1 = ValueData(self.index)
        self.index += 1
        op_output.emit(value1, "out1")

        value2 = ValueData(self.index)
        self.index += 1
        op_output.emit(value2, "out2")
```
- We configure the output ports on lines `35` and `36` using `spec.output()`. There is no need to reference the type (`ValueData`) in Python.
- The values are then sent out using `op_output.emit()` on lines `41` and `45`.
````
`````

We then configure the middle operator - `PingMxOp` - to receive that data on ports `in1` and `in2`:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 40
:emphasize-lines: 8, 9, 16, 17
:name: holoscan-ping-custom-op-app-cpp

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<ValueData>>("in1");
    spec.input<std::shared_ptr<ValueData>>("in2");
    spec.output<std::shared_ptr<ValueData>>("out1");
    spec.output<std::shared_ptr<ValueData>>("out2");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = op_input.receive<std::shared_ptr<ValueData>>("in1").value();
    auto value2 = op_input.receive<std::shared_ptr<ValueData>>("in2").value();

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    HOLOSCAN_LOG_INFO("Middle message value1: {}", value1->data());
    HOLOSCAN_LOG_INFO("Middle message value2: {}", value2->data());

    // Multiply the values by the multiplier parameter
    value1->data(value1->data() * multiplier_);
    value2->data(value2->data() * multiplier_);

    op_output.emit(value1, "out1");
    op_output.emit(value2, "out2");
  };

 private:
  int count_ = 1;
  Parameter<int> multiplier_;
};
```
- We configure the input ports with the `std::shared_ptr<ValueData>` type on lines `47` and `48` using `spec.input<std::shared_ptr<ValueData>>()`.
- The values are received using `op_input.receive()` on lines `55` and `56` using the port names. The received values are of type `std::shared_ptr<ValueData>` as mentioned in the templated `receive()` method.
````

````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 46
:emphasize-lines: 16, 17, 23, 24
:name: holoscan-ping-custom-op-app-python

class PingMxOp(Operator):
    """Example of an operator modifying data.

    This operator has:
        inputs:  "in1", "in2"
        outputs: "out1", "out2"

    The data from each input is multiplied by a user-defined value.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out1")
        spec.output("out2")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        print(f"Middle message received (count: {self.count})")
        self.count += 1

        print(f"Middle message value1: {value1.data}")
        print(f"Middle message value2: {value2.data}")

        # Multiply the values by the multiplier parameter
        value1.data *= self.multiplier
        value2.data *= self.multiplier

        op_output.emit(value1, "out1")
        op_output.emit(value2, "out2")
```
Sending messages of arbitrary data types is pretty straightforward in Python.
The code to define the operator input ports (lines `61-62`), and to receive them (lines `68, 69`) did
not change when we went from passing `int` to `ValueData` objects.
````
`````

`PingMxOp` processes the data, then sends it out on two ports, similarly to what is done by `PingTxOp` above.


## Receiving Any Number of Inputs

In this workflow, `PingRxOp` has a single input port - `receivers` - that is connected to two upstream ports from `PingMxOp`. When an input port needs to connect to multiple upstream ports, we define it with `spec.param()` instead of `spec.input()`. The inputs are then stored in a vector, following the order they were added with `add_flow()`.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 75
:emphasize-lines: 8, 12-13, 22
:name: holoscan-ping-custom-op-app-cpp

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector =
        op_input.receive<std::vector<std::shared_ptr<ValueData>>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    HOLOSCAN_LOG_INFO("Rx message value1: {}", value_vector[0]->data());
    HOLOSCAN_LOG_INFO("Rx message value2: {}", value_vector[1]->data());
  };

 private:
  Parameter<std::vector<IOSpec*>> receivers_;
  int count_ = 1;
};

}  // namespace holoscan::ops
```
- In the operator's `setup()` method, we define a parameter `receivers` (line `82`) that is tied to the private data member `receivers_` (line `96`) of type `Parameter<std::vector<IOSpec*>>`.
- The values are retrieved using `op_input.receive<std::vector<std::shared_ptr<ValueData>>>(...)`.
- `value_vector`'s type is `std::vector<std::shared_ptr<ValueData>>` (lines `86-87`).
````

````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 82
:emphasize-lines: 16, 19
:name: holoscan-ping-custom-op-app-python

class PingRxOp(Operator):
    """Simple receiver operator.

    This operator has:
        input: "receivers"

    This is an example of a native operator that can dynamically have any
    number of inputs connected to is "receivers" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        print(f"Rx message received (count: {self.count}, size: {len(values)})")
        self.count += 1
        print(f"Rx message value1: {values[0].data}")
        print(f"Rx message value2: {values[1].data}")
```
- In Python, a port that can be connected to multiple upstream ports is created by
 defining a parameter and setting the argument `kind="receivers"` (line `97`).
- The call to `receive()` returns a tuple of `ValueData` objects (line `100`).
````
`````

The rest of the code creates the application, operators, and defines the workflow:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 100
:emphasize-lines: 12, 13
:name: holoscan-ping-custom-op-app-cpp

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto mx = make_operator<ops::PingMxOp>("mx", Arg("multiplier", 3));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Define the workflow
    add_flow(tx, mx, {{"out1", "in1"}, {"out2", "in2"}});
    add_flow(mx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
```
````

````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 105
:emphasize-lines: 9, 10
:name: holoscan-ping-custom-op-app-python

class MyPingApp(Application):
    def compose(self):
        # Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMxOp(self, name="mx", multiplier=3)
        rx = PingRxOp(self, name="rx")

        # Define the workflow
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
```
````
`````

- The operators `tx`, `mx`, and `rx` are created in the application's `compose()` similarly to previous examples.
- Since the operators in this example have multiple input/output ports, we need to specify the third, port name pair argument when calling `add_flow()`:
  - `tx/out1` is connected to `mx/in1`, and `tx/out2` is connected to `mx/in2`.
  - `mx/out1` and `mx/out2` are both connected to `rx/receivers`.

## Running the Application

Running the application should give you output similar to the following in your terminal.

```
[info] [gxf_executor.cpp:222] Creating context
[info] [gxf_executor.cpp:1531] Loading extensions from configs...
[info] [gxf_executor.cpp:1673] Activating Graph...
[info] [gxf_executor.cpp:1703] Running Graph...
[info] [gxf_executor.cpp:1705] Waiting for completion...
[info] [gxf_executor.cpp:1706] Graph execution waiting. Fragment:
[info] [greedy_scheduler.cpp:195] Scheduling 3 entities
[info] [ping_multi_port.cpp:80] Middle message received (count: 1)
[info] [ping_multi_port.cpp:82] Middle message value1: 1
[info] [ping_multi_port.cpp:83] Middle message value2: 2
[info] [ping_multi_port.cpp:112] Rx message received (count: 1, size: 2)
[info] [ping_multi_port.cpp:114] Rx message value1: 3
[info] [ping_multi_port.cpp:115] Rx message value2: 6
[info] [ping_multi_port.cpp:80] Middle message received (count: 2)
[info] [ping_multi_port.cpp:82] Middle message value1: 3
[info] [ping_multi_port.cpp:83] Middle message value2: 4
[info] [ping_multi_port.cpp:112] Rx message received (count: 2, size: 2)
[info] [ping_multi_port.cpp:114] Rx message value1: 9
[info] [ping_multi_port.cpp:115] Rx message value2: 12
...
[info] [ping_multi_port.cpp:114] Rx message value1: 51
[info] [ping_multi_port.cpp:115] Rx message value2: 54
[info] [ping_multi_port.cpp:80] Middle message received (count: 10)
[info] [ping_multi_port.cpp:82] Middle message value1: 19
[info] [ping_multi_port.cpp:83] Middle message value2: 20
[info] [ping_multi_port.cpp:112] Rx message received (count: 10, size: 2)
[info] [ping_multi_port.cpp:114] Rx message value1: 57
[info] [ping_multi_port.cpp:115] Rx message value2: 60
[info] [greedy_scheduler.cpp:374] Scheduler stopped: Some entities are waiting for execution, but there are no periodic or async entities to get out of the deadlock.
[info] [greedy_scheduler.cpp:403] Scheduler finished.
[info] [gxf_executor.cpp:1714] Graph execution deactivating. Fragment:
[info] [gxf_executor.cpp:1715] Deactivating Graph...
[info] [gxf_executor.cpp:1718] Graph execution finished. Fragment:
[info] [gxf_executor.cpp:241] Destroying context
```

:::{note}
Depending on your log level you may see more or fewer messages. The output above was generated using the default value of `INFO`.  
Refer to the {ref}`Logging<holoscan-logging>` section for more details on how to set the log level.
:::
