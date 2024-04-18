(ping-simple-example)=
# Ping Simple

Most applications will require more than one operator.  In this example, we will create two operators where one operator will produce and send data while the other operator will receive and print the data.  The code in this example makes use of the built-in **PingTxOp** and **PingRxOp** operators that are defined in the `holoscan::ops` namespace.

In this example we'll cover:

- how to use built-in operators
- how to use `add_flow()` to connect operators together

:::{note}
The example source code and run instructions can be found in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) directory on GitHub, or under `/opt/nvidia/holoscan/examples` in the NGC container and the debian package, alongside their executables.
:::

## Operators and Workflow

Here is a example workflow involving two operators that are connected linearly.

```{digraph} ping_simple
:align: center
:caption: A linear workflow

    rankdir="LR"

    node [shape=record];

    tx [label="PingTxOp| |out(out) : int"];
    rx [label="PingRxOp|[in]in : int | "];
    tx -> rx [label="out...in"]
```

In this example, the source operator **PingTxOp** produces integers from 1 to 10 and passes it to the sink operator **PingRxOp** which prints the integers to standard output.

## Connecting Operators

We can connect two operators by calling `add_flow()` ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) in the application's `compose()` method.


The `add_flow()` method ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) takes the source operator, the destination operator, and the optional port name pairs.
The port name pair is used to connect the output port of the source operator to the input port of the destination operator.
The first element of the pair is the output port name of the upstream operator and the second element is the input port name of the downstream operator.
An empty port name ("") can be used for specifying a port name if the operator has only one input/output port.
If there is only one output port in the upstream operator and only one input port in the downstream operator, the port pairs can be omitted.

The following code shows how to define a linear workflow in the `compose()` method for our example.  Note that when an operator appears
in an `add_flow()` statement, it doesn't need to be added into the workflow separately using `add_operator()`.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 2-3, 10-11, 14
:name: holoscan-one-operator-workflow-cpp

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // Create the tx and rx operators
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Connect the operators into the workflow:  tx -> rx
    add_flow(tx, rx);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
```
- The header files that define **PingTxOp** and **PingRxOp** are included on lines `2` and `3` respectively.
- We create an instance of the **PingTxOp** using the `make_operator()` function (line `10`) with the name "tx" and
  constrain it's `compute()` method to execute 10 times.
- We create an instance of the **PingRxOp** using the `make_operator()` function (line `11`) with the name "rx".
- The tx and rx operators are connected using `add_flow()` (line `14`)
````
````{tab-item} Python
```{code-block} python
:linenos: true
:emphasize-lines: 3, 8, 9, 12
:name: holoscan-one-operator-workflow-python

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp

class MyPingApp(Application):
    def compose(self):
        # Create the tx and rx operators
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
```
- The built-in holoscan operators, **PingRxOp** and **PingTxOp**, are imported on line `3`.
- We create an instance of the **PingTxOp** operator with the name "tx" and constrain it's `compute()` method to execute 10 times (line `8`).
- We create an instance of the **PingRxOp** operator with the name "rx" (line `9`).
- The tx and rx operators are connected using `add_flow()` which defines this application's workflow (line `12`).
````
`````

## Running the Application

Running the application should give you the following output in your terminal:

```
Rx message value: 1
Rx message value: 2
Rx message value: 3
Rx message value: 4
Rx message value: 5
Rx message value: 6
Rx message value: 7
Rx message value: 8
Rx message value: 9
Rx message value: 10
```
