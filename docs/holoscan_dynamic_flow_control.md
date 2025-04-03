(holoscan-dynamic-flow-control)=
# Dynamic Flow Control

Dynamic Flow Control is a feature introduced in Holoscan SDK v3.0 that allows operators to modify their connections with other operators at runtime. This enables creating complex workflows with conditional branching, loops, and dynamic routing patterns.

## Overview

Traditional static workflows in Holoscan define fixed connections between operators at application composition time. Dynamic Flow Control extends this by allowing operators to:

- Modify their connections during execution
- Route data conditionally to different operators
- Create loops and iterative patterns
- Implement complex branching logic

Common use cases include:
- Conditional processing pipelines
- Adaptive workflow routing
- Iterative processing with dynamic termination
- Error handling and recovery flows

## Quick Start

Here's a simple example to get started with Dynamic Flow Control in Holoscan:

### 1. Basic Sequential Flow

The simplest use of dynamic flow control is creating a sequential chain of operators that doesn't have any input or output ports:

`````{tab-set}
````{tab-item} C++
```cpp
#include <holoscan/holoscan.hpp>

// Define a simple operator
class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)
  SimpleOp() = default;

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    // Simple computation
    HOLOSCAN_LOG_INFO("Executing {}", name());
  }
};

// Define the application
class SimpleSequentialApp : public holoscan::Application {
 public:
  void compose() override {
    // Create operators
    auto op1 = make_operator<SimpleOp>("op1");
    auto op2 = make_operator<SimpleOp>("op2");

    // Connect operators sequentially
    add_flow(start_op(), op1);  // Start with op1
    add_flow(op1, op2);         // Then op2
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<SimpleSequentialApp>();
  app->run();
  return 0;
}
```
````

````{tab-item} Python
```python
from holoscan.core import Application, Operator


class SimpleOp(Operator):
    def compute(self, op_input, op_output, context):
        # Simple computation
        print(f"Executing {self.name}")


class SimpleSequentialApp(Application):
    def compose(self):
        # Create operators
        op1 = SimpleOp(self, name="op1")
        op2 = SimpleOp(self, name="op2")

        # Connect operators sequentially
        self.add_flow(self.start_op(), op1)  # Start with op1
        self.add_flow(op1, op2)  # Then op2


def main():
    app = SimpleSequentialApp()
    app.run()


if __name__ == "__main__":
    main()
```
````
`````

### 2. Basic Conditional Flow

Here's a simple example of conditional routing between operators:

`````{tab-set}
````{tab-item} C++
```cpp
#include <holoscan/holoscan.hpp>

// Define a simple operator with a value
class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {}

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    value_++;  // Increment value each time
    HOLOSCAN_LOG_INFO("Executing {} with value {}", name(), value_);
  }

  int get_value() const { return value_; }

 private:
  int value_ = 0;
};

// Define the application
class SimpleConditionalApp : public holoscan::Application {
 public:
  void compose() override {
    // Create operators
    auto op1 = make_operator<SimpleOp>("op1", make_condition<holoscan::CountCondition>(3));
    auto path_a = make_operator<SimpleOp>("path_a");
    auto path_b = make_operator<SimpleOp>("path_b");

    // Define possible flows
    add_flow(op1, path_a);
    add_flow(op1, path_b);

    // Set dynamic flow based on condition
    set_dynamic_flows(op1, [path_a, path_b](const std::shared_ptr<holoscan::Operator>& op) {
      auto simple_op = std::static_pointer_cast<SimpleOp>(op);
      if (simple_op->get_value() % 2 == 0) {
        simple_op->add_dynamic_flow(path_a);  // Even values go to path_a
      } else {
        simple_op->add_dynamic_flow(path_b);  // Odd values go to path_b
      }
    });
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<SimpleConditionalApp>();
  app->run();
  return 0;
}
```
````

````{tab-item} Python
```python
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator


class SimpleOp(Operator):
    def __init__(self, *args, **kwargs):
        self.value = 0
        super().__init__(*args, **kwargs)

    def compute(self, op_input, op_output, context):
        self.value += 1  # Increment value each time
        print(f"Executing {self.name} with value {self.value}")


class SimpleConditionalApp(Application):
    def compose(self):
        # Create operators
        op1 = SimpleOp(self, CountCondition(self, count=3), name="op1")
        path_a = SimpleOp(self, name="path_a")
        path_b = SimpleOp(self, name="path_b")

        # Define possible flows
        self.add_flow(op1, path_a)
        self.add_flow(op1, path_b)

        # Set dynamic flow based on condition
        def route_flow(op):
            if op.value % 2 == 0:
                op.add_dynamic_flow(path_a)  # Even values go to path_a
            else:
                op.add_dynamic_flow(path_b)  # Odd values go to path_b

        self.set_dynamic_flows(op1, route_flow)


def main():
    app = SimpleConditionalApp()
    app.run()


if __name__ == "__main__":
    main()
```
````
`````

### Key Points to Remember:

1. Optionally use `start_op()` ({cpp:func}`C++ <holoscan::Fragment::start_op>`/{py:func}`Python <holoscan.core.Fragment.start_op>`) to get the initial operator in your flow
2. Connect operators with `add_flow()`
3. Use `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`) to define runtime routing logic
4. Implement flow control logic in the callback function passed to `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`)
  - The callback function takes an operator as input and returns void
  - The callback function can add dynamic flows using the operator's `add_dynamic_flow()` ({cpp:func}`C++ <holoscan::Operator::add_dynamic_flow>`/{py:func}`Python <holoscan.core.Operator.add_dynamic_flow>`) methods

For more complex patterns and detailed explanations, see the sections below.

## Key Concepts

### Input and Output Execution Ports

Before Holoscan SDK v3.0, operators needed input and output ports to be connected via `add_flow()` and there is no way to specify the execution dependency if the operator does not have any input or output ports.

However, in some cases, the requirements were different:
- An 'execution order dependency' was needed instead of a 'data flow dependency'.
- Execution control was required rather than keeping a node running continuously.
- The pipeline should run only once unless explicitly specified to loop.

To address these needs, Holoscan SDK v3.0 introduced implicit input/output 'execution ports' (`__input_exec__` / `__output_exec__`), inspired by Unreal Engine's [Blueprints](https://www.unrealengine.com/en-US/blog/introduction-to-blueprints) (particularly [execution pins](https://forums.unrealengine.com/t/execution-pins/226127)).

The output execution port (`__output_exec__`. `holoscan::Operator::kOutputExecPortName` in C++ and `holoscan.core.Operator.OUTPUT_EXEC_PORT_NAME` in Python) of a source operator and the input execution port (`__input_exec__`, `holoscan::Operator::kInputExecPortName` in C++ and `holoscan.core.Operator.INPUT_EXEC_PORT_NAME` in Python) of a target operator are implicitly added when **both** of the following are true:
- Two operators are connected using `add_flow()` without specifying a port map.
- The target operator does not have an explicit input port.

:::{note}
Both the source and target operators must be native Holoscan operators.
Attempting to connect a {ref}`GXF Operator<wrap-gxf-codelet-as-operator>` to a native Holoscan operator, or vice versa, with an empty port map will result in an error.
:::

During execution, after the source operator's `compute()` method is called, the Holoscan executor emits an empty message (`Entity`) to the implicit output execution port as a signal. This can then trigger the target operator’s execution, as the Holoscan executor attaches a `MessageAvailableCondition` to the target operator’s implicit input execution port. Before the target operator’s `compute()` method runs, the executor collects (pops) all messages from the implicit input execution port, enabling execution dependencies without requiring explicit input and output execution ports.

Starting with Holoscan SDK v3.0, operators can be connected via `add_flow()` without the need for explicit input and output execution ports, allowing for more flexible and dynamic operator connections.

### Start Operator

In Holoscan, when the workflow graph is executed, root operators who do not have any input ports are first executed, and unless any condition is specified to the root operator (such as `CountCondition` or `PeriodicCondition`), it will execute continuously.

Inspired by [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup)'s start node (langgraph.graph.START), Holoscan SDK v3.0 introduces a new concept of the **start operator**.

The **start operator** is the first operator in an application fragment, serving as the entry point to the workflow. It is simply the first operator added to the fragment.

This operator is named `<|start|>` and has a condition of `CountCondition(1)`, ensuring it executes only once. Other entry operators that initiate fragment execution should connect to this operator.

In Holoscan, you can retrieve the start operator by calling `start_op()` ({cpp:func}`C++ <holoscan::Fragment::start_op>`/{py:func}`Python <holoscan.core.Fragment.start_op>`) within the `compose()` method. If this method is called multiple times, it will return the same start operator

This API is available in both C++ and Python (see [flow_control/sequential](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/flow_control/sequential) for an example):

`````{tab-set}
````{tab-item} C++
```cpp
class SequentialExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1");
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");

    // Define the three-operator workflow
    add_flow(start_op(), node1);
    add_flow(node1, node2);
    add_flow(node2, node3);
  }
};
```
````

````{tab-item} Python
```python
class SequentialExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")

        # Define the three-operator workflow
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
```
````
`````

In this example, the start operator is connected to `node1`, making `node1` the first operator to execute. Since the start operator has a condition of `CountCondition(1)`, it will only trigger once, ensuring `node1` runs a single time.

In this example, each node (operator) is executed sequentially and executed only once. The `start_op()` method retrieves the start operator, which is connected to `node1`. This makes `node1` the first operator to execute. After `node1` completes its execution, `node2` is triggered, followed by `node3`. The `CountCondition(1)` in the `start_op()` method ensures that each operator in the sequence runs a single time, maintaining a clear and predictable flow of execution.

### Setting Dynamic Flows

The `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`) method allows for dynamic flow control in a Holoscan application. This method sets a callback function that determines the flow of execution based on the state of the operator at runtime.

In the example from [flow_control/conditional](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/flow_control/conditional), the `set_dynamic_flows()` method is used to dynamically control the flow between `node1`, `node2`, and `node4` based on the value of `node1`. The callback function checks the value of `node1` and adds a dynamic flow to either `node2` or `node4`:

```
 Node Graph:

       node1 (launch twice)
       /   \
   node2   node4
     |       |
   node3   node5
```


`````{tab-set}
````{tab-item} C++
```cpp
class ConditionalExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1", make_condition<CountCondition>(2));
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");
    auto node4 = make_operator<SimpleOp>("node4");
    auto node5 = make_operator<SimpleOp>("node5");

    add_flow(node1, node2);
    add_flow(node2, node3);
    add_flow(node1, node4);
    add_flow(node4, node5);

    set_dynamic_flows(node1, [node2, node4](const std::shared_ptr<Operator>& op) {
      auto simple_op = std::static_pointer_cast<SimpleOp>(op);
      if (simple_op->get_value() % 2 == 1) {
        simple_op->add_dynamic_flow(node2);
      } else {
        simple_op->add_dynamic_flow(node4);
      }
    });
  }
};

````

````{tab-item} Python
```python
class ConditionalExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, CountCondition(self, count=2), name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")
        node4 = SimpleOp(self, name="node4")
        node5 = SimpleOp(self, name="node5")

        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
        self.add_flow(node1, node4)
        self.add_flow(node4, node5)

        def dynamic_flow_callback(op):
            if op.value % 2 == 1:
                op.add_dynamic_flow(node2)
            else:
                op.add_dynamic_flow(node4)

        self.set_dynamic_flows(node1, dynamic_flow_callback)
```
````
`````

In the above example, the `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`) methods are used to define and manage dynamic workflows in the application.

The `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`) method takes an operator and a callback function as arguments. The callback function is called with the operator as an argument.
Inside the callback function, the `add_dynamic_flow()` ({cpp:func}`C++ <holoscan::Operator::add_dynamic_flow>`/{py:func}`Python <holoscan.core.Operator.add_dynamic_flow>`) method is used to add a dynamic flow to the operator.

In the example, the callback function checks the value of `node1` and adds a dynamic flow to either `node2` or `node4`.

The `add_dynamic_flow()` ({cpp:func}`C++ <holoscan::Operator::add_dynamic_flow>`/{py:func}`Python <holoscan.core.Operator.add_dynamic_flow>`) method has several overloads to support different ways of adding dynamic flows:

`````{tab-set}
````{tab-item} C++
```cpp
/// Basic connection using default output port. This is the simplest form for connecting
/// two operators when you only need to specify the destination.
void add_dynamic_flow(const std::shared_ptr<Operator>& next_op,
                     const std::string& next_input_port = "");

/// Connection with explicit output port specification. Use this when the source operator
/// has multiple output ports and you need to specify which one to use.
void add_dynamic_flow(const std::string& curr_output_port,
                     const std::shared_ptr<Operator>& next_op,
                     const std::string& next_input_port = "");

/// Connection using a FlowInfo object, which encapsulates all connection details including:
/// - Source operator and its output port specification
/// - Destination operator and its input port specification
/// - Port names and associated IOSpecs
void add_dynamic_flow(const std::shared_ptr<FlowInfo>& flow);

/// Batch connection using multiple FlowInfo objects. Use this to set up multiple
/// connections in a single call, which is more efficient than making multiple
/// individual connections.
void add_dynamic_flow(const std::vector<std::shared_ptr<FlowInfo>>& flows);
```
````

````{tab-item} Python
```python

# 1. Basic connection using default output port. This is the simplest form for connecting
#    two operators when you only need to specify the destination.
op.add_dynamic_flow(next_op: Operator, next_input_port_name: str = '')

# 2. Connection with explicit output port specification. Use this when the source operator
#    has multiple output ports and you need to specify which one to use.
op.add_dynamic_flow(curr_output_port_name: str, next_op: Operator, next_input_port_name: str = '')

# 3. Connection using a FlowInfo object, which encapsulates all connection details including:
#   - Source operator and its output port specification
#   - Destination operator and its input port specification
#   - Port names and associated IOSpecs
#
# This is useful for complex connections or when reusing connection patterns.
op.add_dynamic_flow(flow: FlowInfo)

# 4. Batch connection using multiple FlowInfo objects. Use this to set up multiple
#    connections in a single call, which is more efficient than making multiple
#    individual connections.
op.add_dynamic_flow(flows: list[FlowInfo])
```
````
`````

The simple form of `add_dynamic_flow()` ({cpp:func}`C++ <holoscan::Operator::add_dynamic_flow>`/{py:func}`Python <holoscan.core.Operator.add_dynamic_flow>`) is passing just the next operator (and optionally the next input port name).

If the next operator does not have any explicit input, you can omit the next input port name.
In this case, current operator's implicit output execution port will be connected to the next operator's implicit input execution port.

### Flow Information

The `FlowInfo` ({cpp:class}`C++ <holoscan::Operator::FlowInfo>`/{py:class}`Python <holoscan.core.FlowInfo>`) class represents information about a connection between operators and takes the following arguments in the constructor:
- `curr_operator`: The source operator of the flow connection
- `curr_output_port`: The name of the output port on the source operator
- `next_operator`: The destination operator of the flow connection
- `next_input_port`: The name of the input port on the destination operator

Inside the callback function, you can use the `find_flow_info()` ({cpp:func}`C++ <holoscan::Operator::find_flow_info>`/{py:func}`Python <holoscan.core.Operator.find_flow_info>`) method and `find_all_flow_info()` ({cpp:func}`C++ <holoscan::Operator::find_all_flow_info>`/{py:func}`Python <holoscan.core.Operator.find_all_flow_info>`) method to find the `FlowInfo` object(s) that matches the predicate.

The following example shows how to find the `FlowInfo` object(s) that matches the predicate:

`````{tab-set}
````{tab-item} C++
```cpp
class ConditionalExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1", make_condition<CountCondition>(2));
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");
    auto node4 = make_operator<SimpleOp>("node4");
    auto node5 = make_operator<SimpleOp>("node5");

    add_flow(node1, node2);
    add_flow(node2, node3);
    add_flow(node1, node4);
    add_flow(node4, node5);

    // // If you want to add all the next flows, you can use the following code:
    // set_dynamic_flows(
    //     node1, [](const std::shared_ptr<Operator>& op) { op->add_dynamic_flow(op->next_flows());
    //     });

    set_dynamic_flows(node1, [](const std::shared_ptr<Operator>& op) {
      auto simple_op = std::static_pointer_cast<SimpleOp>(op);
      static const auto& node2_flow = op->find_flow_info(
          [](const auto& flow) { return flow->next_operator->name() == "node2"; });
      static const auto& node4_flow = op->find_flow_info(
          [](const auto& flow) { return flow->next_operator->name() == "node4"; });
      //static const auto& all_next_flows = op->find_all_flow_info(
      //    [](const auto& flow) { return true; });

      //std::cout << "All next flows: ";
      //for (const auto& flow : all_next_flows) {
      //  std::cout << flow->next_operator->name() << " ";
      //}
      //std::cout << std::endl;

      if (simple_op->get_value() % 2 == 1) {
        simple_op->add_dynamic_flow(node2_flow);
      } else {
        simple_op->add_dynamic_flow(node4_flow);
      }
    });
  }
};
```
````

````{tab-item} Python
```python
class ConditionalExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, CountCondition(self, count=2), name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")
        node4 = SimpleOp(self, name="node4")
        node5 = SimpleOp(self, name="node5")

        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
        self.add_flow(node1, node4)
        self.add_flow(node4, node5)

        # # If you want to add all the next flows, you can use the following code:
        # self.set_dynamic_flows(node1, lambda op: op.add_dynamic_flow(op.next_flows))

        # This is another way to add dynamic flows based on the next operator name
        def dynamic_flow_callback(op):
            node2_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node2")
            node4_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node4")

            # all_next_flows = op.find_all_flow_info(lambda flow: True)
            # print(f"All next flows: {[flow.next_operator.name for flow in all_next_flows]}")

            if op.value % 2 == 1:
                op.add_dynamic_flow(node2_flow)
            else:
                op.add_dynamic_flow(node4_flow)

        self.set_dynamic_flows(node1, dynamic_flow_callback)
```
````
`````

In the above example, instead of using `op.add_dynamic_flow(node2)` or `op.add_dynamic_flow(node4)`, we use `op.add_dynamic_flow(node2_flow)` or `op.add_dynamic_flow(node4_flow)`.
And the `node2_flow` and `node4_flow` are `FlowInfo` objects that are found using the `find_flow_info()` method.

The `find_flow_info()` ({cpp:func}`C++ <holoscan::Operator::find_flow_info>`/{py:func}`Python <holoscan.core.Operator.find_flow_info>`) method takes a predicate as an argument and returns a `FlowInfo` object that matches the predicate.

The `find_all_flow_info()` ({cpp:func}`C++ <holoscan::Operator::find_all_flow_info>`/{py:func}`Python <holoscan.core.Operator.find_all_flow_info>`) method takes a predicate as an argument and returns a vector (list) of `FlowInfo` objects that match the predicate.

If you want to get a vector of all the next flows, you can use `op->next_flows()` in C++ or `op.next_flows` in Python.

## Flow Control Pattern Selection Guide

Here's when to choose different flow control patterns:

**start_op() + Cyclic Flow**
- Best for: Dynamic routing, feedback loops, runtime-adaptive flows
- Use when: Flow patterns depend on data content or need to change during execution
- Advantages: Flexible, handles complex routing
- Trade-offs: More complex to debug, slightly higher runtime overhead

**Generator (root operator) with condition (CountCondition, PeriodicCondition, etc.)**
- Best for: Fixed iteration counts, simple linear flows
- Use when: Number of iterations is known in advance (or infinite), static flow patterns
- Advantages: Simple to implement, better performance, easier to debug
- Trade-offs: Less flexible, cannot adapt to runtime conditions

## Examples

[Please see full examples](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/flow_control) under the `examples/flow_control` folder in the Holoscan SDK repository for more detailed implementations and use cases.

Note that the execution control examples are also related to the dynamic behavior of operators, and are available in the [execution_control](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/execution_control) directory.

## Best Practices

1. **Clear Flow Logic**: Keep dynamic flow logic clear and well-documented. Use meaningful names for operators and document the conditions that trigger different flow paths.

2. **Error Handling**: Handle edge cases in flow callbacks. Consider what happens if expected operators are not available or if flow conditions are invalid.

3. **State Management**: Be careful with shared state in dynamic flows. Ensure thread safety when multiple operators access shared resources.

4. **Performance**: Consider the overhead of frequent flow changes. Cache flow information when possible and avoid unnecessary flow modifications.

5. **Testing**: Test all possible flow paths thoroughly. Create unit tests that verify both normal operation and edge cases for each flow pattern.

## Limitations

The dynamic flow control feature has the following limitations:

- It can only be used to connect operators within the same fragment. For inter-fragment flows (connecting operators across different fragments), explicit non-execution ports must be used instead of dynamic flows.
- Both the source and target operators must be native Holoscan operators (not {ref}`GXF Operators<wrap-gxf-codelet-as-operator>`).