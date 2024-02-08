(hello-world-example)=
# Hello World

For our first example, we look at how to create a Hello World example using the Holoscan SDK.

In this example we'll cover:

- how to define your application class
- how to define a one-operator workflow
- how to use a `CountCondition` to limit the number of times an operator is executed

:::{note}
The example source code and run instructions can be found in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) directory on GitHub, or under `/opt/nvidia/holoscan/examples` in the NGC container and the debian package, alongside their executables.
:::

## Defining the HelloWorldApp class

_For more details, see the {ref}`defining-an-application-class` section._

We define the `HelloWorldApp` class that inherits from holoscan's `Application` base class. An instance of the application is created in `main`. The `run()` method will then start the application.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 26
:emphasize-lines: 1, 15, 16
:name: holoscan-hello-world-app-cpp

class HelloWorldApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators, allowing the hello operator to execute once
    auto hello = make_operator<ops::HelloWorldOp>("hello", make_condition<CountCondition>(1));

    // Define the workflow by adding operator into the graph
    add_operator(hello);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<HelloWorldApp>();
  app->run();

  return 0;
}
```
````
````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 21
:emphasize-lines: 1, 11, 12
:name: holoscan-hello-world-app-python

class HelloWorldApp(Application):
    def compose(self):
        # Define the operators
        hello = HelloWorldOp(self, CountCondition(self, 1), name="hello")

        # Define the one-operator workflow
        self.add_operator(hello)

def main():
    app = HelloWorldApp()
    app.run()

if __name__ == "__main__":
    main()
```
````
`````

## Defining the HelloWorldApp workflow

_For more details, see the {ref}`application-workflows` section._

When defining your application class, the primary task is to define the operators used in your application and the interconnectivity between them to define the application workflow. The `HelloWorldApp` uses the simplest form of a workflow which consists of a single operator: `HelloWorldOp`.

For the sake of this first example, we will ignore the details of defining a custom operator to focus on the highlighted information below: when this operator runs (`compute`), it will print out `Hello World!` to the standard output:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 6
:emphasize-lines: 1, 13
:name: holoscan-one-operator-workflow-cpp

class HelloWorldOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HelloWorldOp)

  HelloWorldOp() = default;

  void setup(OperatorSpec& spec) override {
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::cout << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::cout << std::endl;
  }
};
```
````
````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 4
:emphasize-lines: 1, 14
:name: holoscan-one-operator-workflow-python

class HelloWorldOp(Operator):
    """Simple hello world operator.

    This operator has no ports.

    On each tick, this operator prints out hello world.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("")
        print("Hello World!")
        print("")
```
````
`````

Defining the application workflow occurs within the application's `compose()` method. In there, we first create an instance of the `HelloWorldOp` operator defined above, then add it to our simple workflow using `add_operator()`.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:lineno-start: 26
:emphasize-lines: 3, 7, 10
:name: holoscan-hello-world-app-cpp

class HelloWorldApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators, allowing the hello operator to execute once
    auto hello = make_operator<ops::HelloWorldOp>("hello", make_condition<CountCondition>(1));

    // Define the workflow by adding operator into the graph
    add_operator(hello);
  }
};
```
````
````{tab-item} Python
```{code-block} python
:linenos: true
:lineno-start: 21
:emphasize-lines: 2, 4, 7
:name: holoscan-hello-world-app-python

class HelloWorldApp(Application):
    def compose(self):
        # Define the operators
        hello = HelloWorldOp(self, CountCondition(self, 1), name="hello")

        # Define the one-operator workflow
        self.add_operator(hello)
```
````
`````

Holoscan applications deal with streaming data, so an operator's `compute()` method will be called continuously until
some situation arises that causes the operator to stop. For our Hello World example, we want to execute the operator only once. We can impose such a condition by passing a `CountCondition` object as an argument to the operator's constructor.

_For more details, see the {ref}`configuring-app-operator-conditions` section._

## Running the Application

Running the application should give you the following output in your terminal:

```
Hello World!
```

Congratulations! You have successfully run your first Holoscan SDK application!
