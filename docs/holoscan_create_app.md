(creating-holoscan-application)=

# Creating an Application

In this section, we'll address:
- How to {ref}`define an Application class<defining-an-application-class>`.
- How to {ref}`configure an Application<configuring-an-application>`.
- How to {ref}`define different types of workflows<application-workflows>`.
- How to [build and run your application](#building-and-running-your-application).

:::{note}
This section covers basics of applications running as a single fragment. For multi-fragment applications, refer to the [distributed application documentation](./holoscan_create_distributed_app.md).
:::

(defining-an-application-class)=

## Defining an Application Class

The following code snippet shows an example Application code skeleton:

`````{tab-set}
````{tab-item} C++
- We define the `App` class that inherits from the {cpp:class}`Application <holoscan::Application>` base class.
- We create an instance of the `App` class in `main()` using the {cpp:func}`make_application() <holoscan::make_application>` function.
- The {cpp:func}`run()<holoscan::Fragment::run>` method starts the application which will execute its {cpp:func}`compose()<holoscan::Fragment::compose>` method where the custom workflow will be defined.

```{code-block} cpp
:emphasize-lines: 3,5,12-13
:name: holoscan-app-skeleton-cpp

#include <holoscan/holoscan.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    // Define Operators and workflow
    //   ...
  }
};

int main() {
  auto app = holoscan::make_application<App>();
  app->run();
  return 0;
}
```
````
````{tab-item} Python
- We define the `App` class that inherits from the {py:class}`Application <holoscan.core.Application>` base class.
- We create an instance of the `App` class in a `main()` function that is called from `__main__`.
- The {py:func}`run()<holoscan.Application.run>` method starts the application which will execute its {py:func}`compose()<holoscan.Application.compose>` method where the custom workflow will be defined.

```{code-block} python
:emphasize-lines: 3, 5, 11-12
:name: holoscan-app-skeleton-python

from holoscan.core import Application

class App(Application):

    def compose(self):
        # Define Operators and workflow
        #   ...


def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()
```

:::{note}
It is recommended to call {py:func}`run()<holoscan.Application.run>` from within a separate `main()` function rather than calling it directly from `__main__`. This will ensure that the Application's destructor is called before the Python process exits.
:::

`````

:::{tip}
This is also illustrated in the [hello_world](./examples/hello_world.md) example.
:::
___

It is also possible to instead launch the application asynchronously (i.e., non-blocking for the thread launching the application), as shown below:

`````{tab-set}
````{tab-item} C++
This can be done simply by replacing the call to {cpp:func}`run()<holoscan::Fragment::run>` with {cpp:func}`run_async()<holoscan::Fragment::run_async>` which returns a `std::future`. Calling `future.get()` will block until the application has finished running and throw an exception if a runtime error occurred during execution.
```{code-block} cpp
:emphasize-lines: 3-4
:name: holoscan-app-skeleton-cpp-async

int main() {
  auto app = holoscan::make_application<App>();
  auto future = app->run_async();
  future.get();
  return 0;
}
```
````
````{tab-item} Python
This can be done simply by replacing the call to {py:func}`run()<holoscan.Application.run>` with {py:func}`run_async()<holoscan.Application.run_async>` which returns a Python `concurrent.futures.Future`. Calling `future.result()` will block until the application has finished running and raise an exception if a runtime error occurred during execution.
```{code-block} python
:emphasize-lines: 3-4
:name: holoscan-app-skeleton-python-async

def main():
    app = App()
    future = app.run_async()
    future.result()


if __name__ == "__main__":
    main()
````
`````

:::{tip}
This is also illustrated in the [ping_simple_run_async](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/ping_simple_run_async) example.
:::

(configuring-an-application)=

## Configuring an Application

An application can be configured at different levels:

1. {ref}`providing the GXF extensions that need to be loaded<loading-gxf-extensions>` (when using {ref}`GXF operators<wrap-gxf-codelet-as-operator>`).
2. configuring parameters for your application, including for:
   a. {ref}`the operators<configuring-app-operators>` in the workflow.
   b. {ref}`the scheduler<configuring-app-scheduler>` of your application.
3. {ref}`configuring some runtime properties<configuring-app-runtime>` when deploying for production.

The sections below will describe how to configure each of them, starting with a native support for YAML-based configuration for convenience.

(yaml-config-support)=
### YAML configuration support

Holoscan supports loading arbitrary parameters from a YAML configuration file at runtime, making it convenient to configure each item listed above, or other custom parameters you wish to add on top of the existing API. For C++ applications, it also provides the ability to change the behavior of your application without needing to recompile it.

:::{note}
Usage of the YAML utility is optional. Configurations can be hardcoded in your program, or done using any parser that you choose.
:::

Here is an example YAML configuration:

```yaml
string_param: "test"
float_param: 0.50
bool_param: true
dict_param:
  key_1: value_1
  key_2: value_2
```

Ingesting these parameters can be done using the two methods below:

`````{tab-set}
````{tab-item} C++

- The {cpp:func}`~holoscan::Fragment::config` method takes the path to the YAML configuration file. If the input path is relative, it will be relative to the current working directory.
- The {cpp:func}`~holoscan::Fragment::from_config` method returns an {cpp:class}`~holoscan::ArgList` object for a given key in the YAML file. It holds a list of {cpp:class}`~holoscan::Arg` objects, each of which holds a name (key) and a value.
  - If the `ArgList` object has only one `Arg` (when the key is pointing to a scalar item), it can be converted to the desired type using the {cpp:func}`~holoscan::ArgList::as` method by passing the type as an argument.
  - The key can be a dot-separated string to access nested fields.
- The {cpp:func}`~holoscan::Fragment::config_keys` method returns an unordered set of the key names accessible via {cpp:func}`~holoscan::Fragment::from_config`.

```{code-block} cpp
:name: holoscan-from-config-cpp

// Pass configuration file
auto app = holoscan::make_application<App>();
app->config("path/to/app_config.yaml");

// Scalars
auto string_param = app->from_config("string_param").as<std::string>();
auto float_param = app->from_config("float_param").as<float>();
auto bool_param = app->from_config("bool_param").as<bool>();

// Dict
auto dict_param = app->from_config("dict_param");
auto dict_nested_param = app->from_config("dict_param.key_1").as<std::string>();

// Print
std::cout << "string_param: " << string_param << std::endl;
std::cout << "float_param: " << float_param << std::endl;
std::cout << "bool_param: " << bool_param << std::endl;
std::cout << "dict_param:\n" << dict_param.description() << std::endl;
std::cout << "dict_param['key1']: " << dict_nested_param << std::endl;

// // Output
// string_param: test
// float_param: 0.5
// bool_param: 1
// dict_param:
// name: arglist
// args:
//   - name: key_1
//     type: YAML::Node
//     value: value_1
//   - name: key_2
//     type: YAML::Node
//     value: value_2
// dict_param['key1']: value_1
```
````

````{tab-item} Python

- The {py:func}`~holoscan.core.Fragment.config` method takes the path to the YAML configuration file. If the input path is relative, it will be relative to the current working directory.
- The {py:func}`~holoscan.core.Fragment.kwargs` method return a regular Python dict for a given key in the YAML file.
  - **Advanced**: this method wraps the {py:func}`~holoscan.core.Fragment.from_config` method similar to the C++ equivalent, which returns an {py:class}`~holoscan.core.ArgList` object if the key is pointing to a map item, or an {py:class}`~holoscan.core.Arg` object if the key is pointing to a scalar item. An {py:class}`~holoscan.core.Arg` object can be cast to the desired type (e.g., `str(app.from_config("string_param"))`).
- The {py:func}`~holoscan.core.Fragment.config_keys` method returns a set of the key names accessible via {py:func}`~holoscan.core.Fragment.from_config`.

```{code-block} python
:name: holoscan-from-config-python

# Pass configuration file
app = App()
app.config("path/to/app_config.yaml")

# Scalars
string_param = app.kwargs("string_param")["string_param"]
float_param = app.kwargs("float_param")["float_param"]
bool_param = app.kwargs("bool_param")["bool_param"]

# Dict
dict_param = app.kwargs("dict_param")
dict_nested_param = dict_param["key_1"]

# Print
print(f"string_param: {string_param}")
print(f"float_param: {float_param}")
print(f"bool_param: {bool_param}")
print(f"dict_param: {dict_param}")
print(f"dict_param['key_1']: {dict_nested_param}")

# # Output:
# string_param: test
# float_param: 0.5
# bool_param: True
# dict_param: {'key_1': 'value_1', 'key_2': 'value_2'}
# dict_param['key_1']: 'value_1'
```
:::{warning}
{py:func}`~holoscan.core.Fragment.from_config` cannot be used as inputs to the {py:mod}`built-in operators<holoscan.operators>` at this time. Therefore, it's recommended to use {py:func}`~holoscan.core.Fragment.kwargs` in Python.
:::
````
`````

:::{tip}
This is also illustrated in the [video_replayer](./examples/video_replayer.md) example.
:::

:::{attention}
With both `from_config` and `kwargs`, the returned `ArgList`/dictionary will include both the key and its associated item if that item value is a scalar. If the item is a map/dictionary itself, the input key is dropped, and the output will only hold the key/values from that item.
:::

(loading-gxf-extensions)=

### Loading GXF extensions

If you use operators that depend on GXF extensions for their implementations (known as {ref}`GXF operators<wrap-gxf-codelet-as-operator>`), the shared libraries (`.so`) of these extensions need to be dynamically loaded as plugins at runtime.

The SDK already automatically handles loading the required extensions for the [built-in operators](./holoscan_operators_extensions.md) in both C++ and Python, as well as common extensions (listed here). To load additional extensions for your own operators, you can use one of the following approach:

````{tab-set-code}
```{code-block} yaml
extensions:
  - libgxf_myextension1.so
  - /path/to/libgxf_myextension2.so
```
```{code-block} c++
auto app = holoscan::make_application<App>();
auto exts = {"libgxf_myextension1.so", "/path/to/libgxf_myextension2.so"};
for (auto& ext : exts) {
  app->executor().extension_manager()->load_extension(ext);
}
```
```{code-block} python
from holoscan.gxf import load_extensions
from holoscan.core import Application
app = Application()
context = app.executor.context_uint64
exts = ["libgxf_myextension1.so", "/path/to/libgxf_myextension2.so"]
load_extensions(context, exts)
```
````

:::{note}
To be discoverable, paths to these shared libraries need to either be absolute, relative to your working directory, installed in the `lib/gxf_extensions` folder of the holoscan package, or listed under the `HOLOSCAN_LIB_PATH` or `LD_LIBRARY_PATH` environment variables.
:::

Please see other examples in the [system tests](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/tests/system/loading_gxf_extension.cpp) in the Holoscan SDK repository.

(configuring-app-operators)=

### Configuring operators

Operators are defined in the `compose()` method of your application. They are not instantiated
(with the `initialize` method) until an application's `run()` method is called.

Operators have three type of fields which can be configured: parameters, conditions, and resources.

(configuring-app-operator-parameters)=
#### Configuring operator parameters

Operators could have parameters defined in their `setup` method to better control their behavior (see details when [creating your own operators](./holoscan_create_operator.md)). The snippet below would be the implementation of this method for a minimal operator named `MyOp`, that takes a string and a boolean as parameters; we'll ignore any extra details for the sake of this example:

````{tab-set-code}
```{code-block} c++
void setup(OperatorSpec& spec) override {
  spec.param(string_param_, "string_param");
  spec.param(bool_param_, "bool_param");
}
```
```{code-block} python
def setup(self, spec: OperatorSpec):
  spec.param("string_param")
  spec.param("bool_param")
  # Optional in python. Could define `self.<param_name>` instead in `def __init__`
```
````

:::{tip}
Given an instance of an operator class, you can print a human-readable description of its specification to inspect the parameter fields that can be configured on that operator class:

````{tab-set-code}
```{code-block} c++
std::cout << operator_object->spec()->description() << std::endl;
```
```{code-block} python
print(operator_object.spec)
```
````

:::

Given this YAML configuration:

```yaml
myop_param:
  string_param: "test"
  bool_param: true

bool_param: false # we'll use this later
```

We can configure an instance of the `MyOp` operator in the application's `compose` method like this:

````{tab-set-code}
```{code-block} c++
void compose() override {
  // Using YAML
  auto my_op1 = make_operator<MyOp>("my_op1", from_config("myop_param"));

  // Same as above
  auto my_op2 = make_operator<MyOp>("my_op2",
    Arg("string_param", std::string("test")), // can use Arg(key, value)...
    Arg("bool_param") = true                  // ... or Arg(key) = value
  );
}
```
```{code-block} python
def compose(self):
  # Using YAML
  my_op1 = MyOp(self, name="my_op1", **self.kwargs("myop_param"))

  # Same as above
  my_op2 = MyOp(self,
    name="my_op2",
    string_param="test",
    bool_param=True,
  )
```
````

:::{tip}
This is also illustrated in the [ping_custom_op](./examples/ping_custom_op.md) example.
:::

If multiple `ArgList` are provided with duplicate keys, the latest one overrides them:

````{tab-set-code}
```{code-block} c++
void compose() override {
  // Using YAML
  auto my_op1 = make_operator<MyOp>("my_op1",
    from_config("myop_param"),
    from_config("bool_param")
  );

  // Same as above
  auto my_op2 = make_operator<MyOp>("my_op2",
    Arg("string_param", "test"),
    Arg("bool_param") = true,
    Arg("bool_param") = false
  );

  // -> my_op `bool_param_` will be set to `false`
}
```
```{code-block} python
def compose(self):
  # Using YAML
  my_op1 = MyOp(self, name="my_op1",
    from_config("myop_param"),
    from_config("bool_param"),
  )

  # Note: We're using from_config above since we can't merge automatically with kwargs
  # as this would create duplicated keys. However, we recommend using kwargs in Python
  # to avoid limitations with wrapped operators, so the code below is preferred.

  # Same as above
  params = self.kwargs("myop_param").update(self.kwargs("bool_param"))
  my_op2 = MyOp(self, name="my_op2", params)

  # -> my_op `bool_param` will be set to `False`
```
````

(configuring-app-operator-conditions)=
#### Configuring operator conditions

By default, operators with no input ports will continuously run, while operators with input ports will run as long as they receive inputs (as they're configured with the [`MessageAvailableCondition`](./components/conditions.md#messageavailablecondition)).

To change that behavior, one or more other [conditions](./components/conditions.md)' classes can be passed to the constructor of an operator to define when it should execute.

For example, we set three conditions on this operator `my_op`:

````{tab-set-code}
```{code-block} c++
void compose() override {
  using namespace holoscan;
  using namespace std::chrono_literals;

  // Limit to 10 iterations
  auto c1 = make_condition<CountCondition>("my_count_condition", 10);

  // Wait at least 200 milliseconds between each execution
  auto c2 = make_condition<PeriodicCondition>("my_periodic_condition", 200ms);

  // Stop when the condition calls `disable_tick()`
  auto c3 = make_condition<BooleanCondition>("my_bool_condition");

  // Pass directly to the operator constructor
  auto my_op = make_operator<MyOp>("my_op", c1, c2, c3);
}
```
```{code-block} python
def compose(self):
  # Limit to 10 iterations
  c1 = CountCondition(self, 10, name="my_count_condition")

  # Wait at least 200 milliseconds between each execution
  c2 = PeriodicCondition(self, timedelta(milliseconds=200), name="my_periodic_condition")

  # Stop when the condition calls `disable_tick()`
  c3 = BooleanCondition(self, name="my_bool_condition")

  # Pass directly to the operator constructor
  my_op = MyOp(self, c1, c2, c3, name="my_op")
```
````

:::{tip}
This is also illustrated in the [conditions](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/conditions)' examples.
:::

:::{note}
You'll need to specify a unique name for the conditions if there are multiple conditions applied to an operator.
:::

(configuring-app-operator-resources)=
#### Configuring operator resources

Some [resources](./components/resources.md) can be passed to the operator's constructor, typically an [allocator](./components/resources.md#allocator) passed as a regular parameter.

For example:

````{tab-set-code}
```{code-block} c++
void compose() override {
  // Allocating memory pool of specific size on the GPU
  // ex: width * height * channels * channel size in bytes
  auto block_size = 640 * 480 * 4 * 2;
  auto p1 = make_resource<BlockMemoryPool>("my_pool1", 1, size, 1);

  // Provide unbounded memory pool
  auto p2 = make_condition<UnboundedAllocator>("my_pool2");

  // Pass to operator as parameters (name defined in operator setup)
  auto my_op = make_operator<MyOp>("my_op",
                                   Arg("pool1", p1),
                                   Arg("pool2", p2));
}
```
```{code-block} python
def compose(self):
  # Allocating memory pool of specific size on the GPU
  # ex: width * height * channels * channel size in bytes
  block_size = 640 * 480 * 4 * 2;
  p1 = BlockMemoryPool(self, name="my_pool1", storage_type=1, block_size=block_size, num_blocks=1)

  # Provide unbounded memory pool
  p2 = UnboundedAllocator(self, name="my_pool2")

  # Pass to operator as parameters (name defined in operator setup)
  auto my_op = MyOp(self, name="my_op", pool1=p1, pool2=p2)
```
````


(configuring-app-operator-native-resources)=

#### Native resource creation

The resources bundled with the SDK are wrapping an underlying GXF component. However, it is also possible to define a "native" resource without any need to create and wrap an underlying GXF component. Such a resource can also be passed conditionally to an operator in the same way as the resources created in the previous section.

For example:

`````{tab-set}
````{tab-item} C++
To create a native resource, implement a class that inherits from {cpp:class}`Resource <holoscan::Resource>`

```{code-block} cpp
namespace holoscan {

class MyNativeResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MyNativeResource, Resource)

  MyNativeResource() = default;

  // add any desired parameters in the setup method
  // (a single string parameter is shown here for illustration)
  void setup(ComponentSpec& spec) override {
    spec.param(message_, "message", "Message string", "Message String", std::string("test message"));
  }

  // add any user-defined methods (these could be called from an Operator's compute method)
  std::string message() { return message_.get(); }

 private:
  Parameter<std::string> message_;
};
}  // namespace: holoscan
```

The `setup` method can be used to define any parameters needed by the resource.

This resource can be used with a C++ operator, just like any other resource. For example, an operator could have a parameter holding a shared pointer to `MyNativeResource` as below.

```{code-block} cpp
private:

class MyOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MyOperator)

  MyOperator() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(message_resource_, "message_resource", "message resource",
               "resource printing a message");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_TRACE("MyOp::compute()");

    // get a resource based on its name (this assumes the app author named the resource "message_resource")
    auto res = resource<MyNativeResource>("message_resource");
    if (!res) {
      throw std::runtime_error("resource named 'message_resource' not found!");
    }

    // call a method on the retrieved resource class
    auto message = res->message();

  };

private:
    Parameter<std::shared_ptr<holoscan::MyNativeResource> message_resource_;
}
```
The `compute` method above demonstrates how the templated `resource` method can be used to retrieve a resource.


and the resource could be created and passed via a named argument in the usual way
```{code-block} cpp

// example code for within Application::compose (or Fragment::compose)

    auto message_resource = make_resource<holoscan::MyNativeResource>(
        "message_resource", holoscan::Arg("message", "hello world");

    auto my_op = std::make_operator<holoscan::ops::MyOperator>(
        "my_op", holoscan::Arg("message_resource", message_resource));
```

As with GXF-based resources, it is also possible to pass a native resource as a positional argument to the operator constructor.

For a concreate example of native resource use in a real application, see the [volume_rendering_xr application](https://github.com/nvidia-holoscan/holohub/blob/main/applications/volume_rendering_xr/main.cpp) on Holohub. This application uses a native [XrSession resource](https://github.com/nvidia-holoscan/holohub/blob/main/operators/XrFrameOp/xr_session.hpp) type which corresponds to a single OpenXR session. This single "session" resource can then be shared by both the `XrBeginFrameOp` and `XrEndFrameOp` operators.

````
````{tab-item} Python
To create a native resource, implement a class that inherits from {py:class}`Resource <holoscan.core.Resource>`.

```{code-block} python
class MyNativeResource(Resource):
    def __init__(self, fragment, message="test message", *args, **kwargs):
        self.message = message
        super().__init__(fragment, *args, **kwargs)

    # Could optionally define Parameter as in C++ via spec.param as below.
    # Here, we chose instead to pass message as an argument to __init__ above.
    # def setup(self, spec: ComponentSpec):
    #     spec.param("message", "test message")

    # define a custom method
    def message(self):
        return self.message
```

The below shows how some custom operator could use such a resource in its compute method

```{code-block} python
class MyOperator(Operator):
    def compute(self, op_input, op_output, context):
        resource = self.resource("message_resource")
        if resource is None:
            raise ValueError("expected message resource not found")
        assert isinstance(resource, MyNativeResource)

        print(f"message = {resource.message()")
```

where this native resource could have been created and passed positionally to `MyOperator` as follows

```{code-block} python

# example code within Application.compose (or Fragment.compose)

    message_resource = MyNativeResource(
        fragment=self, message="hello world", name="message_resource")

    # pass the native resource as a positional argument to MyOperator
    my_op = MyOperator(fragment=self, message_resource)
```
````
`````

There is a minimal example of native resource use in the [examples/native](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/native/) folder.


(configuring-app-scheduler)=

### Configuring the scheduler

The [scheduler](./components/schedulers.md) controls how the application schedules the execution of the operators that make up its [workflow](application-workflows).

The default scheduler is a single-threaded [`GreedyScheduler`](./components/schedulers.md#greedy-scheduler). An application can be configured to use a different scheduler `Scheduler` ({cpp:class}`C++ <holoscan::Scheduler>`/{py:class}`Python <holoscan.core.Scheduler>`) or change the parameters from the default scheduler, using the `scheduler()` function ({cpp:func}`C++ <holoscan::Fragment::scheduler>`/{py:func}`Python <holoscan.core.Fragment.scheduler>`).

For example, if an application needs to run multiple operators in parallel, the [`MultiThreadScheduler`](./components/schedulers.md#multithread-scheduler) or [`EventBasedScheduler`](./components/schedulers.md#event-based-scheduler) can instead be used. The difference between the two is that the MultiThreadScheduler is based on actively polling operators to determine if they are ready to execute, while the EventBasedScheduler will instead wait for an event indicating that an operator is ready to execute. Additionally, the EventBasedScheduler also offers options for running time-critical operators under real-time scheduling policies supported by Linux kernel (see {ref}`Real-time scheduling with thread pools<configuring-app-thread-pools-realtime>`).

The code snippet below shows how to set and configure a non-default scheduler:

`````{tab-set}
````{tab-item} C++

- We create an instance of a {ref}`holoscan::Scheduler <api/holoscan_cpp_api:schedulers>` derived class by using the {cpp:func}`~holoscan::Fragment::make_scheduler` function. Like operators, parameters can come from explicit {cpp:class}`~holoscan::Arg`s or {cpp:class}`~holoscan::ArgList`, or from a YAML configuration.
- The {cpp:func}`~holoscan::Fragment::scheduler` method assigns the scheduler to be used by the application.

```{code-block} cpp
:emphasize-lines: 2-7
:name: holoscan-config-scheduler-cpp

auto app = holoscan::make_application<App>();
auto scheduler = app->make_scheduler<holoscan::EventBasedScheduler>(
  "myscheduler",
  Arg("worker_thread_number", static_cast<int64_t>(4)),
  Arg("stop_on_deadlock", true)
);
app->scheduler(scheduler);
app->run();
```

````

````{tab-item} Python
- We create an instance of a `Scheduler` class in the {py:mod}`~holoscan.schedulers` module. Like operators, parameters can come from an explicit {py:class}`~holoscan.core.Arg` or {py:class}`~holoscan.core.ArgList`, or from a YAML configuration.
- The {py:func}`~holoscan.core.Fragment.scheduler` method assigns the scheduler to be used by the application.

```{code-block} python
:emphasize-lines: 2-8
:name: holoscan-config-scheduler-python

app = App()
scheduler = holoscan.schedulers.EventBasedScheduler(
    app,
    name="myscheduler",
    worker_thread_number=4,
    stop_on_deadlock=True,
)
app.scheduler(scheduler)
app.run()
```
````
`````

:::{tip}
This is also illustrated in the [multithread](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/multithread) example.
:::


(configuring-app-thread-pools)=
### Configuring worker thread pools

Both the `MultiThreadScheduler` and `EventBasedScheduler` discussed in the previous section automatically create an internal worker thread pool by default. In some scenarios, it may be desirable for users to instead assign operators to specific user-defined thread pools. This also allows optionally pinning operators to a specific thread.

Assume I have three operators, `op1`, `op2` and `op3`, that I want to assign to a thread pool. I would also like to pin `op2` and `op3` to specific threads within the pool. The example below shows the code for configuring thread pools to achieve this from the Fragment `compose` method.

`````{tab-set}
````{tab-item} C++
We create thread pools via calls to the {cpp:func}`~holoscan::Fragment::make_thread_pool` method. The first argument is a user-defined name for the thread pool while the second is the number of threads initially in the thread pool. This `make_thread_pool` method returns a shared pointer to a {cpp:class}`~holoscan::ThreadPool` object. The {cpp:func}`~holoscan::ThreadPool::add` method of that object can then be used to add a single operator or a vector of operators to the thread pool. The second argument to the `add` function is a boolean indicating whether the given operators should be pinned to always run on a specific thread within the thread pool.

```{code-block} cpp
:name: holoscan-thread-pool-example-cpp

    // The following code would be within `Fragment::compose` after operators have been defined
    // Assume op1, op2 and op3 are `shared_ptr<OperatorType>` as returned by `make_operator`

    // create a thread pool with a three threads
    auto pool1 = make_thread_pool("pool1", 3);
    // assign a single operator to the thread pool (unpinned)
    pool1->add(op1, false);
    // assign multiple operators to this thread pool (pinned)
    pool1->add({op2, op3}, true);

```

````

````{tab-item} Python
We create thread pools via calls to the {py:func}`~holoscan.core.Fragment.make_thread_pool` method. The first argument is a user-defined name for the thread pool while the second is the initial size of the thread pool. It is not necessary to modify this as the size will be incremented as needed automatically. This `make_thread_pool` method returns a shared pointer to a {py:class}`~holoscan.resources.ThreadPool` object. The {py:func}`~holoscan.resources.ThreadPool.add` method of that object can then be used to add a single operator or a vector of operators to the thread pool. The second argument to the `add` function is a boolean indicating whether the given operators should be pinned to always run on a specific thread within the thread pool.

```{code-block} python
:name: holoscan-thread-pool-example-python
    # The following code would be within `Fragment::compose` after operators have been defined
    # Assume op1, op2 and op3 are `shared_ptr<OperatorType>` as returned by `make_operator`

    # create a thread pool with a single thread
    pool1 = self.make_thread_pool("pool1", 1);
    # assign a single operator to the thread pool (unpinned)
    pool1.add(op1, True);
    # assign multiple operators to this thread pool (pinned)
    pool1.add([op2, op3], True);
```
````
`````
:::{note}
It is not necessary to define a thread pool for Holoscan applications. There is a default thread pool that gets used for any operators the user did not explicitly assign to a thread pool. The use of thread pools provides a way to explicitly indicate that threads should be pinned.

One case where separate thread pools **must** be used is in order to support pinning of operators involving separate GPU devices. Only a single GPU device should be used from any given thread pool. Operators associated with a GPU device resource are those using one of the CUDA-based allocators like
`BlockMemoryPool`, `CudaStreamPool`, `RMMAllocator` or `StreamOrderedAllocator`.
:::

:::{tip}
A concrete example of a simple application with two pairs of operators in separate thread pools is given in the [thread pool resource example](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/resources/thread_pool).
:::

Note that any given operator can only belong to a single thread pool. Assigning the same operator to multiple thread pools may result in errors being logged at application startup time.

There is also a related boolean parameter, `strict_thread_pinning` that can be passed as a `holoscan::Arg` to the `MultiThreadScheduler` constructor. When this argument is set to `false` and an operator is pinned to a specific thread, it is allowed for other operators to also run on that same thread whenever the pinned operator is not ready to execute. When `strict_thread_pinning` is `true`, the thread can ONLY be used by the operator that was pinned to the thread. For the `EventBasedScheduler`, it is always in strict pinning mode and there is no such parameter.

If a thread pool is configured by the single-thread `GreedyScheduler` is used a warning will be logged indicating that the user-defined thread pools would be ignored. Only `MultiThreadScheduler` and `EventBasedScheduler` can make use of the thread pools.

(configuring-app-thread-pools-realtime)=
#### Linux real-time scheduling with thread pools

The `EventBasedScheduler` offers additional features to pin an operator to a dedicated worker thread scheduled by real-time scheduling policies supported in the Linux kernel. The configuration can be done by using the `add_realtime()` method (in contrast to the `add()` method) in `ThreadPool` to assign an operator with a real-time scheduling policy along with the parameters required for the selected scheduling policy. The supported real-time scheduling policies are:

- **SCHED_FIFO** (`SchedulingPolicy::kFirstInFirstOut`): First-in-first-out scheduling policy that provides priority execution. Operators with this policy will run until completion or until preempted by a higher priority Linux process or thread. Operators with the same priority under `SCHED_FIFO` are scheduled in a first-in-first-out fashion.
- **SCHED_RR** (`SchedulingPolicy::kRoundRobin`): Round-robin scheduling policy that provides execution with CPU time sharing for operators with the same priority level in a round-robin fashion.
- **SCHED_DEADLINE** (`SchedulingPolicy::kDeadline`): Earliest Deadline First scheduling policy that ensures operators meet their specified deadlines. This policy requires setting runtime, deadline, and period parameters.

For more detailed information about Linux kernel schedulers, refer to the [Ubuntu Real-time documentation](https://documentation.ubuntu.com/real-time/latest/explanation/schedulers/#id1).

:::{important}
**Important Notes About Using Real-time Scheduling Polices:**

- **SCHED_DEADLINE Behavior**: Since SCHED_DEADLINE inherently enforces periodic execution, adding a `PeriodicCondition` to these operators is unnecessary.

- **Operator Conditions Still Apply**: Real-time scheduling policies work alongside existing operator conditions. While real-time policies reduce overall scheduling latency, the actual operator execution start timing may still be constrained by conditions defined in the application's graph structure.

- **Understanding the Scope**: The Holoscan SDK integrates with Linux kernel real-time scheduling policies but cannot guarantee real-time performance across your entire application. This feature offers a way to reduce scheduling overhead for specific time-sensitive operators, but the overall system behavior depends on your application design and the underlying Linux kernel configuration.
:::

:::{note}
Using real-time scheduling policies requires appropriate Linux kernel configuration and may require running `sudo sysctl -w kernel.sched_rt_runtime_us=-1` beforehand to disable the real-time runtime limit.

**Container Requirements:**
- **SCHED_DEADLINE**: Requires root privileges and `--cap-add=CAP_SYS_NICE` when running in a container
- **SCHED_FIFO/SCHED_RR**: May require `--ulimit rtprio=99` when running in a container (can replace 99 with the highest value actually used for the `sched_priority` argument to `add_realtime()`)
:::

Here's an example of configuring operators to run with real-time policies:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:name: holoscan-realtime-thread-pool-example-cpp

    // Create a thread pool for real-time operators
    auto realtime_pool = make_thread_pool("realtime_pool", 2);

    // Add operator with SCHED_FIFO policy and priority 1, pinned to CPU core 0
    realtime_pool->add_realtime(op1, SchedulingPolicy::kFirstInFirstOut, true, {0}, 1);

    // Add operator with SCHED_RR policy and priority 2, pinned to CPU core 1
    realtime_pool->add_realtime(op2, SchedulingPolicy::kRoundRobin, true, {1}, 2);

    // Add operator with SCHED_DEADLINE policy, pinned to CPU core 2
    // runtime: 1ms, deadline: 10ms, period: 10ms
    realtime_pool->add_realtime(op3, SchedulingPolicy::kDeadline, true, {2}, 0,
                                1000000, 10000000, 10000000);
```
````

````{tab-item} Python
```{code-block} python
:name: holoscan-realtime-thread-pool-example-python
    # Import required for real-time scheduling
    from holoscan.resources import SchedulingPolicy

    # Create a thread pool for real-time operators
    realtime_pool = self.make_thread_pool("realtime_pool", 2)

    # Add operator with SCHED_FIFO policy and priority 1, pinned to CPU core 0
    realtime_pool.add_realtime(
        op1,
        sched_policy=SchedulingPolicy.SCHED_FIFO,
        pin_operator=True,
        pin_cores=[0],
        sched_priority=1
    )

    # Add operator with SCHED_RR policy and priority 2, pinned to CPU core 1
    realtime_pool.add_realtime(
        op2,
        sched_policy=SchedulingPolicy.SCHED_RR,
        pin_operator=True,
        pin_cores=[1],
        sched_priority=2
    )

    # Add operator with SCHED_DEADLINE policy, pinned to CPU core 2
    # runtime: 1ms, deadline: 10ms, period: 10ms
    realtime_pool.add_realtime(
        op3,
        sched_policy=SchedulingPolicy.SCHED_DEADLINE,
        pin_operator=True,
        pin_cores=[2],
        sched_runtime=1000000,
        sched_deadline=10000000,
        sched_period=10000000,
    )
```
````
`````

(configuring-app-runtime)=
### Configuring runtime properties

As described [below](building-and-running-your-application), applications can run simply by executing the C++ or Python application manually on a given node, or by [packaging it](./holoscan_packager.md) in a [HAP container](./cli/hap.md). With the latter, runtime properties need to be configured: refer to the [App Runner Configuration](./cli/run_config.md) for details.

(application-workflows)=
## Application Workflows

:::{note}
Operators are initialized according to the [topological order](https://en.wikipedia.org/wiki/Topological_sorting)
of its fragment-graph. When an application runs, the operators are executed in the same topological order.
Topological ordering of the graph ensures that all the data dependencies of an operator are satisfied before its instantiation and execution. Currently, we do not support specifying a different and explicit instantiation and execution order of the operators.
:::

### One-operator Workflow

The simplest form of a workflow would be a single operator.

```{digraph} myop
:align: center
:caption: A one-operator workflow

    rankdir="LR"
    node [shape=record];

    myop [label="MyOp| | "];
```

The graph above shows an **Operator** ({cpp:class}`C++ <holoscan::Operator>`/{py:class}`Python <holoscan.core.Operator>`) (named `MyOp`) that has neither inputs nor output ports.
- Such an operator may accept input data from the outside (e.g., from a file) and produce output data (e.g., to a file) so that it acts as both the source and the sink operator.
- Arguments to the operator (e.g., input/output file paths) can be passed as parameters as described in the {ref}`section above<configuring-an-application>`.

We can add an operator to the workflow by calling `add_operator` ({cpp:func}`C++ <holoscan::Fragment::add_operator>`/{py:func}`Python <holoscan.core.Fragment.add_operator>`) method in the `compose()` method.

The following code shows how to define a one-operator workflow in `compose()` method of the `App` class (assuming that the operator class `MyOp` is declared/defined in the same file).

````{tab-set-code}

```{code-block} cpp
:linenos: true
:name: holoscan-one-operator-workflow-cpp

class App : public holoscan::Application {
 public:
  void compose() override {
    // Define Operators
    auto my_op = make_operator<MyOp>("my_op");

    // Define the workflow
    add_operator(my_op);
  }
};
```

```{code-block} python
:linenos: true
:name: holoscan-one-operator-workflow-python

class App(Application):

    def compose(self):
        # Define Operators
        my_op = MyOp(self, name="my_op")

        # Define the workflow
        self.add_operator(my_op)
```
````

### Linear Workflow

Here is an example workflow where the operators are connected linearly:

```{digraph} linear_workflow
:align: center
:caption: A linear workflow

    rankdir="LR"
    node [shape=record];

    sourceop [label="SourceOp| |output(out) : Tensor"];
    processop [label="ProcessOp| [in]input : Tensor | output(out) : Tensor "];
    sinkop [label="SinkOp| [in]input : Tensor | "];
    sourceop -> processop [label="output...input"]
    processop -> sinkop [label="output...input"]
```

In this example, **SourceOp** produces a message and passes it to **ProcessOp**. **ProcessOp** produces another message and passes it to **SinkOp**.

We can connect two operators by calling the `add_flow()` method ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) in the `compose()` method.


The `add_flow()` method ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) takes the source operator, the destination operator, and the optional port name pairs.
The port name pair is used to connect the output port of the source operator to the input port of the destination operator.
The first element of the pair is the output port name of the upstream operator and the second element is the input port name of the downstream operator.
An empty port name ("") can be used for specifying a port name if the operator has only one input/output port.
If there is only one output port in the upstream operator and only one input port in the downstream operator, the port pairs can be omitted.

The following code shows how to define a linear workflow in the `compose()` method of the `App` class (assuming that the operator classes `SourceOp`, `ProcessOp`, and `SinkOp` are declared/defined in the same file).

````{tab-set-code}

```{code-block} cpp
:linenos: true
:name: holoscan-linear-operator-workflow-cpp

class App : public holoscan::Application {
 public:
  void compose() override {
    // Define Operators
    auto source = make_operator<SourceOp>("source");
    auto process = make_operator<ProcessOp>("process");
    auto sink = make_operator<SinkOp>("sink");

    // Define the workflow
    add_flow(source, process); // same as `add_flow(source, process, {{"output", "input"}});`
    add_flow(process, sink);   // same as `add_flow(process, sink, {{"", ""}});`
  }
};
```

```{code-block} python
:linenos: true
:name: holoscan-linear-operator-workflow-python

class App(Application):

    def compose(self):
        # Define Operators
        source = SourceOp(self, name="source")
        process = ProcessOp(self, name="process")
        sink = SinkOp(self, name="sink")

        # Define the workflow
        self.add_flow(source, process) # same as `self.add_flow(source, process, {("output", "input")})`
        self.add_flow(process, sink)   # same as `self.add_flow(process, sink, {("", "")})`
```
````

### Complex Workflow (Multiple Inputs and Outputs)

You can design a complex workflow like below where some operators have multi-inputs and/or multi-outputs:

```{digraph} complex_workflow
:align: center
:caption: A complex workflow (multiple inputs and outputs)

    node [shape=record];

    reader1 [label="{Reader1| |image(out)\nmetadata(out)}"];
    reader2 [label="{Reader2| |roi(out)}"];
    processor1 [label="{Processor1|[in]image1\n[in]image2\n[in]metadata|image(out)}"];
    processor2 [label="{Processor2|[in]image\n[in]roi|image(out)}"];
    processor3 [label="{Processor3|[in]image|seg_image(out)}"];
    writer [label="{Writer|[in]image\n[in]seg_image| }"];
    notifier [label="{Notifier|[in]image| }"];

    reader1->processor1 [label="image...{image1,image2}\nmetadata...metadata"]
    reader2->processor2 [label="roi...roi"]
    processor1->processor2 [label="image...image"]
    processor1->writer [label="image...image"]
    processor2->notifier [label="image...image"]
    processor2->processor3 [label="image...image"]
    processor3->writer [label="seg_image...seg_image"]
```


````{tab-set-code}

```{code-block} cpp
:linenos: true
:name: holoscan-multiio-operator-workflow-cpp

class App : public holoscan::Application {
 public:
  void compose() override {
    // Define Operators
    auto reader1 = make_operator<Reader1>("reader1");
    auto reader2 = make_operator<Reader2>("reader2");
    auto processor1 = make_operator<Processor1>("processor1");
    auto processor2 = make_operator<Processor2>("processor2");
    auto processor3 = make_operator<Processor3>("processor3");
    auto writer = make_operator<Writer>("writer");
    auto notifier = make_operator<Notifier>("notifier");

    // Define the workflow
    add_flow(reader1, processor1, {{"image", "image1"}, {"image", "image2"}, {"metadata", "metadata"}});
    add_flow(reader1, processor1, {{"image", "image2"}});
    add_flow(reader2, processor2, {{"roi", "roi"}});
    add_flow(processor1, processor2, {{"image", "image"}});
    add_flow(processor1, writer, {{"image", "image"}});
    add_flow(processor2, notifier);
    add_flow(processor2, processor3);
    add_flow(processor3, writer, {{"seg_image", "seg_image"}});
  }
};
```

```{code-block} python
:linenos: true
:name: holoscan-multiio-operator-workflow-python

class App(Application):

    def compose(self):
        # Define Operators
        reader1 = Reader1Op(self, name="reader1")
        reader2 = Reader2Op(self, name="reader2")
        processor1 = Processor1Op(self, name="processor1")
        processor2 = Processor2Op(self, name="processor2")
        processor3 = Processor3Op(self, name="processor3")
        notifier = NotifierOp(self, name="notifier")
        writer = WriterOp(self, name="writer")

        # Define the workflow
        self.add_flow(reader1, processor1, {("image", "image1"), ("image", "image2"), ("metadata", "metadata")})
        self.add_flow(reader2, processor2, {("roi", "roi")})
        self.add_flow(processor1, processor2, {("image", "image")})
        self.add_flow(processor1, writer, {("image", "image")})
        self.add_flow(processor2, notifier)
        self.add_flow(processor2, processor3)
        self.add_flow(processor3, writer, {("seg_image", "seg_image")})
```
````

If there is a cycle in the graph with no implicit root operator, the root
operator is either the first operator in the first call to `add_flow` method ({cpp:func}`C++
<holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`), or the
operator in the first
call to `add_operator` method ({cpp:func}`C++ <holoscan::Fragment::add_operator>`/{py:func}`Python
<holoscan.core.Fragment.add_operator>`).

````{tab-set-code}
```{code-block} c++
:emphasize-lines: 5
auto op1 = make_operator<...>("op1");
auto op2 = make_operator<...>("op2");
auto op3 = make_operator<...>("op3");

add_flow(op1, op2);
add_flow(op2, op3);
add_flow(op3, op1);
// There is no implicit root operator
// op1 is the root operator because op1 is the first operator in the first call to add_flow
```
````

If there is a cycle in the graph with an implicit root operator which has no input port, then the initialization and execution orders of the operators are still topologically sorted as far as possible until the cycle needs to be explicitly broken. An example is given below:

![Fragment graph with a cycle and an implicit root operator](Cycle_Implicit_Root.png)

(creating-and-using-subgraphs)=

### Creating and Using Subgraphs

A Subgraph ({cpp:class}`C++ <holoscan::Subgraph>`/{py:class}`Python <holoscan.core.Subgraph>`) encapsulates a group of related operators and their connections behind a clean interface, enabling modular application design and code reuse.

#### Features of Subgraphs

Subgraphs enable:
- **Reusable components**: Create a subgraph once and instantiate it multiple times within an application
- **Encapsulation**: Hide internal complexity behind well-defined interface ports
- **Modular design**: Organize complex applications into logical, maintainable components
- **Hierarchical composition**: Nest subgraphs within other subgraphs for multi-level decomposition
- **Flexible connections**: Connect subgraphs to other subgraphs or operators using the same `add_flow` API

#### Creating a Subgraph

A Subgraph is created by inheriting from the `Subgraph` base class and implementing the `compose()` method. Within `compose()`, you create operators, define flows between them, and expose interface ports that external components can connect to.

The APIs used to add operators, conditions and resources to a subgraph look the same as the ones for adding them to a `Fragment` or `Application`. A unique aspect of Subgraph creation as compared to defining a `Fragment`/`Application` is the definition of "interface ports" (described further below).

`````{tab-set}
````{tab-item} C++

```{code-block} cpp
:emphasize-lines: 8-9,12,16
:name: holoscan-create-subgraph-cpp

class PingTxSubgraph : public holoscan::Subgraph {
 public:
  PingTxSubgraph(holoscan::Fragment* fragment, const std::string& instance_name)
      : holoscan::Subgraph(fragment, instance_name) {}

  void compose() override {
    // Create operators within the subgraph
    auto tx_op = make_operator<ops::PingTxOp>("transmitter", make_condition<CountCondition>(8));
    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // Define internal connections
    add_flow(tx_op, forwarding_op);

    // Expose external interface port
    // The "out" port of forwarding_op is exposed as "data_out"
    add_output_interface_port("data_out", forwarding_op, "out");
  }
};
```

**Key points:**
- The constructor takes a `Fragment*` and `instance_name` which are passed to the base class
- Operators created with `make_operator` are automatically qualified with the instance name. Specifically, the operator added to the fragment via a subgraph will have a name that is the subgraph `instance_name` followed by an underscore and then the operator name provided within `Subgraph::compose`.
- `add_flow` defines internal connections between operators (and/or nested subgraphs)
- `add_output_interface_port` and `add_input_interface_port` expose ports for external connections

````

````{tab-item} Python

```{code-block} python
:emphasize-lines: 7-8,11,15
:name: holoscan-create-subgraph-python

class PingTxSubgraph(Subgraph):
    def __init__(self, fragment, instance_name):
        super().__init__(fragment, instance_name)

    def compose(self):
        # Create operators within the subgraph
        tx_op = PingTxOp(self, CountCondition(self, count=8), name="transmitter")
        forwarding_op = ForwardingOp(self, name="forwarding")

        # Define internal connections
        self.add_flow(tx_op, forwarding_op, {("out", "in")})

        # Expose external interface port
        # The "out" port of forwarding_op is exposed as "data_out"
        self.add_output_interface_port("data_out", forwarding_op, "out")
```

**Key points:**
- The `__init__` method receives `fragment` and `instance_name` and passes them to the base class
- Operators are created with the subgraph (`self`) as their fragment. An operator added to the fragment via a subgraph will have a name that is the subgraph `instance_name` followed by an underscore and then the operator name provided within `Subgraph.compose`.
- `add_flow` defines internal connections between operators (and/or nested subgraphs)
- `add_output_interface_port` and `add_input_interface_port` expose ports for external connections

````
`````


:::{note}
Subgraphs are a convenience for graph composition but do not affect operator scheduling. At runtime, an application using subgraphs will behave exactly the same as one composed without them. Any `add_operator` and `add_flow` calls within a subgraph directly add nodes (with qualified names) and edges to the operator graph maintained by the Fragment passed to the subgraph constructor. It is this final, flattened fragment that the application runs.
:::

#### Interface Ports

Interface ports define the external API of a subgraph. They map external port names to internal operator ports, allowing external components to connect to the subgraph without knowing its internal structure.

- **Input interface ports** (`add_input_interface_port`): Allow data to flow into the subgraph
- **Output interface ports** (`add_output_interface_port`): Allow data to flow out of the subgraph

Interface ports support both single-receiver and multi-receiver patterns, depending on the underlying operator's port configuration. Because interface ports map to an existing operator port, the conditions or other properties defined for the operator port automatically apply to the interface port.

:::{note}
It is not supported to define an input interface port with the same name as an output interface port. This differs from Operators, where such naming is currently allowed but not recommended, as it can lead to ambiguous logging when port names are not unique.
:::

#### Instantiating and Connecting Subgraphs

Once defined, subgraphs are instantiated from a `Fragment` or `Application` using `make_subgraph` and connected like regular operators using `add_flow`.

`````{tab-set}
````{tab-item} C++

```{code-block} cpp
:name: holoscan-use-subgraph-cpp

// compose method override of an Application or Fragment class
void compose() override {
  // Create subgraph instances with unique names
  auto tx_subgraph1 = make_subgraph<PingTxSubgraph>("tx1");
  auto tx_subgraph2 = make_subgraph<PingTxSubgraph>("tx2");

  // Create a multi-receiver subgraph (with interface port defined with `IOSpec::kAnySize`)
  auto rx_subgraph = make_subgraph<PingRxSubgraph>("rx");

  // Connect subgraphs via their interface ports
  add_flow(tx_subgraph1, rx_subgraph, {{"data_out", "data_in"}});
  add_flow(tx_subgraph2, rx_subgraph, {{"data_out", "data_in"}});
}
```

:::{tip}
The {cpp:func}`Fragment::make_subgraph<holoscan::Fragment::make_subgraph>` and {cpp:func}`Subgraph::make_subgraph<holoscan::Subgraph::make_subgraph>` methods create and then automatically call `compose()` on the newly created subgraph. The application author will not need to call `compose` manually.
:::

````

````{tab-item} Python

```{code-block} python
:name: holoscan-use-subgraph-python

# compose method of an Application or Fragment
def compose(self):
    # Create subgraph instances with unique names
    tx_subgraph1 = PingTxSubgraph(self, "tx1")
    tx_subgraph2 = PingTxSubgraph(self, "tx2")

    # Create a multi-receiver subgraph (with interface port defined with `size=IOSpec.ANY_SIZE`)
    rx_subgraph = PingRxSubgraph(self, "rx")

    # Connect subgraphs via their interface ports
    self.add_flow(tx_subgraph1, rx_subgraph, {("data_out", "data_in")})
    self.add_flow(tx_subgraph2, rx_subgraph, {("data_out", "data_in")})
```

:::{tip}
The {py:func}`Fragment.make_subgraph<holoscan.core.Fragment.make_subgraph>` and {py:func}`Subgraph.make_subgraph<holoscan.core.Subgraph.make_subgraph>` methods create and then automatically call `compose()` on the created subgraph. The application author will not need to call `compose` manually.
:::


````
`````

#### Qualified Naming

When a subgraph is instantiated, all operators within it are automatically assigned qualified names by prepending the instance name. This ensures uniqueness when the same subgraph class is used multiple times.

For example, if `PingTxSubgraph` contains a `"transmitter"` operator:
- Instance `"tx1"` creates operator `"tx1_transmitter"`
- Instance `"tx2"` creates operator `"tx2_transmitter"`

This naming scheme extends to nested subgraphs, creating hierarchical names like `"parent_child_operator"`.

Note that it is the qualified name that will show up in tools such as {ref}`NSight Systems traces <nsight-profiling>`, {ref}`data flow tracking <holoscan-flow-tracking>` output, {ref}`GXF JobStatistics <gxf-job-satistics>` reports, and {ref}`DataLogger <holoscan-data-logging>` topic names. This ensures that it is possible to uniquely distinguish which instance of an operator any given log message or measurement corresponds to.

:::{warning}
For the Python API, it is important while in `Subgraph.compose()`, to pass `self` and **not** `self.fragment` as the first argument to any operator constructors. The later would bypass the qualified naming logic and may lead to composition errors due to duplicate node names if there is more than one instance of the subgraph.
:::

#### Mixed Connections

Subgraphs can be connected to both other subgraphs and regular operators interchangeably. As for operator-to-operator connections, in cases where there is only a single port or interface port on the operator/subgraph on either end of a connection, the port name mapping can be omitted.

`````{tab-set}
````{tab-item} C++

```{code-block} cpp
:name: holoscan-subgraph-mixed-connections-cpp

// Subgraph to Subgraph
add_flow(tx_subgraph, rx_subgraph, {{"data_out", "data_in"}});

// Operator to Subgraph
add_flow(tx_operator, rx_subgraph, {{"out", "data_in"}});

// Subgraph to Operator
add_flow(tx_subgraph, rx_operator, {{"data_out", "in"}});

// Operator to Operator (standard)
add_flow(tx_operator, rx_operator);
```

````

````{tab-item} Python

```{code-block} python
:name: holoscan-subgraph-mixed-connections-python

# Subgraph to Subgraph
self.add_flow(tx_subgraph, rx_subgraph, {("data_out", "data_in")})

# Operator to Subgraph
self.add_flow(tx_operator, rx_subgraph, {("out", "data_in")})

# Subgraph to Operator
self.add_flow(tx_subgraph, rx_operator, {("data_out", "in")})

# Operator to Operator (standard)
self.add_flow(tx_operator, rx_operator, {("out", "in")})
```

````
`````

#### Nested Subgraphs

Subgraphs can contain other subgraphs, enabling hierarchical composition. Nested subgraphs are created using `make_subgraph` within a parent subgraph's `compose()` method. Interface ports from nested subgraphs can be exposed as the parent subgraph's interface ports.

`````{tab-set}
````{tab-item} C++

```{code-block} cpp
:emphasize-lines: 8,12,15
:name: holoscan-nested-subgraph-cpp

class NestedSubgraph : public holoscan::Subgraph {
 public:
  NestedSubgraph(holoscan::Fragment* fragment, const std::string& instance_name)
      : holoscan::Subgraph(fragment, instance_name) {}

  void compose() override {
    // Create a nested subgraph
    auto inner_subgraph = make_subgraph<PingTxSubgraph>("inner");
    auto forwarding_op = make_operator<ops::ForwardingOp>("forwarding");

    // Connect nested subgraph to operator
    add_flow(inner_subgraph, forwarding_op, {{"data_out", "in"}});

    // Expose the forwarding operator's port as this subgraph's interface
    add_output_interface_port("data_out", forwarding_op, "out");

    // Alternative: expose the nested subgraph's interface port directly
    // add_output_interface_port("data_out", inner_subgraph, "data_out");
  }
};
```

````

````{tab-item} Python

```{code-block} python
:emphasize-lines: 7,11,14
:name: holoscan-nested-subgraph-python

class NestedSubgraph(Subgraph):
    def __init__(self, fragment, instance_name):
        super().__init__(fragment, instance_name)

    def compose(self):
        # Create a nested subgraph
        inner_subgraph = PingTxSubgraph(self, "inner")
        forwarding_op = ForwardingOp(self, name="forwarding")

        # Connect nested subgraph to operator
        self.add_flow(inner_subgraph, forwarding_op, {("data_out", "in")})

        # Expose the forwarding operator's port as this subgraph's interface
        self.add_output_interface_port("data_out", forwarding_op, "out")

        # Alternative: expose the nested subgraph's interface port directly
        # self.add_output_interface_port("data_out", inner_subgraph, "data_out")
```

````
`````

#### Multi-Receiver Pattern

Subgraphs support the multi-receiver pattern when the underlying operator port is configured with `IOSpec::kAnySize` (C++) or `IOSpec.ANY_SIZE` (Python). This allows multiple sources to connect to a single input interface port of the subgraph.

`````{tab-set}
````{tab-item} C++

```{code-block} cpp
:emphasize-lines: 4
:name: holoscan-subgraph-multireceiver-cpp

// Define multi-receiver operator
void setup(OperatorSpec& spec) override {
  // Port accepts connections from multiple sources
  spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
}

// In subgraph, expose as interface port
add_input_interface_port("data_in", multi_rx_op, "receivers");

// Multiple connections to the same interface port
add_flow(tx_subgraph1, rx_subgraph, {{"data_out", "data_in"}});
add_flow(tx_subgraph2, rx_subgraph, {{"data_out", "data_in"}});
add_flow(tx_subgraph3, rx_subgraph, {{"data_out", "data_in"}});
```

````

````{tab-item} Python

```{code-block} python
:emphasize-lines: 4
:name: holoscan-subgraph-multireceiver-python

# Define multi-receiver operator
def setup(self, spec: OperatorSpec):
    # Port accepts connections from multiple sources
    spec.input("receivers", size=IOSpec.ANY_SIZE)

# In subgraph, expose as interface port
self.add_input_interface_port("data_in", multi_rx_op, "receivers")

# Multiple connections to the same interface port
self.add_flow(tx_subgraph1, rx_subgraph, {("data_out", "data_in")})
self.add_flow(tx_subgraph2, rx_subgraph, {("data_out", "data_in")})
self.add_flow(tx_subgraph3, rx_subgraph, {("data_out", "data_in")})
```

````
`````

:::{tip}
Complete working examples demonstrating subgraph functionality are available in the [subgraph examples](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/subgraph) directory, including the `ping_multi_receiver` example that showcases reusable subgraphs, interface ports, qualified naming, and multi-receiver patterns.
:::

### Dynamic Flow Control for Complex Workflows

As of Holoscan v3.0, the dynamic flow control feature is available, enabling operators to modify their connections with other operators at runtime. This allows for the creation of complex workflows with conditional branching, loops, and dynamic routing patterns.

Key features include:

  - Implicit input/output execution ports for execution dependency control
  - The Start operator concept (`start_op()` ({cpp:func}`C++ <holoscan::Fragment::start_op>`/{py:func}`Python <holoscan.core.Fragment.start_op>`)) for managing workflow entry points
  - Dynamic flow modification using `set_dynamic_flows()` ({cpp:func}`C++ <holoscan::Fragment::set_dynamic_flows>`/{py:func}`Python <holoscan.core.Application.set_dynamic_flows>`) and `add_dynamic_flow()` ({cpp:func}`C++ <holoscan::Operator::add_dynamic_flow>`/{py:func}`Python <holoscan.core.Operator.add_dynamic_flow>`) methods
  - Flow information management via the `FlowInfo` ({cpp:class}`C++ <holoscan::Operator::FlowInfo>`/{py:class}`Python <holoscan.core.FlowInfo>`) class

For details, please refer to the {ref}`Dynamic Flow Control <holoscan-dynamic-flow-control>` section of the user guide.

### Application Execution Control APIs

Holoscan provides APIs for controlling the execution of operators at the application or fragment level.

#### stop_execution

The `stop_execution()` ({cpp:func}`C++ <holoscan::Fragment::stop_execution>`/{py:func}`Python <holoscan.core.Fragment.stop_execution>`) method allows an application to stop the execution of a specific operator or the entire application:

`````{tab-set}
````{tab-item} C++
```cpp
virtual void stop_execution(const std::string& op_name = "");
```

When called with an operator name, this method stops the execution of the specified operator.
When called with an empty string (the default), it stops all operators in the fragment, effectively shutting down the application.

Example usage to stop a specific operator:
```cpp
// From within a Fragment/Application method
stop_execution("source_operator");
```

Example usage to stop the entire application:
```cpp
// From within a Fragment/Application method
stop_execution();
```

Example usage to access `stop_execution()` method from within an operator:
```cpp
// From within an operator's compute method
fragment()->stop_execution(); // `fragment()` returns a pointer to the fragment object
```
````

````{tab-item} Python
```python
def stop_execution(self, op_name="")
```

When called with an operator name, this method stops the execution of the specified operator.
When called with an empty string (the default), it stops all operators in the fragment, effectively shutting down the application.

Example usage to stop a specific operator:
```python
# From within a Fragment/Application method
self.stop_execution("source_operator")
```

Example usage to stop the entire application:
```python
# From within a Fragment/Application method
self.stop_execution()
```

Example usage to access `stop_execution()` method from within an operator:
```python
# From within an operator's compute method
self.fragment.stop_execution() # `self.fragment` is the fragment object
```
````
`````

For a complete example of how to use these methods to implement advanced monitoring behavior, see the [operator_status_tracking](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/execution_control/operator_status_tracking) example, which demonstrates:

1. A source operator that runs for a limited number of iterations
2. A monitor operator that independently tracks the status of other operators
3. Automatic application shutdown when all processing operators have completed

(building-and-running-your-application)=

## Building and running your Application

`````{tab-set}
````{tab-item} C++

You can build your C++ application using CMake, by calling `find_package(holoscan)` in your `CMakeLists.txt` to load the SDK libraries. Your executable will need to link against:
- `holoscan::core`
- any operator defined outside your `main.cpp` which you wish to use in your app workflow, such as:
  - SDK [built-in operators](./holoscan_operators_extensions.md#operators) under the `holoscan::ops` namespace.
  - operators created separately in your project with `add_library`.
  - operators imported externally using with `find_library` or `find_package`.

```{code-block} cmake
:caption: <src_dir>/CMakeLists.txt

# Your CMake project
cmake_minimum_required(VERSION 3.20)
project(my_project CXX)

# Finds the holoscan SDK
find_package(holoscan REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

# Create an executable for your application
add_executable(my_app main.cpp)

# Link your application against holoscan::core and any existing operators you'd like to use
target_link_libraries(my_app
  PRIVATE
    holoscan::core
    holoscan::ops::<some_built_in_operator_target>
    <some_other_operator_target>
    <...>
)
```

:::{tip}
This is also illustrated in all the examples:
- in `CMakeLists.txt` for the SDK installation directory - `/opt/nvidia/holoscan/examples`.
- in `CMakeLists.min.txt` for the SDK [source directory](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#readme).
:::

Once your `CMakeLists.txt` is ready in `<src_dir>`, you can build in `<build_dir>` with the command line below. You can optionally pass `Holoscan_ROOT` if the SDK installation you'd like to use differs from the `PATHS` given to `find_package(holoscan)` above.
```bash
# Configure
cmake -S <src_dir> -B <build_dir> -D Holoscan_ROOT="/opt/nvidia/holoscan"
# Build
cmake --build <build_dir> -j
```

You can then run your application by running `<build_dir>/my_app`.
````

````{tab-item} Python
Python applications do not require building. Simply ensure that:
- The [`holoscan`](./api/holoscan_python_api.md) python module is installed in your `dist-packages` or is listed under the `PYTHONPATH` env variable so you can import {py:mod}`holoscan.core` and any built-in operator you might need in {py:mod}`holoscan.operators`.
- Any external operators are available in modules in your `dist-packages` or contained in `PYTHONPATH`.

:::{note}
While Python applications do not need to be built, they might depend on operators that wrap C++ operators. All Python operators built-in in the SDK already ship with the Python bindings pre-built. Follow {ref}`this section<python-wrapped-operators>` if you are wrapping C++ operators yourself to use in your Python application.
:::

You can then run your application by running `python3 my_app.py`.

````
`````

:::{note}
Given a CMake project, a pre-built executable, or a Python application, you can also use the [Holoscan CLI](./cli/cli.md) to [package and run your Holoscan application](./holoscan_packager.md) in a OCI-compliant container image.
:::

## Dynamic Application Metadata

As of Holoscan v2.3 (for C++) or v2.4 (for Python) it is possible to send metadata alongside the data emitted from an operator's output ports. This metadata can then be used and/or modified by any downstream operators. The subsections below describe how this feature can be used.

### Enabling application metadata

As of Holoscan v3.0, the metadata feature is enabled by default (in older releases it had to be explicitly enabled). If the application author does not wish to use the metadata feature it will not hurt to leave the feature enabled. To avoid even the minor overhead of checking for metadata in received messages, the feature can be explicitly disabled as shown below. 

`````{tab-set}
````{tab-item} C++
```cpp
app = holoscan::make_application<MyApplication>();

// Disable metadata feature before calling app->run() or app->run_async()
app->enable_metadata(false);

app->run();
```
````
````{tab-item} Python
```cpp
app = MyApplication()

# Disable metadata feature before calling app.run() or app.run_async()
app.enable_metadata(False)
app.run()
```
````
`````

None of the built-in operators provided by the SDK itself currently require that the feature be enabled, but it is possible that some third-party operators might require it in order to work as expected. An example is the `V4L2FormatTranslateOp` defined as part of the [v4l2_camera example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/v4l2_camera) (video format information is stored in the metadata).

Note that the `enable_metadata` method exists on the Application, Fragment and Operator classes. Calling this method on the application sets the default for all fragments of a distributed application. Calling the method on an individual fragment sets the default to be used for that fragment (overrides the application-level default). Similarly, calling the method on an individual operator overrides the setting for that specific operator within a fragment.

### Understanding Metadata Flow

Each operator in the workflow has an associated {cpp:class}`~holoscan::MetadataDictionary` object. At the start of each operator's {cpp:func}`~holoscan::Operator::compute` call this metadata dictionary will be empty (i.e. metadata does not persist from previous compute calls). When any call to {cpp:func}`~holoscan::InputContext::receive` data is made, any metadata also found in the input message will be merged into the operator's local metadata dictionary. The operator's compute method can then read, append to or remove metadata as explained in the next section. Whenever the operator emits data via a call to {cpp:func}`~holoscan::OutputContext::emit` the current status of the operator's metadata dictionary will be transmitted on that port alonside the data passed via the first argument to the emit call. Any downstream operators will then receive this metadata via their input ports.

### Working With Metadata from Operator::compute

Within the operator's {cpp:func}`~holoscan::Operator::compute` method, the {cpp:func}`~holoscan::Operator::metadata` method can be called to get a shared pointer to the {cpp:class}`~holoscan::MetadataDictionary` of the operator. The metadata dictionary provides a similar API to a `std::unordered_map` (C++) or `dict` (Python) where the keys are strings (`std::string` for C++) and the values can store any object type (via a C++ {cpp:type}`~holoscan::MetadataObject` holding a `std::any`).


`````{tab-set}
````{tab-item} C++
Templated {cpp:func}`~holoscan::MetadataObject::get` and {cpp:func}`~holoscan::MetadataObject::set` method are provided as demonstrated below to allow directly setting values of a given type without having to explicitly work with the internal {cpp:type}`~holoscan::MetadataObject` type.

```cpp

// Receiving from a port updates operator metadata with any metadata found on the port
auto input_tensors = op_input.receive<TensorMap>("in");

// Get a reference to the shared metadata dictionary
auto meta = metadata();

// Retrieve existing values.
// Use get<Type> to automatically cast the `std::any` contained within the `holoscan::Message`
auto name = meta->get<std::string>("patient_name");
auto age = meta->get<int>("age");

// Get also provides a two-argument version where a default value to be assigned is given by
// the second argument. The type of the default value should match the expected type of the value.
auto flag = meta->get("flag", false);

// Add a new value (if a key already exists, the value will be updated according to the
// operator's metadata_policy).
std::vector<float> spacing{1.0, 1.0, 3.0};
meta->set("pixel_spacing"s, spacing);

// Remove an item
meta->erase("patient_name");

// Check if a key exists
bool has_patient_name = meta->has_key("patient_name");

// Get a vector<std::string> of all keys in the metadata
const auto& keys = meta->keys();

// ... Some processing to produce output `data` could go here ...

// Current state of `meta` will automatically be emitted along with `data` in the call below
op_output.emit(data, "output1");

// Can clear all items
meta->clear();

// Any emit call after this point would not transmit a metadata object
op_output.emit(data, "output2");
```
See the {py:class}`~holoscan.core.MetadataDictionary` API docs for all available methods.
````
````{tab-item} Python
A Pythonic interface is provided for the {py:class}`~holoscan.core.MetadataObject` type.

```python

# Receiving from a port updates operator metadata with any metadata found on the port
input_tensors = op_input.receive("in")

# self.metadata can be used to access the shared MetadataDictionary
# for example we can check if a key exists
has_key = "my_key" in self.metadata

# get the number of keys
num_keys = len(self.metadata)

# get a list of the keys
print(f"metadata keys = {self.metadata.keys()}")

# iterate over the values in the dictionary using the `items()` method
for key, value in self.metadata.items():
    # process item
    pass

# print a Python dict of the keys/values
print(self.metadata)

# Retrieve existing values. If the underlying value is a C++ class, a conversion to an equivalent Python object will be made (e.g. `std::vector<std::string>` to `List[str]`).
name = self.metadata["patient_name"]
age = self.metadata["age"]

# It is also supported to use the get method along with an optional default value to use
# if the key is not present.
flag = self.metadata.get("flag", False)

# print the current metadata policy
print(f"metadata policy = {self.metadata_policy}")

# Add a new value (if a key already exists, the value will be updated according to the
# operator's metadata_policy). If the value is set via the indexing operator as below,
# the Python object itself is stored as the value.
spacing = (1.0, 1.0, 3.0)
self.metadata["pixel_spacing"] = spacing

# In some cases, if sending metadata to downstream C++-based operators, it may be desired
# to instead store the metadata value as an equivalent C++ type. In that case, it is
# necessary to instead set the value using the `set` method with `cast_to_cpp=True`.
# Automatic casting is supported for bool, str, and various numeric and iterator or
# sequence types.

# The following would result in the spacing `Tuple[float]` being stored as a
# C++ `std::vector<double>`. Here we show use of the `pop` method to remove a previous value
# if present.
self.metadata.pop("pixel_spacing", None)
self.metadata.set("pixel_spacing", spacing, cast_to_cpp=True)

# To store floating point elements at a different than the default (double) precision or
# integers at a different precision than int64_t, use the dtype argument and pass a
# numpy.dtype argument corresponding to the desired C++ type. For example, the following
# would instead store `spacing` as a std::vector<float> instead. In this case we show
# use of Python's `del` instead of the pop method to remove an existing item.
del self.metadata["pixel_spacing"]
self.metadata.set("pixel_spacing", spacing, dtype=np.float32, cast_to_cpp=True)

# Remove a value
del self["patient name"]

# ... Some processing to produce output `data` could go here ...

# Current state of `meta` will automatically be emitted along with `data` in the call below
op_output.emit(data, "output1")

# Can clear all items
self.metadata.clear()

# Any emit call after this point would not transmit a metadata object
op_output.emit(data, "output2")
```

See the {py:class}`~holoscan.core.MetadataDictionary` API docs for all available methods.

The above code illustrated various ways of working with and updating an operator's metadata.

:::{note}
Pay particular attention to the details of how metadata is set. When working with pure Python applications it is best to just use `self.metadata[key] = value` or `self.metadata.set(key, value)` to pass Python objects as the value. This will just use a shared object and not result in copies to/from corresponding C++ types. However, when interacting with other operators that wrap a C++ implementation, their `compute` method would expected C++ metadata. In that case, the `set` method with `cast_to_cpp=True` is needed to cast to the expected C++ type. This was shown in some of the "pixel_spacing" set calls in the example above. For convenience, the `value` passed to the `set` method can also be a NumPy array, but note that in this case, a copy into a new C++ std::vector is performed. The dtype of the array will be respected when creating the vector. In general, the types that can currently be cast to C++ are scalar numeric values, strings and Python Iterators or Sequences of these (the sequence will be converted to a 1D or 2D C++ std::vector<T> so the items in the Python sequence cannot be of mixed type).
:::
````
`````

#### Metadata Update Policies

`````{tab-set}
````{tab-item} C++

The operator class also has a {cpp:func}`~holoscan::Operator::metadata_policy` method that can be used to set a {cpp:enum}`~holoscan::MetadataPolicy` to use when handling duplicate metadata keys across multiple input ports of the operator. The available options are:
- "update" (`MetadataPolicy::kUpdate`): replace any existing key from a prior `receive` call with one present in a subsequent `receive` call.
- "reject" (`MetadataPolicy::kReject`): Reject the new key/value pair when a key already exists due to a prior `receive` call.
- "raise" (`MetadataPolicy::kRaise`): Throw a `std::runtime_error` if a duplicate key is encountered. This is the default policy.

The metadata policy would typically be set during {cpp:func}`~holoscan::Application::compose` as in the following example:

```cpp

// Example for setting metadata policy from Application::compose()
my_op = make_operator<MyOperator>("my_op");
my_op->metadata_policy(holoscan::MetadataPolicy::kRaise);

```
````
````{tab-item} Python

The operator class also has a {py:func}`~holoscan.core.Operator.metadata_policy` property that can be used to set a {py:class}`~holoscan.core.MetadataPolicy` to use when handling duplicate metadata keys across multiple input ports of the operator. The available options are:
- "update" (`MetadataPolicy.UPDATE`): replace any existing key from a prior `receive` call with one present in a subsequent `receive` call. This is the default policy.
- "reject" (`MetadataPolicy.REJECT`): Reject the new key/value pair when a key already exists due to a prior `receive` call.
- "raise" (`MetadataPolicy.RAISE`): Throw an exception if a duplicate key is encountered.

The metadata policy would typically be set during {py:func}`~holoscan.core.Application.compose` as in the following example:

```python

# Example for setting metadata policy from Application.compose()
my_op = MyOperator(self, name="my_op")
my_op.metadata_policy = holoscan.core.MetadataPolicy.RAISE

```
````
`````

The policy applied as in the example above only applies to the operator on which it was set. The default metadata policy can also be set for the application as a whole via `Application::metadata_policy` ({cpp:func}`C++ <holoscan::Application::metadata_policy>`/{py:func}`Python <holoscan.core.Application.metadata_policy>`) or for individual fragments of a distributed application via `Fragment::metadata_policy` ({cpp:func}`C++ <holoscan::Fragment::metadata_policy>`/{py:func}`Python <holoscan.core.Fragment.metadata_policy>`).

### Use of Metadata in Distributed Applications

Sending metadata between two fragments of a distributed application is supported, but there are a couple of aspects to be aware of.

1. Sending metadata over the network requires serialization and deserialization of the metadata keys and values. The value types supported for this are the same as for data emitted over output ports (see the table in the section on {ref}`object serialization<object-serialization>`). The only exception is that {cpp:class}`~holoscan::Tensor` and {cpp:class}`~holoscan::TensorMap` values cannot be sent as metadata values between fragments (this restriction also applies to tensor-like Python objects). Any {ref}`custom codecs<object-serialization>` registered for the SDK will automatically also be available for serialization of metadata values.
2. There is a practical size limit of several kilobytes in the amount of metadata that can be transmitted between fragments. This is because metadata is currently sent along with other entity header information in the UCX header, which has fixed size limit (the metadata is stored along with other header information within the size limit defined by the `HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE` {ref}`environment variable<holoscan-distributed-env>`).

The above restrictions only apply to metadata sent **between** fragments. Within a fragment there is no size limit on metadata (aside from system memory limits) and no serialization or deserialization step is needed.

(metadata-limitations)=

### Current limitations

1. The current metadata API is only fully supported for native holoscan Operators and is not currently supported by operators that wrap a GXF codelet (i.e. inheriting from {cpp:class}`~holoscan::GXFOperator` or created via {cpp:class}`~holoscan::ops::GXFCodeletOp`). Aside from `GXFCodeletOp`, the built-in operators provided under the `holoscan::ops` namespace are all native operators, so the feature will work with these. Currently none of these built-in opereators add their own metadata, but any metadata received on input ports will automatically be passed on to their output ports (as long as `app->enable_metadata(false)` was not set to disable the metadata feature).

## CUDA Stream Handling APIs

Please see the dedicated {ref}`Holoscan CUDA stream handling<holoscan-cuda-stream-handling>` page for details on how Holoscan applications using non-default CUDA streams can be written.
