(holoscan-operator-from-decorator)=
# Simplified Python operator creation via the create_op decorator

:::{warning}
The {py:func}`holoscan.decorator.create_op` decorator and the supporting {py:class}`holoscan.decorator.Input` and {py:class}`holoscan.decorator.Output` classes are new in Holoscan v2.2 and are still considered experimental. They are usable now, but it is possible that some backwards incompatible changes to the behavior or API may be made based on initial feedback.
:::

For convenience, a {py:func}`holoscan.decorator.create_op` decorator is provided which can be used to automatically convert a simple Python function/generator or a class into a native Python {py:class}`holoscan.core.Operator`. The wrapped function body (or the `__call__` method if `create_op` is applied to a class) will correspond to the computation to be done in the {py:func}`holoscan.core.Operator.compute` method, but without any need to explicitly make any calls to {py:func}`holoscan.core.InputContext.receive` to receive inputs or {py:func}`holoscan.core.OutputContext.emit` to transmit the output. Any necessary input or output ports will have been automatically generated.

Consider first a simple Python function named `mask_and_offset` that takes `image` and `mask` tensors as input and multiplies them, followed by adding some scalar `offset`.

```{code-block} python
def mask_and_offset(image, mask, offset=1.5):
    return image * mask + offset
```

To turn this into an function that returns a corresponding operator we can add the `create_op` decorator like this:

```{code-block} python
from holoscan.decorator import create_op


@create_op(
    inputs=("image", "mask"),
    outputs="out",
)
def mask_and_offset(image, mask, offset=1.5):
    return image * mask + offset
```

By supplying the `inputs` argument we are specifying that there are two input ports, named "image" and "mask". By setting `outputs="out"` we are indicating that the output will be transmitted on a port named "out". When `inputs` are specified by simple strings in this way, the names used must map to variable names in the wrapped function's signature. We will see later that it is possible to use the {py:class}`holoscan.decorator.Input` class to provide more control over how inputs are mapped to function arguments. Similarly, we will see that the {py:class}`holoscan.decorator.Output` class can be used to provide more control over how the function output is mapped to any output port(s).

There is also an optional, `cast_tensors` argument to `create_op`. For convenience, this defaults to `True`, which results in any tensor-like objects being automatically cast to a NumPy or CuPy array (for host or device tensors, respectively) before they are passed on to the function. If this is not desired (e.g. due to working with a different third party tensor framework than NumPy or CuPy), the user can set `cast_tensors=False`, and manually handle casting of any `holoscan.Tensor` objects to the desired form in the function body. This casting option applies to either single tensors or a tensor map (`dict[Tensor]`).

This decorated function can then be used within the `compose` method of an `Application` to create an operator corresponding to this computation:
```{code-block} python
from holoscan.core import Application, Operator


def MyApp(Application):

    def compose(self)
        mask_op = mask_and_offset(self, name="my_scaling_op", offset=0.0)

        # verify that an Operator was generated
        assert(isinstance(mask_op, Operator))

        # now add any additional operators and create the computation graph using add_flow
```
Note that as for all other Operator classes, it is **required** to supply the application (or fragment) as the first argument (`self` here). The `name` kwarg is always supported and is the name that will be assigned to the operator. Due to the use of this `kwarg` to specify the operator name, the wrapped function (`mask_and_offset` in this case) should not use `name` as an argument name. In this case, we specified `offset=0.0` which would override the default value of `offset=1.5` in the function signature.

For completeness, the use of the `create_op` decorator on `mask_and_offset` is equivalent to if the user had defined the following `MaskAndOffsetOp` class and used it in `MyApp.compose`:

```{code-block} python
def MaskAndOffsetOp(Operator):
    def setup(self, spec):
        spec.input("image")
        spec.input("mask")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Simplified logic here assumes received values are GPU tensors
        # create_op would add additional logic to handle the inputs
        image = op_input.receive("image")
        image = cp.asarray(image)

        mask = op_input.receive("mask")
        mask = cp.asarray(mask)

        out = image * mask + offset
        op_output.emit(out, "out")
```

## Decorating a function that returns a tuple of arrays
Let's consider another example where function takes in multiple arrays, processes them, and returns a tuple of updated arrays:

```{code-block} python
def scale_and_offset(x1, x2, scale=2.0, offset=1.5):
    y1 = x1 * scale
    y2 = x2 + offset
    return y1, y2
```
To turn this into a corresponding operator we can add the `create_op` decorator like this:

```{code-block} python
@create_op(
    inputs=("x1", "x2"),
    outputs=("out1", "ou2"),
)
def scale_and_offset(x1, x2, scale=2.0, offset=1.5):
    y1 = x1 * scale
    y2 = x2 + offset
    return y1, y2
```

As before, the messages received through the ports defined by `inputs`, "x1" and "x2", will be mapped to respective variables `x1` and `x2`. Likewise, the elements of the output tuple, arrays `y1` and `y2`, will be emitted through ports "out1" and "out2", respectively. In contrast to input mapping, which is determined by the naming of ports and variables, the output mapping is determined by the ordering of output ports and elements in the tuple returned by the function.


(holoscan-operator-from-decorator-input)=
## Using the Input class for more control over input ports

This section will cover additional use cases where using a `str` or `Tuple[str]` for the `inputs` argument is insufficient.

**Scenario 1:** Assume that the upstream operator sends a tensormap to a given input port and we need to specify which tensor(s) in the tensormap will map to which input port.

For a concrete example, suppose we want to print a tensor's shape using a function like:
```{code-block} python
def print_shape(tensor):
    print(f"{tensor.shape = }")
```

but the upstream operator outputs a dictionary containing two tensors named "image" and "labels". We could use this operator by specifying which tensor name on a particular input port would map to the function's "tensor" argument. For example:


```{code-block} python
@create_op(inputs=Input("input_tensor", arg_map={"image": "tensor"}))
def print_shape(tensor):
    print(f"{tensor.shape = }")
```

would create an operator with a single input port named "input_tensor" and no output port. The input port may receive a tensormap with any number of tensors, but will only use the tensor named "image", mapping it to the "tensor" argument of the wrapped function. In general, the `arg_map` is a dictionary mapping tensor names found on the port to their corresponding function argument names.

**Scenario 2:** we want to override the scheduling condition present on a port. This can be done by specifying Input with the `condition` and optionally `condition_kwargs` arguments. For example, to override the MessageAvailableCondition that is added to the port by default and allow it to call `compute` even when no input message is available:


```{code-block} python
@create_op(inputs=Input("input_tensor", condition=ConditionType.NONE, condition_kwargs={}))
```

**Scenario 3:** we want to override the parameters of the receiver present on a port. For example, we could specify a different policy for the double buffer receiver that is used by default (policy=1 corresponds to discarding incoming messages when the queue is already full)


```{code-block} python
@create_op(inputs=Input("input_tensor", connector=ConditionType.DOUBLE_BUFFER, connector_kwargs=dict(capacity=1, policy=1)))
```

(holoscan-operator-from-decorator-output)=
## Using the Output class for more control over output ports

To support a case where multiple output ports should be present, the user must have the function return a `dict`. The {py:class}`holoscan.decorator.Output` class then has a `tensor_names` keyword argument that can be specified to indicate which items in the dictionary are to be transmitted on a given output port.

For example, assume we have a function that generates three tensors, `x`, `y` and `z` and we want to transmit `x` and `y` on port "out1" while `z` will be transmitted on port "out2". This can be done by specifying `outputs` as follows in the `create_op` call:

```{code-block} python
@create_op(
    outputs=(
        Output("out1", tensor_names=("x", "y")),
        Output("out2", tensor_names=("z",)),
    ),
)
def xyz_generator(nx=32, ny=32, nz=16):
    x = cp.arange(nx, dtype=np.uint32)
    y = cp.arange(ny, dtype=np.uint32)
    z = cp.arange(nz, dtype=np.uint32)

    # must return a dict type when Output arg(s) with `tensor_names` is used
    return dict(x=x, y=y, z=z)
```

This operator has no input ports and three optional keyword arguments. It splits the output tensors across two ports as described above. All names used in `tensor_names` must correspond to keys present in the `dict` emitted by the object. Often the `dict` values are tensors, but that is not a requirement.


The {py:class}`holoscan.decorator.Output` class also supports `condition`, `condition_kwargs`, `connector` and `connector_kwargs` that work in the same way as shown for {py:class}`holoscan.decorator.Input` above. For example, to override the transmitter queue policy for a single output port named "output_tensor"

```{code-block} python
@create_op(outputs=Output("output_tensor",
                           connector=IOSpec.ConnectorType.DOUBLE_BUFFER,
                           connector_kwargs=dict(capacity=1, policy=1)))
```

note that `tensor_names` was not specified which means that the returned object does not need to be a `dict`. The object itself will be emitted on the "output_tensor" port.

:::{note}
When specifying the `inputs` and `outputs` arguments to `create_op`, please make sure that all ports have unique names. As a concrete example, if an operator has a single input and output port that are used to send images, one should use unique port names like "image_in" and "image_out" rather than using "image" for both.
:::

(holoscan-operator-from-decorator-queue-size-policy)=
## Configuring the queue size and policy for an input or output port

When using the decorator approach to create operators, you can configure the queue size and policy for input and output ports using the `Input` and `Output` classes. Here's how to configure these parameters:

```python
from holoscan.core import ConditionType, IOSpec
from holoscan.decorator import Input, Output, create_op

# Example with input port configuration
@create_op(
    op_param="self",
    inputs=Input(
        "input",
        arg_map="value",
        size=IOSpec.SIZE_ONE,  # Set queue size to 1
        policy=IOSpec.QueuePolicy.POP  # Pop oldest item if queue is full
    ),
    outputs="output"
)
def my_operator(self, value):
    return value * 2

# Example with output port configuration
@create_op(
    op_param="self",
    inputs="input",
    outputs=Output(
        "output",
        size=2,  # Set queue size to 2
        policy=IOSpec.QueuePolicy.REJECT,  # Reject new items if queue is full
        condition_type=ConditionType.NONE
    )
)
def another_operator(self, value):
    return value
```

### Queue Size Options

The `size` parameter can be set to:
- `IOSpec.SIZE_ONE`: Queue size of 1 (default)
- `IOSpec.ANY_SIZE`: Any size queue
- `IOSpec.PRECEDING_COUNT`: Size based on number of preceding connections
- An integer value: Custom queue size

### Queue Policy Options

The `policy` parameter accepts these values:
- `IOSpec.QueuePolicy.POP`: Pop oldest item when queue is full
- `IOSpec.QueuePolicy.REJECT`: Reject new items when queue is full
- `IOSpec.QueuePolicy.FAULT`: Log warning and reject when queue is full

### Example Use Cases

1. Throttling execution with POP policy:
```python
@create_op(
    op_param="self",
    inputs=Input("input", arg_map="value"),
    outputs=Output(
        "output",
        policy=IOSpec.QueuePolicy.POP,
        condition_type=ConditionType.NONE
    )
)
def execution_throttler_op(self, value):
    if value is not None:
        return value
```

2. Multiple input handling:
```python
@create_op(
    op_param="self",
    inputs=Input("input", arg_map="value", size=IOSpec.ANY_SIZE),
    outputs="output"
)
def multi_input_op(self, value):
    return value
```

(holoscan-operator-from-decorator-op-param)=
## Using the op_param argument to access the operator instance

The `op_param` argument to `create_op` can be used to access the operator instance within the function body. This is useful if the operator needs to access its own name or other attributes.

```{code-block} python
@create_op(op_param="self")
def simple_op(self, param1, param2=5):
    print(f"I am here - {self.name} (param1: {param1}, param2: {param2})")
```

The operator can then be used in an application like this:

```{code-block} python
class OpParamApp(Application):
    def compose(self):
        node1 = simple_op(self, param1=1, name="node1")
        node2 = simple_op(self, param1=2, name="node2")
        node3 = simple_op(self, param1=3, name="node3")

        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
```

The output of this application will be:

```
I am here - node1 (param1: 1, param2: 5)
I am here - node2 (param1: 2, param2: 5)
I am here - node3 (param1: 3, param2: 5)
```

When this application runs, each operator instance will print its name along with its parameter values. The `op_param` argument allows the function to access operator attributes like `name` through the specified parameter (in this case `self`). This is particularly useful when you need to access operator-specific information or methods within your function's implementation.


## Interoperability with wrapped C++ operators

The SDK includes a [python_decorator example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/python_decorator) showing interoperability of wrapped C++ operators (`VideoStreamReplayerOp` and `HolovizOp`) alongside native Python operators created via the `create_op` decorator.

The start of this application imports a couple of the built in C++-based operators with Python bindings (`HolovizOp` and `VideoStreamReplayerOp`). In addition to these, two new operators are created via the `create_op` decorator APIs.

```{code-block} python
:emphasize-lines: 4-5, 10-16, 19-22

import os

from holoscan.core import Application
from holoscan.decorator import Input, Output, create_op
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


@create_op(
    inputs="tensor",
    outputs="out_tensor",
)
def invert(tensor):
    tensor = 255 - tensor
    return tensor


@create_op(inputs=Input("in", arg_map="tensor"), outputs=Output("out", tensor_names=("frame",)))
def tensor_info(tensor):
    print(f"tensor from 'in' port: shape = {tensor.shape}, " f"dtype = {tensor.dtype.name}")
    return tensor
```
The first is created by adding the decorator to a function named `invert` which just inverts the (8-bit RGB) color space values. A second operator, is created by adding the decorator to a function named `tensor_info`, which assumes that the input is a CuPy or NumPy tensor, and prints its shape and data type. Note that `create_op`'s default `cast_tensors=True` option ensures that any host or device tensors are cast to NumPy or CuPy arrays, respectively. This is why it is safe to use NumPy APIs in the function bodies. If the user wants to receive the `holoscan.Tensor` object directly and manually handle the casting to a different type of object in the function body, then `cast_tensors=False` should be specified in the keyword arguments to `create_op`.

Now that we have defined or imported all of the operators, we can build an application in the usual way by inheriting from the {py:class}`~holoscan.core.Application` class and implementing the `compose` method. The remainder of the code for this example is shown below.

```{code-block} python
:emphasize-lines: 34-35

class VideoReplayerApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - HolovizOp
    - invert (created via decorator API)
    - tensor_info (created via decorator API)

    `VideoStreamReplayerOp` reads a video file and sends the frames to the HolovizOp.
    The `invert` operator inverts the color map (the 8-bit `value` in each color channel is
    set to `255 - value`).
    The `tensor_info` operator prints information about the tensors shape and data type.
    `HolovizOp` displays the frames.
    """

    def compose(self):
        video_dir = os.path.join(sample_data_path, "racerx")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="racerx",
            frame_rate=0,  # as specified in timestamps
            repeat=False,  # default: false
            realtime=True,  # default: true
            count=40,  # default: 0 (no frame count restriction)
        )
        invert_op = invert(self, name="image_invert")
        info_op = tensor_info(self, name="tensor_info")
        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=854,
            height=480,
            # name="frame" to match Output argument to create_op for tensor_info
            tensors=[dict(name="frame", type="color", opacity=1.0, priority=0)],
        )
        # Define the workflow
        self.add_flow(replayer, invert_op, {("output", "tensor")})
        self.add_flow(invert_op, info_op, {("out_tensor", "in")})
        self.add_flow(info_op, visualizer, {("out", "receivers")})


def main():
    app = VideoReplayerApp()
    app.run()


if __name__ == "__main__":
    main()
```
The highlighted lines show how Operators corresponding to the `invert` and `tensor_info` functions are created by passing the application itself as the first argument. The `invert_op` and `info_op` variables now correspond to a {py:class}`holsocan.core.Operator` class and can be connected in the usual way using `add_flow` to define the computation. Note that a name was provided for these operators, via the optional `name` keyword argument. In this case each operator is only used once, but if the same operator is to be used more than once in an application, each should be given a unique name.


## Using create_op to turn a generator into an Operator

The `create_op` decorator can be applied to a generator in the same way as for a function. In this case, a `BooleanCondition` will automatically be added to the operator that will stop it from trying to call `compute` again once the generator is exhausted (has no more values to yield). The following is a basic example of decorating a generator for integers from 1 to `count`:


```{code-block} python
@create_op(outputs="out")
def source_generator(count):
    yield from range(1, count + 1)
```

The `compose` method can then create an operator from this decorated generator as follows
```{code-block} python
count_op = source_generator(self, count=100, name="int_source")
```


## Using create_op to turn a class into an Operator

The `create_op` decorator can also be applied to a class implementing the `__call__` method, to turn it into an {py:func}`~holoscan.core.Operator`. One reason to choose a class vs. a function is if there is some internal state that needs to be maintained across calls. For example, the operator defined below casts the input data to 32-bit floating point and on even frames also negates the values.

```{code-block} python
@create_op
class NegateEven:
    def __init__(self, start_index=0):
        self.counter = start_index

    def __call__(self, x):
        # cast tensor to 32-bit floating point
        x = x.astype('float32')

        # negate the values if the frame is even
        if self.counter % 2 == 0:
            x = -x
        return x
```

In this case, since there is only a single input and output for the function, we can omit the `inputs` and `outputs` arguments in the call to `create_op`. In this case the input port will have name `"x"`, as determined from the variable name in the function signature. The output port will be have an empty name `""`. To use different port names, the `inputs` and/or `outputs` arguments should be specified.

The `compose` method can then create an operator from this decorated generator as follows. Note that any positional or keyword arguments in the `__init__` method would be supplied during the `NegateEven` call. This returns a function (not yet an operator) that can then be called to generate the operator. This is shown below

```{code-block} python
negate_op_creation_func = NegateEven(start_index=0)                 # `negate_op_creation_func` is a function that returns an Operator
negate_even_op = negate_op_creation_func(self, name="negate_even")  # call the function to create an instance of NegateEvenOp
```

or more concisely as just
```{code-block} python
negate_even_op = NegateEven(start_index=0)(self, name="negate_even")
```

Note that the operator class as defined above is approximately equivalent to the Python native operator defined below. We show it here explicitly for reference.

```{code-block} python
import cupy as cp
import numpy as np


class NegateEvenOp(Operator):

    def __init__(self, fragment, *args, start_index=0, **kwargs):
        self.counter = start_index
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("x")
        spec.output("")

    def compute(op_input, op_output, context):
        x = op_input.receive("x")

        # cast to CuPy or NumPy array
        # (validation that `x` is a holoscan.Tensor is omitted for simplicity)
        if hasattr(x, '__cuda_array_interface__'):
            x = cupy.asarray(x)
        else:
            x = numpy.asarray(x)

        # cast tensor to 32-bit floating point
        x = x.astype('float32')

        # negate the values if the frame is even
        if self.counter % 2 == 0:
            x = -x

        op_output.emit(x, "")
```

The primary differences between this `NegateEvenOp` class and the decorated `NegateEven` above are:
- `NegateEven` does not need to define a `setup` method
- `NegateEven` does not inherit from `Operator` and so does not call its `__init__` from the constructor.
- The `NegateEven::__call__` method is simpler than the `NegateEvenOp::compute` method as `receive` and `emit` methods do not need to be explicitly called and casting to a NumPy or CuPy array is automatically handled for `NegateEven`.

