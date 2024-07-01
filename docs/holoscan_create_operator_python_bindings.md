(holoscan-create-operators-python-bindings)=
# Writing Python bindings for a C++ Operator


For convenience while maintaining high performance, {ref}`operators written in C++<holoscan-defining-operators-cpp>` can be wrapped in Python. The general approach uses [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html) to concisely create bindings that provide a familiar, Pythonic experience to application authors. 

:::{note}
While we provide some utilities to simplify part of the process, this section is designed for advanced developers, since the wrapping of the C++ class using pybind11 is mostly manual and can vary between each operator.
:::

The existing Pybind11 documentation is good and it is recommended to read at least the basics on wrapping [functions](https://pybind11.readthedocs.io/en/stable/basics.html#creating-bindings-for-a-simple-function) and [classes](https://pybind11.readthedocs.io/en/stable/classes.html#object-oriented-code). The material below will assume some basic familiarity with Pybind11, covering the details of creation of the bindings of a C++ {cpp:class}`~holoscan::Operator`. As a concrete example, we will cover creation of the bindings for [ToolTrackingPostprocessorOp](https://github.com/nvidia-holoscan/holohub/tree/main/operators/tool_tracking_postprocessor) from [Holohub](https://github.com/nvidia-holoscan/holohub) as a simple case and then highlight additional scenarios that might be encountered.

:::{tip}
There are several examples of bindings on Holohub in the [operators folder](https://github.com/nvidia-holoscan/holohub/tree/main/operators). The subset of operators that provide a Python wrapper on top of a C++ implementation will have any C++ headers and sources together in a common folder, while any corresponding Python bindings will be in a "python" subfolder (see the [tool_tracking_postprocessor](https://github.com/nvidia-holoscan/holohub/tree/main/operators/tool_tracking_postprocessor) folder layout, for example).

There are also several [examples of bindings](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/python/holoscan/operators) for the built-in operators of the SDK. Unlike on Holohub, for the SDK, the corresponding C++ [headers](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/include/holoscan/operators) and [sources](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/src/operators) of an operator are stored under separate directory trees.
:::

(pybind11-operator-tutorial)=
## Tutorial: binding the ToolTrackingPostprocessorOp class

(pybind11-operator-trampoline)=
### Creating a PyToolTrackingPostprocessorOp trampoline class

In a C++ file ([tool_tracking_postprocessor.cpp](https://github.com/nvidia-holoscan/holohub/blob/main/operators/tool_tracking_postprocessor/python/tool_tracking_postprocessor.cpp) in this case), create a subclass of the C++ Operator class to wrap. The general approach taken is to create a Python-specific class that provides a constructor that takes a `Fragment*`, an explicit list of the operators parameters with default values for any that are optional, and an operator name. This constructor needs to setup the operator as done in [`Fragment::make_operator`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/include/holoscan/core/fragment.hpp#L284), so that it is ready for initialization by the GXF executor. We use the convention of prepending "Py" to the C++ class name for this (so, `PyToolTrackingPostprocessorOp` in this case). :


```{code-block} cpp
:caption: tool_tracking_post_processor/python/tool_tracking_post_processor.cpp

class PyToolTrackingPostprocessorOp : public ToolTrackingPostprocessorOp {
 public:
  /* Inherit the constructors */
  using ToolTrackingPostprocessorOp::ToolTrackingPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PyToolTrackingPostprocessorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> device_allocator,
      std::shared_ptr<Allocator> host_allocator, float min_prob = 0.5f,
      std::vector<std::vector<float>> overlay_img_colors = VIZ_TOOL_DEFAULT_COLORS,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "tool_tracking_postprocessor")
      : ToolTrackingPostprocessorOp(ArgList{Arg{"device_allocator", device_allocator},
                                            Arg{"host_allocator", host_allocator},
                                            Arg{"min_prob", min_prob},
                                            Arg{"overlay_img_colors", overlay_img_colors},
                                            }) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};
```

This constructor will allow providing a Pythonic experience for creating the operator. Specifically, the user can pass Python objects for any of the parameters without having to explicitly create any {cpp:class}`holoscan::Arg` objects via {py:class}`holoscan.core.Arg`. For example, a standard Python float can be passed to `min_prob` and a Python `list[list[float]]` can be passed for `overlay_img_colors` (Pybind11 handles conversion between the C++ and Python types). Pybind11 will also take care of conversion of a Python allocator class like `holoscan.resources.UnboundedAllocator` or `holoscan.resources.BlockMemoryPool` to the underlying C++ `std::shared_ptr<holoscan::Allocator>` type. The arguments `device_allocator` and `host_allocator` correspond to required Parameters of the C++ class and can be provided from Python either positionally or via keyword while the Parameters `min_prob` and `overlay_img_colors` will be optional keyword arguments. `cuda_stream_pool` is also optional, but is only conditionally passed as an argument to the underlying `ToolTrackingPostprocessorOp` constructor when it is not a `nullptr`.

- For all operators, the first argument should be `Fragment* fragment` and is the fragment the operator will be assigned to. In the case of a single fragment application (i.e. not a distributed application), the fragment is just the application itself.
- An (optional) `const std::string& name` argument should be provided to enable the application author to set the operator's name.
- The `const py::args& args` argument corresponds to the `*args` notation in Python. It is a set of 0 or more positional arguments. It is not required to provide this in the function signature, but is recommended in order to enable passing additional conditions such as a `CountCondition` or `PeriodicCondtion` as positional arguments to the operator. The call below to

  ```cpp
  add_positional_condition_and_resource_args(this, args);
  ```

  uses a helper function defined in [operator_util.hpp](https://github.com/nvidia-holoscan/holohub/blob/main/operators/operator_util.hpp) to add any {py:class}`~holoscan.core.Condition` or {py:class}`~holoscan.core.Resource` arguments found in the list of positional arguments.
- The other arguments all correspond to the various parameters ({cpp:class}`holoscan::Parameter`) that are defined for the C++ `ToolTrackingPostProcessorOp` class.
  - All other parameters except `cuda_stream_pool` are passed directly in the argument list to the parent `ToolTrackingPostProcessorOp` class. The parameters present on the C++ operator can be seen in its header [here](https://github.com/grlee77/holohub/blob/3adbba16baafb5958950b261a0d6521f7544cfeb/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.hpp#L46-L52) with default values taken from the `setup` method of the source file [here](https://github.com/grlee77/holohub/blob/3adbba16baafb5958950b261a0d6521f7544cfeb/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.cpp#L77-L89). Note that {cpp:class}`CudaStreamHandler` is a utility that will add a parameter of type `Parameter<std::shared_ptr<CudaStreamPool>>`.
  - The `cuda_stream_pool` argument is only conditionally added if it was not `nullptr` (Python's `None`). This is done via
    ```cpp
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    ```
    instead of passing it as part of the {cpp:class}`holoscan::ArgList` provided to the `ToolTrackingPostprocessorOp` constructor call above.

The remaining lines of the constructor
```cpp
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
```
are required to properly initialize it and should be the same across all operators. These [correspond to equivalent code within the Fragment::make_operator method](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/include/holoscan/core/fragment.hpp#L287-L291).

(pybind11-operator-module-definition)=
### Defining the Python module

For this operator, there are no other custom classes aside from the operator itself, so we define a module using `PYBIND11_MODULE` as shown below with only a single class definition. This is done in the same [tool_tracking_postprocessor.cpp](https://github.com/nvidia-holoscan/holohub/blob/main/operators/tool_tracking_postprocessor/python/tool_tracking_postprocessor.cpp) file where we defined the `PyToolTrackingPostprocessorOp` trampoline class.

The following header will always be needed.
```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;
using pybind11::literals::operator""_a;
```
Here, we typically also add defined the `py` namespace as a shorthand for `pybind11` and indicated that we will use the `_a` literal (it provides a shorthand notation when [defining keyword arguments](https://pybind11.readthedocs.io/en/stable/basics.html#keyword-arguments)).

Often it will be necessary to include the following header if any parameters to the operator involve C++ standard library containers such as `std::vector` or `std::unordered_map`.
```cpp
#include <pybind11/stl.h>
```
This allows pybind11 to cast between the C++ container types and corresponding Python types (Python `dict` / C++ `std::unordered_map`, for example).


```{code-block} cpp
:caption: tool_tracking_post_processor/python/tool_tracking_post_processor.cpp

PYBIND11_MODULE(_tool_tracking_postprocessor, m) {
  py::class_<ToolTrackingPostprocessorOp,
             PyToolTrackingPostprocessorOp,
             Operator,
             std::shared_ptr<ToolTrackingPostprocessorOp>>(
      m,
      "ToolTrackingPostprocessorOp",
      doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python)
      .def(py::init<Fragment*,
                    const py::args& args,
                    std::shared_ptr<Allocator>,
                    std::shared_ptr<Allocator>,
                    float,
                    std::vector<std::vector<float>>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "device_allocator"_a,
           "host_allocator"_a,
           "min_prob"_a = 0.5f,
           "overlay_img_colors"_a = VIZ_TOOL_DEFAULT_COLORS,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "tool_tracking_postprocessor"s,
           doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python);
}  // PYBIND11_MODULE NOLINT
```


:::{note}
- If you are implementing the python wrapping in Holohub, the `<module_name>` passed to `PYBIND_11_MODULE` **must** match `_<CPP_CMAKE_TARGET>` as [covered above](#pybind11-module_name_warning).
- If you are implementing the python wrapping in a standalone CMake project,the `<module_name>` passed to `PYBIND_11_MODULE` **must** match the name of the module passed to the [pybind11-add-module](https://pybind11.readthedocs.io/en/stable/compiling.html#pybind11-add-module) CMake function.

Using a mismatched name in `PYBIND_11_MODULE` will result in failure to import the module from Python.
:::

The order in which the classes are specified in the `py::class_<>` template call is important and should follow the convention shown here. The first in the list is the C++ class name (`ToolTrackingPostprocessorOp`) and second is the `PyToolTrackingPostprocessorOp` class we defined above with the additional, explicit constructor. We also need to list the parent `Operator` class so that all of the methods such as `start`, `stop`, `compute`, `add_arg`, etc. that were already wrapped for the parent class don't need to be redefined here.

The single `.def(py::init<...` call wraps the `PyToolTrackingPostprocessorOp` constructor we wrote above. As such, the argument types provided to `py::init<>` must exactly match the order and types of arguments in that constructor's function signature. The subsequent arguments to `def` are the names and default values (if any) for the named arguments in the same order as the function signature. Note that the `const py::args& args` (Python `*args`) argument is not listed as these are positional arguments that don't have a corresponding name. The use of `py::none()` (Python's `None`) as the default for `cuda_stream_pool` corresponds to the `nullptr` in the C++ function signature. The "_a" literal used in the definition is enabled by the following declaration earlier in the file.

The final argument to `.def` here is a documentation string that will serve as the Python docstring for the function. It is optional and we chose here to define it in a separate header as described in the next section.

(pybind11-operator-docstrings)=
### Documentation strings

Prepare documentation strings (`const char*`) for your python class and its parameters.

:::{note}
Below we use a `PYDOC` macro defined in the [SDK](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/macros.hpp) and available in [HoloHub](https://github.com/nvidia-holoscan/holohub/blob/main/cmake/pydoc/macros.hpp) as a utility to remove leading spaces. In this case, the documentation code is located in header file [tool_tracking_post_processor_pydoc.hpp](https://github.com/nvidia-holoscan/holohub/blob/main/operators/tool_tracking_postprocessor/python/tool_tracking_postprocessor_pydoc.hpp), under a custom `holoscan::doc::ToolTrackingPostprocessorOp` namespace. None of this is required, you just need to make any documentation strings available for use as an argument to the `py::class_` constructor or method definition calls.
:::

```{code-block} cpp
:caption: tool_tracking_post_processor/python/tool_tracking_post_processor_pydoc.hpp

#include "../macros.hpp"

namespace holoscan::doc {

namespace ToolTrackingPostprocessorOp {

// PyToolTrackingPostprocessorOp Constructor
PYDOC(ToolTrackingPostprocessorOp_python, R"doc(
Operator performing post-processing for the endoscopy tool tracking demo.

**==Named Inputs==**

    in : nvidia::gxf::Entity containing multiple nvidia::gxf::Tensor
        Must contain input tensors named "probs", "scaled_coords" and "binary_masks" that
        correspond to the output of the LSTMTensorRTInfereceOp as used in the endoscopy
        tool tracking example applications.

**==Named Outputs==**

    out_coords : nvidia::gxf::Tensor
        Coordinates tensor, stored on the host (CPU).

    out_mask : nvidia::gxf::Tensor
        Binary mask tensor, stored on device (GPU).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
device_allocator : ``holoscan.resources.Allocator``
    Output allocator used on the device side.
host_allocator : ``holoscan.resources.Allocator``
    Output allocator used on the host side.
min_prob : float, optional
    Minimum probability (in range [0, 1]). Default value is 0.5.
overlay_img_colors : sequence of sequence of float, optional
    Color of the image overlays, a list of RGB values with components between 0 and 1.
    The default value is a qualitative colormap with a sequence of 12 colors.
cuda_stream_pool : ``holoscan.resources.CudaStreamPool``, optional
    `holoscan.resources.CudaStreamPool` instance to allocate CUDA streams.
    Default value is ``None``.
name : str, optional
    The name of the operator.
)doc")

}  // namespace ToolTrackingPostprocessorOp
}  // namespace holoscan::doc
```

We tend to use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for parameters, but also encourage adding a custom section at the top that describes the input and output ports and what type of data is expected on them. This can make it easier for developers to use the operator without having to inspect the source code to determine this information.

### Configuring with CMake

We use CMake to configure pybind11 and build the bindings for the C++ operator you wish to wrap. There are two approaches detailed below, one for HoloHub (recommended), one for standalone CMake projects.

:::{tip}
To have your bindings built, ensure the CMake code below is executed as part of a CMake project which already defines the C++ operator as a CMake target, either built in your project (with `add_library`) or imported (with `find_package` or `find_library`).
:::

`````{tab-set}
````{tab-item} In HoloHub
We provide a CMake utility function named [pybind11_add_holohub_module](https://github.com/nvidia-holoscan/holohub/blob/main/cmake/pybind11_add_holohub_module.cmake) in HoloHub to facilitate configuring and building your python bindings.

In our skeleton code below, a top-level CMakeLists.txt which already defined the `tool_tracking_postprocessor` target for the C++ operator would need to do `add_subdirectory(tool_tracking_postprocessor)` to include the following [CMakeLists.txt](https://github.com/nvidia-holoscan/holohub/blob/main/operators/tool_tracking_postprocessor/python/CMakeLists.txt). The `pybind11_add_holohub_module` lists that C++ operator target, the C++ class to wrap, and the path to the C++ binding source code we implemented above.  Note how the module name provided as the first argument to PYPBIND11_MODULE needs to match `_<CPP_CMAKE_TARGET>` (`_tool_tracking_postprocessor_op` in this case).

```{code-block} cmake
:caption: tool_tracking_postprocessor/python/CMakeLists.txt

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET tool_tracking_postprocessor
    CLASS_NAME "ToolTrackingPostprocessorOp"
    SOURCES tool_tracking_postprocessor.cpp
)
```

The key details here are that `CLASS_NAME` should match the name of the C++ class that is being wrapped and is also the name that will be used for the class from Python. `SOURCES` should point to the file where the C++ operator that is being wrapped is defined. The `CPP_CMAKE_TARGET` name will be the name of the holohub package submodule that will contain the operator. 

Note that the python subdirectory where this CMakeLists.txt resides is reachable thanks to the `add_subdirectory(python)` in the [CMakeLists.txt one folder above](https://github.com/nvidia-holoscan/holohub/blob/30d2797e37615f87056075b36ebf1d905b6c770b/operators/tool_tracking_postprocessor/CMakeLists.txt#L33-L35), but that's an arbitrary opinionated location and not a required directory structure.

````
````{tab-item} Standalone CMake

Follow the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake) to configure your CMake project to use pybind11. Then, use the [pybind11_add_module](https://pybind11.readthedocs.io/en/stable/compiling.html#pybind11-add-module) function with the cpp files containing the code above, and link against `holoscan::core` and the library that exposes your C++ operator to wrap.

```{code-block} cmake
:caption: my_op_python/CMakeLists.txt

pybind11_add_module(my_python_module my_op_pybind.cpp)
target_link_libraries(my_python_module
  PRIVATE holoscan::core
  PUBLIC my_op
)
```

**Example**: in the SDK, this is done [here](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/CMakeLists.txt).

````
`````

(pybind11-module_name_warning)=
:::{warning}
The name chosen for `CPP_CMAKE_TARGET` **must** also be used (along with a preceding underscore) as the module name passed as the first argument to the [PYBIND11_MODULE macro in the bindings](https://github.com/grlee77/holohub/blob/3adbba16baafb5958950b261a0d6521f7544cfeb/operators/tool_tracking_postprocessor/python/tool_tracking_postprocessor.cpp#L94).

Note that there is an initial underscore prepended to the name. This is the naming convention used for the shared library and corresponding `__init__.py` file that will be generated by the `pybind11_add_holohub_module` helper function above.

If the name is specified incorrectly, the build will still complete, but at application run time an `ImportError` such as the following would occur

```bash
[command] python3 /workspace/holohub/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --data /workspace/holohub/data/endoscopy
Traceback (most recent call last):
  File "/workspace/holohub/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py", line 38, in <module>
    from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp
  File "/workspace/holohub/build/python/lib/holohub/tool_tracking_postprocessor/__init__.py", line 19, in <module>
    from ._tool_tracking_postprocessor import ToolTrackingPostprocessorOp
ImportError: dynamic module does not define module export function (PyInit__tool_tracking_postprocessor)
```
:::


### Importing the class in Python

`````{tab-set}
````{tab-item} In HoloHub

When building your project, two files will be generated inside `<build_or_install_dir>/python/lib/holohub/<CPP_CMAKE_TARGET>` (e.g. `build/python/lib/holohub/tool_tracking_postprocessor/`):
1. the shared library for your bindings (`_tool_tracking_postprocessor_op.cpython-<pyversion>-<arch>-linux-gnu.so`)
2. an `__init__.py` file that makes the necessary imports to expose this in python

Assuming you have `export PYTHONPATH=<build_or_install_dir>/python/lib/`, you should then be able to create an application in Holohub that imports your class via:

```python
from holohub.tool_tracking_postprocessor_op import ToolTrackingPostProcessorOp
```
**Example**: `ToolTrackingPostProcessorOp` is imported in the Endoscopy Tool Tracking application on HoloHub [here](https://github.com/nvidia-holoscan/holohub/blob/30d2797e37615f87056075b36ebf1d905b6c770b/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py#L38).

````
````{tab-item} Standalone CMake

When building your project, a shared library file holding the python bindings and named `my_python_module.cpython-<pyversion>-<arch>-linux-gnu.so` will be generated inside `<build_or_install_dir>/my_op_python` (configurable with `OUTPUT_NAME` and `LIBRARY_OUTPUT_DIRECTORY` respectively in CMake).

From there, you can import it in python via:

```py
import holoscan.core
import holoscan.gxf  # if your c++ operator uses gxf extensions

from <build_or_install_dir>.my_op_python import MyOp
```

:::{tip}
To imitate HoloHub's behavior, you can also place that file alongside the .so file, name it `__init__.py`, and replace `<build_or_install_dir>.` by `.`. It can then be imported as a python module, assuming `<build_or_install_dir>` is a module under the `PYTHONPATH` environment variable.
:::
````
`````

(pybind11-details)=
## Additional Examples

In this section we will cover other cases that may occasionally be encountered when writing Python bindings for operators.

### Optional arguments

It is also possible to use `std::optional` to handle optional arguments. The `ToolTrackingProcessorOp` example above, for example, has a default argument defined in the spec for `min_prob`.

```cpp
  constexpr float DEFAULT_MIN_PROB = 0.5f;
  // ...

  spec.param(
      min_prob_, "min_prob", "Minimum probability", "Minimum probability.", DEFAULT_MIN_PROB);
```

In the tutorial for `ToolTrackingProcessorOp` above we reproduced this default of 0.5 in both the `PyToolTrackingProcessorOp` constructor function signature as well as the Python bindings defined for it. This carries the risk that the default could change at the C++ operator level without a corresponding change being made for Python.

An alternative way to define the constructor would have been to use `std::optional` as follows

```cpp
  // Define a constructor that fully initializes the object.
  PyToolTrackingPostprocessorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> device_allocator,
      std::shared_ptr<Allocator> host_allocator, std::optional<float> min_prob = 0.5f,
      std::optional<std::vector<std::vector<float>>> overlay_img_colors = VIZ_TOOL_DEFAULT_COLORS,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "tool_tracking_postprocessor")
      : ToolTrackingPostprocessorOp(ArgList{Arg{"device_allocator", device_allocator},
                                            Arg{"host_allocator", host_allocator},
                                            }) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    if (min_prob.has_value()) { this->add_arg(Arg{"min_prob", min_prob.value() }); }
    if (overlay_img_colors.has_value()) {
        this->add_arg(Arg{"overlay_img_colors", overlay_img_colors.value() });
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
```
where now that `min_prob` and `overlay_img_colors` are optional, they are only conditionally added as an argument to ToolTrackingPostprocessorOp when they have a value. If this approach is used, the Python bindings for the constructor should be updated to use `py::none()` as the default as follows:

```cpp
      .def(py::init<Fragment*,
                    const py::args& args,
                    std::shared_ptr<Allocator>,
                    std::shared_ptr<Allocator>,
                    float,
                    std::vector<std::vector<float>>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "device_allocator"_a,
           "host_allocator"_a,
           "min_prob"_a = py::none(),
           "overlay_img_colors"_a = py::none(),
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "tool_tracking_postprocessor"s,
           doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python);
```


### C++ enum parameters as arguments

Sometimes, operators may use a Parameter with an enum type. It is necessary to wrap the C++ enum to be able to use it as a Python type when providing the argument to the operator.

The built-in {cpp:class}`holoscan::ops::AJASourceOp` is an example of a C++ operator that takes a [enum Parameter](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/operators/aja_source/aja_source.cpp#L58) (an `NTV2Channel` enum).

The enum can easily be wrapped for use from Python via `py::enum_` as shown [here](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/operators/aja_source/aja_source.cpp#L98-L108). It is recommended in this case to follow Python's convention of using capitalized names in the enum.

### (Advanced) Custom C++ classes as arguments

Sometimes it is necessary to accept a custom C++ class type as an argument in the operator's constructor. In this case additional interface code and bindings will likely be necessary to support the type.

A relatively simple example of this is the {cpp:class}`~holoscan::ops::InferenceProcessorOp::DataVecMap` type used by {cpp:class}`~holoscan::ops::InferenceProcessorOp`. In that case, the type is a structure that holds an internal `std::map<std:string, std::vector<std::string>>`. The bindings are written to accept a Python dict (`py::dict`) and a [helper function](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/operators/inference_processor/inference_processor.cpp#L52-L58) is used within the constructor to convert that dictionary to the corresponding C++ `DataVecMap`.

A more complicated case is the use of a {cpp:class}`~holoscan::ops::HolovizOp::InputSpec` type in the `HolovizOp` bindings. This case involves creating Python bindings for classes {cpp:class}`~holoscan::ops::HolovizOp::InputSpec` and {cpp:class}`~holoscan::ops::HolovizOp::InputSpec::View` as well as a couple of enum types. To avoid the user having to build a `list[holoscan.operators.HolovizOp.InputSpec]` directly to pass as the `tensors` argument, an [additional Python wrapper class](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v1.0.3/python/holoscan/operators/holoviz/__init__.py#L100-L182) was defined in the `__init__.py` to allow passing a simple Python dict for the `tensors` argument and any corresponding InputSpec classes are automatically created in its constructor before calling the underlying Python bindings class.


### Customizing the C++ types a Python operator can emit or receive

In some instances, users may wish to be able to have a Python operator receive and/or emit a custom C++ type. As a first example, suppose we are wrapping an operator that emits a custom C++ type. We need any downstream native Python operators to be able to receive that type. By default the SDK is able to handle the needed C++ types for built in operators like `std::vector<holoscan::ops::HolovizOp::InputSpec>`. The SDK provides an `EmitterReceiverRegistry` class that 3rd party projects can use to register `receiver` and `emitter` methods for any custom C++ type that needs to be handled. To handle a new type, users should implement an `emitter_receiver<T>` struct for the desired type as in the example below. We will first cover the general steps necessary to register such a type and then cover where some steps may be omitted in certain simple cases.

#### Step 1: define emitter_receiver<T>::emit and emitter_receiver<T>::receive methods

Here is an example for the built-in `std::vector<holoscan::ops::HolovizOp::InputSpec>` used by `HolovizOp` to define the input specifications for its received tensors.

```cpp
#include <holoscan/python/core/emitter_receiver_registry.hpp>

namespace py = pybind11;

namespace holoscan {

/* Implements emit and receive capability for the HolovizOp::InputSpec type.
 */
template <>
struct emitter_receiver<std::vector<holoscan::ops::HolovizOp::InputSpec>> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto input_spec = data.cast<std::vector<holoscan::ops::HolovizOp::InputSpec>>();
    py::gil_scoped_release release;
    op_output.emit<std::vector<holoscan::ops::HolovizOp::InputSpec>>(input_spec, name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    HOLOSCAN_LOG_DEBUG("py_receive: std::vector<HolovizOp::InputSpec> case");
    // can directly return vector<InputSpec>
    auto specs = std::any_cast<std::vector<holoscan::ops::HolovizOp::InputSpec>>(result);
    py::object py_specs = py::cast(specs);
    return py_specs;
  }
};

}
```

This `emitter_receiver` class defines a `receive` method that takes a `std::any` message and casts it to the corresponding Python `list[HolovizOp.InputSpect]` object. Here the `pybind11::cast` call works because we have wrapped the `HolovizOp::InputSpec` class [here](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v2.0.0/python/holoscan/operators/holoviz/holoviz.cpp#L190-L207).

Similarly, the `emit` method takes a `pybind11::object` (of type `list[HolovizOp.InputSpect]`) and casts it to the corresponding C++ type, `std::vector<holoscan::ops::HolovizOp::InputSpec>`. The conversion between `std::vector` and a Python list is one of Pbind11's built-in conversions (available as long as "pybind11/stl.h" has been included).

The signature of the `emit` and `receive` methods must exactly match the case shown here.

#### Step 2: Create a register_types method for adding custom types to the EmitterReceiverRegistry.

The bindings in this operators module, should define a method named `register_types` that takes a reference to an `EmitterReceiverRegistry` as its only argument. Within this function there should be a call to `EmitterReceiverRegistry::add_emitter_receiver` for each type that this operator wished to register. The HolovizOp defines this method using a lambda function

```cpp
  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
        "std::vector<HolovizOp::InputSpec>"s);
    // array camera pose object
    registry.add_emitter_receiver<std::shared_ptr<std::array<float, 16>>>(
        "std::shared_ptr<std::array<float, 16>>"s);
    // Pose3D camera pose object
    registry.add_emitter_receiver<std::shared_ptr<nvidia::gxf::Pose3D>>(
        "std::shared_ptr<nvidia::gxf::Pose3D>"s);
    // camera_eye_input, camera_look_at_input, camera_up_input
    registry.add_emitter_receiver<std::array<float, 3>>("std::array<float, 3>"s);
  });
```

Here the following line registers the `std::vector<holoscan::ops::HolovizOp::InputSpec>` type
that we wrote an `emitter_receiver` for above.

```cpp
registry.add_emitter_receiver<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
        "std::vector<HolovizOp::InputSpec>"s);
```
Internally the registry stores a mapping between the C++ `std::type_index` of the type specified in the template argument and the `emitter_receiver` defined for that type. The second argument is a string that the user can choose which is a label for the type. As we will see later, this label can be used from Python to indicate that we want to emit using the `emitter_receiver::emit` method that was registered for a particular label.

#### Step 3: In the __init__.py file for the Python module defining the operator call register_types

To register types with the core SDK, we need to import the `io_type_registry` class (of type `EmitterReceiverRegistry`) from `holoscan.core`. We then pass that class as input to the `register_types` method defined in step 2 to register the 3rd party types with the core SDK.

```python
from holoscan.core import io_type_registry

from ._holoviz import register_types as _register_types

# register methods for receiving or emitting list[HolovizOp.InputSpec] and camera pose types
_register_types(io_type_registry)
```

where we chose to import `register_types` with an initial underscore as a common Python convention to indicate it is intended to be "private" to this module.

#### In some cases steps 1 and 3 as shown above are not necessary.

When creating Python bindings for an Operator on Holohub, the [pybind11_add_holohub_module.cmake](https://github.com/nvidia-holoscan/holohub/blob/main/cmake/pybind11_add_holohub_module.cmake) utility mentioned above will take care of autogenerating the `__init__.py` as shown in step 3, so it will not be necessary to manually create it in that case.

For types for which Pybind11's default casting between C++ and Python is adequate, it is not necessary to explicitly define the `emitter_receiver` class as shown in step 1. This is true because there are a couple of [default implementations](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v2.1.0/python/holoscan/core/emitter_receiver_registry.hpp) for `emitter_receiver<T>` and `emitter_receiver<std::shared_ptr<T>>` that already cover common cases. The default emitter_receiver works for the `std::vector<HolovizOp::InputSpec>` type shown above, which is why the code shown for illustration there is [not found within the operator's bindings](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/python/holoscan/operators/holoviz/holoviz.cpp). In that case one could immediately implement `register_types` from step 2 without having to explicitly create an `emitter_receiver` class.

An example where the default `emitter_receiver` would not work is the custom one defined by the SDK for `pybind11::dict`. In this case, to provide convenient emit of multiple tensors via passing a `dict[holoscan::Tensor]` to `op_output.emit` we have special handling of Python dictionaries. The dictionary is inspected and if all keys are strings and all values are tensor-like objects, a single C++ `nvidia::gxf::Entity` containing all of the tensors as an `nvidia::gxf::Tensor` is emitted. If the dictionary is not a tensor map, then it is just emitted as a shared pointer to the Python dict object. The `emitter_receiver` implementations used for the core SDK are defined in [emitter_receivers.hpp](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v2.2.0/python/holoscan/core/emitter_receivers.hpp). These can serve as a reference when creating new ones for additional types.

#### Runtime behavior of emit and receive

After registering a new type, receive of that type on any input port will automatically be handled. This is because due to the strong typing of C++, any `op_input.receive` call in an operator's `compute` method can find the registered `receive` method that matches the `std::type_index` of the type and use that to convert to a corresponding Python object.

Because Python is not strongly typed, on `emit`, the default behavior remains emitting a shared pointer to the Python object itself. If we instead want to `emit` a C++ type, we can pass a 3rd argument to `op_output.emit` to specify the name that we used when registering the types via the `add_emitter_receiver` call as above.

#### Example of emitting a C++ type
As a concrete example, the SDK already registers `std::string` by default. If we wanted, for instance, to emit a Python string as a C++ `std::string` for use by a downstream operator that is wrapping a C++ operator expecting string input, we would add a 3rd argument to the `op_output.emit` call as follows

```py
# emit a Python filename string on port "filename_out" using registered type "std::string"
my_string = "filename.raw"
op_output.emit(my_string, "filename_out", "std::string")
```

This specifies that the `emit` method that converts to C++ `std::string` should be used instead of the default behavior of emitting the Python string.

Another example would be to emit a Python `List[float]` as a `std::array<float, 3>` parameter as input to the `camera_eye`, `camera_look_at` or `camera_up` input ports of `HolovizOp`.

```py
op_output.emit([0.0, 1.0, 0.0], "camera_eye_out", "std::array<float, 3>")
```

Only types registered with the SDK can be specified by name in this third argument to `emit`.

#### Table of types registered by the core SDK

The list of types that are registered with the SDK's `EmitterReceiverRegistry` are given in the table below.

C++ Type                                               | name in the EmitterReceiverRegistry
-------------------------------------------------------|-------------------------------------------
holoscan::Tensor                                       | "holoscan::Tensor"
std::shared_ptr&lt;holoscan::GILGuardedPyObject&gt;    | "PyObject"
std::string                                            | "std::string"
pybind11::dict                                         | "pybind11::dict"
holoscan::gxf::Entity                                  | "holoscan::gxf::Entity"
holoscan::PyEntity                                     | "holoscan::PyEntity"
nullptr_t                                              | "nullptr_t"
CloudPickleSerializedObject                            | "CloudPickleSerializedObject"
std::array&lt;float, 3&gt;                             | "std::array&lt;float, 3&gt;"
std::shared_ptr&lt;std::array&lt;float, 16&gt;&gt;     | "std::shared_ptr&lt;std::array&lt;float, 16&gt;&gt;"
std::shared_ptr&lt;nvidia::gxf::Pose3D&gt;             | "std::shared_ptr&lt;nvidia::gxf::Pose3D&gt;"
std::vector&lt;holoscan::ops::HolovizOp::InputSpec&gt; | "std::vector&lt;HolovizOp::InputSpec&gt;"

:::{note}
There is no requirement that the registered name match any particular convention. We generally used
the C++ type as the name to avoid ambiguity, but that is not required.
:::

The sections above explain how a `register_types` function can be added to bindings to expand this list. It is also possible to get a list of all currently registered types, including those that have been registered by any additional imported 3rd party modules. This can be done via

```py
from holoscan.core import io_type_registry

print(io_type_registry.registered_types())
```
