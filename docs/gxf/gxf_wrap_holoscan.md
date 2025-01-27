(gxf_wrap_holoscan)=
# Using Holoscan Operators, Resources, and Types in GXF Applications

For users who are familiar with the GXF development ecosystem, we provide an export feature to leverage native Holoscan code as GXF components to execute in GXF applications and Graph Composer.

For a streamlined approach wrapping a native C++ Holoscan operator as a GXF codelet, review the [`wrap_operator_as_gxf_extension` example on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension) as described below.

For a granular approach wrapping multiple operators and resources, review the [`wrap_holoscan_as_gxf_extension` example on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_holoscan_as_gxf_extension) as described below.

## Wrap An Operator as a GXF Extension

### 1. Creating compatible Holoscan Operators

:::{note}
This section assumes you are already familiar with {ref}`how to create a native C++ operator <native-cpp-operators>`.
:::

To ensure compatibility with GXF codelets, it is recommended to specify `holoscan::gxf::Entity` as the type for input and output ports in `Operator::setup(OperatorSpec& spec)`. This is demonstrated in the implementations of [PingTxNativeOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_tx_native_op/ping_tx_native_op.cpp) and [PingRxNativeOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_rx_native_op/ping_rx_native_op.cpp). In contrast, the built-in operators [PingTxOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/src/operators/ping_tx/ping_tx.cpp) and [PingRxOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/src/operators/ping_rx/ping_rx.cpp) use different specifications. Note that specifying the type is currently for annotation purposes only, as the Holoscan SDK does not validate input and output types. However, this behavior may change in the future.

For more details regarding the use of `holoscan::gxf::Entity`, follow the documentation on {ref}`Interoperability between GXF and native C++ operators <interoperability-with-gxf-operators-cpp>`.

### 2. Creating the GXF extension that wraps the operator

To wrap the native operator as a GXF codelet in a GXF extension, we provide the CMake `wrap_operator_as_gxf_extension` function in the SDK. An example of how it wraps `PingTxNativeOp` and `PingRxNativeOp` can be found [here](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_extension/CMakeLists.min.txt).

- It leverages the CMake target names of the operators defined in their respective `CMakeLists.txt` ([ping_tx_native_op](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_tx_native_op/CMakeLists.min.txt), [ping_rx_native_op](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_rx_native_op/CMakeLists.min.txt))
- The function parameters are documented at the top of the [WrapOperatorAsGXFExtension.cmake](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/cmake/modules/WrapOperatorAsGXFExtension.cmake#L18-42) file (ignore implementation below).

:::{note}
Use the Holoscan SDK script `generate_extension_uuids.py` to generate UUIDs for GXF-wrapped components.
:::

### 3. Using your wrapped operator in a GXF application

:::{note}
This section assumes you are familiar with {ref}`how to create a GXF application <creating-gxf-application>`.
:::

As shown in the `gxf_app/CMakeLists.txt` [here](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_app/CMakeLists.min.txt#L30-33), you need to list the following extensions in `create_gxe_application()` to use your wrapped codelets:
- `GXF::std`
- `gxf_holoscan_wrapper`
- The name of the CMake target for the created extension, defined by the `EXTENSION_TARGET_NAME` argument passed to `wrap_operator_as_gxf_extension` in the previous section.

The codelet class name (defined by the `CODELET_NAMESPACE::CODELET_NAME` arguments passed to `wrap_operator_as_gxf_extension` in the previous section) can then be used as a component `type` in a GXF application node, as shown in the [YAML app definition](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_app/ping.yaml) of the example, connecting the two ping operators.

## Wrap Multiple Components as a GXF Extension

Holoscan SDK provides several granular CMake macros to make resources and operators compatible with a GXF application.

### 1. Wrapping a `holoscan::Resource` as a `GXF::Component`

Holoscan SDK provides the CMake `generate_gxf_resource_wrapper` function to wrap a resource for GXF. The function takes a
Holoscan SDK `Resource` class and applies a wrapper template to produce the following:
- C++ source and header code wrapping the resource for GXF;
- A CMake build target to compile the code to a shared library;
- GXF C++ macro source code to use in an extension.

For instance, the [`wrap_holoscan_as_gxf_extension` example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_holoscan_as_gxf_extension/gxf_extension/CMakeLists.min.txt) demonstrates wrapping a resource as follows:

```cmake
generate_gxf_resource_wrapper(RESOURCE_HDRS RESOURCE_SRCS EXT_CPP_CONTENT
  RESOURCE_CLASS  myres::PingVarCustomNativeRes
  COMPONENT_NAME  PingVarCustomNativeResComponent
  COMPONENT_NAMESPACE myexts
  COMPONENT_TARGET_NAME gxf_wrapped_ping_variant_custom_native_res_lib
  HASH1  0xc4c16b8d6ef94a01
  HASH2  0x8014ce5b3e9602b1
  INCLUDE_HEADERS ping_variant_custom_native_res.hpp
  PUBLIC_DEPENDS ping_variant_custom_native_res
  COMPONENT_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
```
The result of this function call in the example build context is as follows:
- Generates C++ source and header code wrapping the `myres::PingVarCustomNativeRes` resource in the GXF-compatible `myexts::PingVarCustomNativeResComponent`
class. Output C++ file locations are stored in the `RESOURCE_HDRS` and `RESOURCE_SRCS` CMake variables.
- Defines a build target named `gxf_wrapped_ping_variant_custom_native_res_lib`. Running `cmake --build <build-dir> --target <gxf_wrapped_ping_variant_custom_native_res_lib>` will build the GXF resource wrapper depending on `ping_variant_custom_native_res`.
- Appends a GXF factory macro C++ code snippet to the `EXT_CPP_CONTENT` CMake variable for later use.

Repeat this step to wrap any other resources to include in your extension.

:::{note}
Use the Holoscan SDK script `generate_extension_uuids.py` to generate UUIDs for GXF-wrapped components.
:::

### 2. Wrapping a `holoscan::Operator` as a `GXF::Component`

Holoscan SDK provides the CMake `generate_gxf_operator_wrapper` function to wrap an operator for GXF. Like the Resource wrapper above, this function
creates C++ source and header code wrapping the operator as well as a build target to build the operator wrapper's shared lib. Unlike the
[earlier example](#wrap-an-operator-as-a-gxf-extension), this function does not itself generate a GXF extension.

The [`wrap_holoscan_as_gxf_extension` example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_holoscan_as_gxf_extension/gxf_extension/CMakeLists.min.txt) demonstrates wrapping an operator as follows:

```cmake
generate_gxf_operator_wrapper(TX_CODELET_HDRS TX_CODELET_SRCS EXT_CPP_CONTENT
  OPERATOR_CLASS "myops::PingVarTxNativeOp"
  CODELET_NAME PingVarTxNativeOpCodelet
  CODELET_NAMESPACE myexts
  HASH1 0x35545ef8ae1541c5
  HASH2 0x8aef3c2078fc50b4
  CODELET_TARGET_NAME gxf_wrapped_ping_variant_tx_native_op_lib
  DESCRIPTION "Ping Tx Native Operator codelet"
  INCLUDE_HEADERS ping_variant_tx_native_op.hpp
  PUBLIC_DEPENDS ping_variant_tx_native_op
  CODELET_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
```
The result of this function call in the example build context is as follows:
- Generates C++ source and header code wrapping the `myop::PingVarTxNativeOp` operator resource in the GXF-compatible `myexts::PingVarTxNativeOpCodelet` class. Output C++ file locations are stored in the `TX_CODELET_HDRS` and `TX_CODELET_SRCS` CMake variable.
- Defines a build target named `gxf_wrapped_ping_variant_tx_native_op_lib`. Running `cmake --build <build-dir> --target <gxf_wrapped_ping_variant_tx_native_op_lib>` will build the GXF ping operator wrapper depending on `ping_variant_tx_native_op`.
- Appends a GXF factory macro C++ code snippet to the `EXT_CPP_CONTENT` CMake variable for later use.

Repeat this step to wrap any other operators to include in your extension.

### 3. Generating a combined GXF Extension

Holoscan SDK provides the CMake `generate_gxf_extension` function to bundle wrapped components in a GXF extension template. The function
accepts extension details and C++ GXF function content generated from the preceding sequence of wrapping function calls.
Each component registered in the GXF extension can then be instantiated by the GXF component factory at application runtime.

The [`wrap_holoscan_as_gxf_extension` example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_holoscan_as_gxf_extension/gxf_extension/CMakeLists.min.txt) demonstrates generating a GXF extension as follows:

```cmake
generate_gxf_extension(
  EXTENSION_TARGET_NAME gxf_wrapped_ping_variant_ext
  EXTENSION_NAME PingVarCustomNativeResExtension
  EXTENSION_DESCRIPTION
    "Ping Variant Custom Native extension. Includes wrapped Holoscan custom resource and tx/rx operators"
  EXTENSION_AUTHOR "NVIDIA"
  EXTENSION_VERSION "${holoscan_VERSION}"
  EXTENSION_LICENSE "Apache-2.0"
  EXTENSION_ID_HASH1 0x2b8381ed5c2740a1
  EXTENSION_ID_HASH2 0xbe586c019eaa87be
  INCLUDE_HEADERS
    ${RESOURCE_HDRS}
    ${TX_CODELET_HDRS}
    ${RX_CODELET_HDRS}
  PUBLIC_DEPENDS
    gxf_wrapped_ping_variant_custom_native_res_lib
    gxf_wrapped_ping_variant_tx_native_op_lib
    gxf_wrapped_ping_variant_rx_native_op_lib
  EXTENSION_TARGET_PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  EXT_CPP_CONTENT "${EXT_CPP_CONTENT}"
)
```
The `generate_gxf_extension` function call generates a CMake build target `gxf_wrapped_ping_variant_ext`
to build the GXF extension shared library. The resulting extension may be included in a GXF context
to make the custom resource and Tx/Rx operators available to the GXF application.

### 4. Using your wrapped components in a GXF application

:::{note}
This section assumes you are familiar with {ref}`how to create a GXF application<creating-gxf-application>`.
:::

GXF extensions generated with Holoscan SDK wrappings rely on the Holoscan Wrapper extension. When listing
extensions in your application manifest, please ensure dependencies are observed in order:
- `GXF::std`
- `ucx_holoscan_extension`
- `gxf_holoscan_wrapper`
- Your custom wrapper extension name

Review the example [YAML app definition](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_holoscan_as_gxf_extension/gxf_app/ping.yaml)
for a demonstration of how a custom resource can be used as a parameter to the tx operator in the app.
