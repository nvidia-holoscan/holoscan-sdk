(gxf_wrap_holoscan_op)=
# Using Holoscan Operators in GXF Applications

For users who are familiar with the GXF development ecosystem (used in Holoscan SDK 0.2), we provide an export feature to leverage native Holoscan operators as GXF codelets to execute in GXF applications and GraphComposer.

We demonstrate how to wrap a native C++ holoscan operator as a GXF codelet in the [`wrap_operator_as_gxf_extension` example on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension), as described below.

## 1. Creating compatible Holoscan Operators

:::{note}
This section assumes you are already familiar with {ref}`how to create a native C++ operator<native-cpp-operators>`.
:::

To be compatible with GXF codelets, inputs and outputs specified in `Operator::setup(OperatorSpec& spec)` must be of type `holoscan::gxf::Entity`, as shown in the [PingTxNativeOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_tx_native_op/ping_tx_native_op.cpp) and the [PingRxNativeOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_rx_native_op/ping_rx_native_op.cpp) implementations of this example, in contrast to the [PingTxOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/src/operators/ping_tx/ping_tx.cpp) and [PingRxOp](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/src/operators/ping_rx/ping_rx.cpp) built-in operators of the SDK.

For more details regarding the use of `holoscan::gxf::Entity`, follow the documentation on {ref}`Interoperability between GXF and native C++ operators<interoperability-with-gxf-operators-cpp>`.

## 2. Creating the GXF extension that wraps the operator

To wrap the native operator as a GXF codelet in a GXF extension, we provide the CMake `wrap_operator_as_gxf_extension` function in the SDK. An example of how it wraps `PingTxNativeOp` and `PingRxNativeOp` can be found [here](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_extension/CMakeLists.min.txt).
- It leverages the CMake target names of the operators defined in their respective `CMakeLists.txt` ([ping_tx_native_op](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_tx_native_op/CMakeLists.min.txt), [ping_rx_native_op](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/ping_rx_native_op/CMakeLists.min.txt))
- The function parameters are documented at the top of the [WrapOperatorAsGXFExtension.cmake](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/cmake/modules/WrapOperatorAsGXFExtension.cmake#L18-42) file (ignore implementation below).

:::{warning}
- A unique GXF extension is currently needed for each native operator to export (operators cannot be bundled in a single extension at this time).
- Wrapping other GXF entities than operators (as codelets) is not currently supported.
:::

## 3. Using your wrapped operator in a GXF application

:::{note}
This section assumes you are familiar with {ref}`how to create a GXF application<creating-gxf-application>`.
:::

As shown in the `gxf_app/CMakeLists.txt` [here](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_app/CMakeLists.min.txt#L30-33), you need to list the following extensions in `create_gxe_application()` to use your wrapped codelets:
- `GXF::std`
- `gxf_holoscan_wrapper`
- the name of the CMake target for the created extension, defined by the `EXTENSION_TARGET_NAME` argument passed to `wrap_operator_as_gxf_extension` in the previous section

The codelet class name (defined by the `CODELET_NAMESPACE::CODELET_NAME` arguments passed to `wrap_operator_as_gxf_extension` in the previous section) can then be used as a component `type` in a GXF app node, as shown in the [YAML app definition](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/wrap_operator_as_gxf_extension/gxf_app/ping.yaml) of the example, connecting the two ping operators.
