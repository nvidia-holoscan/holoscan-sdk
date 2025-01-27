# Wrap Holoscan As GXF Extension

This example demonstrates how to wrap Holoscan SDK native Operators/Resources/types as GXF codelets/components/types
and add them to a GXF extension so that they can be used in a GXF application.

This example wraps two C++ native operators and one native resource.
  1. a transmitter, which sends empty `gxf::Entity` messages to its "out" port.
     However, in general, this could be a tensor object.
     This operator accepts a Holoscan native resource as a parameter and prints the value of the resource to the terminal, before sending an empty message.
  2. a receiver that receives `gxf::Entity` messages from its "in" port.
     In this example we ignore the empty message and simply print to terminal that we received a ping.
  3. a resource that holds a custom integer type value and a float type value.
     This resource is used as an argument to the transmitter operator.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/gxf/gxf_wrap_holoscan.html) for step-by-step instructions.*

## Run instructions

* **using deb package install and NGC container**:
  ```bash
  cd /opt/nvidia/holoscan # for GXE to find GXF extensions
  ./examples/wrap_holoscan_as_gxf_extension/gxf_app/test_holoscan_as_gxf_ext
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/wrap_holoscan_as_gxf_extension/gxf_app/test_holoscan_as_gxf_ext
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR} # for GXE to find GXF extensions
  ./examples/wrap_holoscan_as_gxf_extension/gxf_app/test_holoscan_as_gxf_ext
  ```
