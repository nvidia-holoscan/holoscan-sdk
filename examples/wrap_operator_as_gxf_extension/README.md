# Wrap Operator As GXF Extension

This example demonstrates how to wrap Holoscan SDK Native Operators as GXF codelets so
that they can be used in a gxf application.

This example wraps two C++ native operators:
  1. a transmitter, which sends empty `gxf::Entity` messages to it's "out" port.
     However, in general, this could be a tensor object.
  2. a receiver that receives `gxf::Entity` messages from it's "in" port.
     In this example we ignore the empty message and simply print to terminal that we received a ping.

## Run instructions

* **using deb package install and NGC container**:
  ```bash
  cd /opt/nvidia/holoscan # for GXE to find GXF extensions
  ./examples/wrap_operator_as_gxf_extension/gxf_app/test_operator_as_gxf_ext
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/wrap_operator_as_gxf_extension/gxf_app/test_operator_as_gxf_ext
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR} # for GXE to find GXF extensions
  ./examples/wrap_operator_as_gxf_extension/gxf_app/test_operator_as_gxf_ext
  ```
