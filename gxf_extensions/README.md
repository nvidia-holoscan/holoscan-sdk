# GXF extensions

See the User Guide for details regarding the extensions in GXF and Holoscan SDK, and for instructions to build your own extensions

- `gxf_holoscan_wrapper`: includes the `holoscan::gxf::OperatorWrapper` codelet. It is used as a utility base class to wrap a holoscan operator to interface with the GXF framework.
- `ucx`: includes `nvidia::holoscan::UcxHoloscanComponentSerializer` which is a `nvidia::gxf::ComponentSerializer` that handles serialization and deserialization of `holoscan::Message` and `holoscan::Tensor` types over a Unified Communication X (UCX) network connection. UCX is used by Holoscan SDK to send data between fragments of distributed applications. This extension must be used in combination with standard GXF UCX extension components. Specifically, this `UcxHoloscanComponentSerializer` is intended for use by the `UcxEntitySerializer` where it can operate alongside the `UcxComponentSerializer` that serializes GXF-specific types (`nvidia::gxf::Tensor`, `nvidia::gxf::VideoBuffer`, etc.).
