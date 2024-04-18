## Troubleshooting the SDK

### X11: Failed to open display :0 [...] Failed to initialize GLFW

Enable permissions to your X server from Docker, either:

- Passing `-u $(id -u):$(id -g)` to `docker run`, or
- Running `xhost +local:docker` on your host

### GLX: Failed to create context: GLXBadFBConfig

You may encounter the error message if the Holoscan Application runs on a Virtual Machine (by a Cloud Service Provider) or without a physical display attached. If you want to run applications that use GPU on x11 (e.g., VNC or NoMachine), the following environment variables need to be set before executing the application to offload the rendering to GPU.

```sh
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

### `GXF_ENTITY_COMPONENT_NOT_FOUND` or `GXF_ENTITY_NOT_FOUND`

Ensure all your application connections in the yaml file (`nvidia::gxf::Connection`) refer to entities or components defined within. This can occur when attempting to remove a component and not cleaning up the stale connections.

### No receiver connected to transmitter of <scheduling term id> of entity <x>. The entity will never tick

Ensure your entity or component is not an orphan, but is connected to a `nvidia::gxf::Connection`.

### AJA device errors

These errors indicate that you don't have AJA support in your environment.

```sh
2022-06-09 18:45:13.826 ERROR gxf_extensions/aja/aja_source.cpp@80: Device 0 not found.
2022-06-09 18:45:13.826 ERROR gxf_extensions/aja/aja_source.cpp@251: Failed to open device 0
```

Double check that you have installed the AJA ntv2 driver, loaded the driver after every reboot, and that you have specified `--device /dev/ajantv20:/dev/ajantv20` in the `docker run` command if you’re running a docker container.

### GXF format converter errors

These errors may indicate that you need to reconfigure your format converter's num_block number.

```sh
2022-06-09 18:47:30.769 ERROR gxf_extensions/format_converter/format_converter.cpp@521: Failed to allocate memory for the channel conversion
2022-06-09 18:47:30.769 ERROR gxf_extensions/format_converter/format_converter.cpp@359: Failed to convert tensor format (conversion type:6)
```

Try increasing the current num_block number by 1 in the yaml file for all format converter entities. This may happen if your yaml file was configured for running with RDMA and you have decided to disable RDMA.

### Video device error

Some of those errors may occur when running the V4L2 codelet:

```
Failed to open device, OPEN: No such file or directory
```

Ensure you have a video device connected (ex: USB webcam) and listed when running `ls -l /dev/video*`.

```
Failed to open device, OPEN: Permission denied
```

This means the `/dev/video*` device is not available to the user from within docker. Give `--group-add video` to the `docker run` command.

### HolovizOp fails on hybrid GPU systems with non-NVIDIA integrated GPU and NVIDIA discrete GPU

You may encounter an error when trying to run the Holoviz operator on a laptop equipped with an integrated and a discrete GPU. By default these systems will be using the integrated GPU when running an application. The integrated GPU does not provide the capabilities the Holoviz operator needs and the operator will fail.

The following environment variables need to be set before executing the application to offload the rendering to the discrete GPU. See [PRIME Render Offload](https://download.nvidia.com/XFree86/Linux-x86_64/535.54.03/README/primerenderoffload.html) for more information.

```sh
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```