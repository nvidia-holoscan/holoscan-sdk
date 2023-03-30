# V4L2 Camera

Minimal examples using GXF YAML API to illustrate the usage of [Video4Linux](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html) to stream from a V4L2 node such as a USB webcam.

## Run instructions

* **using deb package install and NGC container**:
  ```bash
  cd /opt/nvidia/holoscan # for GXE to find GXF extensions
  ./examples/v4l2_camera/gxf/v4l2_camera
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/v4l2_camera/gxf/v4l2_camera
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR} # for GXE to find GXF extensions
  ./examples/v4l2_camera/gxf/v4l2_camera
  ```

> Note: if using a container, add `--device /dev/video0:/dev/video0` (or the ID of whatever device you'd like to use) to the `docker run` command to make your USB camera is available to the V4L2 codelet in the container. This is automatically done by `./run launch`. Also note that your container might not have permissions to open the video devices, run `sudo chmod 666 /dev/video*` to make them available.