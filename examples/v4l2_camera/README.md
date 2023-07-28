# V4L2 Camera

This app captures video streams using [Video4Linux](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html) and visualizes them using Holoviz.

#### Notes on the V4L2 operator

* The V4L2 operator can read a range of pixel formats, though it will always output RGBA32 at this time.
* If pixel format is not specified in the yaml configuration file, it will be automatically selected if `AR24` or `YUYV` is supported by the device.  For other formats, you will need to specify the `pixel_format` parameter in the yaml file which will then be used.  However, note that the operator expects that this format can be encoded as RGA32.  If not, the behavior is undefined.
* The V4L2 operator outputs data on host. In order to move data from host to GPU device, use `holoscan::ops::FormatConverterOp`.

## Requirements

### Containerized Development

If using a container outside the `run` script, add `--group-add video` and `--device /dev/video0:/dev/video0` (or the ID of whatever device you'd like to use) to the `docker run` command to make your camera device available in the container.

### Local Development

Install the following dependency:
```sh
sudo apt-get install libv4l-dev=1.18.0-2build1
```

If you do not have permissions to open the video device, run:
```sh
 sudo usermod -aG video $USER
```

### HDMI IN

The HDMI IN on the dev kit will have to be activated in order for capturing to work via HDMI IN. Please look at the relevant dev kit user guide for instructions.

## Parameters

There are a few parameters that can be specified:

* `device`: The mount point of the device (default=`"/dev/video0"`).
* `pixel_format`: The [V4L2 pixel format](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-intro.html) of the device, as FourCC code (if not specified, app will auto select 'AR24' or 'YUYV' if supported by the device)
* `width`: The frame size width (if not specified, uses device default). Currently, only `V4L2_FRMSIZE_TYPE_DISCRETE` are supported.
* `height`: The frame size height (if not specified, uses device default). Currently, only `V4L2_FRMSIZE_TYPE_DISCRETE` are supported.

**OBS:** Note that specifying both the `width` and `height` parameters will make the app use `BlockMemoryPool` rather than `UnboundedAllocator` which improves the latency (FPS), however
please ensure that your device supports that combination of `width` and `height` (see `v4l2-ctl --list-formats-ext` below) otherwise the application will fail to start.

The parameters of the available V4L2-supported devices can be found with:
```sh
v4l2-ctl --list-devices
```
followed by:
```sh
v4l2-ctl -d /dev/video0 --list-formats-ext
```
If you do not have the `v4l2-ctl` app, it can be installed with (if running via Holoscan Docker image, already available):
```sh
sudo apt-get install v4l-utils
```

## Run Instructions

### C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  cd /opt/nvidia/holoscan
  ./examples/v4l2_camera/cpp/v4l2_camera
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/v4l2_camera/cpp/v4l2_camera
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR}
  ./examples/v4l2_camera/cpp/v4l2_camera
  ```

### Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/v4l2_camera.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/v4l2_camera/python/v4l2_camera.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/v4l2_camera/python/v4l2_camera.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/v4l2_camera/python/v4l2_camera.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/v4l2_camera/python/v4l2_camera.py
  ```

## Use with V4L2 Loopback Devices

The V4L2 operator supports virtual video devices created with the [v4l2loopback kernel module](https://github.com/umlaeute/v4l2loopback). This allows, for example, to open a video file on disk, mount it as a virtual video device, and read it in Holoscan using V4L2. Other use cases are [casting a region of the display](https://wiki.archlinux.org/title/V4l2loopback#Casting_X11_using_FFmpeg) or [using a network stream as a webcam](https://wiki.archlinux.org/title/V4l2loopback#Using_a_network_stream_as_webcam).

### Example: Streaming an mp4 as a Loopback Device

On your local machine, install `v4l2loopback` and `ffmpeg`:
```sh
sudo apt-get install v4l2loopback-dkms ffmpeg
```

Load the `v4l2loopback` kernel module on `/dev/video3`:
```sh
sudo modprobe v4l2loopback video_nr=3 max_buffers=4
```
Note that if you are doing containerized development, the kernel module needs to be loaded before launching the container. Also, if you for want to change parameters given when loading the kernel module, you will have to first unload the kernel module with `sudo modprobe -r v4l2loopback`, for the changes to have any effect.

Next, play a video to `/dev/video3` using `ffmpeg`:
```sh
ffmpeg -stream_loop -1 -re -i /path/to/video.mp4 -pix_fmt yuyv422 -f v4l2 /dev/video3
```

Next, run the `v4l2_camera` application having specified the correct device node in the yaml-configuration file (set the `device` parameter of the V4L2 operator to `/dev/video3`). The mp4 video should be showing in the Holoviz window.