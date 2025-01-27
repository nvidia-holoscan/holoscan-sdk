# V4L2 Camera

This app captures video streams using [Video4Linux](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html) and visualizes them using Holoviz.

#### Notes on the V4L2 operator

* The V4L2 operator can read a range of pixel formats, it will automatically select a format supported by the device. For other formats, you will need to specify the `pixel_format` parameter in the yaml file which will then be used.
* The V4L2 operator outputs data in pinned CUDA memory or in managed CUDA memory on L4T.

## Requirements

### Containerized Development

If using a container outside the `run` script, add `--group-add video` and `--device /dev/video0:/dev/video0` (or the ID of whatever device you'd like to use) to the `docker run` command to make your camera device available in the container.
When using HDMI IN also add `--device /dev/capture-vi-channel0:/dev/capture-vi-channel0`.

### Local Development

Install the following dependency:

```sh
sudo apt-get install libv4l-dev
```

To use `v4l2-ctl` for debugging, also install `v4l-utils`:

```sh
sudo apt-get install v4l-utils
```

If you do not have permissions to open the video device, run:

```sh
sudo usermod -aG video $USER
```

### Updating HDMI IN Firmware

Before using the HDMI IN device on NVIDIA IGX or Clara AGX developer kits, please ensure that it has the latest firmware by following instructions from the [IGX Orin user guide](https://docs.nvidia.com/igx-orin/user-guide/latest/base-os.html#update-the-hdmi-in-input-firmware).

## Parameters

There are a few parameters that can be specified:

* `device`: The mount point of the device
  * Default: `"/dev/video0"`
  * List available options with `v4l2-ctl --list-devices`
* `pixel_format`: The [V4L2 pixel format](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-intro.html) of the device, as FourCC code
  * Default: auto selects `AB24`, `YUYV`, or `MJPG` based on device support
  * List available options with `v4l2-ctl -d /dev/<your_device> --list-formats`
* `width` and `height`, `frame_rate`: The frame dimensions and rate
  * Default: device default
  * List available options with `v4l2-ctl -d /dev/<your_device> --list-formats-ext`
* `exposure_time`: The exposure time of the camera sensor in multiples of 100 μs (e.g. setting exposure_time to 100 is 10 ms)
  * Default: auto exposure, or device default if auto is not supported
  * List supported range with `v4l2-ctl -d /dev/<your_device> -L`
* `gain`: The gain of the camera sensor
  * Default: auto gain, or device default if auto is not supported
  * List supported range with `v4l2-ctl -d /dev/<your_device> -L`

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
ffmpeg -stream_loop -1 -re -i /path/to/video.mp4 -pix_fmt rgba -f v4l2 /dev/video3
```

Next, run the `v4l2_camera` application having specified the correct device node in the yaml-configuration file (set the `device` parameter of the V4L2 operator to `/dev/video3`). The mp4 video should be showing in the Holoviz window.
