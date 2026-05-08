# V4L2 IMX274

This app demonstrates low-latency image sensor pipelines on devices with a MIPI port, such as the NVIDIA Jetson AGX Orin. It captures raw Bayer frames from a Sony IMX274 camera sensor via V4L2 and displays them in real time using Holoviz, with a full processing pipeline: optical black correction → white balance → Bayer demosaicing.

## Pipeline

```
V4L2VideoCaptureOp (pass_through=true)
    ↓  raw uint8 blob + V4L2 metadata
RawFrameConverterOp
    ↓  uint16 Bayer tensor (height × width × 1)
RawImageProcessorOp
    ↓  optical-black corrected, white-balanced Bayer
BayerDemosaicOp
    ↓  RGBA uint16
HolovizOp
```

#### Notes

* `pass_through=true` is required on the V4L2 operator so that raw sensor data is forwarded unchanged and the V4L2 metadata (`pixel_format`, `width`, `height`) is propagated downstream.
* The YAML `receiver.pixel_format` value must match what the camera driver actually negotiates. Run `v4l2-ctl -d /dev/<device> --list-formats-ext` to see supported formats.  For Sony IMX274 camera, this is `RG10`.

## Requirements

### Hardware

An IMX274 sensor attached to the V4L2 subsystem (e.g., on an NVIDIA AGX Orin developer kit). Confirm the device node with:

```sh
v4l2-ctl --list-devices
```

### Containerized Development

When running inside a container, pass the camera device through:

```sh
docker run ... --group-add video --device /dev/video0:/dev/video0 ...
```

Replace `/dev/video0` with your actual device node.

### Local Development

Install the V4L2 development library:

```sh
sudo apt-get install libv4l-dev
```

To inspect device capabilities:

```sh
sudo apt-get install v4l-utils
```

If you do not have permission to open the video device:

```sh
sudo usermod -aG video $USER
```

## Parameters

These parameters are set in the YAML config file (`v4l2_imx274_player.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `receiver.device` | `"/dev/video0"` | V4L2 device node. List options with `v4l2-ctl --list-devices` |
| `receiver.pixel_format` | `"RG10"` | FourCC code. List options with `v4l2-ctl -d /dev/<device> --list-formats` |
| `receiver.width` | `3840` | Capture width in pixels |
| `receiver.height` | `2160` | Capture height in pixels |
| `receiver.frame_rate` | device default | Capture frame rate |
| `raw_image_processor.bayer_grid` | `1` | NppiBayerGridPosition: 0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG |
| `raw_image_processor.raw_depth` | `1` | holoscan::csi::PixelFormat: 1=RAW_10, 2=RAW_12 |
| `raw_image_processor.optical_black` | `50` | Optical black level correction value |

Additional CLI arguments (both Python and C++):

| Argument | Default | Description |
|---|---|---|
| `--frame-limit N` | none | Exit after N frames (useful for testing) |
| `--headless` | false | Run without a display window |
| `--fullscreen` | false | Open Holoviz in fullscreen mode |
| `--use-exclusive-display` | false | Use exclusive display mode |
| `--pixel-format FMT` | `RG10` | V4L2 FourCC pixel format. Overrides `receiver.pixel_format` in YAML |
| `--bayer-grid N` | YAML / `1` | NppiBayerGridPosition: 0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG. Overrides `raw_image_processor.bayer_grid` |
| `--raw-depth N` | YAML / `1` | holoscan::csi::PixelFormat: 1=RAW_10, 2=RAW_12. Overrides `raw_image_processor.raw_depth` |

## Run Instructions

### C++ Run Instructions

* **source (dev container)**:

  ```bash
  ./run launch  # optional: append `install` for the install tree
  ./examples/v4l2_imx274/cpp/v4l2_imx274_player
  ```

* **source (local env)**:

  ```bash
  cd ${BUILD_OR_INSTALL_DIR}
  ./examples/v4l2_imx274/cpp/v4l2_imx274_player
  ```

To use a non-default config file:

```bash
./v4l2_imx274_player /path/to/custom.yaml
```

### Python Run Instructions

* **source (dev container)**:

  ```bash
  ./run launch  # optional: append `install` for the install tree
  python3 ./examples/v4l2_imx274/python/v4l2_imx274_player.py
  ```

* **source (local env)**:

  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/v4l2_imx274/python/v4l2_imx274_player.py
  ```
