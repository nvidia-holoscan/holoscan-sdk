# Video Sources

Minimal examples using GXF YAML API to illustrate the usage of various video sources:
- `aja_capture`: uses the AJA capture card with [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) to avoid copies to the CPU. The renderer (holoviz) leverages the videobuffer from CUDA to Vulkan to avoid copies to the CPU also. Requires to [set up the AJA hardware and drivers](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html).
- `v4l2_camera`: uses [Video4Linux](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html) as a source, to use with a V4L2 node such as a USB webcam (goes through the CPU). It uses CUDA/OpenGL interop to avoid copies to the CPU.
- `video_replayer`: loads a video from the disk, does some format conversions, and provides a basic visualization of tensors.

### Requirements

- `aja_capture`: follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.
- `v4l2_camera`: if using a container, add `--device /dev/video0:/dev/video0` to the `docker run` command to make your USB cameras available to the V4L2 codelet in the container (automatically done by `./run launch`). Note that your container might not have permissions to open the video devices, run `sudo chmod 666 /dev/video*` to make them available.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

```bash
./examples/video_sources/gxf/aja_capture
./examples/video_sources/gxf/v4l2_camera
./examples/video_sources/gxf/video_replayer
```
