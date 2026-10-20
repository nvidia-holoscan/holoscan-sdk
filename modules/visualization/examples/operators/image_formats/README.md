# Image formats

A deterministic RGB gradient or YUV luma ramp rendered by `HolovizOp` from a host or CUDA
Tensor. RGB tensors contain three channels; the renderer adds opaque alpha with a CUDA kernel
writing directly into an exported RGBA image. NV12 and YUYV use BT.709 limited-range conversion.

From the SDK build container:

```bash
image_formats --format rgb --memory cuda --seconds 5
image_formats --format nv12 --memory host --seconds 5
image_formats --format yuyv --memory cuda --seconds 5
```

The defaults are RGB and host memory. `--seconds N` waits N seconds and then stops the session
cooperatively; omit it to run until Ctrl-C, which the managed signal domain
(`ManagedShutdownSignals`) turns into the same cooperative stop. Both exit 0. Use `--device D`
for the CUDA tensor pool; on a multi-GPU host it must be the GPU driving the selected display,
since HolovizOp creates its device from the presentable adapter and a mismatched activation
stream fails the first `compute()` with a cross-device `kIncompatible` error. A working display
is required; `xvfb-run -a` can provide one for smoke tests.
The source runs at 30 Hz and the display at 60 Hz with a bounded 100 ms reuse window.
