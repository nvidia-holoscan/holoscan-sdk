# TensorPreprocessorOp

GPU-native image preprocessing for inference. Accepts a device-resident image frame — either
as a bare Tensor (RGB24 or YUYV) or as a `schema::ImageT` payload carrying NV12/I420/YUYV/RGB8
with per-frame encoding, color-space, and plane-layout metadata — and produces a fixed-shape,
fixed-dtype, fixed-layout tensor suitable for a downstream `InferenceOp`. **All work stays on
the GPU** — every step is an NVIDIA Performance Primitives (NPP) call dispatched onto the
Holoscan CUDA stream.

## Input port shape

The port declared at `setup()` depends on `input_kind`:

| `input_kind` | Declared port | Connection style |
|---|---|---|
| `tensor` (default) | `Input<Tensor>` with `.accepts_projected_tensor()` | Direct Tensor edge, **or** projected from an upstream `Output<schema::ImageT>` (the schema's Tensor field is narrowed onto this port at `graph.add_flow(...)`). Suits pre-decoded RGB24 or packed YUYV producers. |
| `image` | `Input<schema::ImageT>` with `.tensor_field(schema::ImageTensorFields::data)` | Full ImageT payload delivered per frame. Encoding, `color_space`, and `plane_layouts` are read from the metadata; use this for camera producers (V4L2 / Argus / SIPL) that emit NV12 or I420. |

Only `input_kind: image` supports NV12 and I420 — those encodings need `plane_layouts` for
Y/CbCr plane offsets and row strides, which do not exist on a bare Tensor. Attempting NV12 or
I420 through `input_kind: tensor` is rejected with a clear error.

## Supported operations

Every operation runs on the GPU via NPP or a device-to-device `cudaMemcpyAsync`. The exact
pipeline `compute()` runs is determined by these knobs: `input_kind`, `input_pix_fmt`
(`rgb24`, `yuyv`, `nv12`, `i420`), `output_pix_fmt` (`rgb_u8` or `rgb_fp32`), `output_layout`
(`nchw` or `nhwc`), plus the colorimetry knobs `yuv_matrix` (`bt601` or `bt709`) and
`yuv_range` (`full` or `limited`).

### 0. YUV → RGB (any YUV input pix fmt)

Runs a single NPP CSC call sized to the source resolution, into `d_yuv_rgb_u8_` — a device
staging buffer lazily allocated on the first `compute()` and grown (never shrunk) on
resolution bumps. Dispatch by `(pix_fmt, yuv_matrix)`:

| `input_pix_fmt` | `yuv_matrix = bt601` (default) | `yuv_matrix = bt709` |
|---|---|---|
| `yuyv` | `nppiYCbCr422ToRGB_8u_C2C3R_Ctx` | rejected (`kNotSupported`) |
| `nv12` | `nppiNV12ToRGB_8u_P2C3R_Ctx`     | `nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx` |
| `i420` | `nppiYCbCr420ToRGB_8u_P3C3R_Ctx` | rejected (`kNotSupported`) |

Constraints:

- YUV inputs must have an even width (chroma subsampling is 4:2:2 or 4:2:0).
- Bare-Tensor YUYV expects rank-3 `[H, W, 2]` or rank-4 `[1, H, W, 2]` uint8 with a
  row-contiguous stride.
- NV12/I420 require `input_kind: image`; the operator reads plane pointers from
  `ImageT.plane_layouts[i].offset_bytes()` and NPP row steps from
  `plane_layouts[i].row_stride_bytes()`, so producer-side row padding is honored.
- When the ImageT carries `color_space = REC601` or `REC709`, that overrides the operator's
  `yuv_matrix` default per frame; other color spaces fall back to the configured value.
- Only 8-bit YUV is currently supported. NV12 BT.709 uses the `_709CSC` variant which
  expects full-range luma; the `yuv_range` field is accepted and recorded but only
  meaningfully differentiates on the NV12 path today (a follow-up will add
  `_709HDTV`-based limited-range dispatch for the other formats).

Subsequent stages read from the RGB staging buffer instead of the original device pointer, so
the rest of the pipeline is identical to the RGB-input path.

### 1. Resize (always)

`nppiResize_8u_C3R_Ctx` — bilinear (`NPPI_INTER_LINEAR`) resize of the interleaved uint8 RGB
frame from `(src_width, src_height)` to `(output_width, output_height)`. Writes into
`d_resized_u8_`. When `input_pix_fmt == yuyv`, the source is the YUV→RGB staging buffer above;
otherwise it is the incoming tensor's device pointer directly.

### 2. Dtype conversion — only when `output_pix_fmt == rgb_fp32`

`nppiConvert_8u32f_C3R_Ctx` — casts the interleaved uint8 buffer into an interleaved fp32
buffer of the same H×W×3 shape. Writes into `d_resized_f32_`.

### 3. Per-channel normalization — only when `output_pix_fmt == rgb_fp32`

Three fused in-place NPP arithmetic passes on `d_resized_f32_` that together compute
`x = (x/255 − mean[c]) / std[c]` per channel:

- `nppiMulC_32f_C3IR_Ctx` with constants `[1/255, 1/255, 1/255]` — scale to `[0, 1]`.
- `nppiSubC_32f_C3IR_Ctx` with `normalize_mean` — subtract per-channel mean.
- `nppiDivC_32f_C3IR_Ctx` with `normalize_std` — divide by per-channel std.

Set `normalize_mean = [0, 0, 0]`, `normalize_std = [1, 1, 1]` to keep the scaled `[0, 1]`
representation with no mean/std adjustment. Set them to
`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` for ImageNet-style normalization.

### 4. Layout arrangement (always)

Emits into the destination tensor allocated from Holoscan's runtime pool. Which NPP call
runs depends on `output_layout`:

- **`nchw`** (channel-first, `[C, H, W]`):
  - `rgb_u8`: `nppiCopy_8u_C3P3R_Ctx` splits the interleaved u8 buffer into three planar
    channel slabs at contiguous offsets `[0, H*W)`, `[H*W, 2*H*W)`, `[2*H*W, 3*H*W)`.
  - `rgb_fp32`: `nppiCopy_32f_C3P3R_Ctx` — same, for the normalized fp32 buffer.
- **`nhwc`** (channel-last, `[H, W, C]`):
  - Both dtypes use a single `cudaMemcpyAsync(D→D)` of the whole staging buffer into the
    destination. No transpose needed; the staging buffer is already interleaved.

### 5. Optional batch-dim prepend

If `prepend_batch_dim: true` (the default), the emitted tensor's shape is
`[1, C, H, W]` (nchw) or `[1, H, W, C]` (nhwc) — matching what `InferenceOp` typically
expects. Otherwise it is rank-3.

## Full operation matrix

For every YUV `input_pix_fmt`, prepend the matching NPP CSC call from the dispatch table
above (e.g. `nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx →` for `nv12` with `yuv_matrix: bt709`).

| `output_pix_fmt` | `output_layout` | NPP / CUDA calls per frame                                                                                                                                             |
|------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rgb_u8`         | `nchw`          | `nppiResize_8u_C3R_Ctx` → `nppiCopy_8u_C3P3R_Ctx`                                                                                                                       |
| `rgb_u8`         | `nhwc`          | `nppiResize_8u_C3R_Ctx` → `cudaMemcpyAsync(D→D)`                                                                                                                        |
| `rgb_fp32`       | `nchw`          | `nppiResize_8u_C3R_Ctx` → `nppiConvert_8u32f_C3R_Ctx` → `nppiMulC_32f_C3IR_Ctx` → `nppiSubC_32f_C3IR_Ctx` → `nppiDivC_32f_C3IR_Ctx` → `nppiCopy_32f_C3P3R_Ctx`           |
| `rgb_fp32`       | `nhwc`          | `nppiResize_8u_C3R_Ctx` → `nppiConvert_8u32f_C3R_Ctx` → `nppiMulC_32f_C3IR_Ctx` → `nppiSubC_32f_C3IR_Ctx` → `nppiDivC_32f_C3IR_Ctx` → `cudaMemcpyAsync(D→D)`             |

## Input requirements

### Bare-Tensor path (`input_kind: tensor`)

Accepted layouts by `input_pix_fmt`:

- **`rgb24`** — rank-3 `[H, W, 3]` or rank-4 `[1, H, W, 3]` uint8.
- **`yuyv`** — rank-3 `[H, W, 2]`, rank-4 `[1, H, W, 2]`, **or** rank-2 `[H, 2W]` (V4L2-style
  flat byte layout, where the second extent is the row width **in bytes**; `2W` must be
  divisible by 4). Requires an even pixel width.
- **`nv12` / `i420`** — rank-1 `[total_bytes]` uint8. Plane geometry comes from
  `input_width`, `input_height`, and `input_planes` (see below); the shape is only used to
  bound the byte span. Suits Argus / SIPL producers that publish a single `NvSciBuf`-backed
  buffer with the plane pitches known at configuration time.

Common requirements across the four:

- **Dtype:** scalar `uint8` (`dtype.code=kDLUInt`, `dtype.bits=8`, `dtype.lanes=1`).
- **Memory:** device — `kCudaDevice` or `kCudaManaged`. Host tensors are rejected.
- **Strides:** row-contiguous for `rgb24` and `yuyv`; unit byte stride for the rank-1
  `nv12` / `i420` byte buffer.
- **Dimensions:** width, height, and per-plane row strides must fit inside NPP's `int`-typed
  arguments (dimensions above 2 GiB rows / strides above 2 GiB are rejected).

Flat NV12/I420 additionally require:

- `input_width` and `input_height` must be positive and even.
- `input_planes.size()` must equal **2** for NV12 (Y, interleaved CbCr) or **3** for I420
  (Y, Cb, Cr).
- Every plane's `offset_bytes + row_stride_bytes * rows` must fit inside the Tensor's byte
  span. Each `row_stride_bytes` must be at least the plane's row-width-in-bytes
  (`input_width` for Y and NV12 CbCr; `input_width / 2` for I420 Cb / Cr).
- NV12 additionally requires **equal Y and CbCr row strides** — NPP's `nppiNV12ToRGB_*` CSCs
  take a single `rSrcStep`. A producer with mismatched Y/UV pitches must either repack
  upstream or switch to `input_kind: image` (which routes per-plane strides through
  `ImageT.plane_layouts`). I420 Cb and Cr strides may differ (the operator threads each
  stride into NPP independently).

### ImageT path (`input_kind: image`)

- **Dtype of `ImageT.data`:** must be `uint8`.
- **`ImageT.width` / `.height`:** must be positive; YUV encodings additionally require even
  width.
- **`ImageT.plane_layouts`:** for NV12 the vector must have exactly 2 entries (Y plane, CbCr
  interleaved plane); for I420 exactly 3 (Y, Cb, Cr). RGB8 and YUYV honor
  `plane_layouts[0].row_stride_bytes()` when present and fall back to `W * C` when the vector
  is empty.
- **`ImageT.encoding`:** RGB8, YUYV, NV12, I420 are decoded directly. `CUSTOM` falls back to
  the operator's `input_pix_fmt` override. Any other value is rejected.
- **`ImageT.color_space`:** REC601 → BT.601 CSC; REC709 → BT.709 CSC; UNSPECIFIED / SRGB fall
  back to the operator's `yuv_matrix` override.

Anything violating these raises an error out of `compute()` — no silent fallback.

## Example YAML — YUYV bare Tensor source

```yaml
preprocessor:
  input_port_name: raw_frame
  output_port_name: preprocessed
  input_kind: tensor           # default; declares Input<Tensor>.accepts_projected_tensor()
  input_pix_fmt: yuyv
  output_width: 224
  output_height: 224
  output_pix_fmt: rgb_fp32
  output_layout: nchw
  prepend_batch_dim: true
  normalize_mean: [0.485, 0.456, 0.406]   # ImageNet-style
  normalize_std:  [0.229, 0.224, 0.225]
  memory_kind: cuda_device
```

## Example YAML — NV12 camera source (ImageT path) with BT.709

```yaml
preprocessor:
  input_port_name: raw_frame
  output_port_name: preprocessed
  input_kind: image            # declares Input<schema::ImageT>
  input_pix_fmt: nv12          # fallback if the ImageT carries ImageEncoding_CUSTOM
  yuv_matrix: bt709            # fallback if the ImageT carries ColorSpace_UNSPECIFIED / SRGB
  yuv_range: full
  output_width: 224
  output_height: 224
  output_pix_fmt: rgb_fp32
  output_layout: nchw
  prepend_batch_dim: true
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std:  [0.229, 0.224, 0.225]
  memory_kind: cuda_device
```

Pair the NV12 config with a producer that emits `schema::ImageT` with
`encoding = ImageEncoding_NV12`, `color_space = ColorSpace_REC709` (optional; overrides the
YAML `yuv_matrix` per-frame), and two `plane_layouts` entries locating the Y and CbCr planes
inside the underlying Tensor allocation.

## Example YAML — Argus NV12 (flat bare-Tensor path)

Argus publishes a single flat `NvSciBuf`-backed buffer whose plane geometry is known at
configuration time. Use the bare-Tensor path with `input_planes`:

```yaml
preprocessor:
  input_kind: tensor           # default; declares Input<Tensor>.accepts_projected_tensor()
  input_pix_fmt: nv12
  input_width: 1920
  input_height: 1080
  input_planes:
    - {offset_bytes: 0,       row_stride_bytes: 1920}   # Y plane (W bytes/row)
    - {offset_bytes: 2073600, row_stride_bytes: 1920}   # Interleaved CbCr (W bytes/row)
  yuv_matrix: bt601
  yuv_range: full
  output_width: 224
  output_height: 224
  output_pix_fmt: rgb_fp32
  output_layout: nchw
```

For Argus I420 the third plane covers Cr:

```yaml
  input_pix_fmt: i420
  input_planes:
    - {offset_bytes: 0,       row_stride_bytes: 1920}
    - {offset_bytes: 2073600, row_stride_bytes: 960}   # Cb (W/2 bytes/row)
    - {offset_bytes: 2592000, row_stride_bytes: 960}   # Cr
```

For SIPL NV12 with camera-provided Y/UV pitches:

```yaml
  input_pix_fmt: nv12
  input_width:  1920
  input_height: 1080
  input_planes:
    - {offset_bytes: 0,           row_stride_bytes: 2048}   # y_pitch
    - {offset_bytes: 2211840,     row_stride_bytes: 2048}   # uv_pitch (must equal y_pitch)
```

> SIPL NV12 with **mismatched Y and UV pitches** is rejected on the bare-Tensor path (NPP's
> `nppiNV12ToRGB_*` takes a single `rSrcStep`). Use `input_kind: image` with an ImageT that
> carries independent per-plane strides.
>
> SIPL packed **RAW10** cannot be decoded by this operator: setting `input_pix_fmt` to `nv12`
> or `rgb24` will misinterpret the packed 3-samples-per-32-bit-word layout. RAW10 requires an
> upstream unpack + demosaic step before the input reaches TensorPreprocessorOp.

## Parameters (summary)

| Parameter          | Effect                                                                                              |
|--------------------|-----------------------------------------------------------------------------------------------------|
| `input_port_name` / `output_port_name` | Names of the single declared input/output ports.                                |
| `input_kind`       | `tensor` (default: `Input<Tensor>.accepts_projected_tensor()`) or `image` (`Input<schema::ImageT>.tensor_field(schema::ImageTensorFields::data)`). |
| `input_pix_fmt`    | `rgb24`, `yuyv`, `nv12`, or `i420`. Consulted directly on the tensor path; used as a fallback on the image path when the ImageT carries `ImageEncoding_CUSTOM`. |
| `input_width` / `input_height` | Pixel dimensions of the incoming flat NV12 / I420 byte buffer (bare-Tensor path). Ignored for rank-3/4 RGB24, rank-2/3/4 YUYV, and the ImageT path. |
| `input_planes`     | Per-plane `{offset_bytes, row_stride_bytes}` list for the flat NV12 / I420 bare-Tensor path. Exactly 2 entries for NV12 (Y, CbCr) or 3 for I420 (Y, Cb, Cr). |
| `yuv_matrix`       | `bt601` (default) or `bt709`. Overridden per-frame by `ImageT.color_space` when `REC601` / `REC709`. Only NV12 currently supports BT.709; YUYV and I420 reject BT.709 explicitly. |
| `yuv_range`        | `full` (default) or `limited`. Only `full` is currently supported for YUV inputs; `limited` is rejected on the CSC dispatch with `kNotSupported` until upstream `_HDTV`-style variants are wired. |
| `output_width` / `output_height`       | Required. Target size for the bilinear resize.                                  |
| `output_pix_fmt`   | `rgb_u8` (no dtype conversion, no normalization) or `rgb_fp32` (adds convert + normalize passes).   |
| `output_layout`    | `nchw` (channel-first, extra planar-split pass) or `nhwc` (channel-last, single D→D copy).          |
| `prepend_batch_dim`| If true, output shape gets a leading `1` extent.                                                    |
| `normalize_mean` / `normalize_std` | Per-channel `[R, G, B]` normalization constants; ignored for `rgb_u8`. Fixed 3 entries. |
| `memory_kind`      | `cuda_device` or `cuda_managed`. Nothing else is accepted.                                          |

## Notes

- **No stream sync.** Every NPP call is asynchronous on `context.cuda_stream()`; the operator
  neither creates its own stream nor synchronizes. Downstream operators are expected to
  consume on the same stream (Holoscan wires this automatically).
- **Staging buffers are allocated once in `start()`**, not per-compute. `d_resized_u8_` and
  (for fp32 output) `d_resized_f32_` live for the operator's lifetime. `d_yuv_rgb_u8_` (only
  the YUYV path) is a high-water-mark buffer allocated lazily on the first `compute()` and
  grown, never shrunk, on subsequent resolution bumps.
- **Capture-time & frame_id propagation.** The operator receives the full `Sample` (via
  `input.receive()` / `image_input.receive()`), reads `SampleMetadata.capture_timestamp_ns` +
  `.source_clock_id` + `.frame_id`, and forwards them into `EmitOptions.capture_time` /
  `.frame_id` on the output. Absent metadata (`source_clock_id == 0`) stays absent — the
  operator does not stamp its own activation time.
- **No CV-CUDA dependency.** The operator uses only the NPP libraries already required by
  the SDK build (`nppc`, `nppig`, `nppidei`, `nppial`, plus `nppicc` for the YUV → RGB paths);
  it does not pull in CV-CUDA.
