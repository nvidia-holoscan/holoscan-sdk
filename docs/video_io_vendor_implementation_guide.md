# Vendor Implementation Guide: Multi-Stream Video I/O Operators

This guide describes how capture-card vendors subclass the Holoscan SDK
`VideoAcquisitionOperator` and `VideoTransmissionOperator` to provide
SDK-native multi-stream video capture and playout.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Application                        │
│  F.make_operator<ExampleVideoCapture>("cap", 4U,     │
│                   Arg("uri", "sdi://0"), ...);       │
└──────────────────────┬───────────────────────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │   VideoAcquisitionOperator (base)   │
        │   ─ setup(): registers output ports │
        │   ─ emit_capture_stream()           │
        │   ─ query_capture_capabilities()    │
        │   ─ note_acquired_frame() telemetry │
        └──────────────┬──────────────────────┘
                       │ subclass
        ┌──────────────▼──────────────────────┐
        │   ExampleVideoCapture (vendor)      │
        │   ─ start(): open device, alloc DMA │
        │   ─ compute(): dequeue + emit       │
        │   ─ stop(): release device          │
        │   ─ query_capture_capabilities()    │
        └─────────────────────────────────────┘
```

The base class handles:

| Concern | Base provides |
|---------|---------------|
| Port registration | `setup()` registers `signal`, `signal_1`, ... `signal_N` |
| Port name management | `capture_output_port_name()` / `transmit_input_port_name()` |
| Emitting / receiving frames | `emit_capture_stream()` / `receive_transmit_stream()` |
| Common parameters | `uri`, `width`, `height`, `frame_rate`, `pixel_format`, `color_space`, `transport`, `vendor_extensions` |
| Capability reporting | Default `query_capture_capabilities()` from parameters |
| Telemetry counters | `note_acquired_frame()`, `note_dropped_frame()` (atomic) |
| Stream validation | `is_capture_stream_enabled()`, bounds checking |

The vendor subclass provides:

| Concern | Vendor implements |
|---------|-------------------|
| Device lifecycle | `start()`, `stop()` |
| Frame production / consumption | `compute()` |
| Hardware capability query | `query_capture_capabilities()` (optional override) |
| Vendor-specific parameters | Additional `spec.param()` calls in `setup()` |
| Registry enumerator | `register_video_acquisition_enumerator()` (optional) |

---

## Step 1: Subclass the Base Operator

### Acquisition (Capture)

```cpp
#pragma once

#include <holoscan/operators/video_io/video_acquisition_operator.hpp>

namespace example {

class ExampleVideoCapture : public holoscan::ops::VideoAcquisitionOperator {
 public:
  // Required: forward the default and num_streams constructors to the base.
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ExampleVideoCapture, VideoAcquisitionOperator)

  ExampleVideoCapture() = default;

  explicit ExampleVideoCapture(uint32_t num_streams)
      : VideoAcquisitionOperator(num_streams) {}

  // Required if you accept num_streams + Args (the common make_operator pattern).
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit ExampleVideoCapture(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : VideoAcquisitionOperator(num_streams) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  video_io::VideoCaptureCapabilities query_capture_capabilities() const override;

 private:
  // Vendor SDK handle, DMA buffers, etc.
  Parameter<uint32_t> dma_buffer_count_;
  Parameter<bool> gpudirect_enabled_;

  void* device_handle_ = nullptr;
};

}  // namespace example
```

### Transmission (Playout)

```cpp
#pragma once

#include <holoscan/operators/video_io/video_transmission_operator.hpp>

namespace example {

class ExampleVideoPlayout : public holoscan::ops::VideoTransmissionOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ExampleVideoPlayout, VideoTransmissionOperator)

  ExampleVideoPlayout() = default;

  explicit ExampleVideoPlayout(uint32_t num_streams)
      : VideoTransmissionOperator(num_streams) {}

  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit ExampleVideoPlayout(uint32_t num_streams, ArgT&& arg, ArgsT&&... args)
      : VideoTransmissionOperator(num_streams) {
    add_arg(std::forward<ArgT>(arg));
    (add_arg(std::forward<ArgsT>(args)), ...);
  }

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<std::string> output_standard_;
  void* device_handle_ = nullptr;
};

}  // namespace example
```

---

## Step 2: Implement `setup()` — Register Vendor Parameters

Call the base `setup()` first. This registers all common parameters (`uri`, `width`, etc.)
and the correct number of I/O ports based on `num_streams()`. Then add your vendor-specific
parameters.

```cpp
void ExampleVideoCapture::setup(OperatorSpec& spec) {
  // Base registers: output ports (signal, signal_1, ...), uri, width, height,
  // frame_rate, pixel_format, color_space, transport, vendor_extensions, etc.
  VideoAcquisitionOperator::setup(spec);

  // Vendor-specific parameters
  spec.param(dma_buffer_count_,
             "dma_buffer_count",
             "DMA Buffer Count",
             "Number of DMA ring buffers per channel.",
             4U);
  spec.param(gpudirect_enabled_,
             "gpudirect_enabled",
             "GPUDirect RDMA",
             "Enable GPUDirect RDMA if supported.",
             false);
}
```

For transmission:

```cpp
void ExampleVideoPlayout::setup(OperatorSpec& spec) {
  VideoTransmissionOperator::setup(spec);

  spec.param(output_standard_,
             "output_standard",
             "Output Standard",
             "Video standard for output (e.g. 1080p60, 2160p30).",
             std::string("1080p60"));
}
```

---

## Step 3: Implement Lifecycle — `start()` and `stop()`

`start()` is called once after `initialize()`. Open the device, allocate buffers,
configure the hardware. `stop()` tears everything down.

```cpp
void ExampleVideoCapture::start() {
  const std::string device_uri = uri_.get();
  const uint32_t w = width_.get();
  const uint32_t h = height_.get();
  const float fps = frame_rate_.get();
  const uint32_t buf_count = dma_buffer_count_.get();

  // Open vendor device (pseudo-code)
  device_handle_ = example_sdk_open(device_uri.c_str());
  if (!device_handle_) {
    throw std::runtime_error("Failed to open device: " + device_uri);
  }

  // Configure each stream (channel)
  for (uint32_t i = 0; i < num_streams(); ++i) {
    example_sdk_configure_channel(device_handle_, i, w, h, fps, buf_count);
  }

  example_sdk_start_capture(device_handle_);
}

void ExampleVideoCapture::stop() {
  if (device_handle_) {
    example_sdk_stop_capture(device_handle_);
    example_sdk_close(device_handle_);
    device_handle_ = nullptr;
  }
}
```

---

## Step 4: Implement `compute()` — The Frame Loop

`compute()` is called by the scheduler on every tick. For acquisition, dequeue frames from
the hardware and emit them via `emit_capture_stream()`. For transmission, receive frames via
`receive_transmit_stream()` and queue them to the hardware.

### Acquisition `compute()`

```cpp
void ExampleVideoCapture::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  for (uint32_t i = 0; i < num_streams(); ++i) {
    if (!is_capture_stream_enabled(i)) {
      continue;
    }

    void* frame_ptr = nullptr;
    size_t frame_size = 0;
    int status = example_sdk_dequeue_frame(device_handle_, i, &frame_ptr, &frame_size);

    if (status == EXAMPLE_TIMEOUT) {
      note_dropped_frame();
      continue;
    }
    if (status != EXAMPLE_OK) {
      note_dropped_frame();
      HOLOSCAN_LOG_ERROR("Example dequeue failed on channel {}: {}", i, status);
      continue;
    }

    // Wrap the vendor buffer in a GXF Entity with a VideoBuffer or Tensor
    auto entity = holoscan::gxf::Entity::New(&context);
    // ... populate entity with frame data (vendor-specific) ...
    // For GPUDirect: frame_ptr is already a device pointer
    // For CPU DMA: memcpy or cudaMemcpyAsync to a device buffer

    emit_capture_stream(op_output, i, entity);
    note_acquired_frame();

    // Return the buffer to the vendor SDK ring
    example_sdk_requeue_buffer(device_handle_, i, frame_ptr);
  }
}
```

### Transmission `compute()`

```cpp
void ExampleVideoPlayout::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  for (uint32_t i = 0; i < num_streams(); ++i) {
    if (!is_transmit_stream_enabled(i)) {
      continue;
    }

    auto entity = receive_transmit_stream(op_input, i);
    if (!entity) {
      note_dropped_frame();
      continue;
    }

    // Extract the video buffer from the entity (vendor-specific)
    // ... get pointer, size, format from the entity's VideoBuffer/Tensor ...

    int status = example_sdk_queue_output(device_handle_, i, gpu_ptr, frame_size);
    if (status != EXAMPLE_OK) {
      note_dropped_frame();
      HOLOSCAN_LOG_ERROR("Example output queue failed on channel {}: {}", i, status);
      continue;
    }

    note_transmitted_frame();
  }
}
```

---

## Step 5: Override Capability Reporting (Optional but Recommended)

The default `query_capture_capabilities()` derives a minimal snapshot from configured
parameters. Override it to query the actual hardware for supported modes, resolutions,
frame rates, and features.

```cpp
video_io::VideoCaptureCapabilities ExampleVideoCapture::query_capture_capabilities() const {
  video_io::VideoCaptureCapabilities cap;
  cap.backend_id = "vendor.example";
  cap.device_id = uri_.get();
  cap.device_uri = uri_.get();

  // Query hardware for actual capabilities
  int num_inputs = example_sdk_get_input_count(device_handle_);
  cap.max_concurrent_inputs = static_cast<uint32_t>(num_inputs);
  cap.transports.push_back(video_io::VideoTransport::kSdi);

  for (int ch = 0; ch < num_inputs; ++ch) {
    video_io::VideoCaptureChannelCapabilities chan;
    chan.channel_index = static_cast<uint32_t>(ch);
    chan.transport = video_io::VideoTransport::kSdi;
    chan.interface_label = "SDI In " + std::to_string(ch + 1);

    // Query supported resolutions
    int num_modes = 0;
    example_mode_t* modes = example_sdk_get_modes(device_handle_, ch, &num_modes);
    for (int m = 0; m < num_modes; ++m) {
      chan.resolutions.push_back({modes[m].width, modes[m].height});
      chan.framerates.push_back({modes[m].min_fps, modes[m].max_fps});
    }

    // Query pixel formats
    int num_fmts = 0;
    example_pixfmt_t* fmts = example_sdk_get_pixel_formats(device_handle_, ch, &num_fmts);
    for (int f = 0; f < num_fmts; ++f) {
      chan.pixel_formats.push_back({fmts[f].fourcc, fmts[f].description});
    }

    chan.color_spaces.push_back(video_io::VideoColorSpaceKind::kBt709);
    chan.color_spaces.push_back(video_io::VideoColorSpaceKind::kBt2020);
    chan.gpudirect_rdma_supported = example_sdk_supports_gpudirect(device_handle_, ch);
    chan.hardware_timestamp_supported = true;
    chan.progressive_capture_supported = true;
    chan.interlaced_capture_supported = true;

    cap.input_channels.push_back(std::move(chan));
  }

  return cap;
}
```

---

## Step 6: Register a Capability Enumerator (Optional)

The registry allows applications to discover available hardware **before** creating
operator instances. Register an enumerator at library load time (or from a static
initializer) using your vendor `backend_id`.

```cpp
#include <holoscan/operators/video_io/video_io_registry.hpp>

namespace {

struct ExampleRegistrar {
  ExampleRegistrar() {
    holoscan::ops::video_io::register_video_acquisition_enumerator(
        "vendor.example",
        []() -> std::vector<holoscan::ops::video_io::VideoCaptureCapabilities> {
          std::vector<holoscan::ops::video_io::VideoCaptureCapabilities> result;

          int num_devices = example_sdk_enumerate_devices();
          for (int d = 0; d < num_devices; ++d) {
            holoscan::ops::video_io::VideoCaptureCapabilities cap;
            cap.backend_id = "vendor.example";
            cap.device_id = example_sdk_get_device_serial(d);
            cap.device_uri = "example://" + std::to_string(d);
            cap.max_concurrent_inputs = example_sdk_get_input_count_by_index(d);
            cap.transports.push_back(holoscan::ops::video_io::VideoTransport::kSdi);

            for (uint32_t ch = 0; ch < cap.max_concurrent_inputs; ++ch) {
              holoscan::ops::video_io::VideoCaptureChannelCapabilities chan;
              chan.channel_index = ch;
              chan.transport = holoscan::ops::video_io::VideoTransport::kSdi;
              chan.interface_label = "SDI In " + std::to_string(ch + 1);
              cap.input_channels.push_back(std::move(chan));
            }

            result.push_back(std::move(cap));
          }
          return result;
        });
  }
};

static ExampleRegistrar s_registrar;

}  // namespace
```

Applications can then discover devices without instantiating any operator:

```cpp
auto devices = holoscan::ops::video_io::enumerate_video_acquisition_devices("vendor.example");
for (const auto& dev : devices) {
  HOLOSCAN_LOG_INFO("Found device {} with {} inputs",
                    dev.device_id, dev.max_concurrent_inputs);
}
```

---

## Step 7: Wire Into an Application

### Multi-Instance (Preferred — One Operator Per Channel)

The recommended deployment pattern: each operator instance owns one hardware
channel. A shared device resource (vendor-specific) ensures safe access to the
underlying SDK handle with per-channel reservation.

```cpp
// Shared device resource — single SDK handle with per-channel reservation
auto dev = F.make_resource<example::ExampleDeviceResource>(
    "example_dev0", Arg("device", "0"));

// One operator per channel on the same physical device
auto ch0 = F.make_operator<example::ExampleVideoCapture>(
    "sdi_ch0",
    Arg("device_resource", dev),
    Arg("channel_index", 0U),
    Arg("uri", "sdi://0"),
    Arg("width", 1920U),
    Arg("height", 1080U),
    Arg("frame_rate", 60.F),
    Arg("rdma", true));

auto ch1 = F.make_operator<example::ExampleVideoCapture>(
    "sdi_ch1",
    Arg("device_resource", dev),
    Arg("channel_index", 1U),
    Arg("uri", "sdi://0"),
    Arg("width", 1920U),
    Arg("height", 1080U),
    Arg("frame_rate", 60.F),
    Arg("rdma", true));

auto proc0 = F.make_operator<InferenceOp>("proc0");
auto proc1 = F.make_operator<InferenceOp>("proc1");

F.add_flow(ch0, proc0, {{"signal", "input"}});
F.add_flow(ch1, proc1, {{"signal", "input"}});
```

### Multi-Stream (Single Operator, Multiple Ports)

When the vendor SDK manages multiple channels through a single handle and
does not support independent per-channel initialization, use one operator
with `num_streams > 1` to expose each channel on a separate output port.

```cpp
// 4 output ports: signal, signal_1, signal_2, signal_3
auto capture = F.make_operator<example::ExampleVideoCapture>(
    "quad_sdi",
    4U,                              // <-- num_streams
    Arg("uri", "sdi://0"),
    Arg("transport", "sdi"),
    Arg("width", 1920U),
    Arg("height", 1080U),
    Arg("frame_rate", 60.F));

auto proc0 = F.make_operator<InferenceOp>("proc0");
auto proc1 = F.make_operator<InferenceOp>("proc1");
auto proc2 = F.make_operator<InferenceOp>("proc2");
auto proc3 = F.make_operator<InferenceOp>("proc3");

F.add_flow(capture, proc0, {{"signal", "input"}});
F.add_flow(capture, proc1, {{"signal_1", "input"}});
F.add_flow(capture, proc2, {{"signal_2", "input"}});
F.add_flow(capture, proc3, {{"signal_3", "input"}});
```

### YAML Configuration

```yaml
quad_sdi:
  uri: "sdi://0"
  transport: "sdi"
  width: 1920
  height: 1080
  frame_rate: 60.0
  pixel_format: "UYVY"
  color_space: "bt709"
  dma_buffer_count: 8
  gpudirect_enabled: true
  vendor_extensions:
    vendor.example.genlock_source: "ref_in"
    vendor.example.anc_capture: true
```

---

## Port Naming Convention

| `num_streams` | Registered ports | Notes |
|---|---|---|
| 1 (default) | `signal` | Backward compatible with V4L2VideoCaptureOp |
| 2 | `signal`, `signal_1` | Index 0 is always `signal`, not `signal_0` |
| 4 | `signal`, `signal_1`, `signal_2`, `signal_3` | |
| N | `signal`, `signal_1`, ..., `signal_{N-1}` | Max N = 128 (`kVideoIoMaxStreams`) |

Use `capture_output_port_name(i)` / `transmit_input_port_name(i)` if you need the
string programmatically. The base caches these in a vector for zero-allocation lookup
in the hot path (`emit_capture_stream` / `receive_transmit_stream`).

---

## `num_streams` vs. `channel_indices` — When to Use Which

These serve different purposes:

| Parameter | Controls | Set by |
|---|---|---|
| `num_streams` | How many I/O ports `setup()` registers | Constructor argument |
| `channel_indices` | Which hardware channels appear in capability reports | YAML / `Arg()` |

**Common patterns:**

| Scenario | `num_streams` | `channel_indices` | Explanation |
|---|---|---|---|
| Single SDI capture | 1 | `{}` (empty) | One port, one default channel |
| 4x SDI via single SDK handle | 4 | `{0,1,2,3}` | Four ports, four reported channels |
| Singleton SDK that internally muxes 4 channels onto 1 output | 1 | `{0,1,2,3}` | One port, but capability report shows all channels the SDK manages |
| 4 separate operator instances, one per channel | 1 (each) | `{}` (each) | Each operator uses `channel_index` instead |

---

## CMake Integration

```cmake
# Vendor operator library
add_library(example_video_capture
  example_video_capture.cpp
)
target_link_libraries(example_video_capture
  PUBLIC
    holoscan::core
    holoscan::ops::video_io     # base classes + capabilities + registry
  PRIVATE
    example::sdk                # vendor SDK
)
```

---

## Checklist

- [ ] Subclass `VideoAcquisitionOperator` and/or `VideoTransmissionOperator`
- [ ] Forward all three constructor forms (default, `num_streams`, `num_streams` + args)
- [ ] Call base `setup()` first in your override, then add vendor parameters
- [ ] Implement `compute()` — use `emit_capture_stream()` / `receive_transmit_stream()`
- [ ] Call `note_acquired_frame()` / `note_transmitted_frame()` on success
- [ ] Call `note_dropped_frame()` on timeout or error
- [ ] Implement `start()` to open device, `stop()` to close device
- [ ] Override `query_capture_capabilities()` to return real hardware capabilities
- [ ] Use `backend_id` = `"vendor.<yourname>"` for capability reporting
- [ ] Register a capability enumerator for device discovery (optional)
- [ ] Link against `holoscan::ops::video_io`
- [ ] Test with `num_streams` = 1 (backward compat) and > 1 (multi-stream)
- [ ] Validate port naming: `signal`, `signal_1`, ..., `signal_{N-1}`

---

## API Reference (Quick Summary)

### VideoAcquisitionOperator (protected helpers)

| Method | Purpose |
|---|---|
| `emit_capture_stream(op_output, stream_index, entity)` | Emit a frame on port `stream_index` |
| `capture_output_port_name(stream_index)` | Get port name string for index |
| `note_acquired_frame()` | Increment acquired counter (atomic) |
| `note_dropped_frame()` | Increment dropped counter (atomic) |
| `is_capture_stream_enabled(stream_index)` | Check if index < `num_streams()` |
| `build_capture_capabilities_from_parameters()` | Default capability snapshot from params |

### VideoTransmissionOperator (protected helpers)

| Method | Purpose |
|---|---|
| `receive_transmit_stream(op_input, stream_index)` | Receive a frame from port `stream_index` |
| `transmit_input_port_name(stream_index)` | Get port name string for index |
| `note_transmitted_frame()` | Increment transmitted counter (atomic) |
| `note_dropped_frame()` | Increment dropped counter (atomic) |
| `is_transmit_stream_enabled(stream_index)` | Check if index < `num_streams()` |

### Common Parameters (inherited from base `setup()`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend_id` | `string` | `"generic"` | Vendor identifier for capability reporting |
| `channel_index` | `uint32_t` | `0` | Zero-based channel for single-channel mode |
| `channel_indices` | `vector<uint32_t>` | `{}` | Multi-channel index list |
| `uri` | `string` | `""` | Device path, index, or stream URI |
| `width` | `uint32_t` | `0` | Requested width (0 = device default) |
| `height` | `uint32_t` | `0` | Requested height (0 = device default) |
| `frame_rate` | `float` | `0.0` | Requested FPS (0 = device default) |
| `pixel_format` | `string` | `"auto"` | Pixel format label or fourcc |
| `color_space` | `string` | `"auto"` | Color space hint |
| `transport` | `string` | `"auto"` | Transport hint (sdi, hdmi, ethernet, ...) |
| `vendor_extensions` | `YAML::Node` | `{}` | Arbitrary vendor key-value pairs |
