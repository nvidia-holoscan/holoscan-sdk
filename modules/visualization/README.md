# Holoscan Visualization

## Introduction

Holoscan Visualization is the Holoscan SDK 5.x visualization module: a low-latency CUDA/Vulkan
rendering foundation and, above it, a small fixed-scene 2D layer renderer. It is
being written from scratch to replace the vendored `nvpro_core`/`nvvk` helpers
that HoloViz 4.x builds on, so that the frame path — allocation, synchronization,
presentation — is owned by this repository rather than inherited from a general
purpose sample framework.

**Everything in this module is internal for now, and all of it is in progress.**
The 5.0 deliverable is the `HolovizOp` operator — layer rendering and display,
exposed as a Holoscan operator — and that is the only surface 5.0 commits to.
Both layers here sit behind it, with no ABI or source-compatibility promise on
any header yet, so names, signatures, and structure are all still open. That is
the point of reviewing them now: exposing the layer renderer, or HoloViz beneath
it, as supported APIs is a plausible later step, and the design is meant to hold
up if we take it — but nothing in this commit should be read as a commitment
either way.

Holoscan Visualization is the SDK module and project. HoloViz is the rendering backend contained
by that module; `holoviz` remains the backend's CMake target, namespace, include path, and internal
helper prefix. `HolovizOp` is the Holoscan operator that integrates the backend with the SDK.

The module is built in two layers. The bottom one, `holoviz`, is a
**rendering hardware interface**: a thin portability layer that gives one API
over several graphics APIs (Vulkan, D3D12, and so on).
Ours has a working Vulkan backend — enough to run every example and test end to
end, not a finished abstraction. Above it sits the layer renderer
(`HolovizRenderBackend`/`WsiPresentBackend` behind the `RenderBackend`/`PresentBackend`
seams), implemented for the declared layer kinds. The module also owns
`HolovizOp`, the Holoscan operator that adapts SDK tensors and lifecycle semantics
to these backends. HoloViz was designed against that layer API, so most of what
follows reads as "here is the primitive, and here is the thing above it that
needs it."

The module produces a shared library (`libholoviz.so` / `holoviz.dll`), known
as `holoscan::visualization::holoviz` both in the build tree and in the installed
Holoscan package. Every target the module ships lives in the
`holoscan::visualization::` namespace. `HolovizOp` (`holoscan::visualization::op_holoviz`)
links it privately, and external visualization operators reach it through
`find_package(holoscan COMPONENTS visualization)`. It builds as part of the SDK
or as a top-level Holoscan module. It is not linked into `holoscan::core`.

## What is here

```
include/holoviz/         Core rendering API  — namespace holoviz   (Vulkan backend working), plus
                         errors/expected/logger/present_mode: the module's own diagnostic vocabulary
include/holoviz/text/    Text rendering API  — TextRenderer seam, monospace renderer
include/holoviz/imgui/   Dear ImGui integration — namespace holoviz::imgui (Renderer, opt-in ui.hpp)
include/holoviz/integrations/   Optional header-only bridges (Holoscan logger)
operators/include/holoscan/visualization/
                         The installed operator API — namespace holoscan::visualization. Three
                         headers: holoviz_op.hpp holds HolovizOp plus everything else it is declared
                         with (LayerStack and the layer descriptors/enums); image_format.hpp holds
                         the ImageFormat vocabulary; sink_desc.hpp holds the sink descriptors and
                         the operator's own PresentMode (no HoloViz header is included)
src/                     Backend-agnostic dispatch + Slang compiler
src/cuda/                CUDA bootstrap shared across backends (device/context init, error strings)
src/vulkan/              The Vulkan backend  (~8k lines, the bulk of the code)
src/window_systems/      GLFW windowing
src/display_systems/     Direct-display ownership classification/acquisition (WinRT on Windows, libdrm on Linux)
src/text/                The text renderers (ImGui-backed + push-constant monospace renderer)
src/imgui/               The Dear ImGui integration (imgui::Renderer, GLFW input bridge, shader)
operators/src/holoviz_op/
                         HolovizOp's private implementation (one src/<name>_op/ folder per operator):
                         holoviz_op.cpp (the public face), holoviz_op_impl.* (the frame loop),
                         error_translation.* (HoloViz → Holoscan error mapping), types.hpp (re-exports
                         the HoloViz error vocabulary), and backends/ — the
                         RenderBackend/PresentBackend seam (render_backend.hpp/present_backend.hpp,
                         handles.hpp, frame_data.hpp) plus its concrete implementations and layer
                         states (HolovizRenderBackend, WsiPresentBackend)
operators/tests/unit/    Layer-API and backend tests (link the backend objects, not the DSO)
cmake/                   SDK/module dependency resolution logic (recipes: public/cmake/deps/)
docs/                    Design-note index + stable implementation background
examples/core/           Direct core-API samples: textured quad, headless render-to-CUDA, CUDA-interop benchmark, direct display
examples/text/           Text renderer sample: minimal per-frame text loop
examples/imgui/          Dear ImGui debug-UI sample
examples/operators/      Operator samples on a real Holoscan Graph: gray_scale_example (16-bit image layer)
operators/               Holoscan integration, including HolovizOp
tests/unit/              16 gtest targets covering the core API, the Vulkan backend, the text renderers and the ImGui integration
```

The line between the layer renderer and `holoviz` is a substrate boundary, not
just a layering one. What the layer renderer composes layers *with* is meant to
be swappable, so its interfaces name no graphics types — `holoviz` is one
substrate that can sit underneath, and everything Vulkan lives inside it.

The boundary is drawn where it is because the hardware is drawn that way. A GPU
is physically split into compute and graphics modes, and moving work or memory
across that line is not free: CUDA↔graphics interop carries real cost, mostly
driver-side, with some hardware limits underneath it. Every crossing is a price
paid. That cost is also the whole reason a CUDA-only path is interesting — at the
high end it can reach latencies a graphics pipeline paying interop simply cannot,
by never crossing at all. So the interop boundary is not an implementation detail
to hide; it is a hardware feature that deserves its own abstraction, and it is
the seam this module is organized around.

The corollary matters just as much: Vulkan, D3D12, Vulkan SC and the rest all
drive the *same* graphics hardware. They are different spellings of one
substrate, not different substrates, so they belong together behind one
rendering hardware interface rather than being split apart a level up. One
graphics layer with several backends, one compute layer, an explicit boundary
between them — that structure is chosen to match what the hardware actually
does.

**The graphics substrate** is HoloViz, Vulkan today, and is what 5.0 ships. It is
also where the breadth of display paths lives: Win32, X11, Wayland and QNX Screen
for windowed output, plus compositor-bypass modes — Vulkan's exclusive-display
extensions (implemented: `holoviz/display.hpp` and the `direct_display` example), and
the equivalent Windows/D3D12 path for driving a display directly without the
desktop compositor in the loop.

**A pure-CUDA substrate is an architecture we are exploring, not a commitment.**
The idea: for the subset of layer-composition workloads that optimize well as
kernels, implement composition with specialized kernels and shared memory, no
graphics API involved at all, feeding cuDisp (a CUDA-only display output library)
to reach the display. The payoff is the latency floor described above, plus
operation in deployments where a graphics stack is unwanted or unavailable.

The cost is that what graphics hardware does in fixed function is genuinely hard
to reproduce in kernels — true z-depth-based occlusion and alpha blending are the
obvious ones. A CUDA-only path would therefore serve a deliberately narrower set
of layer workloads, not replace the graphics one; the two are complementary
operating modes, which is the other reason neither gets to be *the* substrate in
the interfaces. Nothing of it exists today, and whether it is worth the
constraint it places on the `holoviz` interfaces is an open question —
**feedback on this specifically is welcome**, including "don't design for it
yet."

That constraint is the reason the two layers are kept apart at all, so it is fair
to ask whether the separation earns its keep on the graphics substrate alone.

**Open question: where does the cuDisp display seam go?** We are debating two
placements and would like opinions.

*Inside HoloViz*, as another presentation path behind `Swapchain` and its gates.
Everything above then sees one presentation abstraction no matter which substrate
is underneath, and present timing gets implemented once. The objection is that
`Swapchain`'s whole vocabulary — surfaces, acquire/present gates wrapping binary
semaphores — is shaped by graphics presentation engines, and a CUDA-only path
fits it only by analogy. (An earlier version of this objection was that it would
pull CUDA into a deliberately CUDA-free HoloViz; that ship has sailed — CUDA interop
is now first-class in HoloViz, see below — so the vocabulary mismatch is the
remaining substance.)

*Outside HoloViz*, as a `PresentBackend` implementation that never touches it.
HoloViz stays purely graphics and the substrate boundary stays honest, at the
cost of two independent presentation implementations, with pacing and timing
logic to keep in agreement between them. This is the placement the current
opaque-token seam is built for.

## The core layer

HoloViz's core is a thin, explicit abstraction over a graphics API (a rendering
hardware interface) — closer to WebGPU
than to a scene graph. It is the graphics substrate from the split above, and it
abstracts a graphics API rather than targeting Vulkan directly because that
substrate has to cover more than desktop Vulkan on its own: Vulkan SC and D3D12
are the anticipated additions, across the display paths listed earlier. It also
exists because Holoscan needs CUDA interop and presentation timing to be
first-class rather than bolted on. Interop now is: exported buffers, images and
semaphores are implemented end-to-end (see the header table). Timing is the
part still missing, and the next major piece of HoloViz work (see known gaps).

The closest reference point is **NVRHI**, the rendering hardware interface the RTX team built for their
SDKs, which covers D3D11, D3D12 and Vulkan. This project shares its philosophy:
a thin explicit layer that stays close to the underlying API rather than
inventing a renderer, with the backend differences absorbed at the seam. If you
want a sense of the shape being aimed for, NVRHI is the thing to compare against.

We did not adopt it, for two reasons. First, timing: Holoscan's constraints are
far more extreme than a general-purpose renderer's, and the frame path — pacing,
present timing, the synchronization model — is exactly the part we need to own
rather than inherit. Second, Vulkan SC: QNX is a future release platform for
Holoscan, which means a Vulkan SC backend is a requirement, and that is not
something NVRHI targets. Between those two, the layer we would have kept
customizing is the layer itself.

Every non-trivial entry point returns `holoviz::expected<T, Error>` and is
`[[nodiscard]]`; there are no exceptions and no error globals.

### Object model and ownership

Every public type in `include/holoviz/` is a small **concrete handle** —
`Buffer`, `Image`, `Pipeline`, `CommandList`, `Swapchain`, `Device` and the rest
are move-only value types you get back from `Device::create_*()` as
`expected<Buffer>`, `expected<Image>`, and so on. Each handle wraps a
`std::unique_ptr<detail::XImpl>`, where `detail::XImpl` is the pure abstract
backend interface; the concrete classes that implement those interfaces live
entirely in `src/vulkan/`.

That split exists because each impl is a handle to a **real device object**:
`VulkanBuffer` owns a `VkBuffer` plus its VMA allocation, `VulkanImage` a
`VkImage` and its view, `VulkanCommandList` a command buffer and its pool. The
lifetime of the wrapper tracks the lifetime of the driver-side object — every
backend type releases its Vulkan (and imported CUDA) objects in its destructor —
so the API has to say something definite about ownership rather than leave it to
the caller.

The handle-over-`unique_ptr` shape buys four things at once:

- **exactly one owner** — handles are movable but not copyable, so there is never
  a question of which handle destroys the device object, and no way to end up
  with two handles onto one `VkImage`;
- **a defined destruction point** — dropping the handle destroys the device
  object (creation order is the caller's teardown order in reverse; resources
  must not outlive their `Device`). Every handle also has an explicit
  `destroy(timeout)`: called directly it keeps its fail-fast zero-timeout
  default, while the destructor passes a 1-second timeout — long enough to
  drain a healthy GPU, then log and leak rather than block forever. The render
  backend's destructor enforces the "leak" half structurally: when its bounded
  drain times out it *quarantines* the owned handles into a process-lifetime
  list and poisons the device (`report_leaked_resources()`), because the
  per-handle destructors free unconditionally and freeing what pending GPU/CUDA
  work still references would be a use-after-free, not a leak;
- **a working virtual destructor across the DSO boundary**, since the concrete
  type is hidden inside the library;
- **value-like call sites** — the caller holds `Buffer`, not
  `std::unique_ptr<Buffer>`, and the indirection is an implementation detail
  that could be swapped out without changing a signature.

The `detail::*Impl` interfaces delete copy *and* move, so an impl can never be
detached from the storage that owns it. Everything that is *not* an owner
borrows, explicitly and by raw pointer or value: `BufferSlice` names a range of
a `Buffer` it does not own, `Descriptor` holds borrowed images and buffers,
`RenderPass` points at attachment images, `SwapchainDesc` names a `Window`, and
the swapchain gates are non-owning views into gate slots the swapchain
pre-allocates and recycles. The headers state what must outlive what.

**This is the part of the API most open to a better answer.** A heap allocation
and a vtable indirection per object is more machinery than a fixed-scene module
that allocates everything at setup strictly needs. Alternatives worth weighing:

- **Generational handles into device-owned pools.** The device owns dense arrays
  of concrete objects; callers hold `{index, generation}`. Use-after-free becomes
  a detectable error instead of undefined behavior, and it fits the "allocate
  everything at setup" shape especially well.
- **An explicit non-owning `Ref<T>`** for the borrow sites above, so "borrowed"
  is visible in the type rather than only in the comment.

Neither is obviously right, and the current scheme has the virtue of being
boring and familiar. Opinions welcome — this is cheap to change now and expensive
to change once a renderer is built on it.

### Errors and logging

HoloViz owns its diagnostic vocabulary and does not link Holoscan, following
HoloMQ:

- `<holoviz/expected.hpp>` — `holoviz::expected` / `unexpected` /
  `make_unexpected`, aliased onto `tl::expected`.
- `<holoviz/errors.hpp>` — `holoviz::ErrorCode` and `holoviz::Error {code,
  message}`. The enum carries only the codes the module actually returns, and
  numeric values are append-only once landed (the newest additions are
  `kNullArg` for null/empty handle arguments and `kPushConstantOverflow`). As in HoloMQ, these values belong
  to HoloViz: integrations translate each named meaning into their own domain
  rather than casting or asserting numeric parity.
- `<holoviz/logger.hpp>` — `HOLOVIZ_LOG_*` with compile-time stripping via
  `HOLOVIZ_LOG_ACTIVE_LEVEL`, runtime filtering via `set_log_level()` or the
  `HOLOVIZ_LOG_LEVEL` environment variable, and a `set_log_handler()` sink.

The Holoscan connection is optional and inverted: `<holoviz/integrations/
holoscan_logger.hpp>` is a header-only bridge that installs a handler forwarding
HoloViz logs into `holoscan::Logger`. It is the only file in the module that
includes a Holoscan header, it is never compiled into the library, and the
consumer that includes it is the one that links `holoscan::logger`.

### Symbols

HoloViz is built with hidden visibility and a generated
`<holoviz/holoviz_export.h>`: a declaration is exported only if it is marked
`HOLOVIZ_EXPORT`. Nothing from GLFW, volk or VMA is re-exported
(`--exclude-libs,ALL`), so the DSO's surface is the module's own API and nothing
else. `public/scripts/check_export_macro_semantics.py` enforces this — an
out-of-line public declaration without the macro fails lint.

The vocabulary is small and maps onto what the layer renderer has to do each
frame. Drawing means: describe some GPU work ahead of time (a **pipeline** —
compiled shaders plus fixed state like blending), point
it at inputs (**buffers** and **images**, made visible to the shader through a
**descriptor**), record the actual draw calls into a **command list**, aim them
at an output (a **render pass**), hand the list to the GPU (**submit**), and get
the result on screen (**swapchain**, **present**). A layer — a video image, a set
of overlay rectangles, a run of text — is one or two draws using one of those
pipelines. Compositing layers is doing those draws in order into the same target.

HoloViz's job is to make each of those an explicit object the layer renderer can
create once at startup, so the per-frame path is only recording and submitting.
The headers under `include/holoviz/` carry the design rationale inline; the
table below adds what each piece is *for* one level up.

| Header | The decision, and what it buys the layer renderer |
|---|---|
| `common.hpp` | The shared vocabulary: a deliberately small `Format` set, `Backend`/`ShareMode`/`DevicePolicy`, and the `NativeObject` escape hatch for borrowing raw backend handles. The format list is trimmed to what layers actually need, so an unsupported pixel format is caught up front rather than deep in a backend. |
| `device.hpp` | One device, one graphics queue, no queue object. Every `submit()` advances an internal **device timeline** and returns the `SubmitSyncToken` that completes with the batch — a single "this work is done" token. A batch waits only on the tokens the caller passes in (`SubmitDesc::wait_submit_sync_tokens`), so independent submits are free to overlap on the GPU. That token is what `RenderBackend::submit()` will hand to `PresentBackend::present()` as its `SyncToken`; `wait_on_submit()` is its host-side blocking counterpart. |
| `external_semaphore.hpp` | Timeline values are the *only* sync vocabulary — no fences, no binary semaphores, no events. `ExternalSemaphore` is the CUDA edge of it: created from a `CUstream`, its `wait()` makes the stream wait for a submit's token, and `signal()` returns an `ExternalSyncToken` a later submit can wait on. One mechanism covers "wait for the GPU", "wait for the previous frame", and "wait for CUDA", which is what keeps the frame loop's synchronization small enough to reason about. |
| `swapchain.hpp` | The one place timelines don't reach. Presentation engines only speak binary semaphores, so `acquire()` returns two single-use **gates** per frame that ride the caller's real submit rather than extra shim submits. The gate objects are pre-allocated at swapchain creation in contiguous storage and recycled — `acquire()` allocates nothing. This is the whole acquire → render → present loop `PresentBackend` owns, with the sharp-edged part already handled and exactly-once consumption enforced. |
| `display.hpp` | Exclusive (direct-to-display) output, implemented. `Display` values are enumerated per adapter with an ownership classification (`kCompositor` / `kAvailable` / `kUnknown`), list their modes, and `create_surface()` acquires the display and yields a non-resizable `Surface` for compositor-bypass scanout. The platform acquisition lives in `src/display_systems/`: WinRT + `VK_NV_acquire_winrt_display` on Windows, libdrm + `VK_EXT_acquire_drm_display` on Linux. `examples/core/direct_display/` drives it end to end. |
| `image.hpp` | Zero image-layout vocabulary in the API. Every image is pinned in one *home state* derived from its usage, and anything needing another state transitions and restores internally. The layer renderer's texture pool can therefore be allocated once and handed around without tracking GPU-side state per image — and a CUDA producer writing into one (via `cuda_mipmapped_array()` on an exported image) always finds it in a known state. The device-side transfer helpers (`upload_to_image`/`download_from_image`) honor the caller's `row_pitch_bytes` and currently accept only mip level 0 (images are single-mip for now). |
| `pipeline.hpp` | Shaders are Slang-only and carry their own **reflected** interface, so pipeline creation and descriptor binding are validated against what the shader declares. The pipeline set a render backend needs is enumerated backend-internally from the declared `LayerStack` (it is not described at the seam) and becomes HoloViz pipelines one-for-one, and a mismatch between shader and binding code is a startup error, not a wrong-looking frame. No uniform buffers and no compute: runtime data is push constants or storage buffers. |
| `descriptor.hpp` | One pre-baked binding set per pipeline, bindable by reflected index *or* by shader variable name. Each declared layer gets its descriptor filled in once at `allocate()` time; per frame it is just bound. Chosen over per-draw descriptor writes and buffer-device-address tricks, which do not port to every target. |
| `command_list.hpp` | Two list types encode the two execution styles: `kTransient` (re-record per frame) and `kReusable` (record once at setup, replay forever). A fixed layer set with fixed compositing order is exactly the case `kReusable` exists for. |
| `query.hpp` | GPU timestamp queries for profiling submitted work. Readback is deliberately a three-step recorded contract — reset, write, resolve — because not every planned backend can read results straight from the pool (D3D12 resolves into a readback buffer); `results_ns()` on the host after the resolving submit completes. This measures GPU execution spans; it is *not* the present-feedback API (see known gaps). |
| CUDA interop (cross-cutting) | Not a separate header: interop is built into the resource types. A resource created with `ShareMode::kExported` is born with external-memory handles and imported into CUDA at creation — `Buffer::cuda_device_ptr()` and `Image::cuda_mipmapped_array()` hand back typed CUDA objects, `Device` exposes `cuda_device()`/`cuda_context()`, and `ExternalSemaphore` bridges a `CUstream` onto the device timeline. This is the path by which a CUDA-produced video frame becomes a layer texture without a copy (`examples/core/cuda_interop_benchmark/` exercises it end to end). The CUDA context relationship is decided at device creation: `DeviceDesc` carries `cuda_device`, `cuda_context`, and a `CudaContextMode` — `kExternal` (caller-supplied or primary context) or the implemented CUDA-in-Graphics modes `kCigRequired`/`kCigPreferred`, which create a CiG context sharing the graphics hardware queue via `VK_NV_external_compute_queue` (`reserved_external_queues` sizes the reservation; this extension is why the Vulkan headers are pinned to 1.4.305+). Adapter↔CUDA-device matching is UUID-first: `Adapter::uuid` matches `cuDeviceGetUuid`, so a CUDA ordinal and a HoloViz adapter are identified as the same silicon. The cost of making it first-class: `<cuda.h>` is a hard include of HoloViz headers and `CUDA::cuda_driver` a public link dependency — there is no CUDA-free consumer of HoloViz. |

Three cross-cutting choices drive most of the API shape:

**Allocate at setup, execute allocation-free.** The intended lifecycle is
`create → setup → seal() → execute`, which is the HoloViz-level counterpart of
`RenderBackend`'s `allocate()` / `build()` / per-frame contract.
`DevicePolicy::kSealed` will make post-seal resource creation a hard error;
buffers have no usage flags (pick a size, get everything), samplers are values
that the device dedupes, and pipeline variants are compiled and cached on first
bind against a render pass. The frame loop is meant to touch nothing but
recording and submitting — which is also why every layer spec declares its bounds
up front.

**Allocation-free steady state, not allocation-free lifetime.** The first
target is a pure layer renderer that allocates everything during setup and then
never allocates or frees again, so the hot path stays as lean as possible.
Destruction is real: every backend type releases its Vulkan (and imported CUDA)
objects in its destructor, so create/destroy lifecycles work — resources must
simply be dropped before their `Device`, and nothing is freed on the frame path
by design. The swapchain gates follow the same rule: pre-allocated at creation,
recycled every frame, freed at teardown, and windowed framebuffers are
pre-created at `RenderBackend::register_targets()` (headless ones at `build()`)
so `begin_frame()` never creates one. Two qualifications: a layer's declared
bounds (`max_primitives`, `max_glyphs`) are hard per-frame budgets — the pools
are sized once at `allocate()` and a frame that exceeds its declaration fails
`record()` with `kInvalidArgument` rather than growing anything at runtime (a
prevalidation rejection: the frame stays open and the rejected layer recorded
nothing) — and the first `begin_frame()` after a swapchain recreation
re-creates the per-target framebuffers, the frame path's one documented
allocation exception (the recreation itself already allocated; steady state is
a single integer compare).

**Timelines everywhere they work; gates where they don't.** Timeline semaphores
give a single, CUDA-importable synchronization model. The swapchain gate
abstraction confines the binary-semaphore special case to one file instead of
leaking the presentation engine's validity rules into every call site.

**Reserved seams, not speculative code.** The display-path breadth above is
visible in the enums as commented-out slots rather than code: `WindowSystem`
reserves `kWin32`/`kWayland`/`kX11`/`kQnxScreen` alongside today's GLFW,
`NativeObjectType` reserves the matching native handles, and `Backend` reserves
`kD3D12`/`kVulkanSC`/`kcorVK`. Call sites are written backend-agnostically
today; no non-Vulkan *backend* is implemented, but one of these paths is no
longer reserved: exclusive display is implemented (`holoviz/display.hpp` and the
platform acquisition under `src/display_systems/` — see the header table).
What is available is negotiated, not assumed: `CapabilityReport` reports
`present_completion`, `present_timing`, `exclusive_display`, and
`fill_rectangle`, plus a per-format `FormatCapability` matrix saying what the
adapter actually supports for each use.

## The layer renderer

`operators/src/holoviz_op/backends/` is what `HolovizOp` is built on: a small, rigid 2D layer
renderer for the fixed-scene workloads that dominate real HoloViz usage. The seam
(`RenderBackend`/`PresentBackend`) and its implementations are private to the operator and
sit on top of HoloViz: `HolovizRenderBackend` serves all four layer kinds (image, geometry, text,
LUT — YUV formats deferred) and `WsiPresentBackend` owns the swapchain loop. Applications see only
the declarations in `operators/include/holoscan/visualization/` (`LayerStack`, the layer and
sink descriptors, `HolovizOp` itself). `operators/tests/unit/` drives the seam end to end and
covers every layer kind headless with byte-exact and analytic oracles. API feedback is still cheap to act on — nothing
is committed yet.

- `LayerStack` — the declared layers (image / geometry / text
  / LUT) captured at operator construction time. Every layer declares its bounds
  (`max_width`, `max_primitives`, `max_glyphs`) so the full GPU resource set is
  knowable before the first frame — that declaration is what makes HoloViz's
  allocate-at-setup lifecycle usable rather than aspirational. `LayerDesc` is a
  `std::variant` so adding a layer kind turns non-exhaustive visitors into
  compile errors.
- `backends/render_backend.hpp` — the ownership seam for rasterization, with an explicit
  lifecycle contract: `allocate()` at `initialize()` (all GPU allocation happens
  here; a failure past argument validation is terminal for the backend
  instance), `build()` at `start()` (all pipeline compilation happens here),
  then `begin_frame()`/`record()`/`submit()` per frame with no allocation, no
  compilation, and no blocking sync (one exception: the first `begin_frame()`
  after a swapchain recreation re-creates the per-target framebuffers). On a
  failed `record()` or `submit()`, `abort_frame()` is the universally sound
  recovery: it discards the frame and reopens the slot, and is a no-op success
  on a frame that is not open, so error paths can call it unconditionally. The
  one narrower alternative: `record()`'s `kInvalidArgument` prevalidation
  rejections leave the frame open with the rejected layer contributing nothing,
  so the caller may re-record or submit without aborting. `RenderConfig` carries the frame-loop knobs:
  `max_frames_in_flight` sizes the per-frame ring, `backpressure_timeout_ns`
  bounds `begin_frame()`'s slot-recycle wait — the frame path's one bounded
  block, returning `kBackpressured` on expiry, which leaves the slot un-recycled
  (producer buffers stay retained until a later *successful*
  `begin_frame()`/`abort_frame()`/`destroy()`) — and `clear_color` sets the RGBA
  color every frame's target is cleared to before layer 0 composites.
  `CompositeSpec::order` reorders compositing without new layer data; duplicate
  or out-of-range indices are rejected with `kInvalidArgument`. `stats()`
  reports `texture_pool_bytes`, `geometry_pool_bytes`, and
  `text_glyph_high_water` for diagnostics. For the graphics backend each
  lifecycle phase maps onto HoloViz: pools and images, then pipelines and
  reusable command lists, then record and submit. A CUDA backend, were we to
  build one, would satisfy the same contract with pools and kernels.
- `backends/present_backend.hpp` — owns acquire → render → present, event-driven, never
  busy-polling, with graceful fallback when a present mode is unavailable. The
  graphics implementation is built on HoloViz's `Swapchain` and its gates; it is
  also the interface a cuDisp-backed output would implement if the CUDA substrate
  goes ahead.
- `SinkDesc` — the caller-facing output declaration: a closed
  `std::variant` of `WindowSinkDesc`, `ExclusiveDisplaySinkDesc`,
  `CudaPitchLinearSinkDesc`, and `CudaArraySinkDesc`. Sinks describe *what*, a
  `PresentBackend` executes *how*. The window sink is live (`WsiPresentBackend`);
  the other three are declared now so call sites never change when their present
  paths are wired — a backend rejects an unimplemented sink at construction.
  Exclusive display and headless render-to-CUDA already exist beneath the seam
  (HoloViz display path and `HolovizRenderBackend`'s headless mode).
- `backends/frame_data.hpp` — the per-frame payload shapes the concrete backends
  consume (`operators/src/holoviz_op/backends/holoviz_*_layer.hpp` include it): `ImageFrameData`,
  `GeometryBatch`, `TextRun`, and `LutFrameData` are what
  `LayerFrame::frame_data` points at, with the host/device lifetime contract
  documented in the header (device buffers are released only on a *successful*
  quiescing return — an error from `begin_frame()`/`abort_frame()`/`destroy()`
  keeps the retention duty alive). Each frame carries its own width/height within
  the layer's declared maximum, so a variable-resolution source never triggers a
  runtime reallocation.
- The enums live beside the descriptors: `ImageFormat` in `image_format.hpp`, and
  `YUVModelConversion`/`YUVRange`, `ColorSpace` (how surface data is interpreted
  on screen: `kPassThrough`/`kSRGBNonlinear`/`kBT709Linear`), and
  `GeometryTopology` in `holoviz_op.hpp` — intentionally trimmed sets derived from a survey of real
  applications (RGB/RGBA + NV12/YUYV camera input; 2D overlay primitives only,
  3D deferred). `PresentMode` exists on both sides — HoloViz's in
  `include/holoviz/present_mode.hpp`, the operator's in `sink_desc.hpp` — so the installed
  operator headers include no HoloViz header; `operators/src/common/holoviz_conversions.hpp`
  translates between them (and between `ImageFormat` and `holoviz::Format`) for every operator
  built on HoloViz.

One convention deserves its own paragraph because it is a real foot-gun: **layer
sources are straight (non-premultiplied) alpha** — caller RGBA textures, LUT
palettes, geometry colors, and the text renderers all emit it — but HoloViz's
`BlendState` *defaults* compose premultiplied (`src_color = kOne`). Every layer
pipeline therefore overrides the blend factors explicitly via
`operators/src/holoviz_op/backends/layer_blend.hpp`'s `kStraightAlphaLayerBlend`
(`src_color = kSrcAlpha`, `dst_color = kOneMinusSrcAlpha`, alpha accumulating
coverage). A new pipeline that forgets the override does not fail — it renders
subtly wrong edges.

`FrameCommands`, `SyncToken`, `SurfaceHandle`, and `AcquireToken` (in
`operators/src/holoviz_op/backends/handles.hpp`) are opaque for the substrate reason above: a
CUDA-only backend would have no command buffer and no Vulkan semaphore to name,
so the seam moves opaque tokens rather than typed handles. Each is a *distinct*
one-field struct wrapping a `shared_ptr<void>` rather than a shared alias, so
passing one kind where another is expected is a compile error — the compiler
checks the pairing between token kinds, though not what a given backend's
`impl` actually wraps. That residual opacity is the concrete price of keeping
the CUDA-substrate option open, and the natural place to push back if it does
not look worth designing around yet.

The unit tests under `operators/tests/unit/` — 11 gtest targets — cover this layer: they
build stub backends to confirm the interfaces are concretely implementable and
that a `SyncToken` flows from render to present without either side knowing the
other's type (plus compile-time checks in `test_enums.cpp` that the layer enums stay
deliberately distinct from their HoloViz counterparts, and `holoviz_test_sink_desc` for the
sink declaration surface), and exercise the present backend and every layer kind with
differential image comparisons (`holoviz_test_wsi_present_backend`,
`holoviz_test_backend_*`). They link the backend object library rather than the
operator DSO, which also hands them the `operators/src` and module-private `src/` include
roots for the module-internal surfaces they reach into (frame tables, token payloads, the
headless readback hook). HoloViz and its Vulkan backend have behavioral coverage of their
own under `tests/unit/` — 16 gtest targets (`holoviz_test_destroy`, `holoviz_test_swapchain_resize`
and `holoviz_test_gate_staleness` for device teardown, swapchain resize and gate staleness;
`holoviz_test_display` for the direct-display path), several of which add `src/` to their include path
privately (display-selection helpers). The text renderers get differential coverage of their own:
`holoviz_test_monospace_text` and `holoviz_test_text_renderer_compare` bake
both renderers from the same genuinely monospace font — imgui's bundled Cousine
TTF, passed in as `HOLOVIZ_TEST_MONO_TTF` — and skip when the file is absent.
CTest runs every test with
`HOLOVIZ_VALIDATION=1` in its environment so validation coverage is never
lost, and a GPU-less machine reports the binaries as *Skipped* rather than a
silent green: the "no usable GPU/window system" skips print a `[holoviz-skip]`
sentinel matched by `SKIP_REGULAR_EXPRESSION` (feature-specific skips
deliberately do not, so a partially-skipped binary still reports Passed).

### Fullscreen windows and G-Sync/VRR

`WindowSinkDesc::fullscreen` (and `holoviz::WindowDesc::fullscreen` underneath it) creates an
undecorated, monitor-sized GLFW window on the monitor named by `display_name` (window-system
name, e.g. `DP-2` on X11; empty selects the primary monitor, an unknown name logs the available
names and falls back to the primary; no monitor at all fails window creation with a HoloViz error).
With `width` or `height` left at zero the window takes the monitor's current video mode;
`refresh_hz` asks GLFW for the closest mode at that rate, and zero keeps the monitor's current
rate (not the highest one it supports). The original video mode is restored when the window is
destroyed, and `close_on_escape` makes ESC close the window because an undecorated window has no
close button. The window is created with `GLFW_RESIZABLE` still `TRUE`: Mutter (GNOME) refuses
`_NET_WM_STATE_FULLSCREEN` for fixed-size windows and would leave the window auto-placed on the
primary monitor. The window is not user-resizable, but its surface still opts into the
swapchain's automatic recreation (like a resizable window) so a video-mode or focus change that
invalidates the swapchain recovers instead of freezing the frame loop.

`holoviz_test_fullscreen_window` covers the fullscreen runtime behavior but takes over a monitor,
so it skips unless `HOLOVIZ_FULLSCREEN_TESTS=1` is set in the environment.

Variable refresh (G-Sync / Adaptive-Sync) in such a window is a property of the driver and
desktop, not of this module, and on the NVIDIA X11 driver all of the following have to hold
before the panel follows the presents:

1. **The window is flipped, not composited.** GLFW sets `_NET_WM_BYPASS_COMPOSITOR=1` on
   monitor windows and Mutter unredirects them, but anything that keeps the window
   composited forces the blit path, which never engages VRR. The known offender on Ubuntu
   GNOME is the AppIndicator extension's legacy XEmbed tray (`legacy-tray-enabled`, on by
   default): with it on, the overlay reads `BLIT`/`NORMAL` on an otherwise fully configured
   single-monitor G-Sync setup, and
   `gsettings set org.gnome.shell.extensions.appindicator legacy-tray-enabled false` flips it
   to `FLIP`/`G-SYNC` without a shell restart (verified on driver 595 with GNOME 46 on X11).
2. **The G-Sync panel is the only active output on the X screen.** The driver flips a window
   only when it covers the entire desktop, so on a multi-monitor X screen a fullscreen window
   on one panel is still blitted. Turn the other outputs off for the run
   (`xrandr --output <other> --off`, `--auto` afterwards) or give the panel its own X screen.
3. **The presents pace on the target monitor.** By default the driver paces FIFO presents on
   the primary output's vblank; `__GL_SYNC_DISPLAY_DEVICE=<display_name>` moves that to the
   named monitor. The driver reads the variable once, when the process-wide Vulkan instance is
   first created, so the *application* exports it before launching the graph (before any Vulkan
   use in the process). `HolovizOp` never writes it (a `setenv()` in `start()` would race
   `getenv()` on other scheduler threads and would be too late once any other operator has
   used Vulkan); it logs a hint when the variable is unset for a named fullscreen monitor.
   `gray_scale_example` exports it in `main()` for its `--fullscreen MONITOR` option. Under
   condition 2 the panel is also the primary output, so this only matters for pacing a
   non-primary monitor.
4. **The present rate is below the panel's maximum refresh.** An unpaced FIFO window presents
   once per refresh, so there is nothing for the refresh to follow; hold the frame rate inside
   the panel's VRR range (HolovizOp's `display_rate`, or host pacing in a HoloViz application).
   Presenting at the range floor (`DisplayVRRMinRefreshRate`) still shows `G-SYNC`, but any
   jitter below it makes the driver frame-double instead of tracking the presents.
5. **The panel is in VRR mode on the X screen.** `nvidia-settings -q [dpy:N]/DisplayVRREnabled`
   reads 1; unvalidated Adaptive-Sync panels need the MetaMode attribute
   `AllowGSYNCCompatible=On` first. See `public/docs/enable_gsync.mdx` for the driver setup.

The present mode is not one of the conditions: the driver engages G-Sync for both `kFIFO`
and `kFIFOLatestReady` (what `PresentMode::kAuto` resolves to on drivers exposing
`VK_EXT_present_mode_fifo_latest_ready`), but its overlay labels the latter `VSYNC OFF`, so a
demonstration should pin `WindowSinkDesc::present_mode = kFIFO` to keep the readout unambiguous.

The driver's own overlay is the authority on whether all of this holds:
`nvidia-settings -a ShowGraphicsVisualIndicator=1 -a ShowVRRVisualIndicator=1` prints
`FLIP`/`BLIT` and `G-SYNC`/`NORMAL` in the window. `gray_scale_example --fullscreen <MONITOR>
--fps 90` is the smallest in-tree reproduction; exclusive display (direct-to-display) sidesteps the X
server and needs none of the desktop conditions, but is a separate present path.

## The text module

`include/holoviz/text/` (namespace `holoviz::text`) is the text-rendering seam
the text layer is built on. `text_renderer.hpp` declares `TextRenderer`, a
pure-virtual interface with an ImGui-backed implementation behind it; the
push-constant monospace renderer (`monospace_text_renderer.hpp`) slots in
behind the same API for the fixed-font, no-atlas-update case. The umbrella
header is `<holoviz/text.hpp>`.

## The imgui module

`include/holoviz/imgui/` (namespace `holoviz::imgui`) is the Dear ImGui
integration the ImGui-backed text renderer and the debug UI are built on.
`renderer.hpp` declares `imgui::Renderer`, the HoloViz-driven renderer for
ImGui draw data: beyond drawing, it supports `ImGui::Image` via
`register_image()`/`unregister_image()` (setup-path, borrowed image), and
event-driven GLFW input via `install_input_callbacks()`/
`uninstall_input_callbacks()` (mouse, scroll, keyboard, char, focus;
`forward_input()` remains as a deprecated polling fallback). The umbrella
header `<holoviz/imgui.hpp>` pulls in `renderer.hpp` only. `imgui/ui.hpp` is
the opt-in header for arbitrary ImGui debug UI — it is the only header that
exposes ImGui itself (the bundled imgui headers install under
`holoviz/imgui/vendor/`); the rest of the imgui and text APIs only
forward-declare it.

## Building

HoloViz supports two build contexts: embedded as part of the SDK or as a
top-level Holoscan module against an installed SDK. An independent standalone
backend build without an SDK is not supported. Consuming the backend is a
separate matter: external projects link the exported
`holoscan::visualization::holoviz` through
`find_package(holoscan COMPONENTS visualization)` (see
[Consuming the installed module](#consuming-the-installed-module)). Each
dependency recipe under `public/cmake/deps/` owns its version and uses RAPIDS
CPM in both contexts.

## Installed CMake package

New consumers use the Visualization module package, which imports HolovizOp
(`holoscan::visualization::op_holoviz`) and the HoloViz rendering library
(`holoscan::visualization::holoviz`):

```cmake
find_package(holoscan-visualization CONFIG REQUIRED)
target_link_libraries(my_pipeline PRIVATE holoscan::visualization::op_holoviz)
```

Configure with `-Dholoscan-visualization_DIR=<prefix>/lib/cmake/holoscan`. The required
compatibility form is `find_package(holoscan COMPONENTS visualization REQUIRED)`.

### With the Holoscan SDK

Built by default. From `public/`:

```bash
./run build --config "-DHOLOSCAN_BUILD_EXAMPLES=ON"
```

Dependencies come from the SDK superbuild helper — `holoscan_add_dependency()`
routes shared dependencies (tl-expected and magic_enum) and HoloViz-private
dependencies (GLFW, Slang, Vulkan-Headers, Vulkan-Utility-Libraries, volk, VMA,
and imgui) through `public/cmake/deps/*.cmake`. The shared dependencies are
installed and exported by the SDK; the HoloViz-private ones are header-only,
prebuilt, or (GLFW) built as a static archive that is absorbed into
`libholoviz`, so none of them is installed or exported. HoloViz directly observes
`HOLOSCAN_BUILD_TESTS` and `HOLOSCAN_BUILD_EXAMPLES`. The backend is consumed
in the same build. Everything the module owns ships in the
`holoscan-visualization` install component: both libraries, the `<holoviz/**>`
and `<holoscan/visualization/**>` headers, and the module's package files
(`holoscan-visualization-config.cmake`, its version file and
`holoscan-visualization-targets.cmake`, the module's own export set). The SDK's
`holoscan-config.cmake` loads that targets file as its `visualization`
component (see
[Consuming the installed module](#consuming-the-installed-module)).

#### Coexisting with an application-provided GLFW

HoloViz's GLFW is a private static archive with hidden symbols
(`--exclude-libs,ALL`, see [Symbols](#symbols)), so an application may link its
own GLFW — any 3.x, shared or static — into the same process without a symbol
clash. The two instances are fully independent: each runs its own
`glfwInit`/`glfwTerminate` lifecycle, opens its own display-server connection,
and keeps its own window list. Two consequences follow. The application must
keep calling HoloViz's `poll_window_events()` alongside its own
`glfwPollEvents()`, because each GLFW only dispatches events for its own
windows. And the `GLFWwindow*` returned by `Window::native_object(kGLFWwindow)`
belongs to HoloViz's instance: passing it to the application's GLFW dereferences
a struct laid out by a different build of the library. Deliver input through
`Window::set_input_callbacks()` or `imgui::Renderer::install_input_callbacks()`
instead. This is also why `holoviz_test_imgui_input` skips — it borrows
libholoviz's GLFW key trampoline, which needs the test to share libholoviz's GLFW
instance.

### Top-level Holoscan module build

A top-level Holoscan module configure requires an installed SDK. Loading
`holoscan-config.cmake` supplies the shared tl-expected, magic_enum, and CUDA
targets; HoloViz resolves its private graphics dependencies from the central
SDK recipes in the surrounding source tree.

```bash
cmake -S public/modules/visualization -B /tmp/holoscan-visualization-build -G Ninja \
  -Dholoscan_ROOT=/path/to/holoscan/install \
  -DCMAKE_INSTALL_PREFIX=/path/to/holoscan/install \
  -DHOLOSCAN_BUILD_EXAMPLES=ON
cmake --build /tmp/holoscan-visualization-build
ctest --test-dir /tmp/holoscan-visualization-build --output-on-failure
cmake --install /tmp/holoscan-visualization-build
```

`CMAKE_INSTALL_PREFIX` is the same SDK prefix on purpose: a top-level build
augments the installed SDK in place. Both build contexts install the same
payload in the `holoscan-visualization` component: the libraries, the headers,
the module config/version files and `holoscan-visualization-targets.cmake`
under `<libdir>/cmake/holoscan`. Once they sit beside `holoscan-config.cmake`,
the SDK package exposes them as its `visualization` component and the dedicated
`holoscan-visualization` package finds them directly (see
[Consuming the installed module](#consuming-the-installed-module)).

`holoscan_ROOT` names the SDK install prefix and is the convention used by
Holoscan module CI. For an exact package-config location, use
`-Dholoscan_DIR=/path/to/holoscan/install/lib/cmake/holoscan` instead. A generic
`CMAKE_PREFIX_PATH` or the system package search path also works because every
top-level configure requires Holoscan.

The top-level module build initializes the SDK's pinned RAPIDS CMake helper
from the surrounding source tree. On Unix, libdrm is a system prerequisite for
the direct-display path; on Windows the equivalent is `windowsapp.lib` from
the Windows SDK. The Vulkan SDK is *not* a prerequisite: HoloViz never links
the Vulkan loader, and the pinned Khronos headers are acquired through CPM.

Point `CMAKE_PREFIX_PATH` at prebuilt private dependencies to let CPM find them
before fetching. Slang has an explicit runtime-ownership contract controlled by
`HOLOVIZ_SLANG_PROVIDER`: `AUTO` installs the pinned prebuilt archive with a
source fallback, `PREBUILT` requires that archive, `SOURCE` always builds it,
and `EXTERNAL` uses standard CMake package discovery without installing Slang.
In `EXTERNAL` mode, use `slang_DIR` for an exact package-config directory,
`slang_ROOT` for an installation prefix, or `CMAKE_PREFIX_PATH` and the system
search paths for ordinary discovery. The other exception is imgui: the pinned
source is compiled directly into `libholoviz` and its two public headers are
bundled under `holoviz/imgui/vendor/`, so a system package is never substituted. Tests
default ON and examples OFF.

The HoloViz-private pins are defined once in `public/cmake/deps/*.cmake` and
used in both contexts. A top-level Holoscan module gets shared dependency
versions from the installed SDK; an embedded build gets them from the SDK-wide
recipes.

| Dependency | Pinned version |
|---|---|
| GLFW | 3.4 |
| Slang | 2026.14 |
| Vulkan-Headers / Utility-Libraries / volk | vulkan-sdk-1.4.350.0 |
| VMA | 3.4.0 |
| magic_enum | 0.9.7 |
| tl-expected | 1.1.0 |
| imgui | 1.89.3 |

### Windows development build (clangd / compile_commands.json)

Ninja is the usual generator for `compile_commands.json` but is not required.
NMake ships with Visual Studio and produces the same output.

System prerequisites are the CUDA Toolkit on `PATH` and a Windows SDK (for
`windowsapp.lib`, which the direct-display path links). The LunarG Vulkan SDK is
not required. A Holoscan module build additionally needs an installed SDK.

Open the **x64 Native Tools Command Prompt for VS** from the Start menu — not a
regular Developer Command Prompt or PowerShell, which default to the 32-bit
toolchain and fail at link time against the 64-bit Slang and CUDA binaries.
Then from `public/modules/visualization/`, point CMake at an installed Holoscan SDK:

```cmd
cmake -S . -B build -G "NMake Makefiles" -Dholoscan_ROOT=C:\path\to\holoscan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHOLOSCAN_BUILD_EXAMPLES=ON
cmake --build build
```

The module's `.clangd` config already points at `build/` as the
compilation database, so clangd picks up `build/compile_commands.json`
automatically once it exists. Configuring alone generates the export header
(`holoviz_export.h`); building as well is the belt-and-braces way to be sure
everything clangd indexes actually compiles.

### Consuming the installed module

HoloViz and `HolovizOp` are exported together in the module's own export set,
`<libdir>/cmake/holoscan/holoscan-visualization-targets.cmake`; there is no
separate `holoviz` package or export file. `libholoviz` and
`libholoscan_op_holoviz` (plus their import libraries on Windows), the
`<holoviz/**>` and `<holoscan/visualization/**>` headers, and the module's
config, version and targets files all ship in the `holoscan-visualization`
component, whether the module was built inside the SDK or as a top-level
module. Two entry points import the same targets: the `visualization`
component of the Holoscan package, and the dedicated
`holoscan-visualization` package configured with
`-Dholoscan-visualization_DIR=<prefix>/lib/cmake/holoscan`. External projects
such as Hololink use:

```cmake
find_package(holoscan 5 CONFIG REQUIRED COMPONENTS visualization)
# An operator built directly on HoloViz:
add_library(my_visualization_operator SHARED my_visualization_operator.cpp)
target_link_libraries(my_visualization_operator PRIVATE
    holoscan::core holoscan::visualization::holoviz)
# An application using HolovizOp:
target_link_libraries(my_app PRIVATE holoscan::core holoscan::visualization::op_holoviz)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/installed/sdk`.
`holoscan::visualization::holoviz` supplies the installed include path, C++20 requirement,
CUDA driver dependency, and `tl::expected`. `holoscan::visualization::op_holoviz` needs only
`holoscan::core` from its consumers: the `<holoscan/visualization/**>` headers
include no HoloViz header, and HoloViz is a private runtime dependency of the
operator library. Link `holoscan::visualization::holoviz` only for operators built directly
on HoloViz. An unqualified `find_package(holoscan)` imports
the component too whenever the SDK was built with `HOLOSCAN_BUILD_VISUALIZATION`;
`holoscan_visualization_FOUND` reports whether `holoscan::visualization::op_holoviz`
was imported, and only an explicit `COMPONENTS visualization` request fails when
it was not. Runtime deployments
need the CUDA driver and the Slang runtime described below, with DLLs on `PATH`
on Windows.

The layer renderer's seam and implementations
(`operators/src/holoviz_op/backends/`) remain private to the operator DSO and
are never installed. External operators use the installed HoloViz and text APIs.

Three caveats. Slang is fetched as a prebuilt release archive where one exists
for the host (Linux and Windows on x86_64/aarch64) and is a shared library at
runtime — the official releases ship no static binaries. On any other host the module build falls back to compiling Slang
from source. `HOLOVIZ_SLANG_PROVIDER=SOURCE` makes that choice explicit;
`EXTERNAL` skips both bundled acquisition paths. On Linux, SDK and top-level
module installs use `$ORIGIN` to find a bundled Slang runtime installed
beside `libholoviz`; an external Slang remains its supplier's deployment
responsibility. GLFW is a private static archive absorbed into `libholoviz` in
both build contexts (`public/cmake/deps/glfw.cmake`), so no GLFW runtime is
installed or exported. And `holoscan_security_flags` is an SDK-internal INTERFACE
target that the installed Holoscan package does not carry, so a top-level module
build gets only the toolchain's default hardening; the SDK-embedded path remains
the shipping configuration.

There is no module-local `run` wrapper or Dockerfile — use the SDK's
`public/run` for containerized workflows.

### Running the examples

The textured-quad demo needs a window, a CUDA-capable GPU (it initializes the
CUDA driver before creating the device), and takes the directory holding its
resources — the image and the shader — as its argument:

```bash
./holoviz_textured_quad <path-to>/examples/core/textured_quad
```

It brings up a window, compiles a Slang shader at runtime, uploads a texture,
and spins a blended textured quad driven by a push constant.
`examples/core/textured_quad/Main.cpp` is the best single read for what the API
feels like in practice.

Headless rendering has its own sample — no display required:

```bash
./holoviz_render_to_cuda
```

renders to a Vulkan image, accesses the result as a CUDA array, and writes the
final frame to `render_to_cuda_output.bmp` for offline inspection.

`holoviz_direct_display` is the exclusive-display (compositor-bypass)
sample: it enumerates the adapter's displays with their ownership
classification, acquires an available one, and renders straight to the panel
with no window system in the loop. `--width W`/`--height H`/`--hz R` pick the
closest supported mode (unset means largest/fastest) and `--frames N` bounds
the run.

The text and imgui examples need a window:

```bash
./holoviz_text_minimal      # per-frame text loop; --monospace / --fillrect / --benchmark
./holoviz_imgui             # arbitrary ImGui debug UI via <holoviz/imgui/ui.hpp>
```

`holoviz_cuda_interop_benchmark` takes the same resource-directory argument and is
the CUDA-interop reference: each frame a CUDA FFT Gaussian blur writes into a
scratch buffer, the result is copied device-to-device into a
`ShareMode::kExported` image (the equivalent of the 4.x `ImageCudaDevice()`
path), sampled onto a full-screen quad and presented, with FPS measured over a
fixed window and printed on exit.

Verified on Thor IGX (aarch64), x86_64 Linux, and Windows x64 — the demo and
the benchmark, CUDA interop included, run on all three.

## Known gaps and caveats

All of these are known and on the roadmap; they are listed so a reviewer knows
where the edges are, roughly in order of significance:

- **Present feedback is wired for waiting, not yet for measuring.**
  `VK_KHR_present_id`/`present_wait` are negotiated at device creation
  (`CapabilityReport::present_completion`), `Swapchain::present()` tags frames
  and `Swapchain::wait_for_present()` blocks until display — and
  `WsiPresentBackend::wait_for_present()` uses that natively, falling back to a
  render-done wait on devices without the extensions. What is still missing is
  *measured* present timing (`PresentTiming`/`query_timing()` still return
  nothing): no timestamps to pace against and no deadline scheduling until a
  display-timing extension (e.g. `VK_EXT_present_timing`) is wired up.
- **YUV image formats are deferred.** Declaring `kNV12`/`kYUYV422` (or
  `kR8G8B8Unorm`) in an image layer fails `allocate()` with `kNotImplemented`;
  the planned lowerings are recorded in `operators/src/holoviz_op/backends/holoviz_image_layer.hpp`. The
  frame-data shapes (`ImageFrameData::planes[2]`) already accommodate them.
- **CUDA interop is implemented but young.** Exported buffers, images and
  external semaphores run end-to-end (the benchmark and the layer backends
  exercise the full path), but the failure paths (import failures, context
  mismatches) have seen little exercise. It deserves the closest review.
- **Vulkan validation is off by default**, enabled with `HOLOVIZ_VALIDATION`
  set to `1`/`true`/`on` (validation layer when the loader finds it, plus the
  debug-utils messenger). The same switch gates extra module-side debug checks,
  e.g. the device's live-child count at teardown. The variable is read once,
  before the lazily created instance exists. CTest sets it for every test, so
  the whole suite runs validated by default.
- **HolovizOp parity scope is deliberately deferred** (decision recorded at the
  5.0 review): image uploads are `kR8G8B8A8Unorm`/`kB8G8R8A8Unorm`/`kR8Unorm`/`kR16Unorm`/
  `kR16G16B16A16Unorm` only (unorm 8- and 16-bit; no float/sRGB/segmentation-integer formats,
  and the operator drops samples whose dtype/channel count disagree with the declared layer
  format — covered at the operator boundary by `holoviz_test_holoviz_op_image_admission`;
  non-pass-through
  `color_space` is rejected, not converted), no depth/camera/3D topologies, no
  per-layer opacity or per-vertex colors, headless output is reachable only
  through the test accessor `HolovizRenderBackend::headless_target()` (no
  production render-to-CUDA-buffer path yet), and a backend drives a single
  device (no multi-GPU peer/host-copy fallback).
- **Swapchain resize is automatic, with one caveat.** On resizable surfaces
  `holoviz::Swapchain::acquire()/present()` recreate in place (`oldSwapchain`
  handover, image indices identity-stable) when the surface changes; a frame
  loop needs no resize code and only sees `kFrameDropped` (a recreation
  consumed the frame) and `kImageCountChanged` (rebuild per-image resources).
  Staleness is checked, not assumed: `AcquireSyncToken` carries the swapchain
  `generation` that minted it — gates from a superseded generation return
  `kInvalidState`, and a stale token passed to `present()` returns
  `kFrameDropped`.
  Non-resizable surfaces skip all of this and keep the explicit-`resize()` /
  `kOutOfDate` contract. The caveat: detection polls the surface's
  `currentExtent` once per acquire, so window systems whose surfaces report no
  fixed extent (Wayland) still need an explicit `resize()` with the window's
  framebuffer size.
- **`DevicePolicy::kSealed` is not enforced.** Only `kDynamic` exists in 5.0
  (requesting `kSealed` fails at device creation, and `seal()` itself currently
  returns an error), so the allocation-free steady state is a convention today,
  not a guarantee — `HolovizRenderBackend` upholds it by construction (pools at
  allocate(), pipelines at build(), warmed lazy variants), but nothing checks.
- **Render pass load/store ops are hardcoded** (always clear on load, store on
  end); the desc fields are present but commented out. Depth attachments
  (`set_depth_image`) are unimplemented and return errors.
- **Shader compilation is runtime-only.** Slang → SPIR-V happens in-process via
  a process-wide session; there is no offline/precompiled path and no
  `VkPipelineCache` persistence. The lazy per-render-pass pipeline variants
  *are* now warmed at `build()` time by the render backend, so the frame path
  compiles nothing.
- **Lifetime rules are documented, not enforced.** Slices, descriptors, and
  render-pass attachments all borrow; the headers say what must outlive what,
  and nothing checks it.
- `Adapter::supports_surface(const Surface&)` probes against a caller-owned
  surface, so adapter selection is setup-time work, not something to repeat per
  frame; `create_device_for_surface()` is the convenience wrapper that picks the
  first adapter that can present to it.
- Line-family geometry (`kLineList`/`kLineStrip`/`kCrossList`) rasterizes as
  width-1 GPU lines — HoloViz exposes no line-width state. Rectangle and oval
  outlines honor `line_width` via instanced triangle rings; wide line segments
  are a follow-up on the same expansion infrastructure.

## Design notes

[`docs/`](docs/README.mdx) carries six short notes on background this README has no
room for: why 4.x immediate mode was abandoned, where copies are structurally
required, the CUDA synchronization contract, what the timing API has to provide
(and where its wait/measure halves stand), the split between this module and
`HolovizOp`, and how backends, adapters, and graphics capability relate (including
the corVK question).
