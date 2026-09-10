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
the point of reviewing them now: exposing the layer renderer, or the RHI beneath
it, as supported APIs is a plausible later step, and the design is meant to hold
up if we take it — but nothing in this commit should be read as a commitment
either way.

Holoscan Visualization is the SDK module and project. HoloViz is the rendering backend contained
by that module; `holoviz` remains the backend's CMake target, namespace, include path, and internal
helper prefix. `HolovizOp` is the Holoscan operator that integrates the backend with the SDK.

The module is built in two layers. The bottom one, `holoviz::rhi`, is an
**RHI — a rendering hardware interface**, the usual term for a thin portability
layer that gives one API over several graphics APIs (Vulkan, D3D12, and so on).
Ours has a working Vulkan backend — enough to run every example and test end to
end, not a finished abstraction. Above it sits the layer renderer
(`RhiRenderBackend`/`WsiPresentBackend` behind the `RenderBackend`/`PresentBackend`
seams), implemented for the declared layer kinds. The module also owns
`HolovizOp`, the Holoscan operator that adapts SDK tensors and lifecycle semantics
to these backends. The RHI was designed against that layer API, so most of what
follows reads as "here is the primitive, and here is the thing above it that
needs it."

The module produces a static library by default (`libholoviz.a` / `holoviz.lib`),
exported as the build-tree target `holoviz::holoviz`, for linking into the
Holoscan Visualization module, tests, and examples. `HOLOVIZ_BUILD_SHARED=ON`
builds it as a shared implementation library instead (`libholoviz.so` /
`holoviz.dll`). It builds as part of the SDK or as a top-level Holoscan module.
It is not linked into `holoscan::core`; the operator integration consumes it.

## What is here

```
include/holoviz/rhi/     RHI API             — namespace holoviz::rhi   (Vulkan backend working)
include/holoviz/*.hpp    Layer renderer API  — namespace holoviz        (implemented on the RHI)
include/holoviz/text/    Text rendering API  — TextRenderer seam, ImGui backend, monospace renderer
include/holoviz/backends/  Factories for the concrete backends (RhiRenderBackend, WsiPresentBackend)
include/holoviz/         errors/expected/logger: the module's own diagnostic vocabulary — plus the
                         seam headers: sink_desc.hpp, handles.hpp, frame.hpp/frame_data.hpp,
                         backend_preference.hpp, color_space.hpp
include/holoviz/integrations/   Optional header-only bridges (Holoscan logger)
src/rhi/                 Backend-agnostic dispatch + Slang compiler
src/rhi/cuda/            CUDA bootstrap shared across backends (device/context init, error strings)
src/rhi/vulkan/          The Vulkan backend  (~8k lines, the bulk of the code)
src/rhi/window_systems/  GLFW windowing
src/rhi/display_systems/ Direct-display ownership classification/acquisition (WinRT on Windows, libdrm on Linux)
src/text/                The text renderers (ImGui backend + push-constant monospace renderer)
src/backends/            The concrete RenderBackend/PresentBackend implementations + layer states
cmake/                   SDK/module dependency resolution logic (recipes: public/cmake/deps/)
docs/                    Design-note index + stable implementation background
examples/rhi/            Direct RHI samples: textured quad, headless render-to-CUDA, CUDA-interop benchmark, direct display
examples/text/           Text renderer samples: minimal per-frame text loop, ImGui debug UI
examples/layers/         The RenderBackend/PresentBackend seam driven end to end (HolovizOp's frame loop)
operators/               Holoscan integration, including HolovizOp
tests/unit/              15 gtest targets covering the layer renderer API, the backends, and every layer kind
```

The line between `holoviz` and `holoviz::rhi` is a substrate boundary, not just a
layering one. What `holoviz` composes layers *with* is meant to be swappable, so
its interfaces name no graphics types — `holoviz::rhi` is one substrate that can
sit underneath, and everything Vulkan lives inside it.

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
substrate, not different substrates, so they belong together behind one RHI
rather than being split apart a level up. One graphics layer with several
backends, one compute layer, an explicit boundary between them — that structure
is chosen to match what the hardware actually does.

**The graphics substrate** is the RHI, Vulkan today, and is what 5.0 ships. It is
also where the breadth of display paths lives: Win32, X11, Wayland and QNX Screen
for windowed output, plus compositor-bypass modes — Vulkan's exclusive-display
extensions (implemented: `rhi/display.hpp` and the `direct_display` example), and
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

*Inside the RHI*, as another presentation path behind `Swapchain` and its gates.
Everything above then sees one presentation abstraction no matter which substrate
is underneath, and present timing gets implemented once. The objection is that
`Swapchain`'s whole vocabulary — surfaces, acquire/present gates wrapping binary
semaphores — is shaped by graphics presentation engines, and a CUDA-only path
fits it only by analogy. (An earlier version of this objection was that it would
pull CUDA into a deliberately CUDA-free RHI; that ship has sailed — CUDA interop
is now first-class in the RHI, see below — so the vocabulary mismatch is the
remaining substance.)

*Outside the RHI*, as a `PresentBackend` implementation that never touches it.
The RHI stays purely graphics and the substrate boundary stays honest, at the
cost of two independent presentation implementations, with pacing and timing
logic to keep in agreement between them. This is the placement the current
opaque-token seam is built for.

## The RHI (rendering hardware interface) layer

The RHI is a thin, explicit abstraction over a graphics API — closer to WebGPU
than to a scene graph. It is the graphics substrate from the split above, and it
abstracts a graphics API rather than targeting Vulkan directly because that
substrate has to cover more than desktop Vulkan on its own: Vulkan SC and D3D12
are the anticipated additions, across the display paths listed earlier. It also
exists because Holoscan needs CUDA interop and presentation timing to be
first-class rather than bolted on. Interop now is: exported buffers, images and
semaphores are implemented end-to-end (see the header table). Timing is the
part still missing, and the next major piece of RHI work (see known gaps).

The closest reference point is **NVRHI**, the RHI the RTX team built for their
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

Every public type in `include/holoviz/rhi/` is a small **concrete handle** —
`Buffer`, `Image`, `Pipeline`, `CommandList`, `Swapchain`, `Device` and the rest
are move-only value types you get back from `Device::create_*()` as
`expected<Buffer>`, `expected<Image>`, and so on. Each handle wraps a
`std::unique_ptr<detail::XImpl>`, where `detail::XImpl` is the pure abstract
backend interface; the concrete classes that implement those interfaces live
entirely in `src/rhi/vulkan/`.

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

`holoviz::rhi` re-exports the two error types with `using` aliases in
`rhi/common.hpp` so RHI call sites keep reading in terms of the RHI namespace.

### Symbols

HoloViz is built with hidden visibility and a generated
`<holoviz/holoviz_export.h>`: a declaration is exported only if it is marked
`HOLOVIZ_EXPORT`. In the default static build the
macros compile away (`HOLOVIZ_STATIC_DEFINE`); in the shared build
(`HOLOVIZ_BUILD_SHARED=ON`) nothing from Slang, GLFW, volk or VMA is re-exported
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

The RHI's job is to make each of those an explicit object the layer renderer can
create once at startup, so the per-frame path is only recording and submitting.
The headers under `include/holoviz/rhi/` carry the design rationale inline; the
table below adds what each piece is *for* one level up.

| Header | The decision, and what it buys the layer renderer |
|---|---|
| `common.hpp` | The shared vocabulary: a deliberately small `Format` set, `Backend`/`ShareMode`/`DevicePolicy`, and the `NativeObject` escape hatch for borrowing raw backend handles. The format list is trimmed to what layers actually need, so an unsupported pixel format is caught up front rather than deep in a backend. |
| `device.hpp` | One device, one graphics queue, no queue object. Every `submit()` advances an internal **device timeline** and returns the `SubmitSyncToken` that completes with the batch — a single "this work is done" token. A batch waits only on the tokens the caller passes in (`SubmitDesc::wait_submit_sync_tokens`), so independent submits are free to overlap on the GPU. That token is what `RenderBackend::submit()` will hand to `PresentBackend::present()` as its `SyncToken`; `wait_on_submit()` is its host-side blocking counterpart. |
| `external_semaphore.hpp` | Timeline values are the *only* sync vocabulary — no fences, no binary semaphores, no events. `ExternalSemaphore` is the CUDA edge of it: created from a `CUstream`, its `wait()` makes the stream wait for a submit's token, and `signal()` returns an `ExternalSyncToken` a later submit can wait on. One mechanism covers "wait for the GPU", "wait for the previous frame", and "wait for CUDA", which is what keeps the frame loop's synchronization small enough to reason about. |
| `swapchain.hpp` | The one place timelines don't reach. Presentation engines only speak binary semaphores, so `acquire()` returns two single-use **gates** per frame that ride the caller's real submit rather than extra shim submits. The gate objects are pre-allocated at swapchain creation in contiguous storage and recycled — `acquire()` allocates nothing. This is the whole acquire → render → present loop `PresentBackend` owns, with the sharp-edged part already handled and exactly-once consumption enforced. |
| `display.hpp` | Exclusive (direct-to-display) output, implemented. `Display` values are enumerated per adapter with an ownership classification (`kCompositor` / `kAvailable` / `kUnknown`), list their modes, and `create_surface()` acquires the display and yields a non-resizable `Surface` for compositor-bypass scanout. The platform acquisition lives in `src/rhi/display_systems/`: WinRT + `VK_NV_acquire_winrt_display` on Windows, libdrm + `VK_EXT_acquire_drm_display` on Linux. `examples/rhi/direct_display/` drives it end to end. |
| `image.hpp` | Zero image-layout vocabulary in the API. Every image is pinned in one *home state* derived from its usage, and anything needing another state transitions and restores internally. The layer renderer's texture pool can therefore be allocated once and handed around without tracking GPU-side state per image — and a CUDA producer writing into one (via `cuda_mipmapped_array()` on an exported image) always finds it in a known state. The device-side transfer helpers (`upload_to_image`/`download_from_image`) honor the caller's `row_pitch_bytes` and currently accept only mip level 0 (images are single-mip for now). |
| `pipeline.hpp` | Shaders are Slang-only and carry their own **reflected** interface, so pipeline creation and descriptor binding are validated against what the shader declares. The pipeline set a render backend needs is enumerated backend-internally from the declared `LayerStack` (it is not described at the seam) and becomes RHI pipelines one-for-one, and a mismatch between shader and binding code is a startup error, not a wrong-looking frame. No uniform buffers and no compute: runtime data is push constants or storage buffers. |
| `descriptor.hpp` | One pre-baked binding set per pipeline, bindable by reflected index *or* by shader variable name. Each declared layer gets its descriptor filled in once at `allocate()` time; per frame it is just bound. Chosen over per-draw descriptor writes and buffer-device-address tricks, which do not port to every target. |
| `command_list.hpp` | Two list types encode the two execution styles: `kTransient` (re-record per frame) and `kReusable` (record once at setup, replay forever). A fixed layer set with fixed compositing order is exactly the case `kReusable` exists for. |
| `query.hpp` | GPU timestamp queries for profiling submitted work. Readback is deliberately a three-step recorded contract — reset, write, resolve — because not every planned backend can read results straight from the pool (D3D12 resolves into a readback buffer); `results_ns()` on the host after the resolving submit completes. This measures GPU execution spans; it is *not* the present-feedback API (see known gaps). |
| CUDA interop (cross-cutting) | Not a separate header: interop is built into the resource types. A resource created with `ShareMode::kExported` is born with external-memory handles and imported into CUDA at creation — `Buffer::cuda_device_ptr()` and `Image::cuda_mipmapped_array()` hand back typed CUDA objects, `Device` exposes `cuda_device()`/`cuda_context()`, and `ExternalSemaphore` bridges a `CUstream` onto the device timeline. This is the path by which a CUDA-produced video frame becomes a layer texture without a copy (`examples/rhi/cuda_interop_benchmark/` exercises it end to end). The CUDA context relationship is decided at device creation: `DeviceDesc` carries `cuda_device`, `cuda_context`, and a `CudaContextMode` — `kExternal` (caller-supplied or primary context) or the implemented CUDA-in-Graphics modes `kCigRequired`/`kCigPreferred`, which create a CiG context sharing the graphics hardware queue via `VK_NV_external_compute_queue` (`reserved_external_queues` sizes the reservation; this extension is why the Vulkan headers are pinned to 1.4.305+). Adapter↔CUDA-device matching is UUID-first: `Adapter::uuid` matches `cuDeviceGetUuid`, so a CUDA ordinal and an RHI adapter are identified as the same silicon. The cost of making it first-class: `<cuda.h>` is a hard include of the RHI headers and `CUDA::cuda_driver` a public link dependency — there is no CUDA-free consumer of the RHI. |

Three cross-cutting choices drive most of the API shape:

**Allocate at setup, execute allocation-free.** The intended lifecycle is
`create → setup → seal() → execute`, which is the RHI-level counterpart of
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
longer reserved: exclusive display is implemented (`rhi/display.hpp` and the
platform acquisition under `src/rhi/display_systems/` — see the header table).
What is available is negotiated, not assumed: `CapabilityReport` reports
`present_completion`, `present_timing`, `exclusive_display`, and
`fill_rectangle`, plus a per-format `FormatCapability` matrix saying what the
adapter actually supports for each use.

## The layer renderer

`include/holoviz/` is what `HolovizOp` is built on: a small, rigid 2D layer
renderer for the fixed-scene workloads that dominate real HoloViz usage. The
interfaces are implemented on top of the RHI (`src/backends/`): `RhiRenderBackend`
serves all four layer kinds (image, geometry, text, LUT — YUV formats deferred)
and `WsiPresentBackend` owns the swapchain loop. `examples/layers/backend_seam/` drives
the seam end to end, and `tests/unit/` covers every layer kind headless with
byte-exact and analytic oracles. API feedback is still cheap to act on — nothing
is committed yet.

- `layer_desc.hpp` — a `LayerStack` of declared layers (image / geometry / text
  / LUT) captured at operator construction time. Every layer declares its bounds
  (`max_width`, `max_primitives`, `max_glyphs`) so the full GPU resource set is
  knowable before the first frame — that declaration is what makes the RHI's
  allocate-at-setup lifecycle usable rather than aspirational. `LayerDesc` is a
  `std::variant` so adding a layer kind turns non-exhaustive visitors into
  compile errors.
- `render_backend.hpp` — the ownership seam for rasterization, with an explicit
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
  lifecycle phase maps onto the RHI: pools and images, then pipelines and
  reusable command lists, then record and submit. A CUDA backend, were we to
  build one, would satisfy the same contract with pools and kernels.
- `present_backend.hpp` — owns acquire → render → present, event-driven, never
  busy-polling, with graceful fallback when a present mode is unavailable. The
  graphics implementation is built on the RHI's `Swapchain` and its gates; it is
  also the interface a cuDisp-backed output would implement if the CUDA substrate
  goes ahead.
- `sink_desc.hpp` — the caller-facing output declaration: `SinkDesc` is a closed
  `std::variant` of `WindowSinkDesc`, `ExclusiveDisplaySinkDesc`,
  `CudaPitchLinearSinkDesc`, and `CudaArraySinkDesc`. Sinks describe *what*, a
  `PresentBackend` executes *how*. The window sink is live (`WsiPresentBackend`);
  the other three are declared now so call sites never change when their present
  paths are wired — a backend rejects an unimplemented sink at construction.
  Exclusive display and headless render-to-CUDA already exist beneath the seam
  (the RHI display path and `RhiRenderBackend`'s headless mode).
- `frame_data.hpp` vs `frame.hpp` — two deliberately separate per-frame shapes.
  `frame_data.hpp` is the live seam the concrete backends consume today
  (`src/backends/rhi_*_layer.hpp` include it): `ImageFrameData`,
  `GeometryBatch`, `TextRun`, and `LutFrameData` are what
  `LayerFrame::frame_data` points at, with the host/device lifetime contract
  documented in the header (device buffers are released only on a *successful*
  quiescing return — an error from `begin_frame()`/`abort_frame()`/`destroy()`
  keeps the retention duty alive). `frame.hpp` is the `HolovizOp`
  binding surface — `Subregion` (the per-frame active sub-rectangle of a
  layer's declared maximum) and `ImageFrame`/`ImagePlaneBinding`, which carry
  the Holoscan 5.x stream-and-lifetime contract for frames derived from
  received `Tensor`s.
- `backend_preference.hpp` — `BackendPreference` (`kAuto`/`kBuiltinVulkan`) is
  the caller's knob for choosing which `RenderBackend` implementation the module
  resolves to; future ICD-level backends are added here as first-class options.
- `color_space.hpp` — `ColorSpace` says how surface data is interpreted on
  screen (`kPassThrough`/`kSRGBNonlinear`/`kBT709Linear`).
- `image_format.hpp`, `image_color_space.hpp`, `geometry_topology.hpp`,
  `present_mode.hpp` — intentionally trimmed sets derived from a survey of real
  applications (RGB/RGBA + NV12/YUYV camera input; 2D overlay primitives only,
  3D deferred).

One convention deserves its own paragraph because it is a real foot-gun: **layer
sources are straight (non-premultiplied) alpha** — caller RGBA textures, LUT
palettes, geometry colors, and the text renderers all emit it — but the RHI's
`BlendState` *defaults* compose premultiplied (`src_color = kOne`). Every layer
pipeline therefore overrides the blend factors explicitly via
`src/backends/layer_blend.hpp`'s `kStraightAlphaLayerBlend`
(`src_color = kSrcAlpha`, `dst_color = kOneMinusSrcAlpha`, alpha accumulating
coverage). A new pipeline that forgets the override does not fail — it renders
subtly wrong edges.

`FrameCommands`, `SyncToken`, `SurfaceHandle`, and `AcquireToken` (in
`include/holoviz/handles.hpp`) are opaque for the substrate reason above: a
CUDA-only backend would have no command buffer and no Vulkan semaphore to name,
so the seam moves opaque tokens rather than typed handles. Each is a *distinct*
one-field struct wrapping a `shared_ptr<void>` rather than a shared alias, so
passing one kind where another is expected is a compile error — the compiler
checks the pairing between token kinds, though not what a given backend's
`impl` actually wraps. That residual opacity is the concrete price of keeping
the CUDA-substrate option open, and the natural place to push back if it does
not look worth designing around yet.

The unit tests under `tests/unit/` — 15 gtest targets — cover this layer: they
build stub backends to confirm the interfaces are concretely implementable and
that a `SyncToken` flows from render to present without either side knowing the
other's type (plus compile-time enum-parity checks against the RHI headers in
`test_enums.cpp`, `holoviz_test_sink_desc` for the sink declaration surface,
and `holoviz_test_rhi_display` for the direct-display path). The RHI and the
Vulkan backend also have behavioral coverage (`test_rhi_*`: device/teardown,
swapchain resize, present backend, and every layer kind with differential image
comparisons). The text renderers get differential coverage of their own:
`holoviz_test_monospace_text` and `holoviz_test_text_renderer_compare` bake
both renderers from the same genuinely monospace font — imgui's bundled Cousine
TTF, passed in as `HOLOVIZ_TEST_MONO_TTF` — and skip when the file is absent.
Several targets add `src/` to their include path privately to exercise
module-internal surfaces (frame tables, token payloads, display-selection
helpers, the headless readback hook) directly. CTest runs every test with
`HOLOVIZ_RHI_VALIDATION=1` in its environment so validation coverage is never
lost, and a GPU-less machine reports the binaries as *Skipped* rather than a
silent green: the "no usable GPU/window system" skips print a `[holoviz-skip]`
sentinel matched by `SKIP_REGULAR_EXPRESSION` (feature-specific skips
deliberately do not, so a partially-skipped binary still reports Passed).

## The text module

`include/holoviz/text/` is the text-rendering seam the text layer and the debug
UI are built on. `text_renderer.hpp` declares `TextRenderer`, a pure-virtual
interface with an ImGui-backed implementation behind it; the push-constant
monospace renderer (`monospace_text_renderer.hpp`) slots in behind the same API
for the fixed-font, no-atlas-update case. `imgui_backend.hpp` is the RHI-driven
ImGui backend underneath both the text path and the debug UI: beyond drawing, it
supports `ImGui::Image` via `register_image()`/`unregister_image()`
(setup-path, borrowed image), and event-driven GLFW input via
`install_input_callbacks()`/`uninstall_input_callbacks()` (mouse, scroll,
keyboard, char, focus; `forward_input()` remains as a deprecated polling
fallback). `imgui_ui.hpp` is the opt-in header for arbitrary ImGui debug UI —
it is the only header that exposes ImGui itself (the imgui headers install
under `holoviz/imgui/`); the rest of the text API only forward-declares it.

## Building

HoloViz supports two build contexts: embedded as part of the SDK or as a
top-level Holoscan module against an installed SDK. An independent standalone
backend build and direct downstream use of the backend library are not
supported. Each dependency recipe under `public/cmake/deps/` owns its version
and uses RAPIDS CPM in both contexts.

### With the Holoscan SDK

Built by default. From `public/`:

```bash
./run build --config "-DHOLOSCAN_BUILD_EXAMPLES=ON"
```

Dependencies come from the SDK superbuild helper — `holoscan_add_dependency()`
routes shared dependencies (tl-expected and magic_enum) and HoloViz-private
dependencies (GLFW, Slang, Vulkan-Headers, Vulkan-Utility-Libraries, volk, VMA,
and imgui) through `public/cmake/deps/*.cmake`. The helper registers both groups
in the SDK's install component and export set. HoloViz directly observes
`HOLOSCAN_BUILD_TESTS` and `HOLOSCAN_BUILD_EXAMPLES`. The backend is consumed
in the same build and ships no headers or package config of its own.

### Top-level Holoscan module build

A top-level Holoscan module configure requires an installed SDK. Loading
`holoscan-config.cmake` supplies the shared tl-expected, magic_enum, and CUDA
targets; HoloViz resolves its private graphics dependencies from the central
SDK recipes in the surrounding source tree.

```bash
cmake -S public/modules/visualization -B /tmp/holoscan-visualization-build -G Ninja \
  -Dholoscan_ROOT=/path/to/holoscan/install \
  -DCMAKE_INSTALL_PREFIX=/tmp/holoscan-visualization-install \
  -DHOLOSCAN_BUILD_EXAMPLES=ON
cmake --build /tmp/holoscan-visualization-build
ctest --test-dir /tmp/holoscan-visualization-build --output-on-failure
cmake --install /tmp/holoscan-visualization-build
```

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
bundled under `holoviz/imgui/`, so a system package is never substituted. Tests
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

The default build is a static library embedded into the surrounding SDK or
Holoscan Visualization module build. `HOLOVIZ_BUILD_SHARED=ON` changes that
internal linkage choice and installs the runtime library needed by the module.
Neither build type installs the backend headers or exports a
`find_package(holoviz)` package. The supported 5.0 surface is `HolovizOp`; the
lower-level backend can be reconsidered as a standalone product if a concrete
requirement emerges later.

Three caveats. Slang is fetched as a prebuilt release archive where one exists
for the host (Linux and Windows on x86_64/aarch64), and stays a shared library
at runtime even in the static build — the official releases ship no static
binaries. On any other host the module build falls back to compiling Slang
from source. `HOLOVIZ_SLANG_PROVIDER=SOURCE` makes that choice explicit;
`EXTERNAL` skips both bundled acquisition paths. On Linux, SDK and top-level
shared-module installs use `$ORIGIN` to find a bundled Slang runtime installed
beside `libholoviz`; an external Slang remains its supplier's deployment
responsibility. Fetched GLFW is pinned to a static build regardless of the
backend library type, so it is absorbed rather than becoming another module
runtime dependency. And `holoscan_security_flags` is an SDK-internal INTERFACE
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
./holoviz_rhi_textured_quad <path-to>/examples/rhi/textured_quad
```

It brings up a window, compiles a Slang shader at runtime, uploads a texture,
and spins a blended textured quad driven by a push constant.
`examples/rhi/textured_quad/Main.cpp` is the best single read for what the API
feels like in practice.

Headless rendering has its own sample — no display required:

```bash
./holoviz_rhi_render_to_cuda
```

renders to a Vulkan image, accesses the result as a CUDA array, and writes the
final frame to `render_to_cuda_output.bmp` for offline inspection.

`holoviz_rhi_direct_display` is the exclusive-display (compositor-bypass)
sample: it enumerates the adapter's displays with their ownership
classification, acquires an available one, and renders straight to the panel
with no window system in the loop. `--width W`/`--height H`/`--hz R` pick the
closest supported mode (unset means largest/fastest) and `--frames N` bounds
the run.

The text examples need a window:

```bash
./holoviz_text_minimal      # per-frame text loop; --monospace / --fillrect / --benchmark
./holoviz_text_imgui        # arbitrary ImGui debug UI via <holoviz/text/imgui_ui.hpp>
```

`holoviz_layers_backend_seam` needs no arguments and is the layer-renderer counterpart:
it drives the full `RenderBackend`/`PresentBackend` seam — declared layers
(geometry overlay + text), `allocate()`/`build()`, then the acquire → record →
submit → present loop with present-id tagging and `wait_for_present()` — in a
resizable window. `--frames N` exits after N frames for automated smoke runs.

`holoviz_rhi_cuda_interop_benchmark` takes the same resource-directory argument and is
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
  the planned lowerings are recorded in `src/backends/rhi_image_layer.hpp`. The
  frame-data shapes (`ImageFrameData::planes[2]`) already accommodate them.
- **CUDA interop is implemented but young.** Exported buffers, images and
  external semaphores run end-to-end (the benchmark and the layer backends
  exercise the full path), but the failure paths (import failures, context
  mismatches) have seen little exercise. It deserves the closest review.
- **Vulkan validation is off by default**, enabled with `HOLOVIZ_RHI_VALIDATION`
  set to `1`/`true`/`on` (validation layer when the loader finds it, plus the
  debug-utils messenger). The same switch gates extra module-side debug checks,
  e.g. the device's live-child count at teardown. The variable is read once,
  before the lazily created instance exists. CTest sets it for every test, so
  the whole suite runs validated by default.
- **HolovizOp parity scope is deliberately deferred** (decision recorded at the
  5.0 review): image uploads are `kR8G8B8A8Unorm`/`kR8Unorm` only (no
  float/16-bit/sRGB/segmentation-integer formats; non-pass-through
  `color_space` is rejected, not converted), no depth/camera/3D topologies, no
  per-layer opacity or per-vertex colors, headless output is reachable only
  through the test accessor `RhiRenderBackend::headless_target()` (no
  production render-to-CUDA-buffer path yet), and a backend drives a single
  device (no multi-GPU peer/host-copy fallback).
- **Swapchain resize is automatic, with one caveat.** On resizable surfaces
  `rhi::Swapchain::acquire()/present()` recreate in place (`oldSwapchain`
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
  not a guarantee — `RhiRenderBackend` upholds it by construction (pools at
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
  width-1 GPU lines — the RHI exposes no line-width state. Rectangle and oval
  outlines honor `line_width` via instanced triangle rings; wide line segments
  are a follow-up on the same expansion infrastructure.

## Design notes

[`docs/`](docs/README.mdx) carries six short notes on background this README has no
room for: why 4.x immediate mode was abandoned, where copies are structurally
required, the CUDA synchronization contract, what the timing API has to provide
(and where its wait/measure halves stand), the split between this module and
`HolovizOp`, and how backends, adapters, and graphics capability relate (including
the corVK question).
