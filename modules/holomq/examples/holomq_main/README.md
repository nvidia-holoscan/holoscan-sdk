# holomq_main

End-to-end publish–subscribe demo for HoloMQ over iceoryx2.
A publisher loans samples, fills an external `BufferPool`, exports a
`BufferHandle` into `FramePayload`, and sends; a subscriber blocks on
`Subscriber::wait_and_take()`, imports the buffer with `Context::import_buffer()`,
and verifies strict sequence `0..N-1`.

Iceoryx2 service name: `holomq_main_v4` (bump when `FramePayload` or
`BufferHandle` layout changes).

HoloMQ library diagnostics use `<holomq/logger.hpp>`; this example keeps its
own stderr helpers for application messages.

## Prerequisites

- NVIDIA GPU with a working CUDA driver (`libcuda.so.1`)
- HoloMQ built with iceoryx2
- Shared memory suitable for iceoryx2 (`/dev/shm`; SDK container workflows use
  `--ipc=host`)

## Build

From `public`:

```bash
./run build --config "-DHOLOMQ_BUILD_EXAMPLES=ON"
```

Inside an already configured SDK build container:

```bash
./run launch --run-cmd "cmake --build . --target holomq_main"
```

Binary path inside the default SDK build tree:

`modules/holomq/examples/holomq_main/holomq_main`

The example uses `holomq_main::DemoCudaStream` (see `cuda_raii.hpp`), which synchronizes
the stream in its destructor before `cuStreamDestroy` so pending HoloMQ deferred ref
releases complete safely.

## Run the demo

Use **two processes** with the same `--samples` count. Start the **subscriber
first** (the publisher waits up to five seconds for a connected subscriber).
The default `--samples` matches `holomq::k_max_in_flight` (currently 8). For
larger counts without pacing, add `--publish-pace-ms` so the publisher does not
outrun the external `BufferPool`.

Host or container shell (with GPU visible):

```bash
BIN=modules/holomq/examples/holomq_main/holomq_main

# Terminal 1
${BIN} --role subscriber

# Terminal 2
${BIN} --role publisher
```

Single-shell smoke test (subscriber in background; keep `--samples` at or below
`holomq::k_max_in_flight`, currently 8, unless using `--publish-pace-ms`):

```bash
BIN=modules/holomq/examples/holomq_main/holomq_main
${BIN} --role subscriber --samples 3 &
sleep 1
${BIN} --role publisher --samples 3
wait
```

From the SDK root:

```bash
cd public
./run launch --run-cmd '
  BIN=./modules/holomq/examples/holomq_main/holomq_main
  ${BIN} --role subscriber --samples 3 &
  sleep 1
  ${BIN} --role publisher --samples 3
  wait
'
```

## Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--role` | `publisher` | `publisher` or `subscriber` |
| `--samples` | `8` (`k_max_in_flight`) | Number of samples to send/receive (`0..N-1`) |
| `--fd-pass-socket` | *(empty)* | Publisher: AF_UNIX listen path for FD-pass; empty picks a unique `/tmp` path |
| `--publisher-hold-ms` | `2000` | Publisher: sleep after all sends before exit |
| `--publish-pace-ms` | `0` | Publisher: optional delay between sends (`0` = no pacing) |
| `--pool-type` | `device` | `device` or `host` (`host-pinned` accepted as alias) |
| `--host-register-flags` | `devicemap` | Host pools only: `pageable`, `portable`, `devicemap`, `portable-devicemap` |

Run `${BIN} --help` for the full list.

### Pool matrix (publisher)

| `--pool-type` | `--host-register-flags` | Pool factory |
|---------------|-------------------------|--------------|
| `device` | *(ignored)* | `Context::create_device_buffer_pool()` |
| `host` | `pageable` | `Context::create_host_buffer_pool()` |
| `host` | `portable` | `Context::create_host_pinned_buffer_pool()` (portable) |
| `host` | `devicemap` | `Context::create_host_pinned_buffer_pool()` (devicemap) |
| `host` | `portable-devicemap` | `Context::create_host_pinned_buffer_pool()` (both flags) |

The subscriber imports buffers using `BufferHandle::memory_type()` from each
received payload; no pool-type flag is required on the subscriber side.

Example host pool publisher:

```bash
${BIN} --role publisher --samples 5 --pool-type host --host-register-flags devicemap
```

## What this example exercises

- `ContextBuilder` with optional VMM FD-pass for device and split-generation host pools
- External `BufferPool` sized to `holomq::k_max_in_flight` for this bounded demo workload. General
  publishers must also budget distinct history, unread, and borrowed backing retained after a
  producer loan ends (see `<holomq/limits.hpp>` and `docs/PUBLIC_API.md`).
- `Publisher::loan()`, `send()` with a release callback keeping owner `Buffer`s alive
- `Subscriber::wait_and_take()` with `WaitMode::KernelWait`
- `Buffer::export_handle()` / `Context::import_buffer()` across processes
- `FramePayload` metadata plus embedded `BufferHandle` (see `frame_payload.hpp`)

## Files

| File | Role |
|------|------|
| `main.cpp` | Publisher/subscriber loops and CLI |
| `frame_payload.hpp` | Demo message type in `holomq_main` (`sequence`, frame size, `holomq::BufferHandle`) |
| `../common/holoargs.hpp` | Example CLI parsing in `holomq_examples::args` |
| `CMakeLists.txt` | Builds the `holomq_main` executable |

Further API detail: [`../../docs/PUBLIC_API.md`](../../docs/PUBLIC_API.md).
