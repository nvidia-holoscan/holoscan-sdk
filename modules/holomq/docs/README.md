# HoloMQ documentation

## Application developers

| Document | Description |
|----------|-------------|
| [`PUBLIC_API.md`](PUBLIC_API.md) | Application-facing C++ API (draft design review) |
| [`presentations/`](presentations/) | Slide-style API tour and sync deep dive |

HoloMQ builds with the iceoryx2 transport (`HOLOMQ_TRANSPORT=iox2`). The Holoscan SDK
maps `HOLOSCAN_MQ_BACKEND` / `HOLOMQ_DEFAULT_BACKEND` (`iceoryx2`) to that cache when
`HOLOMQ_TRANSPORT` is not set explicitly.

## Contributors

| Document | Description |
|----------|-------------|
| [`design-notes/README.md`](design-notes/README.md) | Design notes (buffer pools, sync, transport, FD-pass) |
| [`CONTRIBUTOR_READING_ORDER.md`](CONTRIBUTOR_READING_ORDER.md) | Dependency order for navigating and reviewing the codebase |
| [`abi_and_exports.md`](abi_and_exports.md) | Header guards, visibility, and DSO export policy |
