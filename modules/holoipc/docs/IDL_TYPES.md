# IDL types (Holoscan IPC, Fast DDS)

The DDS types used by the IPC library are defined in `../idl/` and the C++ code is generated into transport-specific `gen/` trees. Generated headers under `include/holoscan/ipc/transport/fastdds/gen/` are part of the public API: use angle-bracket includes with the full path (e.g. `#include <holoscan/ipc/transport/fastdds/gen/pointer_descriptor.hpp>`). For the example apps and build options, see [fastdds_example README](../examples/fastdds_example/README.md#ipc-and-build-options).

## IDL files

| File | Generated types | Purpose |
|------|-----------------|---------|
| `pointer_descriptor.idl` | `PointerDescriptor`, `PointerDescriptorPubSubType`, … | Wire format for pointer metadata (`version`, `key`, `handle_type`, opaque `handle`, `reply_to_topic_name` — publisher inbox for control messages from subscribers). |
| `control_message.idl` | `ControlMessage`, `ControlMessageType`, `ControlMessagePubSubType`, … | Lifecycle control on the control channel: **ACQUIRED**, **RELEASED**, **ACK**, **NACK** (see [PROTOCOL_SPEC.md](PROTOCOL_SPEC.md)). |

## Regenerating C++ from IDL

**Default:** C++ is generated **at CMake build time** into **`${PROJECT_BINARY_DIR}/generated/`** (not checked into git):

- **Public headers (installed):** `generated/include/holoscan/ipc/transport/fastdds/gen/*.hpp`
- **Private implementation:** `generated/fastdds_private/` (`*CdrAux*`, `ControlMessageTypeObjectSupport.hpp`, `*.cxx`, …)

CMake runs `cmake_regen_fastdds_idl.py ipc-transport` (raw `fastddsgen` + `postprocess_idl_gen.py`). The same driver also supports a `flat` subcommand for example IDLs. See `src/transport/fastdds/fast_dds_idl_gen.cmake` and `src/transport/fastdds/CMakeLists.txt`.

**Tooling:** **Fast-DDS-Gen** ([eProsima/Fast-DDS-Gen](https://github.com/eProsima/Fast-DDS-Gen)) — `fastddsgen` on `PATH`, or **`FAST_DDS_GEN`** set to the install root (see `public/Dockerfile`). **Python 3** is required for the CMake driver and post-process step. The types in this repo target the **Fast DDS 3** API; distro `fastddsgen` packages may be older.

**Preview / diff only:** run the same driver as CMake into a throwaway directory (see [Fast DDS `scripts/README.md`](../src/transport/fastdds/scripts/README.md) “Preview / diff”).

The `holo_ipc` target compiles generated `.cxx` via the `ipc_fastdds_gen_objs` object library.

**Holoscan SDK Docker image (`public/Dockerfile`):** the `build` stage **copies** the Fast-DDS-Gen tree from the **`fastdds-gen-builder`** stage (where the jar is built with **`openjdk-17-jdk-headless`**) into `/opt/fastdds-gen/${FASTDDS_GEN_VERSION}` (default `4.3.0`), sets `FAST_DDS_GEN` to that directory, adds `fastddsgen` to `PATH`, and **`apt-get install`s `openjdk-17-jre-headless`** so `java -jar` can run. Override at image build with `--build-arg FASTDDS_GEN_VERSION=...` (must match a `v${FASTDDS_GEN_VERSION}` tag on [Fast-DDS-Gen](https://github.com/eProsima/Fast-DDS-Gen)). Example:

```bash
command -v fastddsgen
ls "${FAST_DDS_GEN}/fastddsgen.jar"
```

### After changing IDL

1. Rebuild `holo_ipc` (CMake re-runs codegen when `.idl` inputs change).
2. Optionally run the preview command in `scripts/README.md` to emit outputs under a temp directory.
3. If you rename or remove an IDL file, update `src/transport/fastdds/CMakeLists.txt` (`_ipc_fd_gen_outputs` / `add_custom_command`) and any `#include` paths.
