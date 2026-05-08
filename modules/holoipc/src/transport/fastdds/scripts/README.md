# Fast DDS transport — IDL codegen helpers

## Build-time generation (default)

`holo_ipc` generates C++ from `public/modules/holoipc/idl/{pointer_descriptor,control_message}.idl`
during the CMake build. Outputs go under **`${PROJECT_BINARY_DIR}/generated/`** (not the source tree):

- Public API headers: `generated/include/holoscan/ipc/transport/fastdds/gen/*.hpp`
- Private sources: `generated/fastdds_private/*` (compiled into `holo_ipc` only)

Requirements: **`fastddsgen`** on `PATH` (or `FAST_DDS_GEN` pointing at the Fast-DDS-Gen install root, as in `public/Dockerfile`), plus **Python 3** (for `cmake_regen_fastdds_idl.py ipc-transport` and `postprocess_idl_gen.py`).

Then configure and build as usual, e.g.:

```bash
cmake --build <build-dir> --target holo_ipc -j"$(nproc)"
```

CMake details: `fast_dds_idl_gen.cmake` and `CMakeLists.txt` in `src/transport/fastdds/`.

## Preview / diff (optional)

From the Holoscan SDK repo root, with `fastddsgen` on `PATH`:

```bash
OUT_BASE=$(mktemp -d "${TMPDIR:-/tmp}/holoscan_ipc_idl_preview.XXXXXX")
SCRIPT_DIR=public/modules/holoipc/src/transport/fastdds/scripts
IDL_DIR=public/modules/holoipc/idl
PUB="${OUT_BASE}/include/holoscan/ipc/transport/fastdds/gen"
PRIV="${OUT_BASE}/private"
mkdir -p "${PUB}" "${PRIV}"
python3 "${SCRIPT_DIR}/cmake_regen_fastdds_idl.py" ipc-transport \
  --fastddsgen "$(command -v fastddsgen)" \
  --idl-dir "${IDL_DIR}" \
  --idl pointer_descriptor.idl control_message.idl \
  --out-public-gen "${PUB}" \
  --out-private "${PRIV}" \
  --postprocess "${SCRIPT_DIR}/postprocess_idl_gen.py"
echo "Preview: ${OUT_BASE}"
```

## Manual raw fastddsgen (advanced)

From `public/modules/holoipc/idl/`:

```bash
OUT=/tmp/ipc_fastdds_raw
mkdir -p "$OUT"
fastddsgen -replace -d "$OUT" pointer_descriptor.idl -no-dependencies
fastddsgen -replace -d "$OUT" control_message.idl -no-dependencies
```

Then route and post-process with `cmake_regen_fastdds_idl.py ipc-transport` (same routing rules as CMake) or copy by hand per the routing rules in that script.

## Post-process behavior

`postprocess_idl_gen.py` wraps types in `holoscan::ipc::transport::fastdds::gen` and adjusts C++ qualifiers; quoted DDS type names stay `holoscan::ipc::*`. Re-running on already-processed files is safe.
