# IDL types for the fastdds_benchmark app

DDS types defined in **`Buffer.idl`** are generated **at CMake build time** into **`${CMAKE_CURRENT_BINARY_DIR}/gen/`** for the `fastdds_benchmark` target (not under the source tree). One IDL file may define **multiple structs**; fastddsgen emits a single set of `Buffer*` sources (`Buffer.hpp` holds all C++ structs from that file, e.g. `AccelBuffer` and `Buffer`).

## IDL files

| File | Generated types | Purpose |
|------|-----------------|---------|
| `Buffer.idl` | `AccelBuffer`, `Buffer`, `AccelBufferPubSubType`, `BufferPubSubType` | **AccelBuffer:** `PointerDescriptor` + `size` + `index`. **Buffer:** `sequence<octet> data` + `index` (host-side bytes; index reserved for future use). |

`Buffer.idl` uses `#include "pointer_descriptor.idl"`, which CMake resolves by passing **`-I`** to the SDK IDL directory (`public/modules/holoipc/idl/`). Generated C++ references SDK types under `#include <holoscan/ipc/transport/fastdds/gen/...>` after **`postprocess_example_fastdds_idl.py`** (invoked by `cmake_regen_fastdds_idl.py flat` from `fastdds_benchmark/CMakeLists.txt`).

## Regenerating C++ from IDL

**Normal workflow:** change `Buffer.idl`, then rebuild `fastdds_benchmark` (same **Fast-DDS-Gen** / **`fastddsgen`** requirement as `holo_ipc`; see [IDL_TYPES.md](../../../docs/IDL_TYPES.md)).

**Manual preview** (optional):

```bash
OUT=/tmp/fastdds_benchmark_idl_gen
mkdir -p "$OUT"
python3 public/modules/holoipc/src/transport/fastdds/scripts/cmake_regen_fastdds_idl.py flat \
  --fastddsgen "$(command -v fastddsgen)" \
  --idl-file "$(pwd)/public/modules/holoipc/examples/fastdds_benchmark/idl/Buffer.idl" \
  --ipc-idl-include "$(pwd)/public/modules/holoipc/idl" \
  --out-dir "$OUT" \
  --postprocess "$(pwd)/public/modules/holoipc/examples/postprocess_example_fastdds_idl.py"
```

Raw **fastddsgen** without that pipeline leaves `holoscan::ipc::PointerDescriptor` includes; the postprocess step rewrites them to the SDK gen layout under `holoscan::ipc::transport::fastdds::gen`.
