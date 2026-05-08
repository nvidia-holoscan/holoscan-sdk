# IDL types for the fastdds_example app

DDS types defined here (e.g. **CudaBuffer**) are generated **at CMake build time** into **`${CMAKE_CURRENT_BINARY_DIR}/gen/`** for the `fastdds_example` target (not under the source tree).

## IDL files

| File | Generated types | Purpose |
|------|-----------------|---------|
| `CudaBuffer.idl` | `CudaBuffer`, `CudaBufferPubSubType` | Composite message: PointerDescriptor + size. |

`CudaBuffer.idl` uses `#include "pointer_descriptor.idl"`, which CMake resolves by passing **`-I`** to the SDK IDL directory (`public/modules/holoipc/idl/`). Generated C++ references SDK types under `#include <holoscan/ipc/transport/fastdds/gen/...>` after **`postprocess_example_fastdds_idl.py`** (invoked by `cmake_regen_fastdds_idl.py flat` from `fastdds_example/CMakeLists.txt`).

## Regenerating C++ from IDL

**Normal workflow:** change `CudaBuffer.idl`, then rebuild `fastdds_example` (same **Fast-DDS-Gen** / **`fastddsgen`** requirement as `holo_ipc`; see [IDL_TYPES.md](../../../docs/IDL_TYPES.md)).

**Manual preview** (optional): from the SDK repo root, use the same driver CMake uses:

```bash
OUT=/tmp/fastdds_example_idl_gen
mkdir -p "$OUT"
python3 public/modules/holoipc/src/transport/fastdds/scripts/cmake_regen_fastdds_idl.py flat \
  --fastddsgen "$(command -v fastddsgen)" \
  --idl-file "$(pwd)/public/modules/holoipc/examples/fastdds_example/idl/CudaBuffer.idl" \
  --ipc-idl-include "$(pwd)/public/modules/holoipc/idl" \
  --out-dir "$OUT" \
  --postprocess "$(pwd)/public/modules/holoipc/examples/postprocess_example_fastdds_idl.py"
```

Raw **fastddsgen** without that pipeline leaves `holoscan::ipc::PointerDescriptor` includes; the postprocess step rewrites them to the SDK gen layout under `holoscan::ipc::transport::fastdds::gen`.
