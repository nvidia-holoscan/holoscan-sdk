# Tensor Python Protocol Benchmark

This pure-Python benchmark compares `holoscan.core.Tensor.as_tensor()` import
and `holoscan.core.Tensor` export across the supported zero-copy Python
protocols.

For import, the DLPack C exchange fast path is measured by passing a
`torch.Tensor` directly. The PyTorch Python DLPack fallback is measured with a
small proxy object that exposes only `__dlpack__` and `__dlpack_device__`,
hiding the type-level `__dlpack_c_exchange_api__` capsule. CuPy DLPack import
is measured separately because CuPy does not currently expose the C exchange
API and its Python DLPack path has different cost from PyTorch. Host
array-interface import is measured with a NumPy-backed proxy that exposes only
NumPy `__array_interface__`. Device array-interface import is measured with
separate PyTorch- and CuPy-backed proxies that expose only
`__cuda_array_interface__`. PyTorch CPU tensors do not expose
`__array_interface__`, so there is no PyTorch host array-interface row.

For export, the benchmark measures Holoscan Tensor conversion through the
DLPack C exchange test helper, which validates the C exchange table and
round-trips the exported tensor, Python `__dlpack__` consumed by PyTorch, host
`__array_interface__` consumed by NumPy, and device `__cuda_array_interface__`
consumed by CuPy.

The benchmark also includes stable-ABI evaluation rows: repeated
`Tensor.shape` and `Tensor.strides` property access, versioned `Tensor.__dlpack__`
capsule creation, and `Tensor.as_tensor()` round-trip from another Holoscan
Tensor. Use a multi-dimensional shape when comparing sequence builders so the
measurement includes more than one tuple entry.

Run directly from the SDK source root after building Python bindings.
Use the same `HOLOSCAN_BUILD_ARCH` setting as the build when selecting a target explicitly:

```bash
BUILD_DIR=$(./run get_build_dir)
PYTHONPATH="${BUILD_DIR}/python/lib" \
  python3 benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py \
  --device auto --shape 2x2x2x2x2x2x2x2x2x2x2x2 \
  --iterations 50000 --warmup 5000 --repeats 11 \
  --configuration nanobind-3.0.1-abi3 \
  --benchmark_out=nanobind-3.0.1-abi3.json --benchmark_out_format=json
```

To run in the build container, invoke `launch` from the same host source root.
It selects the build directory and sets the Python module search path:

```bash
./run launch --run-cmd \
  "python3 ../benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py"
```

For the measurement-only nanobind split-mode prototype, install the official
`nanobind-backend` package into the benchmark environment and configure with:

```sh
cmake -S . -B build-python-split \
  -DHOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE=ON
cmake --build build-python-split --target holoscan-python
```

The prototype option is off by default. Production builds continue to use the
linked `STABLE_ABI` module unless recorded performance evidence changes that
decision. This manual path was last built and imported on x86_64 during the
2026-09-01 S0 evaluation; CI does not exercise it.
