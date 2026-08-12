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

Run directly from the repository root after building Python bindings:

```sh
PYTHONPATH=build-cu13-x86_64/python/lib \
  python3 benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py \
  --device auto --shape 1024 --iterations 10000 --repeats 7
```

From a build container:

```sh
./run launch build-cu13-x86_64 --run-cmd \
  "python3 /workspace/holoscan-sdk/benchmark/tensor_python_as_tensor/tensor_as_tensor_fastpath_benchmark.py"
```
