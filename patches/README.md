# Patches used by the Holoscan SDK

This folder contains patches that are either applied during the build process of Holoscan SDK
or have been used to build artifacts used by the SDK

- `libtorch`:
  - Inline ([Dockerfile](../Dockerfile)): Remove unused `kineto` references to silence warning
  - `libtorch.Caffe2.cmake.patch`: Patches `Caffe2/public/cuda.cmake` to address configuration warning ([GitHub Issue#129777](https://github.com/pytorch/pytorch/issues/129777)):
```
CMake Warning at /opt/libtorch/2.5.0_24.08/share/cmake/Caffe2/public/cuda.cmake:143 (message):
  Failed to compute shorthash for libnvrtc.so
```
