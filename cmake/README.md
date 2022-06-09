# CMake support

### Test CMakeLists.txt

The CMakeLists.txt in the `test` folder is meant to demonstrate the use of:
- `find_package(GXF ...)`: search and import GXF targets given a root directory that contains its artifacts.
- `create_gxe_application`: generates a manifest file that lists the path to the necessary extensions, and a run script (bash) that runs GXE with the adequate arguments and environment variables.

Call the following command to configure and build it:
```bash
cmake \
  -S gxf_extensions/cmake/test \
  -B ${build_dir} \
  -D GFX_DIR:PATH="/path/to/gxf-release/" \
  -D yaml-cpp_DIR:PATH="/path/to/yaml-cpp-install/" \
  -D CUDAToolkit_ROOT:PATH="/path/to/cuda-install/"
cmake -build ${build_dir} -j
```

### GXF Extensions in Holoscan SDK

These extensions use the same mechanism as the test CMakeLists.txt above, but also require additional dependencies. These dependencies need to be installed ahead of the CMake configuration, and their root path passed to the CMake configuration command. Use the `${dep}_DIR` format to satisfy the requirements set by each extension (leveraging [`define_search_dir_for.cmake`](./define_search_dir_for.cmake))

```cmake
cmake \
  -S gxf_extensions/${holoscan_gxf_extension} \
  -B build \
  -D ${dep1}_DIR:PATH="/path/to/${dep1_install_dir}" \
  -D ${dep2}_DIR:PATH="/path/to/${dep2_install_dir}" \
  ...
```
