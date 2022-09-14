# Clara Holoviz SDK

## Prerequisites

- Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

## Building

Clara Holoviz is build with a docker container.

All build operations are executed using the 'run.sh' batch file, use

```shell
$ ./run.sh -h
```

to see the options. The script supports

-   creating the docker image needed for building Clara Holoviz
-   building Clara Holoviz

The docker image _clara-holoviz-dev_ARCH:?.?.?_ (where `ARCH` is `x86_64`
or `aarch64` and `?.?.?` is the version) is used to build the Clara Holoviz
executable.

First the _dev_ docker image needs to be build by, this need to be done only once or when the dev docker image changes.

```shell
$ ./run.sh image_build_dev
```

When this is done the Clara Holoviz executable can be build.

```shell
$ ./run.sh build
```

The build results are written to the `build_x86_64` or `build_aarch64` directory, dependent on the target architecture.

The final binary package is written to the `package` directory in the build directory. It includes the Holoviz shared library, header files, CMake setup files to include Holoviz in CMake based projects, a static library of ImGUI and example source files.

To build the examples, go to the `package/src/examples/xyz` directory (where `xyx` is the name of the example) and execute these steps:

```shell
$ cmake -B build .
$ cmake --build build/
```

### Cross-compilation: x86_64 to aarch64

While the development dockerfile does not currently support true cross-compilation,
you can compile for Holoviz for aarch64 from a x86_64 host using an emulation environment.

Run the following steps on your host.

Follow the [installation steps for `qemu`](https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#emulation-environment)

Set the `TARGET_ARCH` environment variable:

```shell
$ export TARGET_ARCH=aarch64
```

Follow the build steps above by first building the docker _dev_ image and then building Clara Holoviz.
