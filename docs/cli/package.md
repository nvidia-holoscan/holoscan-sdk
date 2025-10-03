(holoscan-cli-package)=

# Holoscan CLI - Package Command

`holoscan package` - generate [HAP-compliant](./hap.md) container for your application.

## Synopsis

`holoscan package` [](#cli-help) [](#cli-log-level) [](#cli-package-add) [](#cli-package-config) [](#cli-package-cuda) [](#cli-package-docs) [](#cli-package-models) [](#cli-package-platform) [](#cli-package-timeout) [](#cli-package-version) [](#cli-package-add-host) [](#cli-package-base-image) [](#cli-package-build-image) [](#cli-package-build-cache) [](#cli-package-cmake-args) [](#cli-package-holoscan-sdk-file) [](#cli-package-includes) [](#cli-package-input-data) [](#cli-package-monai-deploy-sdk-file) [](#cli-package-no-cache) [](#cli-package-sdk) [](#cli-package-sdk-version) [](#cli-package-source) [](#cli-package-output) [](#cli-package-tag) [](#cli-package-username) [](#cli-package-uid) [](#cli-package-gid) [](#cli-package-application)


## Examples

The code below package a python application for x86_64 systems:

```bash
# Using a Python directory as input
# Required: a `__main__.py` file in the application directory to execute
# Optional: a `requirements.txt` file in the application directory to install dependencies
holoscan package --platform x86_64 --tag my-awesome-app --config /path/to/my/awesome/application/config.yaml /path/to/my/awesome/application/

# Using a Python file as input
holoscan package --platform x86_64 --tag my-awesome-app --config /path/to/my/awesome/application/config.yaml /path/to/my/awesome/application/my-app.py
```

The code below package a C++ application for the IGX Orin DevKit (aarch64) with a discrete GPU:

```bash
# Using a C++ source directory as input
# Required: a `CMakeLists.txt` file in the application directory
holoscan package --platform igx-dgpu --tag my-awesome-app --config /path/to/my/awesome/application/config.yaml /path/to/my/awesome/application/

# Using a C++ pre-compiled executable as input
holoscan package --platform igx-dgpu --tag my-awesome-app --config /path/to/my/awesome/application/config.yaml /path/to/my/awesome/bin/application-executable
```

:::{note}
The commands above load the generated image onto Docker to make the image accessible with `docker images`.

If you need to package for a different platform or want to transfer the generated image to another system, use the `--output /path/to/output` flag so the generated package can be saved to the specified location.
:::

## Positional Arguments

(cli-package-application)=

#### `application`

Path to the application to be packaged. The following inputs are supported:

- **C++ source code**: you may pass a directory path with your C++ source code with a `CMakeLists.txt` file in it, and the **Packager** will attempt to build your application using CMake and include the compiled application in the final package.
- **C++ pre-compiled executable**: A pre-built executable binary file may be directly provided to the **Packager**.
- **Python application**: you may pass either:
  - a directory which includes a `__main__.py` file to execute (required) and an optional `requirements.txt` file that defined dependencies for your Python application, or
  - the path to a single python file to execute

:::{warning}
Python (PyPI) modules are installed into the user's (via [](#cli-package-username) argument) directory with the user ID specified via [](#cli-package-uid).
Therefore, when running a packaged Holoscan application on Kubernetes or other service providers, running Docker with non root user, and running Holoscan CLI `run` command where the logged-on user's ID is different, ensure to specify the `USER ID` that is used when building the application package.


For example, include the `securityContext` when running a Holoscan packaged application with `UID=1000` using Argo:

```yaml
spec:
  securityContext:
    runAsUser: 1000
    runAsNonRoot: true
```
:::

## Flags

### Options

(#cli-package-add)=

#### `[--add DIR_PATH]`

`--add` enables additional files to be added to the application package. Use this option to include additional Python modules, files, or static objects (.so) on which the application depends.

- `DIR_PATH` must be a directory path. The packager recursively copies all the files and directories inside `DIR_PATH` to `/opt/holoscan/app/lib`.
- `--add` may be specified multiple times.

For example:
```bash
holoscan package --add /path/to/python/module-1 --add /path/to/static-objects
```

With the example above, assuming the directories contain the following:

```bash
/path/to/
├── python
│   ├── module-1
│   │   ├── __init__.py
│   │   └── main.py
└── static-objects
    ├── my-lib.so
    └── my-other-lib.so
```

The resulting package will contain the following:

```bash
/opt/holoscan/
├── app
│   └── my-app
└──lib/
    ├── module-1
    │   ├── __init__.py
    │   └── main.py
    ├── my-lib.so
    └── my-other-lib.so

```
(#cli-package-config)=

#### `--config|-c CONFIG`

Path to the application's [configuration file](./run_config.md). The configuration file must be in `YAML` format with a `.yaml` file extension.

(#cli-package-docs)=

#### `[--docs|-d DOCS]`

An optional directory path of documentation, README, licenses that shall be included in the package.


(#cli-package-models)=

#### `[--models|-m MODELS]`

An optional directory path to a model file, a directory with a single model, or a directory with multiple models.

Single model example:

```bash
my-model/
├── surgical_video.gxf_entities
└── surgical_video.gxf_index

my-model/
└── model
    ├── surgical_video.gxf_entities
    └── surgical_video.gxf_index
```

Multi-model example:

```bash
my-models/
├── model-1
│   ├── my-first-model.gxf_entities
│   └── my-first-model.gxf_index
└── model-2
    └── my-other-model.ts
```
(#cli-package-cuda)=

#### `--cuda CUDA_VERSION`

The CUDA version used to build the application is specified. If not specified, CUDA `13` would be used.


`CUDA_VERSION` must be one of: `12`, `13`. 

(#cli-package-platform)=

#### `--platform PLATFORM`

A comma-separated list of platform types to generate. Each platform value specified generates a standalone container image. If you are running the **Packager** on the same architecture, the generated image is automatically loaded onto Docker and is available with `docker images`. Otherwise, use `--output` flag to save the generated image onto the disk.

`PLATFORM` must be one of: `jetson`, `igx-igpu`, `igx-dgpu`, `sbsa`, `x86_64`.

- `jetson`: Orin AGX DevKit
- `igx-igpu`: IGX Orin DevKit with integrated GPU (iGPU)
- `igx-dgpu`: IGX Orin DevKit with dedicated GPU (dGPU)
- `sbsa`: Server Base System Architecture (64-bit ARM processors)
- `x86_64`: systems with a [x86-64](https://en.wikipedia.org/wiki/X86-64) processor(s)

(#cli-package-timeout)=

#### `[--timeout TIMEOUT]`

An optional timeout value of the application for the supported orchestrators to manage the application's lifecycle.
Defaults to `0`.

(#cli-package-version)=

#### `[--version VERSION]`

An optional version number of the application. When specified, it overrides the value specified in the [configuration file](./run_config.md).


### Advanced Build Options

(#cli-package-add-host)=

#### `[--add-host ADD_HOSTS]`

Optionally add one or more host-to-IP mapping (format: host:ip).

(#cli-package-base-image)=

#### `[--base-image BASE_IMAGE]`

Optionally specifies the base container image for building packaged application. It must be a valid Docker image tag either accessible online or via `docker images. By default, the **Packager** picks a base image to use from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan).

(#cli-package-build-image)=

#### `[--build-image BUILD_IMAGE]`

Optionally specifies the build container image for building C++ applications. It must be a valid Docker image tag either accessible online or via `docker images. By default, the **Packager** picks a build image to use from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan).


(#cli-package-build-cache)=

#### `[--build-cache BUILD_CACHE]`

Specifies a directory path for storing Docker cache. Defaults to `~/.holoscan_build_cache`. If the `$HOME` directory is inaccessible, the CLI uses the `/tmp` directory.

(#cli-package-cmake-args)=

#### `[--cmake-args CMAKE_ARGS]`

A comma-separated list of *cmake* arguments to be used when building C++ applications.

For example:

```bash
holoscan package --cmake-args "-DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_ARG=VALUE"
```

(#cli-package-holoscan-sdk-file)=

#### `[--holoscan-sdk-file HOLOSCAN_SDK_FILE]`

Path to the Holoscan SDK Debian or PyPI package. If not specified, the packager downloads the SDK file from the internet depending on the SDK version detected/specified. The `HOLOSCAN_SDK_FILE` filename must have `.deb` or `.whl` file extension for Debian package or PyPI wheel package, respectively.


(#cli-package-includes)=

#### `[--includes  [{debug,holoviz,torch,onnx}]]`

To reduce the size of the packaged application container, the CLI Packager, by default, includes minimum runtime dependencies to run applications designed for Holoscan. You can specify additional runtime dependencies to be included in the packaged application using this option. The following options are available:

- `debug`: includes debugging tools, such as `gdb`
- `holoviz`: includes dependencies for Holoviz rendering on x11 and Wayland
- `torch`: includes `libtorch` runtime dependencies
- `onnx`: includes `onnxruntime` runtime, `libnvinfer-plugin8`, `libnvonnxparser8` dependencies.

:::{note}
Refer to [Developer Resources](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/DEVELOP.md#advanced-local-environment--cmake) for dependency versions.
:::



Usage:
```bash
holoscan package --includes holoviz torch onnx
```


(#cli-package-input-data)=

#### `[--input-data INPUT_DATA]`

Optionally, embed input data in the package. `INPUT_DATA` must be a valid path to a directory containing the data to be embedded in `/var/holoscan/input`, and it can be accessed inside the container via the `HOLOSCAN_INPUT_PATH` environment variable.

(#cli-package-monai-deploy-sdk-file)=

#### `[--monai-deploy-sdk-file MONAI_DEPLOY_SDK_FILE]`

Path to the MONAI Deploy App SDK Debian or PyPI package. If not specified, the packager downloads the SDK file from the internet based on the SDK version. The `MONAI_DEPLOY_SDK_FILE` package filename must have `.whl` or `.gz` file extension.


(#cli-package-no-cache)=

#### `[--no-cache|-n]`

Do not use cache when building image.

(#cli-package-sdk)=

#### `[--sdk SDK]`

SDK for building the application: Holoscan or MONAI-Deploy. `SDK` must be one of: holoscan, monai-deploy.


(#cli-package-sdk-version)=

#### `[--sdk-version SDK_VERSION]`

Set the version of the SDK that is used to build and package the Application. If not specified, the packager attempts to detect the installed version.

(#cli-package-source)=

#### `[--source URL|DIR|FILE]`

Override the artifact manifest source with a securely hosted file or from the local file system.

If the value is a valid directory, the Package scans it for `artifacts.json` or `artifacts-cu12.json` if `--cuda` is 12 or 13, respectively.

For example:

- *URL*: https://my.domain.com/my-file.json
- *DIR*: /home/me/path/to/artifact-json-files/
- *FILE*: /home/me/path/to/artifact.json


### Output Options

(#cli-package-output)=

#### `[--output|-o OUTPUT]`

Output directory where result images will be written.

:::{note}
If this flag isn't present, the packager will load the generated image onto Docker to make the image accessible with `docker images`. The `--output` flag is therefore required when building a packaging for a different target architecture than the host system that runs the packaer.
:::

(#cli-package-tag)=

#### `--tag|-t TAG`

Name and optionally a tag (format: `name:tag`).

For example:

```bash
my-company/my-application:latest
my-company/my-application:1.0.0
my-application:1.0.1
my-application
```

### Security Options

(#cli-package-username)=

#### `[--username USERNAME]`

Optional *username* to be created in the container execution context. Defaults to `holoscan`.

(#cli-package-uid)=

#### `[--uid UID]`

Optional *user ID* to be associated with the user created with `--username` with default of `1000`.

:::{warning}
A very large UID value may result in a very large image due to an open [issue](https://github.com/docker/hub-feedback/issues/2263) with Docker.
It is recommended to use the default value of `1000` when packaging an application and use your current UID/GID when running the application.
:::

(#cli-package-gid)=

#### `[--gid GID]`

Optional *group ID* to be associated with the user created with `--username` with default of `1000`
