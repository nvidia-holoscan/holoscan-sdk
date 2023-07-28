# CLI Packager

This example demonstrates how to package a Holoscan Application (in this example, [hello_world](../hello_world/)) into a [HAP-compliant](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/hap.html) container using the Holoscan CLI.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/ping_multi_port.html) to learn more about the Holoscan Packager.*

## Run Instructions

### Setup the Holoscan CLI

Refer to the documentation in the [user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html).

### Define configurations

The packager will require a `--platform` and a `--platform-config`. We set them here as a prerequisite for this example. Refer to the user guide for other possible configurations.

```bash
export gpu_mode=dgpu
export platform=x64-workstation
```

The packager will also need the path to the configuration file passed to `--config`. We point to the one in this folder:

```bash
# Note: This is the install path for debian and ngc container.
# You could point to a cloned repository or a build/install directory instead
export holoscan_dir=/opt/nvidia/holoscan
export holoscan_app_config_path=$holoscan_dir/examples/cli_packager/app.yaml
```

We then define the path to the application to package:

```bash
# To build and package C++ executable
export holoscan_app_path=$holoscan_dir/examples/hello_world/cpp
## Note: only run this if `holoscan_dir` is the holoscan source directory and not build or install directory
mv $holoscan_app_path/CMakeLists.min.txt $holoscan_app_path/CMakeLists.txt

# To package prebuilt C++ executable
## Note: needs executable to exist, won't work from the source directory
export holoscan_app_path=$holoscan_dir/examples/hello_world/cpp/hello_world

# To package python app
## Note: could omit `hello_world.py` if there was a __main__.py file in the directory
export holoscan_app_path=$holoscan_dir/examples/hello_world/python/hello_world.py
```

### Run the packager

This command will create a docker container that includes the application:

```bash
holoscan package -t holoscan-hello-world-app \
  --platform $platform \
  --platform-config $gpu_mode \
  --config $holoscan_app_config_path \
  $holoscan_app_path
```

### Run the containerized application

Given the configurations listed in the instructions above, that would be:

```bash
holoscan run holoscan-hello-world-app-x64-workstation-dgpu-linux-amd64:<version-of-image>
```
