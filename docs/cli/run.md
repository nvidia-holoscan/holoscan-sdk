(holoscan-cli-run)=

# Holoscan CLI - Run Command

`holoscan run` - simplifies running a packaged Holoscan application by reducing the number of arguments required compared to `docker run`. In addition, it follows the guidelines of [HAP specification](./hap.md) when launching your packaged Holoscan application.


:::{warning}
When running a packaged Holoscan application on Kubernetes or other service providers, running Docker with non root user, and running Holoscan CLI `run` command where the logged-on user's ID is different, ensure to specify the `USER ID` that is used when building the application package.


For example, include the `securityContext` when running a Holoscan packaged application with `UID=1000` using Argo:

```yaml
spec:
  securityContext:
    runAsUser: 1000
    runAsNonRoot: true
```
:::

## Synopsis

`holoscan run` [](#cli-help) [](#cli-log-level) [](#cli-run-address) [](#cli-run-driver) [](#cli-run-input) [](#cli-run-output) [](#cli-run-fragments) [](#cli-run-worker) [](#cli-run-worker-address) [](#cli-run-config) [](#cli-run-health-check) [](#cli-run-network) [](#cli-run-nic) [](#cli-run-use-all-nics) [](#cli-run-render) [](#cli-run-quiet) [](#cli-run-shm-size)[](#cli-run-terminal) [](#cli-run-device) [](#cli-run-gpu) [](#cli-run-uid) [](#cli-run-gid)[](#cli-run-image-tag)

## Examples

To run a packaged Holoscan application:

```bash
holoscan run -i /path/to/my/input -o /path/to/application/generated/output my-application:1.0.1
```

## Positional Arguments

(#cli-run-image-tag)=

### `image:[tag]`

Name and tag of the Docker container image to execute.

## Flags

(#cli-run-address)=

### `[--address ADDRESS]`

Address (`[<IP or hostname>][:<port>]`) of the *App Driver*. If not specified, the *App Driver* uses the default host address (`0.0.0.0`) with the default port number (`57777`).

For example:

```bash
--address my_app_network
--address my_app_network:57777
```

:::{note}
Ensure that the IP address is not blocked and the port is configured with the firewall accordingly.
:::

(#cli-run-driver)=

### `[--driver]`

Run the **App Driver** on the current machine. Can be used together with the [](#cli-run-worker) option to run both the **App Driver** and the **App Worker** on the same machine.

(#cli-run-input)=

### `[--input|-i INPUT]`

Specifies a directory path with input data for the application to process. When specified, a directory mount is set up to the value defined in the environment variable `HOLOSCAN_INPUT_PATH`.

:::{note}
Ensure that the directory on the host is accessible by the current user or the user specified with [--uid](#cli-run-uid).
:::

:::{note}
Use the host system path when running applications inside Docker (DooD).
:::


(#cli-run-output)=

### `[--output|-o OUTPUT]`

Specifies a directory path to store application-generated artifacts. When specified, a directory mount is set up to the value defined in the environment variable `HOLOSCAN_OUTPUT_PATH`.

:::{note}
Ensure that the directory on the host is accessible by the current user or the user specified with [--uid](#cli-run-uid).
:::

(#cli-run-fragments)=

### `[--fragments|-f FRAGMENTS]`

Comma-separated names of the fragments to be executed by the **App Worker**. If not specified, only one fragment (selected by the **App Driver**) will be executed. `all` can be used to run all the fragments.

(#cli-run-worker)=

### `[--worker]`

Run the **App Worker**.

(#cli-run-worker-address)=

### `[--worker-address WORKER_ADDRESS]`

The address (`[<IP or hostname>][:<port>]`) of the **App Worker**. If not specified, the **App Worker** uses the default host address (`0.0.0.0`) with a randomly chosen port number between `10000` and `32767` that is not currently in use. This argument automatically sets the `HOLOSCAN_UCX_SOURCE_ADDRESS` environment variable if the worker address is a local IP address. Refer to [](#creating-holoscan-distributed-application-env-vars) for details.

For example:

```bash
--worker-address my_app_network
--worker-address my_app_network:10000
```

:::{note}
Ensure that the IP address is not blocked and the port is configured with the firewall accordingly.
:::

(#cli-run-config)=

### `[--config CONFIG]`

Path to the application configuration file. If specified, it overrides the embedded configuration file found in the environment variable `HOLOSCAN_CONFIG_PATH`.

(#cli-run-health-check)=

### `[--health-check HEALTH_CHECK]`

Enables the health check service for [distributed applications](../holoscan_create_distributed_app.md) by setting the `HOLOSCAN_ENABLE_HEALTH_CHECK` environment variable to `true`. This allows [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe) to monitor the application's liveness and readiness.

(#cli-run-network)=

### `[--network|-n NETWORK]`

The Docker network that the application connects to for communicating with other containers. The **Runner** use the `host` network by default if not specified. Otherwise, the specified value is used to create a network with the `bridge` driver.

For advanced uses, first create a network using `docker network create` and pass the name of the network to the `--network` option. Refer to [Docker Networking](https://docs.docker.com/network/) documentation for additional details.

(#cli-run-nic)=

### `[--nic NETWORK_INTERFACE]`

Name of the network interface to use with a distributed multi-fragment application. This option sets `UCX_NET_DEVICES` environment variable with the value specified and is required when running a distributed multi-fragment application across multiple nodes. See {ref}`UCX Network Interface Selection <ucx-network-selection>` for details.


(#cli-run-use-all-nics)=

### `[--use-all-nics]`

When set, this option allows UCX to control the selection of network interface cards for data transfer. Otherwise, the network interface card specified with '--nic' is used. This option sets the environment variable `UCX_CM_USE_ALL_DEVICES` to `y` (default: False).

When this option is not set, the CLI runner always sets `UCX_CM_USE_ALL_DEVICES` to `n`.


(#cli-run-render)=

### `[--render|-r]`

Enable graphic rendering from your application. Defaults to `False`.

(#cli-run-quiet)=

### `[--quiet|-q]`

Suppress the STDOUT and print only STDERR from the application. Defaults to `False`.

(#cli-run-shm-size)=

### `[--shm-size]`

Sets the size of `/dev/shm`. The format is <number(int,float)>[MB|m|GB|g|Mi|MiB|Gi|GiB]. Use `config` to read the shared memory value defined in the `app.json` manifest. By default, the container is launched using `--ipc=host` with host system's `/dev/shm` mounted.

(#cli-run-terminal)=

### `[--terminal]`

Enters terminal with all configured volume mappings and environment variables.

(#cli-run-device)=

### `[--device]`

Map host devices into the application container.

By default, the CLI searches the `/dev/` path for devices unless the specified string starts with `/`.

For example:

```bash
# mount all AJA capture cards
--device ajantv*
# mount AJA capture card 0 and 1
--device ajantv0 ajantv1
# mount V4L2 video device 1 and AJAX capture card 2
--device video1 --device /dev/ajantv2
```


:::{warning}
When using the `--device` option, append `--` after the last item to avoid misinterpretation by the CLI. For example:

```bash
holoscan run --render --device ajantv0 video1 -- my-application-image:1.0

```
:::


(#cli-run-gpu)=

### `[--gpu]`

Override the value of the `NVIDIA_VISIBLE_DEVICES` environment variable with the default value set to 
the value defined in the [package manifest file](./hap.md#package-manifest) or `all` if undefined.

Refer to the [GPU Enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration)
page for all available options.

:::{note}
The default value is `nvidia.com/igpu=0` when running a HAP built for iGPU on a system with both iGPU and dGPU,
:::

:::{note}
A single integer value translates to the device index, not the number of GPUs.
:::

(#cli-run-uid)=

### `[--uid UID]`

Run the application with the specified user ID (UID). Defaults to the current user's UID.

(#cli-run-gid)=

### `[--gid GID]`

Run the application with the specified group ID (GID). Defaults to the current user's GID.

:::{note}
The Holoscan Application supports various environment variables for configuration. Refer to [](#creating-holoscan-distributed-application-env-vars) for details.
:::
