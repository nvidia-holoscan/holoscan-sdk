(creating-holoscan-distributed-application)=

# Creating a Distributed Application

Distributed applications refer to those where the workflow is divided into multiple fragments that may be run on separate nodes. For example, data might be collected via a sensor at the edge, sent to a separate workstation for processing, and then the processed data could be sent back to the edge node for visualization. Each node would run a single fragment consisting of a computation graph built up of operators. Thus one fragment is the equivalent of a non-distributed application. In the distributed context, the application initializes the different fragments and then defines the connections between them to build up the full distributed application workflow.

In this section we'll describe:

- How to {ref}`define a distributed application<defining-a-distributed-application-class>`.
- How to {ref}`build and run a distributed application<building-and-running-a-distributed-application>`.

(defining-a-distributed-application-class)=

## Defining a Distributed Application Class

:::{tip}
Defining distributed applications is also illustrated in the [video_replayer_distributed](./examples/video_replayer_distributed.md) and [ping_distributed](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/ping_distributed) examples. The `ping_distributed` examples also illustrate how to update C++ or Python applications to parse user-defined arguments in a way that works without disrupting support for distributed application command line arguments (e.g., `--driver`, `--worker`).
:::

Defining a single fragment ({cpp:class}`C++ <holoscan::Fragment>`/{py:class}`Python <holoscan.core.Fragment>`) involves adding operators using `make_operator()` ({cpp:func}`C++ <holoscan::Fragment::make_operator>`) or the operator constructor ({py:func}`Python <holoscan.core.Operator>`), and defining the connections between them using the `add_flow()` method ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) in the `compose()` method. Thus, defining a fragment is just like defining a non-distributed application except that the class should inherit from fragment instead of application.

The application will then be defined by initializing fragments within the application's `compose()` method. The `add_flow()` method ({cpp:func}`C++ <holoscan::Application::add_flow>`/{py:func}`Python <holoscan.core.Application.add_flow>`) can be used to define the connections across fragments.


`````{tab-set}
````{tab-item} C++
- We define the `Fragment1` and `Fragment2` classes that inherit from the {cpp:class}`Fragment <holoscan::Fragment>` base class.
- We define the `App` class that inherits from the {cpp:class}`Application <holoscan::Application>` base class.
- The `App` class initializes any fragments used and defines the connections between them. Here we have used dummy port and operator names in the example `add_flow` call connecting the fragments since no specific operators are shown in this example.
- We create an instance of the `App` class in `main()` using the {cpp:func}`make_application() <holoscan::make_application>` function.
- The {cpp:func}`run()<holoscan::Fragment::run>` method starts the application which will execute its {cpp:func}`compose()<holoscan::Fragment::compose>` method where the custom workflow will be defined.

```{code-block} cpp
:emphasize-lines: 3, 11, 19, 24-25, 28-29, 35-36
:name: holoscan-app-skeleton-cpp-distributed

#include <holoscan/holoscan.hpp>

class Fragment1 : public holoscan::Fragment {
 public:
  void compose() override {
    // Define Operators and workflow for Fragment1
    //   ...
  }
};

class Fragment2 : public holoscan::Fragment {
 public:
  void compose() override {
    // Define Operators and workflow for Fragment2
    //   ...
  }
};

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto fragment1 = make_fragment<Fragment1>("fragment1");
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    // Define the workflow: replayer -> holoviz
    add_flow(fragment1, fragment2, {{"fragment1_operator_name.output_port_name",
                                     "fragment2_operator_name.input_port_name"}});
  }
};


int main() {
  auto app = holoscan::make_application<App>();
  app->run();
  return 0;
}
```
````
````{tab-item} Python
- We define the `Fragment1` and `Fragment2` classes that inherit from the {py:class}`Fragment <holoscan.core.Fragment>` base class.
- We define the `App` class that inherits from the {py:class}`Application <holoscan.core.Application>` base class.
- The `App` class initializes any fragments used and defines the connections between them. Here we have used dummy port and operator names in the example add_flow call connecting the fragments since no specific operators are shown in this example.
- We create an instance of the `App` class in `__main__`.
- The {py:func}`run()<holoscan.Application.run>` method starts the application which will execute its {py:func}`compose()<holoscan.Application.compose>` method where the custom workflow will be defined.

```{code-block} python
:emphasize-lines: 3, 9, 15, 18-19, 21-22, 26-27
:name: holoscan-app-skeleton-python-distributed

from holoscan.core import Application, Fragment

class Fragment1(Fragment):

    def compose(self):
        # Define Operators and workflow
        #   ...

class Fragment2(Fragment):

    def compose(self):
        # Define Operators and workflow
        #   ...

class App(Application):

    def compose(self):
        fragment1 = Fragment1(self, name="fragment1")
        fragment2 = Fragment2(self, name="fragment2")

        self.add_flow(fragment1, fragment2, {("fragment1_operator_name.output_port_name",
                                              "fragment2_operator_name.input_port_name")})


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
```
````
`````

### Serialization of Custom Data Types for Distributed Applications

Transmission of data between fragments of a multi-fragment application is done via the [Unified Communications X (UCX)](https://openucx.org/) library. In order to transmit data, it must be serialized into a binary form suitable for transmission over a network. For Tensors ({cpp:class}`C++ <holoscan::Tensor>`/{py:class}`Python <holoscan.core.Tensor>`), strings and various scalar and vector numeric types, serialization is already built in. For more details on concrete examples of how to extend the data serialization support to additional user-defined classes, see the separate page on {ref}`serialization<object-serialization>`.


(building-and-running-a-distributed-application)=
## Building and running a Distributed Application

`````{tab-set}
````{tab-item} C++

Building a distributed application works in the same way as for a non-distributed one. See {ref}`building-and-running-your-application`
````
````{tab-item} Python
Python applications do not require building. See {ref}`building-and-running-your-application`.
````
`````

Running an application in a distributed setting requires launching the application binary on all nodes involved in the distributed application. A single node must be selected to act as the application driver. This is achieved by using the `--driver` command-line option. Worker nodes are initiated by launching the application with the `--worker` command-line option. It's possible for the driver node to also serve as a worker if both options are specified.

The address of the driver node must be specified for each process (both the driver and worker(s)) to identify the appropriate network interface for communication. This can be done via the `--address` command-line option, which takes a value in the form of `[<IPv4/IPv6 address or hostname>][:<port>]` (e.g., `--address 192.168.50.68:10000`):
- The driver's IP (or hostname) **MUST** be set for each process (driver and worker(s)) when running distributed applications on multiple nodes (default: `0.0.0.0`). It can be set without the port (e.g., `--address 192.168.50.68`).
- In a single-node application, the driver's IP (or hostname) can be omitted, allowing any network interface (`0.0.0.0`) to be selected by the [UCX](https://openucx.readthedocs.io/en/master/faq.html#which-network-devices-does-ucx-use) library.
- The port is always optional (default: `8765`). It can be set without the IP (e.g., `--address :10000`).

The worker node's address can be defined using the `--worker-address` command-line option (`[<IPv4/IPv6 address or hostname>][:<port>]`). If it's not specified, the application worker will default to the host address (`0.0.0.0`) with a randomly chosen port number between `10000` and `32767` that is not currently in use.
This argument automatically sets the `HOLOSCAN_UCX_SOURCE_ADDRESS` environment variable if the worker address is a local IP address. Refer to [this section](#creating-holoscan-distributed-application-env-vars) for details.

The `--fragments` command-line option is used in combination with `--worker` to specify a comma-separated list of fragment names to be run by a worker. If not specified, the application driver will assign a single fragment to the worker. To indicate that a worker should run all fragments, you can specify `--fragments all`.

The `--config` command-line option can be used to designate a path to a configuration file to be used by the application.

Below is an example launching a three fragment application named `my_app` on two separate nodes:
- The application driver is launched at `192.168.50.68:10000` on the first node (A), with a worker running two fragments, "fragment1" and "fragment3."
- On a separate node (B), the application launches a worker for "fragment2," which will connect to the driver at the address above.

`````{tab-set}
````{tab-item} C++
```bash
# Node A
my_app --driver --worker --address 192.168.50.68:10000 --fragments fragment1,fragment3
# Node B
my_app --worker --address 192.168.50.68:10000 --fragments fragment2
```
````
````{tab-item} Python
```bash
# Node A
python3 my_app.py --driver --worker --address 192.168.50.68:10000 --fragments fragment1,fragment3
# Node B
python3 my_app.py --worker --address 192.168.50.68:10000 --fragments fragment2
```

````
`````

(ucx-network-selection)=
`````{note}
### UCX Network Interface Selection

[UCX](https://openucx.org/) is used in the Holoscan SDK for communication across fragments in distributed applications. It is designed to [select the best network device based on performance characteristics (bandwidth, latency, NUMA locality, etc.)](https://openucx.readthedocs.io/en/master/faq.html#which-network-devices-does-ucx-use). In some scenarios (under investigation), UCX cannot find the correct network interface to use, and the application fails to run. In this case, you can manually specify the network interface to use by setting the `UCX_NET_DEVICES` environment variable.

For example, if the user wants to use the network interface `eth0`, you can set the environment variable as follows, before running the application:

```bash
export UCX_NET_DEVICES=eth0
```

Or, if you are running a packaged distributed application with the {ref}`Holoscan CLI<holoscan-cli-run>`, use the `--nic eth0` option to manually specify the network interface to use.

The available network interface names can be found by running the following command:

```bash
ucx_info -d | grep Device: | awk '{print $3}' | sort | uniq
# or
ip -o -4 addr show | awk '{print $2, $4}' # to show interface name and IP
```
`````
`````{warning}
### Known limitations

The following are known limitations of the distributed application support in the SDK, some of which will be addressed in future updates:

#### 1. A connection error message is displayed even when the distributed application is running correctly.

The message `Connection dropped with status -25 (Connection reset by remote peer)` appears in the console even when the application is functioning properly. This is a known issue and will be addressed in future updates, ensuring that this message will only be displayed in the event of an actual connection error. It currently is printed once some fragments complete their work and start shutdown. Any connections from those fragments to ones that remain open are disconnected at that point, resulting in the logged message.

#### 2. GPU tensors can only currently be sent/received by UCX from a single device on a given node.

By default, device ID 0 is used by the UCX extensions to send/receive data between fragments. To override this default, the user can set environment variable `HOLOSCAN_UCX_DEVICE_ID`.

#### 3. Health check service is turned off by default

The health checker service in a distributed application is turned off by default. However, it can be enabled by setting the environment variable `HOLOSCAN_ENABLE_HEALTH_CHECK` to `true` (can use `1` and `on,` case-insensitive). If the environment variable is not set or is invalid, the default value (disabled) is used.


#### 4. The use of the management port is unsupported on the NVIDIA IGX Orin Developer Kit.

IGX devices come with two ethernet ports, noted as port #4 and #5 in the [NVIDIA IGX Orin User Guide](https://docs.nvidia.com/igx-orin/user-guide/latest/system-overview.html#i-o-and-external-interfaces). To run distributed applications on these devices, the user must ensure that ethernet port #4 is used to connect the driver and the workers.
`````

`````{note}
### GXF UCX Extension

Holoscan's distributed application feature makes use of the [GXF UCX Extension](https://docs.nvidia.com/metropolis/deepstream/dev-guide/graphtools-docs/docs/text/ExtensionsManual/UcxExtension.html). Its documentation may provide useful additional context into how data is transmitted between fragments.
`````
:::{tip}
Given a CMake project, a pre-built executable, or a Python application, you can also use the [Holoscan CLI](./cli/cli.md) to [package and run your Holoscan application](./holoscan_packager.md) in a OCI-compliant container image.
:::

(#creating-holoscan-distributed-application-env-vars)=

### Environment Variables for Distributed Applications

(holoscan-distributed-env)=

#### Holoscan SDK environment variables

You can set environment variables to modify the default actions of services and the scheduler when executing a distributed application.

- **HOLOSCAN_ENABLE_HEALTH_CHECK**: determines whether the health check service should be active for distributed applications. Accepts values such as "true," "1," or "on" (case-insensitive) as true, enabling the health check. If unspecified, it defaults to "false." When enabled, the [gRPC Health Checking Service](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) is activated, allowing tools like [grpc-health-probe](https://github.com/grpc-ecosystem/grpc-health-probe) to monitor liveness and readiness.
This environment variable is only used when the distributed application is launched with `--driver` or `--worker` options. The health check service in a distributed application runs on the same port as the App Driver (default: `8765`) and/or the App Worker.

- **HOLOSCAN_DISTRIBUTED_APP_SCHEDULER** : controls which scheduler is used for distributed applications. It can be set to either `greedy`, `multi_thread` or `event_based`. `multithread` is also allowed as a synonym for `multi_thread` for backwards compatibility. If unspecified, the default scheduler is `multi_thread`.

- **HOLOSCAN_STOP_ON_DEADLOCK** : can be used in combination with `HOLOSCAN_DISTRIBUTED_APP_SCHEDULER` to control whether or not the application will automatically stop on deadlock. Values of "True", "1" or "ON" will be interpreted as true (enable stop on deadlock). It is "true" if unspecified. This environment variable is only used when `HOLOSCAN_DISTRIBUTED_APP_SCHEDULER` is explicitly set.

- **HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT** : controls the delay (in ms) without activity required before an application is considered to be in deadlock. It must be an integer value (units are ms).

- **HOLOSCAN_MAX_DURATION_MS** : sets the application to automatically terminate after the requested maximum duration (in ms) has elapsed. It must be an integer value (units are ms). This environment variable is only used when `HOLOSCAN_DISTRIBUTED_APP_SCHEDULER` is explicitly set.

- **HOLOSCAN_CHECK_RECESSION_PERIOD_MS** : controls how long (in ms) the scheduler waits before re-checking the status of operators in an application. It must be a floating point value (units are ms). This environment variable is only used when `HOLOSCAN_DISTRIBUTED_APP_SCHEDULER` is explicitly set.

- **HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE** : can be used to override the default 7 kB serialization buffer size. This should typically not be needed as tensor types store only a small header in this buffer to avoid explicitly making a copy of their data. However, other data types do get directly copied to the serialization buffer and in some cases it may be necessary to increase it.

- **HOLOSCAN_UCX_ASYNCHRONOUS** : If set to false, asynchronous transmit of UCX messages is disabled (this is the default since Holoscan 3.0, which matches the prior behavior from Holoscan 1.0). In Holoscan 2.x, the default was true, but this was deemed harder for application developers to use, so that decision has been reverted. Synchronous mode makes it easier to use an allocator like `BlockMemoryPool` as additional tensors would not be queued before the prior one was sent.

- **HOLOSCAN_UCX_DEVICE_ID** : The GPU ID of the device that will be used by UCX transmitter/receivers in distributed applications. If unspecified, it defaults to 0. A list of discrete GPUs available in a system can be obtained via `nvidia-smi -L`. GPU data sent between fragments of a distributed application must be on this device.

- **HOLOSCAN_UCX_PORTS** : This defines the preferred port numbers for the SDK when specific ports for UCX communication need to be predetermined, such as in a Kubernetes environment. If the distributed application requires three ports (UCX receivers) and the environment variable is unset, the SDK chooses three unused ports sequentially from the range 10000~32767. Specifying a value, for example, `HOLOSCAN_UCX_PORTS=10000`, results in the selection of ports 10000, 10001, and 10002. Multiple starting values can be comma-separated. The system increments from the last provided port if more ports are needed. Any unused specified ports are ignored.

- **HOLOSCAN_UCX_SOURCE_ADDRESS** : This environment variable specifies the local IP address (source) for the UCX connection. This variable is especially beneficial when a node has multiple network interfaces, enabling the user to determine which one should be utilized for establishing a UCX client (UcxTransmitter). If it is not explicitly specified, the default address is set to `0.0.0.0`, representing any available interface.

#### UCX-specific environment variables
Transmission of data between fragments of a multi-fragment application is done via the [Unified Communications X (UCX)](https://openucx.readthedocs.io) library, a point-to-point communication framework designed to utilize the best available hardware resources (shared memory, TCP, GPUDirect RDMA, etc). UCX has many parameters that can be controlled via environment variables. A few that are particularly relevant to Holoscan SDK distributed applications are listed below:

- The [`UCX_TLS`](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use) environment variable can be used to control which transport layers are enabled. By default, `UCX_TLS=all` and UCX will attempt to choose the optimal transport layer automatically.
- The `UCX_NET_DEVICES` environment variable is by default set to `all`, meaning that UCX may choose to use any available network interface controller (NIC). In some cases it may be necessary to restrict UCX to a specific device or set of devices, which can be done by setting `UCX_NET_DEVICES` to a comma separated list of the device names (i.e., as obtained by Linux command `ifconfig -a` or `ip link show`).
- Setting `UCX_TCP_CM_REUSEADDR=y` is recommended to enable ports to be reused without having to wait the full socket TIME_WAIT period after a socket is closed.
- The [`UCX_LOG_LEVEL`](https://openucx.readthedocs.io/en/master/faq.html#how-can-i-tell-which-protocols-and-transports-are-being-used-for-communication) environment variable can be used to control the logging level of UCX. The default is setting is WARN, but changing to a lower level such as INFO will provide more verbose output on which transports and devices are being used.
- By default, Holoscan SDK will automatically set `UCX_PROTO_ENABLE=y` upon application launch to enable the newer "v2" UCX protocols. If for some reason, the older v1 protocols are needed, one can set `UCX_PROTO_ENABLE=n` in the environment to override this setting. When the v2 protocols are enabled, one can optionally set `UCX_PROTO_INFO=y` to enable detailed logging of what protocols are being used at runtime.
- By default, Holoscan SDK will automatically set `UCX_MEMTYPE_CACHE=n` upon application launch to disable the UCX memory type cache (See [UCX documentation](https://openucx.readthedocs.io/en/master/faq.html#i-m-running-ucx-with-gpu-memory-and-geting-a-segfault-why) for more information. It can cause about [0.2 microseconds of pointer type checking overhead with the cudacudaPointerGetAttributes() CUDA API](https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues)). If for some reason the memory type cache is needed, one can set `UCX_MEMTYPE_CACHE=y` in the environment to override this setting.
- By default, the Holoscan SDK will automatically set `UCX_CM_USE_ALL_DEVICES=n` at application startup to disable consideration of all devices for data transfer. If for some reason the opposite behavior is desired, one can set `UCX_CM_USE_ALL_DEVICES=y` in the environment to override this setting. Setting `UCX_CM_USE_ALL_DEVICES=n` can be used to workaround an issue where UCX sometimes defaults to a device that might not be the most suitable for data transfer based on the host's available devices. On a host with address 10.111.66.60, UCX, for instance, might opt for the `br-80572179a31d` (192.168.49.1) device due to its superior bandwidth as compared to `eno2` (10.111.66.60). With `UCX_CM_USE_ALL_DEVICES=n`, UCX will ensure consistency by using the same device for data transfer that was initially used to establish the connection. This ensures more predictable behavior and can avoid potential issues stemming from device mismatches during the data transfer process.
- Setting `UCX_TCP_PORT_RANGE=<start>-<end>` can be used to define a specific range of ports that UCX should utilize for data transfer. This is particularly useful in environments where ports need to be predetermined, such as in a Kubernetes setup. In such contexts, Pods often have ports that need to be exposed, and these ports must be specified ahead of time. Moreover, in scenarios where firewall configurations are stringent and only allow specified ports, having a predetermined range ensures that the UCX communication does not get blocked. This complements the `HOLOSCAN_UCX_SOURCE_ADDRESS`, which specifies the local IP address for the UCX connection, by giving further control over which ports on that specified address should be used. By setting a port range, users can ensure that UCX operates within the boundaries of the network and security policies of their infrastructure.

:::{tip}
A list of all available UCX environment variables and a brief description of each can be obtained by running `ucx_info -f` from the Holoscan SDK container. Holoscan SDK uses UCX's active message (AM) protocols, so environment variables related to other protocols such as tag-mat.
:::

(object-serialization)=

## Serialization

Distributed applications must serialize any objects that are to be sent between the fragments of a multi-fragment application. Serialization involves binary serialization to a buffer that will be sent from one fragment to another via the Unified Communications X (UCX) library. For tensor types (e.g., holoscan::Tensor), no actual copy is made, but instead transmission is done directly from the original tensor's data and only a small amount of header information is copied to the serialization buffer.

A table of the types that have codecs pre-registered so that they can be serialized between fragments using Holoscan SDK is given below.

| Type Class                              | Specific Types                                                                            |
|-----------------------------------------|-------------------------------------------------------------------------------------------|
| integers                                | int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t                  |
| floating point                          | float, double, complex &lt;float&gt;, complex&lt;double&gt;                               |
| boolean                                 | bool                                                                                      |
| strings                                 | std::string                                                                               |
| std::vector&lt;T&gt;                    | T is std::string or any of the boolean, integer or floating point types above             |
| std::vector&lt;std::vector&lt;T&gt;&gt; | T is std::string or any of the boolean, integer or floating point types above             |
| std::vector&lt;HolovizOp::InputSpec&gt; | a vector of InputSpec objects that are specific to HolovizOp                              |
| std::shared_ptr&lt;T&gt;                | T is any of the scalar, vector or std::string types above |
| tensor types                            | holoscan::Tensor, nvidia::gxf::Tensor, nvidia::gxf::VideoBuffer, nvidia::gxf::AudioBuffer |
| GXF-specific types                      | nvidia::gxf::TimeStamp, nvidia::gxf::EndOfStream                                          |


:::{warning}
If an operator transmitting both CPU and GPU tensors is to be used in distributed applications, the same output port cannot mix both GPU and CPU tensors. CPU and GPU tensor outputs should be placed on separate output ports. This is a limitation of the underlying UCX library being used for zero-copy tensor serialization between operators.

As a concrete example, assume an operator, `MyOperator` with a single output port named "out" defined in it's setup method. If the output port is only ever going to connect to other operators within a fragment, but never across fragments then it is okay to have a `TensorMap` with a mixture of host and device arrays on that single port.

`````{tab-set}
````{tab-item} C++

```cpp
void MyOperator::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("out");
}

void MyOperator::compute(OperatorSpec& spec) {

  // omitted: some computation resulting in multiple holoscan::Tensors
  // (two on CPU ("cpu_coords_tensor" and "cpu_metric_tensor") and one on device ("gpu_tensor").

  TensorMap out_message;

  // insert all tensors in one TensorMap (mixing CPU and GPU tensors is okay when ports only connect within a Fragment)
  out_message.insert({"coordinates", cpu_coords_tensor});
  out_message.insert({"metrics", cpu_metric_tensor});
  out_message.insert({"mask", gpu_tensor});

  op_output.emit(out_message, "out");
}

```

````
````{tab-item} Python

```python
class MyOperator:

    def setup(self, spec: OperatorSpec):
        spec.output("out")


    def compute(self, op_input, op_output, context):
        # Omitted: assume some computation resulting in three holoscan::Tensor or tensor-like
        # objects. Two on CPU ("cpu_coords_tensor" and "cpu_metric_tensor") and one on device
        # ("gpu_tensor").

        # mixing CPU and GPU tensors in a single dict is okay only for within-Fragment connections
        op_output.emit(
            dict(
                coordinates=cpu_coords_tensor,
                metrics=cpu_metrics_tensor,
                mask=gpu_tensor,
            ),
            "out"
        )
```
`````

However, this mixing of CPU and GPU arrays on a single port will not work for distributed apps and instead separate ports should be used if it is necessary for an operator to communicate across fragments.

`````{tab-set}
````{tab-item} C++

```cpp
void MyOperator::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("out_host");
  spec.output<holoscan::TensorMap>("out_device");
}

void MyOperator::compute(OperatorSpec& spec) {

  // some computation resulting in a pair of holoscan::Tensor, one on CPU ("cpu_tensor") and one on device ("gpu_tensor").
  TensorMap out_message_host;
  TensorMap out_message_device;

  // put all CPU tensors on one port
  out_message_host.insert({"coordinates", cpu_coordinates_tensor});
  out_message_host.insert({"metrics", cpu_metrics_tensor});
  op_output.emit(out_message_host, "out_host");

  // put all GPU tensors on another
  out_message_device.insert({"mask", gpu_tensor});
  op_output.emit(out_message_device, "out_device");
}
```

````
````{tab-item} Python

```python
class MyOperator:

    def setup(self, spec: OperatorSpec):
        spec.output("out_host")
        spec.output("out_device")


    def compute(self, op_input, op_output, context):
        # Omitted: assume some computation resulting in three holoscan::Tensor or tensor-like
        # objects. Two on CPU ("cpu_coords_tensor" and "cpu_metric_tensor") and one on device
        # ("gpu_tensor").

        # split CPU and GPU tensors across ports for compatibility with inter-fragment communication
        op_output.emit(
            dict(coordinates=cpu_coords_tensor, metrics=cpu_metrics_tensor),
            "out_host"
        )
        op_output.emit(dict(mask=gpu_tensor), "out_device")
```

````
`````
:::


### Python

For the Python API, any array-like object supporting the [DLPack](https://dmlc.github.io/dlpack/latest/) interface, [`__array_interface__`](https://numpy.org/doc/stable/reference/arrays.interface.html) or [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) will be transmitted using  {py:class}`~holoscan.core.Tensor` serialization. This is done to avoid data copies for performance reasons. Objects of type `list[holoscan.HolovizOp.InputSpec]` will be sent using the underlying C++ serializer for `std::vector<HolovizOp::InputSpec>`. All other Python objects will be serialized to/from a `std::string` using the [cloudpickle](https://github.com/cloudpipe/cloudpickle) library.

:::{warning}
A restriction imposed by the use of cloudpickle is that all fragments in a distributed application [must be running the same Python version](https://github.com/cloudpipe/cloudpickle/blob/v2.2.1/README.md?plain=1#L17-L18).
:::

:::{warning}
Distributed applications behave differently than single fragment applications when {py:func}`op_output.emit() <holoscan.core.OutputContext.emit>` is called to emit a tensor-like Python object. Specifically, for array-like objects such as a PyTorch tensor, the same Python object will **not** be received by any call to {py:func}`op_input.receive() <holoscan.core.InputContext.receive>` in a downstream Python operator (even if the upstream and downstream operators are part of the same fragment). An object of type `holoscan.Tensor` will be received as a `holoscan.Tensor`. Any other array-like objects with data stored on device (GPU) will be received as a CuPy tensor. Similarly, any array-like object with data stored on the host (CPU) will be received as a NumPy array. The user must convert back to the original array-like type if needed (typically possible in a zero-copy fashion via DLPack or array interfaces).
:::


### C++

For any additional C++ classes that need to be serialized for transmission between fragments in a distributed application, the user must create their own codec and register it with the Holoscan SDK framework. As a concrete example, suppose that we had the following simple Coordinate class that we wish to send between fragments.

```cpp
struct Coordinate {
  float x;
  float y;
  float z;
};
```

To create a codec capable of serializing and deserializing this type, one should define a {cpp:class}`holoscan::codec` class for it as shown below.

```cpp
#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"


namespace holoscan {

template <>
struct codec<Coordinate> {
  static expected<size_t, RuntimeError> serialize(const Coordinate& value, Endpoint* endpoint) {
    return serialize_trivial_type<Coordinate>(value, endpoint);
  }
  static expected<Coordinate, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_trivial_type<Coordinate>(endpoint);
  }
};

}  // namespace holoscan
```

In this example, the first argument to `serialize` is a const reference to the type to be serialized and the return value is an {cpp:type}`~holoscan::expected` containing the number of bytes that were serialized. The `deserialize` method returns an {cpp:type}`~holoscan::expected` containing the deserialized object. The {cpp:class}`~holoscan::Endpoint` class is a base class representing the serialization endpoint (For distributed applications, the actual endpoint class used is {cpp:class}`~holoscan::UcxSerializationBuffer`).

The helper functions `serialize_trivial_type` (`deserialize_trivial_type`) can be used to serialize (deserialize) any plain-old-data (POD) type. Specifically, POD types can be serialized by just copying `sizeof(Type)` bytes to/from the endpoint. The {cpp:func}`~holoscan::Endpoint::read_trivial_type` and `~holoscan::Endpoint::write_trivial_type` methods could be used directly instead.

```cpp

template <>
struct codec<Coordinate> {
  static expected<size_t, RuntimeError> serialize(const Coordinate& value, Endpoint* endpoint) {
      return endpoint->write_trivial_type(&value);
  }
  static expected<Coordinate, RuntimeError> deserialize(Endpoint* endpoint) {
      Coordinate encoded;
    auto maybe_value = endpoint->read_trivial_type(&encoded);
    if (!maybe_value) { return forward_error(maybe_value); }
    return encoded;
  }
};

```

In practice, one would not actually need to define `codec<Coordinate>` at all since `Coordinate` is a trivially serializable type and the existing `codec` treats any types for which there is not a template specialization as a trivially serializable type. It is, however, still necessary to register the codec type with the {cpp:class}`~holoscan::gxf::CodecRegistry` as described below.

For non-trivial types, one will likely also need to use the {cpp:func}`~holoscan::Endpoint::read` and {cpp:func}`~holoscan::Endpoint::write` methods to implement the codec. Example use of these for the built-in codecs can be found in [`holoscan/core/codecs.hpp`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/include/holoscan/core/codecs.hpp).

Once such a codec has been defined, the remaining step is to register it with the static {cpp:class}`~holoscan::gxf::CodecRegistry` class. This will make the UCX-based classes used by distributed applications aware of the existence of a codec for serialization of this object type. If the type is specific to a particular operator, then one can register it via the {cpp:func}`~holoscan::gxf::GXFExecutor::register_codec` class.

```cpp
#include <holsocan/core/executors/gxf/gxf_executor.h>  // for holoscan::gxf::GXFExecutor

namespace holoscan::ops {

void MyCoordinateOperator::initialize() {
  gxf::GXFExecutor::register_codec<Coordinate>("Coordinate");

  // ...

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

}  // namespace holoscan::ops
```

Here, the argument provided to `register_codec` is the name the registry will use for the codec. This name will be serialized in the message header so that the deserializer knows which deserialization function to use on the received data. In this example, we chose a name that matches the class name, but that is not a requirement. If the name matches one that is already present in the {cpp:class}`~holoscan::gxf::CodecRegistry` class, then any existing codec under that name will be replaced by the newly registered one.

It is also possible to directly register the type outside of the context of {cpp:func}`~holoscan::Operator::initialize` by directly retrieving the static instance of the codec registry as follows.

```cpp

namespace holoscan {

gxf::CodecRegistry::get_instance().add_codec<Coordinate>("Coordinate");

}  // namespace holoscan
```

:::{tip}
CLI arguments (such as `--driver`, `--worker` ,`--fragments`)  are parsed by the `Application` ({cpp:class}`C++ <holoscan::Application>`/{py:class}`Python <holoscan.core.Application>`) class and the remaining arguments are available as `app.argv` ({cpp:func}`C++ <holoscan::Application::argv>`/{py:func}`Python <holoscan.core.Application.argv>`).

`````{tab-set}
````{tab-item} C++
A concrete example of using `app->argv()` in the [ping_distributed.cpp](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/ping_distributed/cpp/ping_distributed.cpp#:~:text=int%20main) example is covered in the section on {ref}`user-defined command line arguments<adding-user-defined-cli-arguments>`.

If you want to get access to the arguments before creating the {cpp:class}`C++ <holoscan::Application>` instance, you can access them through `holoscan::Application().argv()`.

The following example shows how to access the arguments in your application.

```cpp
#include <holoscan/holoscan.hpp>

class MyPingApp : public holoscan::Application {
// ...
};

int main(int argc, char** argv) {
  auto my_argv =
      holoscan::Application({"myapp", "--driver", "my_arg1", "--address=10.0.0.1"}).argv();
  HOLOSCAN_LOG_INFO(" my_argv: {}", fmt::join(my_argv, " "));

  HOLOSCAN_LOG_INFO(
      "    argv: {} (argc: {}) ",
      fmt::join(std::vector<std::string>(argv, argv + argc), " "),
      argc);

  auto app_argv = holoscan::Application().argv();  // do not use reference ('auto&') here (lifetime issue)
  HOLOSCAN_LOG_INFO("app_argv: {} (size: {})", fmt::join(app_argv, " "), app_argv.size());

  auto app = holoscan::make_application<MyPingApp>();
  HOLOSCAN_LOG_INFO("app->argv() == app_argv: {}", app->argv() == app_argv);

  app->run();
  return 0;
}

// $ ./myapp --driver --input image.dat --address 10.0.0.20

//  my_argv: myapp my_arg1
//     argv: ./myapp --driver --input image.dat --address 10.0.0.20 (argc: 6)
// app_argv: ./myapp --input image.dat (size: 3)
// app->argv() == app_argv: true

```

Please see other examples in the [Application unit tests](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/tests/core/application.cpp#:~:text=TestAppCustomArguments) in the Holoscan SDK repository.

````
````{tab-item} Python

A concrete example of usage of `app.argv` in the [ping_distributed.py](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/ping_distributed/python/ping_distributed.py#:~:text=__main__) example is covered in the section on {ref}`user-defined command line arguments<adding-user-defined-cli-arguments>`.

If you want to get access to the arguments before creating the {py:class}`Python <holoscan.core.Application>` instance, you can access them through `Application().argv`.

The following example shows how to access the arguments in your application.

```python
import argparse
import sys
from holoscan.core import Application


class MyApp(Application):
    def compose(self):
        pass


def main():
    app = MyApp()  # or alternatively, MyApp([sys.executable, *sys.argv])
    app.run()


if __name__ == "__main__":

    print("sys.argv:", sys.argv)
    print("Application().argv:", app.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args(app.argv[1:])
    print("args:", args)

    main()

# $ python cli_test.py --address 10.0.0.20 --input image.dat
# sys.argv: ['cli_test.py', '--address', '10.0.0.20', '--input', 'image.dat']
# Application().argv: ['cli_test.py', '--input', 'image.dat']
# args: Namespace(input='a')
```

```python
>>> from holoscan.core import Application
>>> import sys
>>> Application().argv == sys.argv
True
>>> Application([]).argv == sys.argv
True
>>> Application([sys.executable, *sys.argv]).argv == sys.argv
True
>>> Application(["python3", "myapp.py", "--driver", "my_arg1", "--address=10.0.0.1"]).argv
['myapp.py', 'my_arg1']
```

Please see other examples in the [Application unit tests (TestApplication class)](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/python/tests/unit/test_core.py#:~:text=TestApplication) in the Holoscan SDK repository.
````
`````
:::

(adding-user-defined-cli-arguments)=

### Adding user-defined command line arguments

When adding user-defined command line arguments to an application, one should avoid the use of any of the default command line argument names as `--help`, `--version`, `--config`, `--driver`, `--worker`, `--address`, `--worker-address`, `--fragments` as covered in the section on {ref}`running a distributed application<building-and-running-a-distributed-application>`. It is recommended to parse user-defined arguments from the `argv` (({cpp:func}`C++ <holoscan::Application::argv>`/{py:func}`Python <holoscan.core.Application.argv>`)) method/property of the application as covered in the note above, instead of using C++ `char* argv[]` or Python `sys.argv` directly. This way, only the new, user-defined arguments will need to be parsed.

A concrete example of this for both C++ and Python can be seen in the existing [ping_distributed](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/ping_distributed) example where an application-defined boolean argument (`--gpu`) is specified in addition to the default set of application arguments.


`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 5-7
:name: holoscan-app-skeleton-cpp

int main() {
  auto app = holoscan::make_application<App>();

  // Parse args
  bool tensor_on_gpu = false;
  auto& args = app->argv();
  if (std::find(args.begin(), args.end(), "--gpu") != std::end(args)) { tensor_on_gpu = true; }

  // configure tensor on host vs. GPU
  app->gpu_tensor(tensor_on_gpu);

  // run the application
  app->run();

  return 0;
}
```
````
````{tab-item} Python
```{code-block} python
:emphasize-lines: 14-23
:name: holoscan-app-skeleton-python

def main(on_gpu=False):
    app = MyPingApp()

    tensor_str = "GPU" if on_gpu else "host"
    print(f"Configuring application to use {tensor_str} tensors")
    app.gpu_tensor = on_gpu

    app.run()


if __name__ == "__main__":

    # get the Application's arguments
    app_argv = Application().argv

    parser = ArgumentParser(description="Distributed ping application.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a GPU tensor instead of a host tensor",
    )
    # pass app_argv[1:] to parse_args (app_argv[0] is the path of the application)
    args = parser.parse_args(app_argv[1:])
    main(on_gpu=args.gpu)
```
For Python, `app.argv[1:]` can be used with an `ArgumentParser` from Python's [argparse](https://docs.python.org/3/library/argparse.html) module.

Alternatively, it may be preferable to instead use `parser.parse_known_args()` to allow any arguments not defined by the user's parser to pass through to the application class itself. If one also sets `add_help=False` when constructing the `ArgumentParser`, it is possible to print the parser's help while still preserving the default application help (covering the default set of distributed application arguments). An example of this style is shown in the code block below.
```{code-block} Python
    parser = ArgumentParser(description="Distributed ping application.", add_help=False)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a GPU tensor instead of a host tensor",
    )

    # use parse_known_args to ignore other CLI arguments that may be used by Application
    args, remaining = parser.parse_known_args()

    # can print the parser's help here prior to the Application's help output
    if "-h" in remaining or "--help" in remaining:
        print("Additional arguments supported by this application:")
        print(textwrap.indent(parser.format_help(), "  "))
    main(on_gpu=args.gpu)
```
````
`````
