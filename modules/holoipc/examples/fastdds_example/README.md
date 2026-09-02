# Holoscan IPC with Fast DDS (`fastdds_example`)

This directory is a **Holoscan SDK** sample, not a drop-in copy of the upstream eProsima hello world. It uses **`holoscan::ipc`** (`Context`, `FastDdsTransport`, `make_context`), **CUDA IPC** via **`PointerDescriptor`**, Holoscan logging (`HOLOSCAN_IPC_LOG_*`), and app IDL types (**CudaBuffer**). The layout (publisher/subscriber, listener vs wait-set) is similar to classic Fast DDS tutorials, but the **process and types are Holoscan-specific**.

**Where to read more**

- Broader Holoscan examples and install layout: [Holoscan by Example](../../../../docs/holoscan_by_example.mdx) (this sample lives under `modules/holoipc/examples/` in the source tree).
- Generic Fast DDS concepts (participants, readers/writers, listeners, wait-sets, XML profiles): [Fast DDS documentation](https://fast-dds.docs.eprosima.com/).

The sections below focus on **how to run this binary** and **Holoscan IPC / build** notes. For full DDS entity theory, see the eProsima docs above.

- [Run the example](#run-the-example)
- [Wait-set subscriber](#wait-set-subscriber)
- [IPC and build options](#ipc-and-build-options)
- [XML profile playground](#xml-profile-playground)

## Description

Publisher and subscriber build nested DDS entities (participant → pub/sub → writer/reader) with optional XML profiles (see [XML profile playground](#xml-profile-playground)). The subscriber can use **listener callbacks** (default) or a **wait-set** (`--waitset`); both paths exercise the same Holoscan IPC + sample flow. For callback vs wait-set details, see the Fast DDS docs linked above.

## Run the example

To launch this example, two different terminals are required: one for the publisher and one for the subscriber. These steps assume a **Linux** environment aligned with Holoscan SDK-supported platforms (this sample uses CUDA IPC).

### Hello world publisher

```shell
user@machine:example_path$ ./fastdds_example publisher
Publisher running. Please press Ctrl+C to stop the Publisher at any time.
```

### Hello world subscriber

```shell
user@machine:example_path$ ./fastdds_example subscriber
Subscriber running. Please press Ctrl+C to stop the Subscriber at any time.
```

Run ``./fastdds_example -h`` (or ``--help``) to list available flags.

### Expected output

Regardless of which application is run first, since the publisher will not start sending data until a subscriber is discovered, the expected output both for publishers and subscribers is a first displayed message acknowledging the match, followed by the amount of samples sent or received until Ctrl+C is pressed.

### Publisher

```shell
Publisher running. Please press Ctrl+C to stop the Publisher at any time.
Publisher matched.
Buffer with size: 4194304 bytes (sample 1) SENT
Buffer with size: 4194304 bytes (sample 2) SENT
...
```

### Subscriber

```shell
Subscriber running. Please press Ctrl+C to stop the Subscriber at any time.
Subscriber matched.
Buffer with size: 4194304 bytes, device_ptr: 0x... (sample 1) ACQUIRED
[Subscriber] Read from GPU: "Hello from GPU!"
...
```

When Ctrl+C is pressed to stop one of the applications, the other one will show the unmatched status, displaying an informative message, and it will stop sending / receiving messages.
The following is a possible output of the publisher application when stopping the subscriber app.

```shell
...
Buffer with size: 4194304 bytes (sample 9) SENT
Buffer with size: 4194304 bytes (sample 10) SENT
Publisher unmatched.
```

## Wait-set subscriber

As described [above](#description), this sample supports two subscriber implementations. Launching the subscriber with ``-w`` or ``--waitset`` uses the wait-set path instead of the listener callback.

```shell
user@machine:example_path$ ./fastdds_example subscriber --waitset
```

The expected output matches the *[Expected output](#expected-output)* section above.

## IPC and build options

This example uses the **holoscan::ipc** (IPC) library for participant-scoped pointer sharing (CUDA IPC descriptors over DDS). Generated headers from IDL are treated as part of the public API and should be included with angle brackets:

- Library types (e.g. `PointerDescriptor`): `#include <holoscan/ipc/transport/fastdds/gen/pointer_descriptor.hpp>`
- App-generated types (e.g. `CudaBuffer`): `#include <gen/CudaBufferPubSubTypes.hpp>` (headers are produced under the example’s **build** tree `gen/`, not in source control).

The example target gets SDK headers via `holoscan::ipc`; the build adds the example binary dir so `<gen/...>` resolves.

## XML profile playground

The *eProsima Fast DDS* entities can be configured through an XML profile from the environment.
This is accomplished by setting the environment variable ``FASTDDS_DEFAULT_PROFILES_FILE`` to the path of the XML profiles file:

```shell
user@machine:example_path$ export FASTDDS_DEFAULT_PROFILES_FILE=hello_world_profile.xml
```

The example provides an XML profile file with certain QoS:

- Reliable reliability: avoid sample loss.
- Transient local durability: enable late-join subscriber applications to receive previous samples.
- Keep-last history with high depth: ensure certain amount of previous samples for late-joiners.

Applying different configurations to the entities will change to a greater or lesser extent how the application behaves in relation to sample management.
Even when these settings affect the behavior of the sample management, the applications' output will be similar.
