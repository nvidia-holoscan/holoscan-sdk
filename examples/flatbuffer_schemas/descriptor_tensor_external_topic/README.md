# Descriptor-backed schema external topic

This example connects an HSDK Graph to a separate standalone process through two HoloMQ external
topics. The application-owned `tensor_topic.ExternalFrame` FlatBuffer carries ordinary frame
metadata plus `pixels` and `calibration` Tensor fields. Both fields have independent layout, shape,
and capacity contracts.

The graph executable imports a frame, updates its ordinary source metadata while retaining both
Tensor handles, and exports the resulting frame. The standalone client loads the compiled
endpoint catalog, publishes the input through
`create_tensor_schema_publisher<Payload>()`, and receives the result through
`create_tensor_schema_subscriber<Payload>()`.

## What it demonstrates

- `DescriptorTensorTopicContract<ExternalFrameT>` with one complete field-qualified inventory for
  both `Graph::import_topic()` and `Graph::export_topic()`.
- A graph that continues to use only typed `Input<ExternalFrameT>` and
  `Output<ExternalFrameT>` ports.
- A standalone process that attaches only through `plan.endpoint_catalog()` and the public
  `ChannelRuntime` Tensor-schema endpoint factories.
- Multi-field metadata construction with `create_tensor_metadata()`, `emit(metadata, tensors)`,
  and `receive_tensors()`; the same publisher surface also supports capture-aware
  `emit_captured()`, and the returned bundle preserves transport-neutral sample metadata alongside
  the schema metadata and Tensor handles.
- A finite two-process round trip used as the public CTest smoke test.

## Descriptor metadata and Tensor storage

The FlatBuffer envelope contains the sequence, source string, and each Tensor's shape, dtype,
strides, and storage metadata. It never contains the pixel or calibration bytes. HoloMQ transfers
fixed-size descriptors and retains or imports the corresponding Tensor backing instead of
serializing Tensor data into the schema envelope.

The client in this portable host-only example starts from ordinary application-owned arrays, so
the imported boundary permits one bounded `kCopyIfNeeded` staging step when that backing is not
already HoloMQ-exportable. The graph retains the imported Tensor handles while changing only
ordinary metadata. Its export policy also permits bounded staging for portability, but compatible
HoloMQ-imported backing can be exported directly. After any required staging, HoloMQ transfers
only the fixed descriptor and shares the backing handle across the process boundary; it does not
serialize or copy Tensor bytes as part of message delivery.

## Run in two terminals

Build the example against an installed SDK:

```bash
cmake -S cpp -B build
cmake --build build -j
```

Then start the graph first:

```bash
./build/descriptor_tensor_external_topic_graph \
  --catalog /tmp/descriptor-topic-catalog.json \
  --instance demo
```

After it prints `catalog ready`, run the standalone peer with the same catalog and instance:

```bash
./build/descriptor_tensor_external_topic_client \
  --catalog /tmp/descriptor-topic-catalog.json \
  --instance demo
```

The client publishes a data frame, validates the exported frame, and sends a second acknowledged
frame so the graph can stop cleanly. Both processes then exit.

The `run_example.py` helper performs the same recipe with a unique endpoint instance and is used by
`EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_TEST`.
