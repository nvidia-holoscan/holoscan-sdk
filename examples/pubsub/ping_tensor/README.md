# PubSub Ping Tensor

This variant of `ping_tensor` uses Fast-DDS pub/sub to transmit tensors between separate
publisher and subscriber processes. Run the same binary (or script) from separate terminals
and choose a role with `--role`.

Both C++ and Python implementations are provided.

## C++ Binary

`pubsub_ping_tensor`

### Usage

Launch the subscriber in one terminal:

```bash
./pubsub_ping_tensor --role subscriber
```

To opt into eager CUDA IPC acquire on the subscriber:

```bash
./pubsub_ping_tensor --role subscriber --eager
```

Launch the publisher in a second terminal:

```bash
./pubsub_ping_tensor --role publisher --gpu --count 10
```

## Python Script

`pubsub_ping_tensor.py`

### Usage

Launch the subscriber in one terminal:

```bash
python pubsub_ping_tensor.py --role subscriber
```

Launch the publisher in a second terminal:

```bash
python pubsub_ping_tensor.py --role publisher --gpu --count 10
```

## Options

| Option | Description |
|--------|-------------|
| `--role ROLE` | `publisher` or `subscriber` (required) |
| `--gpu` | Place tensors in GPU memory |
| `--count COUNT` | Message count: publisher send count or subscriber receive target (default: 10, negative = indefinite) |
| `--batch_size BATCH` | Batch size of the tensor (dimension omitted if 0) |
| `--rows ROWS` | Number of rows in the tensor (default: 32) |
| `--columns COLUMNS` | Number of columns in the tensor (default: 64) |
| `--channels CHANNELS` | Number of channels in the tensor (dimension omitted if 0) |
| `--data_type TYPE` | Tensor element type (default: `uint8_t` in C++, `uint8` in Python) |
| `--native_buffer_policy POLICY` | `disabled`, `preferred`, or `required` (default: `preferred`) |
| `--eager` | Subscriber-side: enable `holoipc` `acquire_pointer_eager()` for CUDA IPC import |
| `--track` | Enable Data Flow Tracking output |
| `--tx_period_ms MS` | Publisher delay between tensor messages (default: 100 ms) |

## Runtime Topology

Run one subscriber process and one publisher process in separate terminals. The publisher waits
for at least one matched subscriber before emitting tensors on `ping_tensor_topic`.

```text
Terminal 1
  SubscriberApp
    rx <- topic ping_tensor_topic

Terminal 2
  PublisherApp
    tick -> tx -> topic ping_tensor_topic
```

## Notes

- The example uses pub/sub connectors with topic name `ping_tensor_topic`.
- Discovery is handled by the DDS middleware itself (no additional setup needed).
- `--eager` is a subscriber-side option that enables `holoipc` eager CUDA IPC import
  on the subscriber and is ignored by the publisher role.
- The C++ and Python publishers/subscribers are interoperable -- you can mix them
  (e.g. C++ publisher with Python subscriber).
