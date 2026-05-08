# PubSub Video Replayer

A pub/sub adaptation of the single-process `video_replayer` example using the Fast-DDS
backend. Both C++ and Python implementations are provided.

- The **publisher** process runs a `VideoStreamReplayerOp` and broadcasts decoded video
  frames on the `video_replayer_topic` pub/sub topic.  It waits for at least one matched
  subscriber before starting playback.
- Each **subscriber** process receives frames from the topic and displays them via
  `HolovizOp`.

## Prerequisites

The racerx sample dataset must be available.  Set the `HOLOSCAN_INPUT_PATH` environment
variable to its parent directory (the replayer will look for `$HOLOSCAN_INPUT_PATH/racerx`),
or edit the `replayer.directory` key in `video_replayer.yaml`.

## C++ Binary

`pubsub_video_replayer`

### Usage

Launch one or more subscribers first (each opens its own HolovizOp window):

```bash
./pubsub_video_replayer --role subscriber
```

Then start the publisher:

```bash
./pubsub_video_replayer --role publisher
```

## Python Script

`pubsub_video_replayer.py`

### Usage

Launch the subscriber in one terminal:

```bash
python pubsub_video_replayer.py --role subscriber --config video_replayer.yaml
```

Launch the publisher in a second terminal:

```bash
python pubsub_video_replayer.py --role publisher --config video_replayer.yaml
```

## Options

| Option | Description |
|--------|-------------|
| `--role ROLE` | `publisher` or `subscriber` (required) |
| `--config PATH` | Path to the YAML config file (default: `video_replayer.yaml` next to the binary/script) |
| `--native_buffer_policy POLICY` | `disabled`, `preferred`, or `required` (default: `preferred`) |
| `--eager` | Subscriber-side: enable `holoipc` `acquire_pointer_eager()` for CUDA IPC import |
| `--disable_pending_export_condition` | Publisher-side debug option: skip PendingExportCondition |
| `--track` | Enable Holoscan data flow tracking |

## Runtime Topology

```text
Terminal 1  PublisherApp
              replayer (VideoStreamReplayerOp)
                └─ output ──► topic video_replayer_topic

Terminal 2  SubscriberApp
              holoviz (HolovizOp)
                └─ receivers ◄── topic video_replayer_topic
```

Multiple subscriber terminals can be connected simultaneously.

## YAML Configuration

`video_replayer.yaml` contains configuration for both roles:

- `replayer` -- `VideoStreamReplayerOp` settings used by the publisher.
- `rmm_allocator` -- memory pool for the publisher's video decoder.
- `holoviz` -- `HolovizOp` display settings used by each subscriber.

Refer to the file for available keys and defaults.

## Notes

- The publisher loops the recording by default (`repeat: true`).  Subscribers that connect
  after playback starts will see frames on the next loop.
- Discovery is handled by the DDS middleware itself (no additional setup needed).
- `--eager` is a subscriber-side option that enables `holoipc` eager CUDA IPC import
  on the subscriber and is ignored by the publisher role.
- The C++ and Python publishers/subscribers are interoperable -- you can mix them
  (e.g. C++ publisher with Python subscriber).
- When native buffers are enabled (the default `preferred` policy), the publisher attaches
  a `PendingExportCondition` to throttle execution while CUDA IPC exports are in flight.
  This is needed on the zero-copy native buffer (CUDA IPC) path to prevent GPU memory
  pool exhaustion when the publisher runs faster than the subscriber can consume.  It does
  not apply to the host-buffer byte path, which copies data and releases pool memory
  immediately.  For `BlockMemoryPool`, which has a fixed number of blocks,
  `MemoryAvailableCondition` is a simpler alternative that blocks until a block is free.
  Use `--disable_pending_export_condition` to bypass this for debugging.
