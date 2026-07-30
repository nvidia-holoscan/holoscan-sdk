# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pub/sub video replayer example using the Fast-DDS backend.

Run the subscriber (display) in one terminal:
    python pubsub_video_replayer.py --role subscriber --config video_replayer.yaml

Run the publisher (replayer) in a second terminal:
    python pubsub_video_replayer.py --role publisher --config video_replayer.yaml
"""

import argparse
import os

from holoscan.conditions import PublisherAvailableCondition, SubscriberAvailableCondition
from holoscan.core import Application, IOSpec, Tracker
from holoscan.network_contexts import FastDdsPubSubNetworkContext
from holoscan.operators import HolovizOp, VideoStreamReplayerOp
from holoscan.resources import RMMAllocator
from holoscan.schedulers import EventBasedScheduler

try:
    from holoscan.conditions import PendingExportCondition
except ImportError:
    PendingExportCondition = None

TOPIC_NAME = "video_replayer_topic"
MAX_PENDING_NATIVE_EXPORTS = 4


class SubscriberApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        ctx = FastDdsPubSubNetworkContext(
            self,
            native_buffer_policy=self._args.native_buffer_policy,
            native_buffer_use_eager_acquire=self._args.eager,
            name="pubsub_context",
        )
        self.network_context(ctx)

        self.scheduler(
            EventBasedScheduler(
                self,
                worker_thread_number=2,
                stop_on_deadlock_timeout=300000,
                name="scheduler",
            )
        )

        pub_ready = PublisherAvailableCondition(
            self,
            receiver="receivers",
            poll_period_ms=100,
            latch_ready=True,
            name="publisher_available",
        )

        holoviz = HolovizOp(
            self,
            pub_ready,
            name="holoviz",
            **self.kwargs("holoviz"),
        )

        # Bind HolovizOp's "receivers" input to the pub/sub topic.
        inputs = holoviz.spec.inputs
        if "receivers" in inputs:
            inputs["receivers"].connector(
                IOSpec.ConnectorType.PUBSUB,
                topic_name=TOPIC_NAME,
                capacity=2,
                policy=0,  # pop
            )

        self.add_operator(holoviz)


class PublisherApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        ctx = FastDdsPubSubNetworkContext(
            self,
            native_buffer_policy=self._args.native_buffer_policy,
            name="pubsub_context",
        )
        self.network_context(ctx)

        self.scheduler(
            EventBasedScheduler(
                self,
                worker_thread_number=2,
                stop_on_deadlock_timeout=300000,
                name="scheduler",
            )
        )

        replayer_kwargs = dict(self.kwargs("replayer"))
        env_data_dir = os.environ.get("HOLOSCAN_INPUT_PATH")
        if env_data_dir:
            replayer_kwargs["directory"] = os.path.join(env_data_dir, "racerx")

        rmm_allocator = RMMAllocator(self, name="rmm_allocator", **self.kwargs("rmm_allocator"))

        sub_ready = SubscriberAvailableCondition(
            self,
            transmitter="output",
            min_subscriber_count=1,
            poll_period_ms=100,
            stabilization_ms=500,
            latch_ready=True,
            name="subscriber_available",
        )

        conditions = [sub_ready]
        if (
            PendingExportCondition is not None
            and not self._args.disable_pending_export_condition
            and self._args.native_buffer_policy != "disabled"
        ):
            pending_export_cond = PendingExportCondition(
                self,
                max_pending=MAX_PENDING_NATIVE_EXPORTS,
                network_context=ctx,
                name="pending_export_cond",
            )
            conditions.append(pending_export_cond)

        replayer = VideoStreamReplayerOp(
            self,
            *conditions,
            name="replayer",
            allocator=rmm_allocator,
            **replayer_kwargs,
        )

        # Bind the replayer's "output" port to the pub/sub topic.
        outputs = replayer.spec.outputs
        if "output" in outputs:
            outputs["output"].connector(
                IOSpec.ConnectorType.PUBSUB,
                topic_name=TOPIC_NAME,
            )

        self.add_operator(replayer)


def main():
    parser = argparse.ArgumentParser(description="Pub/sub video replayer example (Fast-DDS)")
    parser.add_argument(
        "--role", required=True, choices=["publisher", "subscriber"], help="Process role"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "video_replayer.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--native_buffer_policy",
        default="preferred",
        choices=["disabled", "preferred", "required"],
        help="Native buffer policy (default: preferred)",
    )
    parser.add_argument(
        "--eager", action="store_true", help="Subscriber-side: enable eager CUDA IPC acquire"
    )
    parser.add_argument("--track", action="store_true", help="Enable Data Flow Tracking output")
    parser.add_argument(
        "--disable_pending_export_condition",
        action="store_true",
        help="Publisher-side debug option: skip PendingExportCondition even when native buffers "
        "are enabled",
    )
    args = parser.parse_args()

    app = PublisherApp(args) if args.role == "publisher" else SubscriberApp(args)
    app.config(args.config)
    if args.track:
        with Tracker(app, num_start_messages_to_skip=0, num_last_messages_to_discard=0) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()


if __name__ == "__main__":
    main()
