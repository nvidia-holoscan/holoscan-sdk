# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch interoperability coverage for standalone Tensor channel clients."""

import uuid

import holoscan_test_support
import pytest

import holoscan as hs


# REQ: PY-CHANNEL-007
# REQ: TENSOR-PY-017
def test_catalog_tensor_dlpack_copy_from_torch():
    """Copy a PyTorch DLPack producer through catalog-attached Tensor clients."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    # Key setup: Attach a host-Tensor publisher and subscriber using independent catalogs.
    endpoint_name = f"python-channel-torch-{uuid.uuid4().hex}"
    publisher_json, subscriber_json = holoscan_test_support._tensor_client_catalogs(
        endpoint_name, "host"
    )
    publisher_catalog = hs.topics.EndpointCatalog.parse(publisher_json)
    subscriber_catalog = hs.topics.EndpointCatalog.parse(subscriber_json)

    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(publisher_catalog, endpoint_name)
        subscriber = runtime.subscriber(subscriber_catalog, endpoint_name)
        publisher.wait_for_subscribers(2.0)
        subscriber.refresh_connections()

        value = torch.full((2, 4), 12, dtype=torch.uint8)

        # Key assertion: A PyTorch DLPack producer follows the bounded Tensor copy path and arrives
        # as a read-only SDK Tensor.
        assert publisher.emit(value, timeout=1.0) == 1
        sample = subscriber.receive(1.0)
        assert sample.data.is_read_only
        np.testing.assert_array_equal(np.asarray(sample.data), value.numpy())
