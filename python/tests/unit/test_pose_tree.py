"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import numpy as np
import pytest

from holoscan.pose_tree import Pose3, PoseTree, PoseTreeAccessMethod


@pytest.fixture
def tree():
    tree_instance = PoseTree()
    tree_instance.init()
    return tree_instance


class TestPoseTree:
    def test_init(self):
        # Test invalid initialization parameters
        with pytest.raises(RuntimeError):
            PoseTree().init(number_frames=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(number_edges=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(history_length=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(default_number_edges=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(default_history_length=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(edges_chunk_size=0)
        with pytest.raises(RuntimeError):
            PoseTree().init(history_chunk_size=0)

        # Test valid initialization
        tree = PoseTree()
        assert tree.init() is None

    def test_id(self):
        # Test frame creation and lookup multiple times
        for _i in range(10):
            tree = PoseTree()
            tree.init()

            # Get initial version
            version = tree.get_pose_tree_version()

            # Create two frames
            frame_a = tree.create_frame("a")
            frame_b = tree.create_frame("b")

            # Test that we can't create duplicate frames
            with pytest.raises(RuntimeError):
                tree.create_frame("a")
            with pytest.raises(RuntimeError):
                tree.create_frame("b")

            # Test frame lookup
            assert tree.find_frame("a") == frame_a
            assert tree.find_frame("b") == frame_b

            # Test non-existent frame lookup
            with pytest.raises(RuntimeError):
                tree.find_frame("aa")

            # Create new frame
            frame_aa = tree.create_frame("aa")
            assert frame_aa is not None
            # Version should not change until we set a pose
            assert tree.get_pose_tree_version() == version
            # Set a pose between frames
            pose = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
            assert tree.set("a", "b", 0.0, pose) is not None
            # Version should increase after setting pose
            assert tree.get_pose_tree_version() > version

            # Test identity poses for frames to themselves
            result = tree.get(frame_b, frame_b, 0.0)
            np.testing.assert_array_almost_equal(result.translation, np.array([0.0, 0.0, 0.0]))
            result = tree.get(frame_a, frame_a, 0.0)
            np.testing.assert_array_almost_equal(result.translation, np.array([0.0, 0.0, 0.0]))

    def test_create_edges(self):
        tree = PoseTree()
        tree.init(8, 256, 1024, 4, 4, 1, 1)

        # Create frames
        frame_a = tree.create_frame("a", 1)
        frame_b = tree.create_frame("b")
        frame_c = tree.create_frame("c")
        frame_d = tree.create_frame("d")
        frame_e = tree.create_frame("e")

        # Delete frame e
        tree.delete_frame(frame_e)

        # Test creating edges
        assert tree.create_edges(frame_a, frame_b) is not None

        # Test duplicate edges
        with pytest.raises(RuntimeError):
            tree.create_edges(frame_a, frame_b)
        with pytest.raises(RuntimeError):
            tree.create_edges(frame_b, frame_a)

        # Test creating edge with history length
        assert tree.create_edges(frame_b, frame_c, 16) is not None

        # Test out of memory error
        with pytest.raises(RuntimeError):
            tree.create_edges(frame_c, frame_d, 1024)

        # Test creating edge to deleted frame
        with pytest.raises(RuntimeError):
            tree.create_edges(frame_d, frame_e)

        # Test out of memory error
        with pytest.raises(RuntimeError):
            tree.create_edges(frame_d, frame_a)

        # Test valid edge creation
        assert tree.create_edges(frame_d, frame_c, 16) is not None

    def test_two_nodes(self, tree):
        # Create two frames and test basic pose operations
        frame1_id = tree.create_frame()
        frame2_id = tree.create_frame()

        # Set a pose between frames
        pose = Pose3.from_translation(np.array([1.0, 2.0, 3.0]))
        version = tree.set(frame1_id, frame2_id, 0.0, pose)
        assert version is not None

        # Get the pose back
        result = tree.get(frame1_id, frame2_id, 0.0)
        np.testing.assert_array_almost_equal(result.translation, pose.translation)

        # Test inverse relationship
        result_inverse = tree.get(frame2_id, frame1_id, 0.0)
        np.testing.assert_array_almost_equal(result_inverse.translation, -pose.translation)

    def test_three_nodes(self, tree):
        # Create three frames and test pose chain
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Set poses between frames
        pose_ab = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose_bc = Pose3.from_translation(np.array([0.0, 1.0, 0.0]))

        tree.set(a, b, 0.0, pose_ab)
        tree.set(b, c, 0.0, pose_bc)

        # Test composed transformation
        result = tree.get(a, c, 0.0)
        expected_translation = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

    def test_tree(self, tree):
        # Create a tree structure and test traversal from root to leaf
        depth = 10
        branching = 10

        # Create root node
        nodes = [[tree.create_frame()]]
        actual = Pose3.identity()

        # Build tree level by level
        for d in range(1, depth):
            nodes.append([])
            for b in range(branching):
                # Create child node
                nodes[d].append(tree.create_frame())

                # Create random pose from parent to child
                rng = np.random.default_rng()
                dx = rng.normal(0, 1)
                dy = rng.normal(0, 1)
                dz = rng.normal(0, 1)
                pose = Pose3.from_translation(np.array([dx, dy, dz]))

                # Connect to parent
                tree.set(nodes[d - 1][0], nodes[d][b], 0.0, pose)

                # Track actual transform along main branch
                if b == 0:
                    actual = actual @ pose

        # Test transform from root to leaf along main branch
        result = tree.get(nodes[0][0], nodes[depth - 1][0], 0.0)
        np.testing.assert_array_almost_equal(result.translation, actual.translation)
        # Test inverse transform
        result_inv = tree.get(nodes[depth - 1][0], nodes[0][0], 0.0)
        np.testing.assert_array_almost_equal(result_inv.translation, -actual.translation)

    def test_get_pose2_xy(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        # Set 3D pose with translation
        pose = Pose3.from_translation(np.array([1.0, 2.0, 3.0]))
        tree.set(a, b, 0.0, pose)

        # Test getting 2D projection
        result = tree.get_pose2_xy(a, b, 1.0)
        expected_translation = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

        # Test getting 2D projection using frame names
        result = tree.get_pose2_xy("a", "b", 1.0)
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

    def test_check_cycle(self, tree):
        # Create frames
        a = tree.create_frame()
        b = tree.create_frame()
        c = tree.create_frame()

        # Set poses to form chain a->b->c
        assert tree.set(a, b, 0.0, Pose3.identity())
        assert tree.set(b, c, 0.0, Pose3.identity())

        # Try to create cycle by connecting a->c directly
        # This should fail since it would create cycle a->b->c->a
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.0, Pose3.identity())

    def test_check_interpolation(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        # Set two poses at different times
        pose1 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose2 = Pose3.from_translation(np.array([3.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose1)
        tree.set(a, b, 1.0, pose2)

        # Test default interpolation at t=0.5
        result = tree.get(a, b, 0.5)
        expected = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test with frame names
        result = tree.get("a", "b", 0.5)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear interpolation at t=0.5
        result = tree.get(a, b, 0.5, method=PoseTreeAccessMethod.INTERPOLATE_LINEARLY)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear interpolation with names
        result = tree.get("a", "b", 0.5, method=PoseTreeAccessMethod.INTERPOLATE_LINEARLY)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear interpolation at t=1.5
        result = tree.get(a, b, 1.5, method=PoseTreeAccessMethod.INTERPOLATE_LINEARLY)
        expected = Pose3.from_translation(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear interpolation with names at t=1.5
        result = tree.get("a", "b", 1.5, method=PoseTreeAccessMethod.INTERPOLATE_LINEARLY)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear extrapolation at t=0.5
        result = tree.get(a, b, 0.5, method=PoseTreeAccessMethod.EXTRAPOLATE_LINEARLY)
        expected = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear extrapolation with names at t=0.5
        result = tree.get("a", "b", 0.5, method=PoseTreeAccessMethod.EXTRAPOLATE_LINEARLY)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear extrapolation at t=1.5
        result = tree.get(a, b, 1.5, method=PoseTreeAccessMethod.EXTRAPOLATE_LINEARLY)
        expected = Pose3.from_translation(np.array([4.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test linear extrapolation with names at t=1.5
        result = tree.get("a", "b", 1.5, method=PoseTreeAccessMethod.EXTRAPOLATE_LINEARLY)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test nearest at t=0.4
        result = tree.get(a, b, 0.4, method=PoseTreeAccessMethod.NEAREST)
        expected = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test nearest with names at t=0.4
        result = tree.get("a", "b", 0.4, method=PoseTreeAccessMethod.NEAREST)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test nearest at t=0.6
        result = tree.get(a, b, 0.6, method=PoseTreeAccessMethod.NEAREST)
        expected = Pose3.from_translation(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test nearest with names at t=0.6
        result = tree.get("a", "b", 0.6, method=PoseTreeAccessMethod.NEAREST)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

    def test_versioning(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        # Get initial version
        v1 = tree.get_pose_tree_version()

        # Set initial pose
        pose1 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose1)

        # Get version after first update
        v2 = tree.get_pose_tree_version()

        # Set new pose
        pose2 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        tree.set(a, b, 1.0, pose2)

        # Test that we can't get pose with old version
        with pytest.raises(RuntimeError):
            tree.get(a, b, 0.0, version=v1)

        # Test we can get pose with intermediate version
        result1 = tree.get(a, b, 1.0, version=v2)
        np.testing.assert_array_almost_equal(result1.translation, pose1.translation)
        result2 = tree.get(a, b, 1.0)
        np.testing.assert_array_almost_equal(result2.translation, pose2.translation)

    def test_ordered_updates(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        # Set initial pose
        pose1 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose1)

        # Set second pose
        pose2 = Pose3.from_translation(np.array([3.0, 0.0, 0.0]))
        tree.set(a, b, 1.0, pose2)

        # Try to set same timestamp again - should fail
        pose3 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        with pytest.raises(RuntimeError):
            tree.set(a, b, 1.0, pose3)

        # Verify poses
        result1 = tree.get(a, b, 1.0)
        np.testing.assert_array_almost_equal(result1.translation, pose2.translation)

        result2 = tree.get(b, a, 1.0)
        np.testing.assert_array_almost_equal(result2.translation, -pose2.translation)

        # Set later timestamp - should succeed
        pose4 = Pose3.from_translation(np.array([4.0, 0.0, 0.0]))
        tree.set(a, b, 2.0, pose4)

        # Try to set intermediate timestamp - should fail
        with pytest.raises(RuntimeError):
            tree.set(a, b, 1.5, Pose3.identity())

    def test_temporal_direct(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        # Set initial poses
        pose1 = Pose3.from_translation(np.array([0.7, -1.3, 2.0]))
        pose2 = Pose3.from_translation(np.array([-0.7, -1.6, 2.4]))

        tree.set(a, b, 0.0, pose1)
        tree.set(a, b, 1.0, pose2)

        # Test exact timestamps
        result1 = tree.get(a, b, 0.0)
        np.testing.assert_array_almost_equal(result1.translation, pose1.translation)

        result2 = tree.get(a, b, 1.0)
        np.testing.assert_array_almost_equal(result2.translation, pose2.translation)

        # Test extrapolation
        result3 = tree.get(a, b, 12.6)
        np.testing.assert_array_almost_equal(result3.translation, pose2.translation)

        # Test interpolation at t=0.5
        result4 = tree.get(a, b, 0.5)
        expected_translation = np.array([0.0, -1.45, 2.2])
        np.testing.assert_array_almost_equal(result4.translation, expected_translation)

        # Test interpolation at various points
        for t in [0.3, 0.6, 0.9]:
            result = tree.get(a, b, t)
            expected = Pose3.from_translation((1 - t) * pose1.translation + t * pose2.translation)
            np.testing.assert_array_almost_equal(result.translation, expected.translation)

    def test_temporal_indirect(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Set initial poses for a->b chain
        pose_ab0 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose_ab1 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose_ab0)
        tree.set(a, b, 1.0, pose_ab1)

        # Set initial poses for b->c chain
        pose_bc0 = Pose3.from_translation(np.array([0.0, 1.0, 0.0]))
        pose_bc1 = Pose3.from_translation(np.array([0.0, 2.0, 0.0]))
        tree.set(b, c, 0.0, pose_bc0)
        tree.set(b, c, 1.0, pose_bc1)

        # Test direct transforms at exact timestamps
        result = tree.get(a, b, 0.0)
        np.testing.assert_array_almost_equal(result.translation, pose_ab0.translation)
        result = tree.get(a, b, 1.0)
        np.testing.assert_array_almost_equal(result.translation, pose_ab1.translation)
        result = tree.get(a, b, 12.6)
        np.testing.assert_array_almost_equal(result.translation, pose_ab1.translation)

        result = tree.get(b, c, 0.0)
        np.testing.assert_array_almost_equal(result.translation, pose_bc0.translation)
        result = tree.get(b, c, 1.0)
        np.testing.assert_array_almost_equal(result.translation, pose_bc1.translation)
        result = tree.get(b, c, 12.6)
        np.testing.assert_array_almost_equal(result.translation, pose_bc1.translation)

        # Test interpolated transforms at various timestamps
        for t in [0.3, 0.5, 0.9]:
            # Test a->b interpolation
            result = tree.get(a, b, t)
            expected_ab = Pose3.from_translation(
                (1 - t) * pose_ab0.translation + t * pose_ab1.translation
            )
            np.testing.assert_array_almost_equal(result.translation, expected_ab.translation)

            # Test b->c interpolation
            result = tree.get(b, c, t)
            expected_bc = Pose3.from_translation(
                (1 - t) * pose_bc0.translation + t * pose_bc1.translation
            )
            np.testing.assert_array_almost_equal(result.translation, expected_bc.translation)

            # Test composed a->c interpolation
            result = tree.get(a, c, t)
            expected_ab = Pose3.from_translation(
                (1 - t) * pose_ab0.translation + t * pose_ab1.translation
            )
            expected_bc = Pose3.from_translation(
                (1 - t) * pose_bc0.translation + t * pose_bc1.translation
            )
            expected_ac = expected_ab @ expected_bc
            np.testing.assert_array_almost_equal(result.translation, expected_ac.translation)

    def test_get_latest(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Set poses at different times
        pose_ab0 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose_ab1 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose_ab0)
        tree.set(a, b, 2.0, pose_ab1)

        pose_bc0 = Pose3.from_translation(np.array([0.0, 1.0, 0.0]))
        pose_bc1 = Pose3.from_translation(np.array([0.0, 2.0, 0.0]))
        tree.set(b, c, 0.0, pose_bc0)
        tree.set(b, c, 2.5, pose_bc1)

        # Test getting latest poses
        result = tree.get_latest(a, b)
        assert result is not None
        assert result[1] == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(result[0].translation, pose_ab1.translation)

        result = tree.get_latest(b, a)
        assert result is not None
        assert result[1] == pytest.approx(2.0)
        np.testing.assert_array_almost_equal(result[0].translation, -pose_ab1.translation)

        result = tree.get_latest(b, c)
        assert result is not None
        assert result[1] == pytest.approx(2.5)
        np.testing.assert_array_almost_equal(result[0].translation, pose_bc1.translation)

        result = tree.get_latest(c, b)
        assert result is not None
        assert result[1] == pytest.approx(2.5)
        np.testing.assert_array_almost_equal(result[0].translation, -pose_bc1.translation)

        # Test getting latest for non-existent transforms
        with pytest.raises(RuntimeError):
            tree.get_latest(a, c)
        with pytest.raises(RuntimeError):
            tree.get_latest(a, a)

        # Test getting latest after many updates
        for i in range(10, 2049):
            pose = Pose3.from_translation(np.array([float(i), 0.0, 0.0]))
            tree.set(a, b, float(i), pose)
            result = tree.get_latest(a, b)
            assert result is not None
            assert result[1] == pytest.approx(float(i))

    def test_get_latest_edge(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Set initial poses
        pose_ab0 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose_ab1 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))
        pose_bc0 = Pose3.from_translation(np.array([0.0, 1.0, 0.0]))
        pose_bc1 = Pose3.from_translation(np.array([0.0, 2.0, 0.0]))

        tree.set(a, b, 0.0, pose_ab0)
        tree.set(a, b, 2.5, pose_ab1)
        tree.set(b, c, 0.0, pose_bc0)
        version = tree.get_pose_tree_version()
        tree.set(b, c, 2.0, pose_bc1)

        # Test direct edges
        result = tree.get(a, b)
        assert result is not None
        np.testing.assert_array_almost_equal(result.translation, pose_ab1.translation)

        result = tree.get(b, a)
        assert result is not None
        np.testing.assert_array_almost_equal(result.translation, -pose_ab1.translation)

        result = tree.get(b, c)
        assert result is not None
        np.testing.assert_array_almost_equal(result.translation, pose_bc1.translation)

        result = tree.get(c, b)
        assert result is not None
        np.testing.assert_array_almost_equal(result.translation, -pose_bc1.translation)

        # Test composed transform
        result = tree.get(a, c)
        assert result is not None
        expected = Pose3.from_translation(pose_ab1.translation + pose_bc1.translation)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test versioned query
        result = tree.get(a, c, version=version)
        assert result is not None
        expected = Pose3.from_translation(pose_ab1.translation + pose_bc0.translation)
        np.testing.assert_array_almost_equal(result.translation, expected.translation)

        # Test identity and invalid frames
        result = tree.get(a, a)
        assert result is not None
        np.testing.assert_array_almost_equal(result.translation, np.zeros(3))

    def test_disconnect_edge(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Set initial poses
        pose = Pose3.identity()
        tree.set(a, b, 0.0, pose)
        tree.set(b, c, 0.0, pose)

        # Cannot set direct edge between a and c initially
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.0, pose)

        # Cannot disconnect at t=0.0 (edge exists)
        with pytest.raises(RuntimeError):
            tree.disconnect_edge(a, b, 0.0)

        # Can disconnect at t=1.0
        assert tree.disconnect_edge(a, b, 1.0) is not None

        # Cannot disconnect again
        with pytest.raises(RuntimeError):
            tree.disconnect_edge(a, b, 1.0)

        # Check connectivity before disconnect time
        assert tree.get(a, c, 0.0) is not None
        assert tree.get(c, a, 0.0) is not None

        # Check connectivity after disconnect time
        with pytest.raises(RuntimeError):
            tree.get(a, c, 1.0)
        with pytest.raises(RuntimeError):
            tree.get(c, a, 1.0)

        # Cannot set edge before disconnect time
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.99, pose)

        # Can set edge at/after disconnect time
        assert tree.set(a, c, 1.0, pose) is not None

        # Test disconnecting indirect edge
        a = tree.create_frame("a2")
        b = tree.create_frame("b2")
        c = tree.create_frame("c2")
        tree.set(a, b, 0.0, pose)
        tree.set(b, c, 0.0, pose)

        # Cannot disconnect non-existent edge
        with pytest.raises(RuntimeError):
            tree.disconnect_edge(a, c, 1.0)

        # Cannot set direct edge
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.0, pose)

        # Can disconnect via c->b
        assert tree.disconnect_edge(c, b, 1.0) is not None

        # Check connectivity before disconnect
        assert tree.get(a, c, 0.0) is not None
        assert tree.get(c, a, 0.0) is not None

        # Check connectivity after disconnect
        with pytest.raises(RuntimeError):
            tree.get(a, c, 1.0)
        with pytest.raises(RuntimeError):
            tree.get(c, a, 1.0)

        # Cannot set edge before disconnect time
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.99, pose)

        # Can set edge at/after disconnect time
        assert tree.set(a, c, 1.0, pose) is not None

    def test_disconnect_frame(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        pose = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        tree.set(a, b, 0.0, pose)
        tree.set(b, c, 0.0, pose)

        # Cannot set direct edge between a and c when connected through b
        with pytest.raises(RuntimeError):
            tree.set(a, c, 0.0, pose)

        # Cannot disconnect frame at t=0.0
        with pytest.raises(RuntimeError):
            tree.disconnect_frame(b, 0.0)

        # Can disconnect frame at t=1.0
        assert tree.disconnect_frame(b, 1.0) is not None

        # Check connectivity before disconnect time
        assert tree.get(a, c, 0.0) is not None
        assert tree.get(c, a, 0.0) is not None

        # Check connectivity after disconnect time
        with pytest.raises(RuntimeError):
            tree.get(a, c, 1.0)
        with pytest.raises(RuntimeError):
            tree.get(c, a, 1.0)

        # Check direct connections to disconnected frame
        assert tree.get(a, b, 0.0) is not None
        with pytest.raises(RuntimeError):
            tree.get(a, b, 1.0)
        assert tree.get(b, a, 0.0) is not None
        with pytest.raises(RuntimeError):
            tree.get(b, a, 1.0)

        # Can set new edges after disconnect time
        assert tree.set(a, c, 1.01, pose) is not None
        assert tree.set(a, b, 1.01, pose) is not None

    def test_delete_frame(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        pose = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        tree.set(a, b, 1.0, pose)
        tree.set(b, c, 1.0, pose)

        # Cannot set direct edge between a and c when connected through b
        with pytest.raises(RuntimeError):
            tree.set(a, c, 1.0, pose)

        # Can get pose between a and c through b
        assert tree.get(a, c, 1.0) is not None

        # Delete frame b
        assert tree.delete_frame(b) is not None

        # Frame b no longer exists
        with pytest.raises(RuntimeError):
            tree.find_frame("b")

        # Cannot set pose using deleted frame
        with pytest.raises(RuntimeError):
            tree.set(a, b, 1.0, pose)

        # Connection between a and c is broken after deleting b
        with pytest.raises(RuntimeError):
            tree.get(a, c, 1.0)

        # Can now set direct edge between a and c
        assert tree.set(a, c, 1.0, pose) is not None

    def test_delete_edge(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        pose = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        tree.set(a, b, 1.0, pose)
        tree.set(b, c, 1.0, pose)

        # Cannot set direct edge between a and c when connected through b
        with pytest.raises(RuntimeError):
            tree.set(a, c, 1.0, pose)

        # Can get pose between a and c through b
        assert tree.get(a, c, 1.0) is not None

        # Delete edge between a and b
        assert tree.delete_edge(a, b) is not None

        # Edge b-c still exists
        assert tree.get(b, c, 1.0) is not None

        # Edge a-b no longer exists
        with pytest.raises(RuntimeError):
            tree.get(a, b, 1.0)

        # Connection between a and c is broken after deleting a-b
        with pytest.raises(RuntimeError):
            tree.get(a, c, 1.0)

        # Can now set direct edge between a and c
        assert tree.set(a, c, 1.0, pose) is not None

        # Cannot set b-c in past before deleting edge
        with pytest.raises(RuntimeError):
            tree.set(b, c, 0.0, pose)

        # Delete edge between b and c
        assert tree.delete_edge(c, b) is not None

        # Can now set b-c edge in past
        assert tree.set(b, c, 0.0, pose) is not None

    def test_disconnect_edge_and_queries(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")
        d = tree.create_frame("d")

        # Set initial poses
        tree.set(a, b, 0.0, Pose3.from_translation(np.array([1.0, 0.0, 0.0])))
        tree.set(b, c, 0.0, Pose3.from_translation(np.array([0.0, 1.0, 0.0])))
        tree.set(c, d, 0.0, Pose3.from_translation(np.array([0.0, 0.0, 1.0])))

        # Check initial pose chain
        result = tree.get(a, d, 1.0)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 1.0, 1.0]))

        # Disconnect b-c edge at t=1.0
        tree.disconnect_edge(b, c, 1.0)
        with pytest.raises(RuntimeError):
            tree.get(a, d, 1.0)

        # Add new b-c edge at t=2.0
        tree.set(b, c, 2.0, Pose3.from_translation(np.array([0.0, 2.0, 0.0])))
        tree.disconnect_edge(b, c, 3.0)
        tree.set(b, c, 4.0, Pose3.from_translation(np.array([0.0, 4.0, 0.0])))
        tree.set(b, c, 5.0, Pose3.from_translation(np.array([0.0, 5.0, 0.0])))
        tree.disconnect_edge(b, c, 6.0)
        tree.set(b, c, 7.0, Pose3.from_translation(np.array([0.0, 7.0, 0.0])))
        tree.disconnect_edge(b, c, 8.0)
        tree.set(b, c, 8.01, Pose3.from_translation(np.array([0.0, 8.0, 0.0])))

        # Check poses at various times
        result = tree.get(a, d, 2.5)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 2.0, 1.0]))

        with pytest.raises(RuntimeError):
            tree.get(a, d, 3.5)

        result = tree.get(a, d, 4.5)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 4.5, 1.0]))

        result = tree.get(a, d, 5.5)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 5.0, 1.0]))

        with pytest.raises(RuntimeError):
            tree.get(a, d, 6.5)

        result = tree.get(a, d, 7.99)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 7.0, 1.0]))

        result = tree.get(a, d, 8.01)
        np.testing.assert_array_almost_equal(result.translation, np.array([1.0, 8.0, 1.0]))

    def test_default_access(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Create edges with different default access methods
        tree.create_edges(a, b, 16, method=PoseTreeAccessMethod.NEAREST)
        tree.create_edges(b, c, method=PoseTreeAccessMethod.PREVIOUS)

        # Set poses
        tree.set(a, b, 0.0, Pose3.from_translation(np.array([1.0, 0.0, 0.0])))
        tree.set(a, b, 1.0, Pose3.from_translation(np.array([2.0, 0.0, 0.0])))
        tree.set(b, c, 0.0, Pose3.from_translation(np.array([1.0, 0.0, 0.0])))
        tree.set(b, c, 1.0, Pose3.from_translation(np.array([2.0, 0.0, 0.0])))

        # Check poses using default access methods
        result_ab = tree.get(a, b, 0.6)
        result_bc = tree.get(b, c, 0.6)

        # a-b uses NEAREST so should get pose at t=1.0
        np.testing.assert_array_almost_equal(result_ab.translation, np.array([2.0, 0.0, 0.0]))

        # b-c uses PREVIOUS so should get pose at t=0.0
        np.testing.assert_array_almost_equal(result_bc.translation, np.array([1.0, 0.0, 0.0]))

    def test_get_content(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")
        c = tree.create_frame("c")

        # Create edges with different access methods
        tree.create_edges(a, b, 16, method=PoseTreeAccessMethod.NEAREST)
        tree.create_edges(b, c, method=PoseTreeAccessMethod.PREVIOUS)

        # Set some poses
        tree.set(a, b, 1.0, Pose3.from_translation(np.array([2.0, 0.0, 0.0])))
        tree.set(b, c, 0.0, Pose3.from_translation(np.array([1.0, 0.0, 0.0])))

        # Test getting content with empty vectors

        frames = tree.get_frame_uids()
        frame_names = tree.get_frame_names()
        edges = tree.get_edge_uids()
        edge_names = tree.get_edge_names()

        assert len(frames) == 3
        assert len(frame_names) == 3
        assert len(edges) == 2
        assert len(edge_names) == 2

    def test_access_methods(self, tree):
        a = tree.create_frame("a")
        b = tree.create_frame("b")

        pose1 = Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose2 = Pose3.from_translation(np.array([2.0, 0.0, 0.0]))

        tree.set(a, b, 0.0, pose1)
        tree.set(a, b, 1.0, pose2)

        # Test different access methods
        result_nearest = tree.get(a, b, 0.7, method=PoseTreeAccessMethod.NEAREST)
        result_previous = tree.get(a, b, 0.7, method=PoseTreeAccessMethod.PREVIOUS)
        result_linear = tree.get(a, b, 0.7, method=PoseTreeAccessMethod.INTERPOLATE_LINEARLY)

        # Nearest should return pose2 (closer to t=1.0)
        np.testing.assert_array_almost_equal(result_nearest.translation, pose2.translation)

        # Previous should return pose1
        np.testing.assert_array_almost_equal(result_previous.translation, pose1.translation)

        # Linear interpolation should return intermediate value
        expected_translation = np.array([1.7, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result_linear.translation, expected_translation)
