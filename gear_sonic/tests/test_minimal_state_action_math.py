from __future__ import annotations

import math

import torch

from gear_sonic.envs.manager_env.mdp.commands import _repeated_eval_motion_ids
from gear_sonic.envs.manager_env.mdp.recorders import (
    _gravity_in_frame,
    _relative_heading_sin_cos,
)


def _yaw_quaternion(degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor(
        [[math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)]],
        dtype=torch.float32,
    )


def test_relative_heading_uses_reference_minus_robot_and_wraps_continuously():
    encoded = _relative_heading_sin_cos(_yaw_quaternion(170.0), _yaw_quaternion(-170.0))
    expected = torch.tensor(
        [[math.sin(math.radians(20.0)), math.cos(math.radians(20.0))]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(encoded, expected)


def test_identity_frame_keeps_world_down_direction():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    torch.testing.assert_close(_gravity_in_frame(identity, down), down.unsqueeze(0))


def test_repeated_eval_motion_ids_are_contiguous():
    expected = torch.arange(8, dtype=torch.long).repeat_interleave(4)
    torch.testing.assert_close(_repeated_eval_motion_ids(32, 4, "cpu"), expected)
