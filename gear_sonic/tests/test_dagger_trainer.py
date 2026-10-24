"""CPU-only contract tests for the standalone DAgger trainer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from gear_sonic.trl.trainer.dagger_trainer import (
    DaggerRound,
    DaggerTrainer,
    _adapt_std_parameterization,
)


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _Motion:
    g1_encoder_index = 0
    encoder_index = torch.tensor([[1, 0], [1, 0]])
    motion_ids = torch.tensor([3, 4])


class _Commands:
    def get_term(self, _name):
        return _Motion()


class _InnerEnv:
    action_space = SimpleNamespace(shape=(1,))
    command_manager = _Commands()


class _FakeEnv:
    num_envs = 2
    env = _InnerEnv()
    action_transform_module = None
    config = _Cfg(meta_action_dim=None)

    def __init__(self):
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return {
            "actor_obs": torch.zeros(2, 3),
            "teacher": torch.zeros(2, 2),
            "tokenizer": torch.zeros(2, 1),
            "critic_obs": torch.zeros(2, 1),
        }

    def step(self, _actions):
        self.step_index += 1
        done = torch.tensor([self.step_index % 2 == 0, False])
        return self.reset_obs(), torch.ones(2), done, {}

    def reset_obs(self):
        return {
            "actor_obs": torch.ones(2, 3) * self.step_index,
            "teacher": torch.zeros(2, 2),
            "tokenizer": torch.zeros(2, 1),
            "critic_obs": torch.zeros(2, 1),
        }


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_actions = 1
        self.weight = nn.Parameter(torch.zeros(3, 1))

    def forward(self, obs):
        return obs["actor_obs"] @ self.weight


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1))

    def act_inference(self, obs_dict, cur_dones=None):  # noqa: ARG002
        return torch.ones(2, 1) * self.bias

    def init_rollout(self):
        pass

    def clear_rollout(self):
        pass


class DaggerTrainerTest(unittest.TestCase):
    def test_std_parameterization_conversion(self):
        model = nn.Module()
        model.register_parameter("std", nn.Parameter(torch.ones(2)))
        converted = _adapt_std_parameterization(model, {"std": torch.ones(2) * 2.0})
        self.assertIn("std", converted)
        self.assertNotIn("log_std", converted)

    def test_round_rejects_mismatched_sample_counts(self):
        with self.assertRaises(ValueError):
            DaggerRound(
                student_obs=torch.zeros(2, 3),
                teacher_action=torch.zeros(1, 1),
                student_action=torch.zeros(2, 1),
                executed_action=torch.zeros(2, 1),
                done=torch.zeros(2, dtype=torch.bool),
                episode_id=torch.zeros(2, dtype=torch.long),
                timestep=torch.zeros(2, dtype=torch.long),
                motion_id=torch.zeros(2, dtype=torch.long),
                mixture_teacher=torch.zeros(2, dtype=torch.bool),
            )

    def test_cpu_smoke_collects_and_marks_round(self):
        cfg = _Cfg(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            rollout_steps=3,
            train_batch_size=4,
            train_steps_per_round=2,
            beta_schedule=[1.0],
            num_rounds=1,
            action_clip_value=20.0,
            teacher_encoder="g1",
            smoke=True,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "teacher.pt"
            torch.save({"policy_state_dict": _Teacher().state_dict()}, checkpoint)
            trainer = DaggerTrainer(
                env=_FakeEnv(),
                student=_Student(),
                teacher=_Teacher(),
                config=cfg,
                run_dir=temp_dir,
                device=torch.device("cpu"),
                teacher_checkpoint=checkpoint,
            )
            metrics = trainer.run()
            self.assertEqual(metrics["rounds"][0]["dataset_samples"], 6.0)
            self.assertTrue((Path(temp_dir) / "dagger_smoke.ok").is_file())


if __name__ == "__main__":
    unittest.main()
