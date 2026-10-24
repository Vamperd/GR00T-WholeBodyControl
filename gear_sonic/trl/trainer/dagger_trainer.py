"""Standalone DAgger trainer for a frozen SONIC teacher and an MLP student.

This module deliberately does not reuse PPO's ``ref_model``.  The teacher is
queried on observations visited by the student, while the student is trained
with ordinary supervised action regression on the aggregated data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


@dataclass
class DaggerRound:
    """One immutable round of DAgger samples kept on CPU."""

    student_obs: torch.Tensor
    teacher_action: torch.Tensor
    student_action: torch.Tensor
    executed_action: torch.Tensor
    done: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    motion_id: torch.Tensor
    mixture_teacher: torch.Tensor

    def __post_init__(self) -> None:
        count = self.student_obs.shape[0]
        tensors = (
            self.teacher_action,
            self.student_action,
            self.executed_action,
            self.done,
            self.episode_id,
            self.timestep,
            self.motion_id,
            self.mixture_teacher,
        )
        if any(t.shape[0] != count for t in tensors):
            raise ValueError("all DAgger round tensors must have the same sample count")
        if self.student_obs.ndim != 2:
            raise ValueError(f"student_obs must be [N,D], found {tuple(self.student_obs.shape)}")
        for name, tensor in (
            ("student_obs", self.student_obs),
            ("teacher_action", self.teacher_action),
            ("student_action", self.student_action),
            ("executed_action", self.executed_action),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite values found in {name}")

    @property
    def count(self) -> int:
        return int(self.student_obs.shape[0])


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("policy_state_dict", "actor_model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint):
            return checkpoint
    raise ValueError("checkpoint does not contain a policy state dict")


def _adapt_std_parameterization(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert legacy ``std``/``log_std`` state to the model's parameterization."""

    model_state = model.state_dict()
    has_model_std = "std" in model_state
    has_model_log_std = "log_std" in model_state
    result = dict(state_dict)
    if has_model_std and "log_std" in result and "std" not in result:
        result["std"] = torch.exp(result.pop("log_std"))
    elif has_model_log_std and "std" in result and "log_std" not in result:
        result["log_std"] = torch.log(result.pop("std"))
    return result


def load_policy_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> str:
    """Load a policy checkpoint and return its SHA256 digest."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"policy checkpoint is missing: {checkpoint_path}")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _adapt_std_parameterization(model, _checkpoint_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class DaggerTrainer:
    """Collect and train DAgger rounds for vectorized ManagerEnvWrapper."""

    def __init__(
        self,
        *,
        env: Any,
        student: nn.Module,
        teacher: nn.Module,
        config: Any,
        run_dir: str | Path,
        device: torch.device,
        teacher_checkpoint: str | Path,
        student_checkpoint: str | Path | None = None,
    ) -> None:
        self.env = env
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data" / "dagger").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "manifests").mkdir(parents=True, exist_ok=True)

        self.teacher_checkpoint = Path(teacher_checkpoint).expanduser()
        self.teacher_checkpoint_sha256 = load_policy_checkpoint(
            self.teacher, self.teacher_checkpoint, self.device
        )
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.student_checkpoint_sha256: str | None = None
        if student_checkpoint:
            self.student_checkpoint_sha256 = load_policy_checkpoint(
                self.student, student_checkpoint, self.device
            )

        self._freeze_student_noise()
        self.rounds: list[DaggerRound] = []
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.get("weight_decay", 1.0e-4)),
        )
        self.num_envs = int(env.num_envs)
        self.action_dim = int(env.env.action_space.shape[-1])
        self._check_action_contract()

        self._write_manifest(
            {
                "teacher_checkpoint": str(self.teacher_checkpoint.resolve()),
                "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
                "student_checkpoint_sha256": self.student_checkpoint_sha256,
                "teacher_encoder": str(config.get("teacher_encoder", "g1")),
                "num_envs": self.num_envs,
                "action_dim": self.action_dim,
                "run_dir": str(self.run_dir.resolve()),
                "source_commit": self._source_commit(),
            }
        )

    def _freeze_student_noise(self) -> None:
        for name in ("std", "log_std"):
            parameter = getattr(self.student, name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(False)

    def _check_action_contract(self) -> None:
        if int(self.student.num_actions) != self.action_dim:
            raise ValueError(
                "student action dimension does not match env action space: "
                f"student={self.student.num_actions}, env={self.action_dim}"
            )
        if getattr(self.env, "action_transform_module", None) is not None:
            raise ValueError(
                "standalone DAgger requires direct environment actions; "
                "action transform is enabled"
            )
        if self.env.config.get("meta_action_dim", None) is not None:
            raise ValueError("standalone DAgger does not support meta_action_dim")

    @staticmethod
    def _source_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    def _write_manifest(self, values: dict[str, Any]) -> None:
        manifest_path = self.run_dir / "manifests" / "dagger_manifest.json"
        payload = {}
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update(_jsonable(values))
        payload["config"] = _jsonable(self.config)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _validate_g1_encoder(self) -> None:
        if str(self.config.get("teacher_encoder", "g1")) != "g1":
            raise ValueError("the standalone release teacher currently supports only teacher_encoder=g1")
        motion = self.env.env.command_manager.get_term("motion")
        g1_index = getattr(motion, "g1_encoder_index", None)
        encoder_index = getattr(motion, "encoder_index", None)
        if g1_index is None or encoder_index is None:
            raise RuntimeError("motion command does not expose encoder_index/g1_encoder_index")
        if not torch.all(encoder_index[:, g1_index] == 1):
            raise RuntimeError("not every environment is assigned to the G1 encoder")
        if not torch.all(encoder_index.sum(dim=-1) == 1):
            raise RuntimeError("teacher encoder selection is not one-hot")

    def _motion_ids(self) -> torch.Tensor:
        motion = self.env.env.command_manager.get_term("motion")
        value = getattr(motion, "motion_ids", None)
        if value is None:
            return torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        return value.detach().clone().to(device=self.device, dtype=torch.long)

    def _student_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        student_obs = obs["actor_obs"]
        if student_obs.ndim != 2:
            raise ValueError(f"student actor_obs must be [B,D], found {tuple(student_obs.shape)}")
        with torch.no_grad():
            action = self.student.forward({"actor_obs": student_obs})
        if action.ndim != 2:
            raise ValueError(f"student action must be [B,A], found {tuple(action.shape)}")
        return action

    def _teacher_action(self, obs: dict[str, torch.Tensor], done: torch.Tensor) -> torch.Tensor:
        if "teacher" not in obs:
            raise ValueError("teacher observation group is missing from the environment output")
        teacher_obs = dict(obs)
        # UniversalTokenModule uses actor_obs to infer batch/sequence shape,
        # while its release proprioception is supplied by the teacher group.
        # This keeps the teacher's original input contract separate from the
        # larger G1-reference actor_obs seen by the MLP student.
        teacher_obs["actor_obs"] = obs["teacher"]
        with torch.no_grad():
            action = self.teacher.act_inference(obs_dict=teacher_obs, cur_dones=done)
        if action.ndim != 2:
            raise ValueError(f"teacher action must be [B,A], found {tuple(action.shape)}")
        return action

    def collect_round(self, round_index: int, beta: float) -> DaggerRound:
        obs = self.env.reset()
        if not isinstance(obs, dict) or "actor_obs" not in obs:
            raise RuntimeError("environment reset did not return a flattened actor_obs dictionary")
        self._validate_g1_encoder()
        self.student.eval()
        self.teacher.eval()
        if hasattr(self.teacher, "init_rollout"):
            self.teacher.init_rollout()

        steps = int(self.config.rollout_steps)
        beta = float(beta)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0,1], found {beta}")
        mixture_teacher = torch.rand(self.num_envs, device=self.device) < beta
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        episode_id = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        next_episode_id = self.num_envs
        timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        buffers: dict[str, list[torch.Tensor]] = {
            "student_obs": [],
            "teacher_action": [],
            "student_action": [],
            "executed_action": [],
            "done": [],
            "episode_id": [],
            "timestep": [],
            "motion_id": [],
            "mixture_teacher": [],
        }

        for _ in range(steps):
            if done.any():
                done_ids = done.nonzero(as_tuple=False).flatten()
                mixture_teacher[done_ids] = torch.rand(len(done_ids), device=self.device) < beta
                episode_id[done_ids] = torch.arange(
                    next_episode_id, next_episode_id + len(done_ids), device=self.device
                )
                next_episode_id += len(done_ids)
                timestep[done_ids] = 0

            student_obs = obs["actor_obs"].detach()
            teacher_action = self._teacher_action(obs, done)
            student_action = self._student_action(obs)
            if teacher_action.shape != student_action.shape:
                raise ValueError(
                    "teacher/student action shape mismatch: "
                    f"teacher={tuple(teacher_action.shape)}, student={tuple(student_action.shape)}"
                )
            clip_value = self.config.get("action_clip_value", None)
            if clip_value is not None and float(clip_value) > 0:
                clip_value = float(clip_value)
                teacher_action = torch.clamp(teacher_action, -clip_value, clip_value)
                student_action = torch.clamp(student_action, -clip_value, clip_value)
            executed_action = torch.where(
                mixture_teacher[:, None], teacher_action, student_action
            )

            motion_id = self._motion_ids()
            next_obs, _rewards, next_done, _infos = self.env.step({"actions": executed_action})
            next_done = next_done.to(device=self.device).bool()

            buffers["student_obs"].append(student_obs.cpu())
            buffers["teacher_action"].append(teacher_action.cpu())
            buffers["student_action"].append(student_action.cpu())
            buffers["executed_action"].append(executed_action.cpu())
            buffers["done"].append(next_done.cpu())
            buffers["episode_id"].append(episode_id.cpu())
            buffers["timestep"].append(timestep.cpu())
            buffers["motion_id"].append(motion_id.cpu())
            buffers["mixture_teacher"].append(mixture_teacher.cpu())

            timestep += 1
            done = next_done
            obs = next_obs

        if hasattr(self.teacher, "clear_rollout"):
            self.teacher.clear_rollout()

        result = DaggerRound(
            student_obs=torch.cat(buffers["student_obs"], dim=0),
            teacher_action=torch.cat(buffers["teacher_action"], dim=0),
            student_action=torch.cat(buffers["student_action"], dim=0),
            executed_action=torch.cat(buffers["executed_action"], dim=0),
            done=torch.cat(buffers["done"], dim=0),
            episode_id=torch.cat(buffers["episode_id"], dim=0),
            timestep=torch.cat(buffers["timestep"], dim=0),
            motion_id=torch.cat(buffers["motion_id"], dim=0),
            mixture_teacher=torch.cat(buffers["mixture_teacher"], dim=0),
        )
        shard_path = self.run_dir / "data" / "dagger" / f"round_{round_index:02d}.pt"
        torch.save(result.__dict__, shard_path)
        return result

    def _all_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rounds:
            raise RuntimeError("cannot train student before collecting a DAgger round")
        observations = torch.cat([item.student_obs for item in self.rounds], dim=0)
        actions = torch.cat([item.teacher_action for item in self.rounds], dim=0)
        return observations, actions

    def _freeze_normalizer_after_update(self, observations: torch.Tensor) -> None:
        normalizer = getattr(self.student, "running_mean_std", None)
        if normalizer is None:
            return
        self.student.train()
        normalizer.unfreeze()
        with torch.no_grad():
            for chunk in observations.split(int(self.config.train_batch_size)):
                normalizer(chunk.to(self.device))
        normalizer.freeze()

    def train_student(self, round_index: int) -> dict[str, float]:
        observations, targets = self._all_samples()
        self._freeze_normalizer_after_update(observations)
        self.student.train()
        batch_size = min(int(self.config.train_batch_size), observations.shape[0])
        steps = int(self.config.train_steps_per_round)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.get("seed", 0)) + round_index)
        losses: list[float] = []
        for _ in range(steps):
            indices = torch.randint(
                observations.shape[0], (batch_size,), generator=generator, device="cpu"
            )
            batch_obs = observations[indices].to(self.device)
            batch_targets = targets[indices].to(self.device)
            prediction = self.student.forward({"actor_obs": batch_obs})
            loss = torch.mean(torch.square(prediction - batch_targets))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.student.eval()
        with torch.no_grad():
            latest_prediction = self.student.forward(
                {"actor_obs": self.rounds[-1].student_obs.to(self.device)}
            ).cpu()
        latest_target = self.rounds[-1].teacher_action
        action_rmse = torch.sqrt(torch.mean(torch.square(latest_prediction - latest_target))).item()
        metrics = {
            "round": float(round_index),
            "dataset_samples": float(observations.shape[0]),
            "train_loss_last": losses[-1] if losses else float("nan"),
            "train_loss_mean_last_100": sum(losses[-100:]) / max(1, len(losses[-100:])),
            "latest_round_teacher_action_rmse": float(action_rmse),
        }
        checkpoint = {
            "policy_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "round": round_index,
            "metrics": metrics,
        }
        torch.save(checkpoint, self.run_dir / f"round_{round_index:02d}.pt")
        return metrics

    @torch.no_grad()
    def evaluate_student_only(self, name: str, steps: int) -> dict[str, float]:
        obs = self.env.reset()
        self.student.eval()
        done_count = 0
        action_saturation = 0
        total_actions = 0
        reward_sum = 0.0
        info_sums: dict[str, float] = {}
        info_counts: dict[str, int] = {}
        for _ in range(int(steps)):
            action = self._student_action(obs)
            clip_value = self.config.get("action_clip_value", None)
            if clip_value is not None and float(clip_value) > 0:
                action_saturation += int((action.abs() >= float(clip_value)).sum().item())
                total_actions += int(action.numel())
                action = torch.clamp(action, -float(clip_value), float(clip_value))
            obs, rewards, done, infos = self.env.step({"actions": action})
            done_count += int(done.to(torch.bool).sum().item())
            reward_sum += float(rewards.detach().mean().cpu())
            episode_info = infos.get("episode", {}) if isinstance(infos, dict) else {}
            if isinstance(episode_info, dict):
                for key, value in episode_info.items():
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        scalar = float(value.detach().float().mean().cpu())
                    elif isinstance(value, (int, float)):
                        scalar = float(value)
                    else:
                        continue
                    info_sums[str(key)] = info_sums.get(str(key), 0.0) + scalar
                    info_counts[str(key)] = info_counts.get(str(key), 0) + 1
        metrics = {
            "steps": float(steps),
            "done_count": float(done_count),
            "mean_reward": reward_sum / max(1, int(steps)),
            "action_saturation_ratio": action_saturation / max(1, total_actions),
        }
        metrics.update(
            {
                f"episode.{key}.mean": info_sums[key] / info_counts[key]
                for key in info_sums
                if info_counts.get(key, 0) > 0
            }
        )
        (self.run_dir / "manifests" / f"student_only_{name}.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        return metrics

    def run(self) -> dict[str, Any]:
        beta_schedule = [float(value) for value in self.config.beta_schedule]
        num_rounds = int(self.config.num_rounds)
        if len(beta_schedule) != num_rounds:
            raise ValueError(
                f"beta_schedule length {len(beta_schedule)} does not match num_rounds {num_rounds}"
            )
        all_metrics: dict[str, Any] = {
            "rounds": [],
            "student_only": {},
            "quality_pass": None,
            "quality_note": "execution metrics are recorded; formal tracking quality requires pilot review",
        }
        for round_index, beta in enumerate(beta_schedule):
            round_data = self.collect_round(round_index, beta)
            self.rounds.append(round_data)
            metrics = self.train_student(round_index)
            all_metrics["rounds"].append(metrics)
            all_metrics["student_only"][f"round_{round_index:02d}"] = self.evaluate_student_only(
                f"round_{round_index:02d}", int(self.config.rollout_steps)
            )

        (self.run_dir / "manifests" / "dagger_metrics.json").write_text(
            json.dumps(_jsonable(all_metrics), indent=2, sort_keys=True), encoding="utf-8"
        )
        self._write_manifest(
            {
                "beta_schedule": beta_schedule,
                "round_sample_counts": [item.count for item in self.rounds],
                "rollout_steps": int(self.config.rollout_steps),
                "execution_pass": True,
            }
        )
        marker_name = "dagger_smoke.ok" if bool(self.config.get("smoke", False)) else "dagger_execution.ok"
        (self.run_dir / marker_name).write_text("execution_pass=true\n", encoding="utf-8")
        return all_metrics
