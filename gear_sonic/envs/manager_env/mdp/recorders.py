"""Custom recorder terms for the manager environment MDP."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import TYPE_CHECKING

import cv2
import imageio
from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from isaaclab import envs


@configclass
class RecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    render_envs = None
    running_ref_root_height = None
    trajectory = None
    minimal_state_t = None
    minimal_goal_t = None
    minimal_actions = None
    minimal_state_tp1 = None
    minimal_outcome = None
    minimal_motion = None
    minimal_context_t = None
    minimal_metadata = None
    physics_state_t = None
    physics_actions = None
    physics_state_tp1 = None
    physics_diagnostics = None
    physics_metadata = None


class RenderEnvsRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term for rendering environments with advanced features like text overlay and frame skipping."""

    cfg: RenderEnvsRecorderCfg

    def __init__(self, cfg: RenderEnvsRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        # Determine save directory (backward compatibility)
        self.save_dir = self.cfg.video_save_path
        logger.info(f"=== Start recording video to {self.save_dir} ===")

        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.video_writers = []
        self._writers_closed = False
        self.frame_id = 0
        self.first_render = True
        self._fixed_eye = None
        self._fixed_target = None

    def _initialize_writers(self):
        """Initialize video writers for each environment."""
        logger.info(f"Saving rendering to {self.save_dir}")
        # Get configuration parameters with defaults
        self.group_camera = self.env.wrapper.config.get("group_camera", False)
        self.max_render_envs = self.env.wrapper.config.get("max_render_envs", self.env.num_envs)
        if self.group_camera:
            self.max_render_envs = 1  # single video from overview_camera
        self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        self.start_idx = self.env.wrapper.start_idx

        for i in range(self.max_render_envs):
            file_name = f"{self.save_dir}/{self.start_idx+i:06d}.mp4"
            fps = 1 / (self.env.step_dt * self.render_frame_skip)
            writer = imageio.get_writer(
                file_name,
                fps=fps,
                codec="libx264",
                quality=self.cfg.video_quality,
                pixelformat="yuv420p",
            )
            self.video_writers.append(writer)

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record video frames after each step with frame skipping and text overlay support."""
        if len(self.video_writers) == 0:
            self._initialize_writers()

        # Check if we should render this frame
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

        # Set camera position based on robot root position
        root_pos = self.env.command_manager.get_term("motion").robot_body_pos_w[:, 0]
        camera_offset = self.env.wrapper.config.get("eval_camera_offset", [2, 2, 1])
        fix_camera = self.env.wrapper.config.get("fix_camera_after_first_frame", False)
        cam = self.env.scene["eval_camera"]

        if fix_camera and self._fixed_eye is not None:
            # Reuse the camera position from the first frame
            eye, target = self._fixed_eye, self._fixed_target
        elif self.group_camera:
            center = root_pos.mean(dim=0, keepdim=True).expand_as(root_pos)
            eye = center + torch.tensor(camera_offset, device=self.env.device)
            target = center
            if fix_camera:
                self._fixed_eye = eye.clone()
                self._fixed_target = center.clone()
        else:
            eye = root_pos + torch.tensor(camera_offset, device=self.env.device)
            target = root_pos
            if fix_camera:
                self._fixed_eye = eye.clone()
                self._fixed_target = root_pos.clone()

        # Write world poses to Fabric AND sync to USD so both renderer paths see it
        cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
        cam.set_world_poses_from_view(eye, target)

        # Two render calls: 1st flushes pose to render pipeline, 2nd captures at new pose
        if hasattr(self.env, "sim"):
            self.env.sim.render()
            self.env.sim.render()

        # Mark sensor as outdated so update actually re-reads the annotator buffers
        cam._is_outdated[:] = True  # noqa: SLF001
        cam.update(dt=0.0, force_recompute=True)

        # Get RGB data
        rgb_viewer = cam.data.output["rgb"].clone()

        # Get render info if available
        cur_render_info = None
        if self.env.wrapper.config.get("render_info", None) is not None:
            end_idx = self.start_idx + self.max_render_envs
            cur_render_info = self.env.wrapper.config.render_info[self.start_idx : end_idx]

        # Process each environment, loop over the video writers
        if self.frame_id >= 1:
            loop = (
                tqdm(range(self.max_render_envs))
                if self.first_render
                else range(self.max_render_envs)
            )
            for i in loop:
                frame = rgb_viewer[i].cpu().numpy()

                # Add text overlay if render info is provided
                if cur_render_info is not None and i < len(cur_render_info):
                    for j, text in enumerate(cur_render_info[i]):
                        frame = cv2.putText(
                            frame,
                            str(text),
                            (10, 30 + j * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1,
                        )

                self.video_writers[i].append_data(frame)
            self.first_render = False

        self.frame_id += 1
        return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Explicitly close all video writers."""
        if not self._writers_closed:
            for i, writer in enumerate(self.video_writers):
                try:
                    writer.close()
                    logger.info(f"Closed video writer {i}")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Error closing video writer {i}: {e}")
            self.video_writers.clear()
            self._writers_closed = True
            self.frame_id = 0
            self.first_render = True
            self._fixed_eye = None
            self._fixed_target = None
            logger.info("=== All video writers closed ===")

    def __del__(self):
        """Ensure writers are closed when object is destroyed."""
        self.close_writers()


@configclass
class RenderEnvsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for environment rendering recorder with advanced features."""

    class_type = RenderEnvsRecorderTerm
    video_save_path: str = None
    video_quality: int = 5


class TrajectoryRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term that saves per-environment trajectory data (joint positions, root pose, object/table state).

    Saves .trajectory.pkl files alongside the video output, enabling kinematic replay
    in multi-scene composite renders.
    """

    cfg: TrajectoryRecorderCfg

    def __init__(self, cfg: TrajectoryRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        self.save_dir = self.cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"=== TrajectoryRecorder: saving to {self.save_dir} ===")

        self._initialized = False
        self._closed = False
        self._frame_data: dict[int, dict] = {}  # env_idx -> {field: [frames]}
        self.frame_id = 0

    def _initialize(self):
        """Initialize per-env data buffers after environment is ready."""
        self.num_record_envs = self.env.num_envs
        self.start_idx = (
            getattr(self.env.wrapper, "start_idx", 0) if hasattr(self.env, "wrapper") else 0
        )

        # Match video recorder's frame skip to keep trajectory in sync with video
        if hasattr(self.env, "wrapper"):
            self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        else:
            self.render_frame_skip = 2

        # Detect available scene entities
        self._has_object = "object" in self.env.scene.rigid_objects
        self._has_table = "table" in self.env.scene.rigid_objects

        # Get motion command for root pose
        try:
            self._motion_cmd = self.env.command_manager.get_term("motion")
        except Exception:  # noqa: BLE001
            self._motion_cmd = None

        for i in range(self.num_record_envs):
            self._frame_data[i] = self._create_empty_data()

        self._initialized = True

    def _create_empty_data(self) -> dict:
        data = {
            "dof_pos": [],
            "root_pos_w": [],
            "root_quat_w": [],
        }
        if self._has_object:
            data["object_pos_w"] = []
            data["object_quat_w"] = []
        if self._has_table:
            data["table_pos_w"] = []
            data["table_quat_w"] = []
        return data

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record trajectory state after each step, synced with video frame skip."""
        if not self._initialized:
            self._initialize()

        # Skip frames to match video recorder cadence
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

        robot = self.env.scene["robot"]
        env_origins = self.env.scene.env_origins

        for i in range(self.num_record_envs):
            # Joint positions
            joint_pos = robot.data.joint_pos[i].cpu().numpy().copy()
            self._frame_data[i]["dof_pos"].append(joint_pos)

            # Root position (relative to env origin)
            if self._motion_cmd is not None:
                root_pos = self._motion_cmd.robot_body_pos_w[i, 0].cpu().numpy().copy()
            else:
                root_pos = robot.data.root_pos_w[i].cpu().numpy().copy()
            root_pos_rel = root_pos - env_origins[i].cpu().numpy()
            self._frame_data[i]["root_pos_w"].append(root_pos_rel)

            # Root quaternion (wxyz)
            root_quat = robot.data.root_quat_w[i].cpu().numpy().copy()
            self._frame_data[i]["root_quat_w"].append(root_quat)

            # Object state
            if self._has_object:
                obj = self.env.scene["object"]
                obj_pos = obj.data.root_pos_w[i].cpu().numpy().copy()
                obj_pos_rel = obj_pos - env_origins[i].cpu().numpy()
                obj_quat = obj.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["object_pos_w"].append(obj_pos_rel)
                self._frame_data[i]["object_quat_w"].append(obj_quat)

            # Table state
            if self._has_table:
                table = self.env.scene["table"]
                table_pos = table.data.root_pos_w[i].cpu().numpy().copy()
                table_pos_rel = table_pos - env_origins[i].cpu().numpy()
                table_quat = table.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["table_pos_w"].append(table_pos_rel)
                self._frame_data[i]["table_quat_w"].append(table_quat)

        self.frame_id += 1
        return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Save all trajectory data to pkl files."""
        if self._closed or not self._initialized:
            return
        self._closed = True

        # FPS matches the video (after frame skip)
        effective_fps = 1.0 / (self.env.step_dt * self.render_frame_skip)

        scene_metadata = {}

        for i in range(self.num_record_envs):
            env_idx = self.start_idx + i
            data = self._frame_data[i]

            if not data["dof_pos"]:
                continue

            # Stack frame arrays
            trajectory = {
                "dof_pos": np.array(data["dof_pos"]),
                "root_pos_w": np.array(data["root_pos_w"]),
                "root_quat_w": np.array(data["root_quat_w"]),
                "quat_format": "wxyz",
                "fps": effective_fps,
                "num_joints": data["dof_pos"][0].shape[0],
                "total_frames": len(data["dof_pos"]),
            }

            if data.get("object_pos_w"):
                trajectory["object_pos_w"] = np.array(data["object_pos_w"])
                trajectory["object_quat_w"] = np.array(data["object_quat_w"])
            else:
                trajectory["object_pos_w"] = None
                trajectory["object_quat_w"] = None

            if data.get("table_pos_w"):
                trajectory["table_pos_w"] = np.array(data["table_pos_w"])
                trajectory["table_quat_w"] = np.array(data["table_quat_w"])
            else:
                trajectory["table_pos_w"] = None
                trajectory["table_quat_w"] = None

            # Save pkl
            pkl_path = os.path.join(self.save_dir, f"{env_idx:06d}.trajectory.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved trajectory: {pkl_path} ({trajectory['total_frames']} frames)")

            # Build metadata entry
            meta = {
                "trajectory_file": f"{env_idx:06d}.trajectory.pkl",
                "video_file": f"{env_idx:06d}.mp4",
                "num_frames": trajectory["total_frames"],
                "num_joints": trajectory["num_joints"],
                "fps": effective_fps,
                "has_object": trajectory["object_pos_w"] is not None,
                "has_table": trajectory["table_pos_w"] is not None,
            }

            # Add object USD path if available from config
            if hasattr(self.env, "wrapper"):
                obj_usd = self.env.wrapper.config.get("object_usd_path", None)
                if obj_usd:
                    meta["object_usd_path"] = obj_usd

            scene_metadata[str(env_idx)] = meta

        # Save scene metadata JSON
        meta_path = os.path.join(self.save_dir, "scene_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)
        logger.info(f"Saved scene metadata: {meta_path}")
        logger.info("=== TrajectoryRecorder: all data saved ===")

    def __del__(self):
        self.close_writers()


@configclass
class TrajectoryRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for trajectory recording alongside video."""

    class_type = TrajectoryRecorderTerm
    save_path: str = None


MINIMAL_ACTION_DIM = 29
MINIMAL_TERMINATION_TERM_ALIASES = {"time_out": "motion_time_out"}


def _action_term(env):
    """Return the single 29-DoF joint action term used by released SONIC."""
    term_names = list(env.action_manager.active_terms)
    if len(term_names) != 1:
        raise RuntimeError(
            "Minimal state-action recording requires exactly one action term; "
            f"found {term_names}"
        )
    term = env.action_manager.get_term(term_names[0])
    if type(term).__name__ != "JointPositionAction":
        raise RuntimeError(
            "Minimal state-action recording requires JointPositionAction; "
            f"found {type(term).__name__}"
        )
    if env.action_manager.total_action_dim != MINIMAL_ACTION_DIM:
        raise RuntimeError(
            f"Expected a {MINIMAL_ACTION_DIM}-D action, found "
            f"{env.action_manager.total_action_dim}"
        )
    joint_names = list(getattr(term, "_joint_names", []))
    if len(joint_names) != MINIMAL_ACTION_DIM:
        raise RuntimeError(
            f"Expected {MINIMAL_ACTION_DIM} action joint names, found {len(joint_names)}"
        )
    return term


def _gravity_in_frame(quat_w: torch.Tensor, down_dir: torch.Tensor) -> torch.Tensor:
    """Express the world down unit vector in the frame represented by ``quat_w``."""
    if down_dir.ndim == 1:
        down_dir = down_dir.unsqueeze(0).expand(quat_w.shape[0], -1)
    return quat_apply(quat_inv(quat_w), down_dir)


def _relative_heading_sin_cos(
    robot_quat_w: torch.Tensor, reference_quat_w: torch.Tensor
) -> torch.Tensor:
    """Encode reference yaw relative to robot yaw without an angle discontinuity."""

    def yaw(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    delta = yaw(reference_quat_w) - yaw(robot_quat_w)
    return torch.stack((torch.sin(delta), torch.cos(delta)), dim=-1)


def _minimal_state(env, previous_action: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the current-frame 93-D state in the released policy's coordinates."""
    term = _action_term(env)
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    if command.use_ref_motion_root_quat_w_as_anchor or getattr(command, "_offline", False):
        raise RuntimeError(
            "Minimal state-action recording requires the simulated robot pelvis orientation"
        )
    joint_ids = getattr(term, "_joint_ids")
    joint_pos = robot.data.joint_pos[:, joint_ids]
    joint_vel = robot.data.joint_vel[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    default_joint_vel = robot.data.default_joint_vel[:, joint_ids]
    if previous_action.shape[-1] != MINIMAL_ACTION_DIM:
        raise RuntimeError(
            f"Expected previous action shape (..., 29), found {previous_action.shape}"
        )
    return {
        "joint_pos": joint_pos - default_joint_pos,
        "joint_vel": joint_vel - default_joint_vel,
        "base_ang_vel": robot.data.root_ang_vel_b,
        "gravity_robot": _gravity_in_frame(command.robot_anchor_quat_w, command.down_dir),
        "previous_action": previous_action,
    }


def _minimal_goal(env) -> dict[str, torch.Tensor]:
    """Build the current-frame 63-D goal in state-compatible coordinates."""
    term = _action_term(env)
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    joint_ids = getattr(term, "_joint_ids")
    reference_joint_pos = command.joint_pos[:, joint_ids]
    reference_joint_vel = command.joint_vel[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    default_joint_vel = robot.data.default_joint_vel[:, joint_ids]
    return {
        "reference_joint_pos": reference_joint_pos - default_joint_pos,
        "reference_joint_vel": reference_joint_vel - default_joint_vel,
        "gravity_reference": _gravity_in_frame(command.anchor_quat_w, command.down_dir),
        "relative_heading": _relative_heading_sin_cos(
            command.robot_anchor_quat_w, command.anchor_quat_w
        ),
    }


def _json_parameter(value, num_envs: int, width: int, trailing_dim: int | None = None):
    """Serialize a resolved action parameter without hiding per-environment values."""
    expected_tail = (width,) if trailing_dim is None else (width, trailing_dim)
    if isinstance(value, (int, float)):
        scalar_values = torch.full(expected_tail, float(value), dtype=torch.float32)
        return {"scope": "global", "values": scalar_values.tolist()}
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().cpu()
    if tensor.ndim == len(expected_tail):
        if tuple(tensor.shape) != expected_tail:
            raise RuntimeError(
                f"Expected action parameter shape {expected_tail}, found {tensor.shape}"
            )
        return {"scope": "global", "values": tensor.tolist()}
    if tensor.ndim != len(expected_tail) + 1 or tuple(tensor.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"Expected action parameter shape ({num_envs}, {expected_tail}), found {tensor.shape}"
        )
    if tensor.shape[0] != num_envs:
        raise RuntimeError(f"Expected {num_envs} action parameter rows, found {tensor.shape[0]}")
    if torch.equal(tensor, tensor[0:1].expand_as(tensor)):
        return {"scope": "global", "values": tensor[0].tolist()}
    return {"scope": "per_environment", "values": tensor.tolist()}


def _json_runtime_tensor(value, num_envs: int):
    """Serialize a runtime physics tensor and retain every environment row."""
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().cpu()
    if tensor.ndim < 1 or tensor.shape[0] != num_envs:
        raise RuntimeError(
            f"Expected runtime tensor with first dimension {num_envs}, found {tensor.shape}"
        )
    return {"scope": "per_environment", "values": tensor.tolist()}


def _minimal_collection_identity(env) -> dict[str, torch.Tensor]:
    """Resolve stable motion identity and reset-attempt counters for every env."""
    command = env.command_manager.get_term("motion")
    local_motion_ids = command.motion_ids.to(torch.int64)
    global_motion_ids = command.motion_lib.get_motion_ids_in_dataset(local_motion_ids).to(
        torch.int64
    )
    batch_ids = torch.full_like(global_motion_ids, int(command.eval_batch_id))
    repeat = int(command.cfg.eval_motion_repeat)
    variant_offset = int(command.cfg.eval_variant_offset)
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    variant_ids = torch.remainder(env_ids, repeat) + variant_offset

    if not hasattr(env, "_minimal_attempt_ids"):
        env._minimal_attempt_ids = torch.zeros_like(global_motion_ids)  # noqa: SLF001
        env._minimal_identity_motion_ids = torch.full_like(global_motion_ids, -1)  # noqa: SLF001
        env._minimal_identity_batch_ids = torch.full_like(batch_ids, -1)  # noqa: SLF001
    identity_changed = (env._minimal_identity_motion_ids != global_motion_ids) | (  # noqa: SLF001
        env._minimal_identity_batch_ids != batch_ids  # noqa: SLF001
    )
    env._minimal_attempt_ids[identity_changed] = 0  # noqa: SLF001
    env._minimal_identity_motion_ids.copy_(global_motion_ids)  # noqa: SLF001
    env._minimal_identity_batch_ids.copy_(batch_ids)  # noqa: SLF001
    return {
        "env_id": env_ids,
        "motion_id": local_motion_ids,
        "global_motion_id": global_motion_ids,
        "variant_id": variant_ids,
        "batch_id": batch_ids,
        "attempt_id": env._minimal_attempt_ids.clone(),  # noqa: SLF001
    }


def _minimal_termination_term_mapping(termination_manager) -> dict[str, str]:
    """Map stable dataset names to every active runtime termination term."""
    mapping: dict[str, str] = {}
    for runtime_name in termination_manager.active_terms:
        output_name = MINIMAL_TERMINATION_TERM_ALIASES.get(runtime_name, runtime_name)
        if output_name in mapping:
            raise RuntimeError(
                f"Termination terms {mapping[output_name]!r} and {runtime_name!r} "
                f"both map to {output_name!r}"
            )
        mapping[output_name] = runtime_name
    required = {"anchor_pos", "anchor_ori_full", "ee_body_pos", "motion_time_out"}
    missing = sorted(required - mapping.keys())
    if missing:
        raise RuntimeError(
            f"Minimal recorder is missing required termination terms: {missing}; "
            f"active runtime terms are {list(termination_manager.active_terms)}"
        )
    return mapping


class MinimalStateTRecorderTerm(recorder_manager.RecorderTerm):
    """Record state ``t`` after action caching and before physics stepping."""

    def record_pre_step(self):
        return "state_t", _minimal_state(self._env, self._env.action_manager.prev_action)


class MinimalGoalTRecorderTerm(recorder_manager.RecorderTerm):
    """Record the current G1 reference goal aligned with state ``t``."""

    def record_pre_step(self):
        return "goal_t", _minimal_goal(self._env)


class MinimalActionsRecorderTerm(recorder_manager.RecorderTerm):
    """Record the raw 29-D action actually received by Isaac Lab."""

    def record_pre_step(self):
        action = self._env.action_manager.action
        term = _action_term(self._env)
        if not torch.equal(action, term.raw_actions):
            raise RuntimeError("ActionManager action and joint action term raw_actions differ")
        expected_processed = action * getattr(term, "_scale", 1.0) + getattr(
            term, "_offset", 0.0
        )
        clip = getattr(term, "_clip", None)
        if clip is not None:
            expected_processed = torch.clamp(
                expected_processed, min=clip[:, :, 0], max=clip[:, :, 1]
            )
        if not torch.allclose(expected_processed, term.processed_actions, rtol=1.0e-5, atol=1.0e-6):
            max_error = torch.max(torch.abs(expected_processed - term.processed_actions)).item()
            raise RuntimeError(f"Runtime action mapping check failed; max error={max_error}")
        return "actions", action


class MinimalStateTp1RecorderTerm(recorder_manager.RecorderTerm):
    """Record state ``t+1`` after physics and before automatic reset."""

    def record_post_step(self):
        return "state_tp1", _minimal_state(self._env, self._env.action_manager.action)


class MinimalOutcomeRecorderTerm(recorder_manager.RecorderTerm):
    """Record terminal boundaries for each transition."""

    def record_post_step(self):
        termination_manager = self._env.termination_manager
        termination_terms = {
            output_name: termination_manager.get_term(runtime_name)
            for output_name, runtime_name in _minimal_termination_term_mapping(
                termination_manager
            ).items()
        }
        outcome = {
            "terminated": self._env.reset_terminated,
            "truncated": self._env.reset_time_outs,
            "termination_terms": termination_terms,
        }
        done = self._env.reset_terminated | self._env.reset_time_outs
        if hasattr(self._env, "_minimal_attempt_ids"):
            self._env._minimal_attempt_ids[done] += 1  # noqa: SLF001
        return "outcome", outcome


class MinimalMotionRecorderTerm(recorder_manager.RecorderTerm):
    """Record reference indexing needed to audit episode alignment."""

    def record_pre_step(self):
        command = self._env.command_manager.get_term("motion")
        identity = _minimal_collection_identity(self._env)
        identity["motion_step"] = command.motion_start_time_steps + command.time_steps
        return "motion", identity


class MinimalContextTRecorderTerm(recorder_manager.RecorderTerm):
    """Record episode-constant reset deltas for state-randomized collection."""

    def record_pre_step(self):
        command = self._env.command_manager.get_term("motion")
        joint_ids = getattr(_action_term(self._env), "_joint_ids")
        return "context_t", {
            "reset_root_pose_delta": command.reset_root_pose_delta,
            "reset_root_velocity_delta": command.reset_root_velocity_delta,
            "reset_joint_pos_delta": command.reset_joint_pos_delta[:, joint_ids],
            "reset_joint_vel_delta": command.reset_joint_vel_delta[:, joint_ids],
        }


class MinimalMetadataRecorderTerm(recorder_manager.RecorderTerm):
    """Write the runtime-resolved state/action schema after the HDF5 stream closes."""

    cfg: MinimalMetadataRecorderCfg

    def close(self, file_path: str):
        term = _action_term(self._env)
        robot = self._env.scene["robot"]
        command = self._env.command_manager.get_term("motion")
        joint_ids = getattr(term, "_joint_ids")
        joint_names = list(getattr(term, "_joint_names"))
        default_joint_pos = robot.data.default_joint_pos[:, joint_ids].detach().cpu()
        default_joint_vel = robot.data.default_joint_vel[:, joint_ids].detach().cpu()
        nominal_default_joint_pos = getattr(robot.data, "default_joint_pos_nominal", None)
        if nominal_default_joint_pos is not None:
            nominal_default_joint_pos = nominal_default_joint_pos[joint_ids].detach().cpu()

        clip = getattr(term, "_clip", None)
        motion_keys = getattr(command.motion_lib, "_motion_data_keys", [])
        wrapper = getattr(self._env, "wrapper", None)
        wrapper_clip = None
        if wrapper is not None:
            wrapper_clip = wrapper.config.get("action_clip_value", None)
        termination_term_mapping = _minimal_termination_term_mapping(
            self._env.termination_manager
        )
        metadata = {
            "schema_version": self.cfg.schema_version,
            "dataset_file": os.path.basename(f"{file_path}.hdf5"),
            "dimensions": {"state": 93, "goal": 63, "action": 29},
            "state_field_order": [
                "joint_pos",
                "joint_vel",
                "base_ang_vel",
                "gravity_robot",
                "previous_action",
            ],
            "goal_field_order": [
                "reference_joint_pos",
                "reference_joint_vel",
                "gravity_reference",
                "relative_heading",
            ],
            "joint_names": joint_names,
            "default_joint_pos": _json_parameter(
                default_joint_pos, self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "default_joint_vel": _json_parameter(
                default_joint_vel, self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "nominal_default_joint_pos": (
                _json_parameter(
                    nominal_default_joint_pos, self._env.num_envs, MINIMAL_ACTION_DIM
                )
                if nominal_default_joint_pos is not None
                else None
            ),
            "action_scale": _json_parameter(
                getattr(term, "_scale", 1.0), self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "action_offset": _json_parameter(
                getattr(term, "_offset", 0.0), self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "action_clip": (
                _json_parameter(clip, self._env.num_envs, MINIMAL_ACTION_DIM, 2)
                if clip is not None
                else None
            ),
            "wrapper_action_clip": wrapper_clip,
            "wrapper_action_transform_enabled": bool(
                wrapper is not None and wrapper.action_transform_module is not None
            ),
            "action_term_type": type(term).__name__,
            "action_mapping": "processed = clip(actions * action_scale + action_offset)",
            "quaternion_order": "wxyz",
            "gravity_definition": "world down unit vector expressed in pelvis frame",
            "relative_heading_definition": (
                "[sin(yaw_reference-yaw_robot), cos(yaw_reference-yaw_robot)]"
            ),
            "sim_dt": self._env.physics_dt,
            "control_dt": self._env.step_dt,
            "sample_rate_hz": 1.0 / self._env.step_dt,
            "motion_id_to_key": {
                str(index): str(key) for index, key in enumerate(motion_keys)
            },
            "motion_collection": {
                "eval_motion_repeat": int(command.cfg.eval_motion_repeat),
                "eval_require_full_batch": bool(command.cfg.eval_require_full_batch),
                "variant_offset": int(command.cfg.eval_variant_offset),
                "randomization_profile": str(command.cfg.collection_randomization_profile),
                "seed": int(command.cfg.collection_seed),
                "motion_manifest": command.cfg.collection_motion_manifest,
                "env_to_variant": [
                    int(command.cfg.eval_variant_offset)
                    + env_id % int(command.cfg.eval_motion_repeat)
                    for env_id in range(self._env.num_envs)
                ],
                "env_to_motion_slot": [
                    env_id // int(command.cfg.eval_motion_repeat)
                    for env_id in range(self._env.num_envs)
                ],
            },
            "reset_randomization": {
                "enabled": bool(command.cfg.randomize_eval_resets),
                "pose_range": {
                    str(name): list(bounds)
                    for name, bounds in command.cfg.pose_range.items()
                },
                "velocity_range": {
                    str(name): list(bounds)
                    for name, bounds in command.cfg.velocity_range.items()
                },
                "joint_position_range": list(command.cfg.joint_position_range),
                "joint_velocity_range": list(command.cfg.joint_velocity_range),
            },
            "runtime_physics": {
                "body_names": list(robot.body_names),
                "body_com": _json_runtime_tensor(
                    robot.root_physx_view.get_coms(), self._env.num_envs
                ),
                "material_columns": [
                    "static_friction",
                    "dynamic_friction",
                    "restitution",
                ],
                "material_properties": _json_runtime_tensor(
                    robot.root_physx_view.get_material_properties(), self._env.num_envs
                ),
            },
            "termination_terms": list(self._env.termination_manager.active_terms),
            "termination_term_mapping": termination_term_mapping,
        }
        output_path = self.cfg.schema_output_path or f"{file_path}.schema.json"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)


PHYSICS_STATE_DIM = 70
PHYSICS_CONTACT_THRESHOLD_N = 10.0
PHYSICS_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _physics_nominal_joint_pos(env, joint_ids) -> torch.Tensor:
    robot = env.scene["robot"]
    nominal = getattr(robot.data, "default_joint_pos_nominal", None)
    if nominal is None:
        raise RuntimeError(
            "Physics state-action recording requires default_joint_pos_nominal; "
            "enable the startup default-joint event"
        )
    if nominal.ndim == 1:
        nominal = nominal[joint_ids].unsqueeze(0).expand(env.num_envs, -1)
    elif nominal.ndim == 2:
        nominal = nominal[:, joint_ids]
    else:
        raise RuntimeError(f"Unexpected nominal default shape {tuple(nominal.shape)}")
    if nominal.shape != (env.num_envs, MINIMAL_ACTION_DIM):
        raise RuntimeError(f"Unexpected nominal default shape {tuple(nominal.shape)}")
    return nominal


def _physics_contact_sensor(env):
    try:
        return env.scene["contact_forces"]
    except KeyError as error:
        raise RuntimeError("Physics recording requires the contact_forces sensor") from error


def _physics_foot_ids(env) -> list[int]:
    sensor = _physics_contact_sensor(env)
    ids, names = sensor.find_bodies(list(PHYSICS_FOOT_BODY_NAMES), preserve_order=True)
    if names != list(PHYSICS_FOOT_BODY_NAMES):
        raise RuntimeError(f"Could not resolve ordered G1 feet; found {names}")
    return ids


def _physics_state(env) -> dict[str, torch.Tensor]:
    """Return the canonical 70-D Markov state plus replay-only root/body pose."""
    term = _action_term(env)
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    joint_ids = getattr(term, "_joint_ids")
    nominal = _physics_nominal_joint_pos(env, joint_ids)
    foot_forces = _physics_contact_sensor(env).data.net_forces_w[:, _physics_foot_ids(env)]
    foot_contact = torch.linalg.vector_norm(foot_forces, dim=-1) > PHYSICS_CONTACT_THRESHOLD_N
    base_height = (
        robot.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
    )
    return {
        "joint_pos_canonical": robot.data.joint_pos[:, joint_ids] - nominal,
        "joint_vel": robot.data.joint_vel[:, joint_ids],
        "base_lin_vel_robot": robot.data.root_lin_vel_b,
        "base_ang_vel_robot": robot.data.root_ang_vel_b,
        "gravity_robot": _gravity_in_frame(robot.data.root_quat_w, command.down_dir),
        "base_height": base_height,
        "foot_contact": foot_contact,
        "root_pos_w": robot.data.root_pos_w,
        "root_quat_w": robot.data.root_quat_w,
        "body_pos_w": robot.data.body_pos_w,
    }


def _processed_from_raw(term, raw_action: torch.Tensor) -> torch.Tensor:
    processed = raw_action * getattr(term, "_scale", 1.0) + getattr(term, "_offset", 0.0)
    clip = getattr(term, "_clip", None)
    if clip is not None:
        processed = torch.clamp(processed, min=clip[:, :, 0], max=clip[:, :, 1])
    return processed


class PhysicsStateTRecorderTerm(recorder_manager.RecorderTerm):
    """Record S_t after Action processing and before the first physics substep."""

    def record_pre_step(self):
        return "physics_state_t", _physics_state(self._env)


class PhysicsActionsRecorderTerm(recorder_manager.RecorderTerm):
    """Record raw, absolute processed, and canonical processed Action_t."""

    def record_pre_step(self):
        term = _action_term(self._env)
        raw = self._env.action_manager.action
        if not torch.equal(raw, term.raw_actions):
            raise RuntimeError("ActionManager action and joint action term raw_actions differ")
        expected = _processed_from_raw(term, raw)
        if not torch.allclose(expected, term.processed_actions, rtol=1.0e-5, atol=1.0e-6):
            error = torch.max(torch.abs(expected - term.processed_actions)).item()
            raise RuntimeError(f"Runtime Action mapping failed; max error={error}")
        joint_ids = getattr(term, "_joint_ids")
        nominal = _physics_nominal_joint_pos(self._env, joint_ids)
        initial_processed = _processed_from_raw(term, self._env.action_manager.prev_action)
        return "physics_action", {
            "action_target_canonical": term.processed_actions - nominal,
            "raw_policy_action": raw,
            "processed_joint_target_abs": term.processed_actions,
            "initial_processed_target_canonical": initial_processed - nominal,
        }


class PhysicsStateTp1RecorderTerm(recorder_manager.RecorderTerm):
    """Record S_{t+1} after all physics substeps and before reset."""

    def record_post_step(self):
        return "physics_state_tp1", _physics_state(self._env)


class PhysicsDiagnosticsRecorderTerm(recorder_manager.RecorderTerm):
    """Aggregate actuator torque and foot impulse over one control interval."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # RecorderManager is constructed before ActionManager in Isaac Lab.
        # Resolve Action/contact terms lazily on the first actual environment step.
        self._joint_ids = None
        self._foot_ids = None
        self._torque_sum = None
        self._substeps = 0

    def _initialize(self):
        if self._torque_sum is not None:
            return
        self._joint_ids = getattr(_action_term(self._env), "_joint_ids")
        self._foot_ids = _physics_foot_ids(self._env)
        self._torque_sum = torch.zeros(
            self._env.num_envs, MINIMAL_ACTION_DIM, device=self._env.device
        )

    def record_pre_step(self):
        self._initialize()
        self._torque_sum.zero_()
        self._substeps = 0
        return None, None

    def record_post_physics_decimation_step(self):
        robot = self._env.scene["robot"]
        self._torque_sum += robot.data.applied_torque[:, self._joint_ids]
        self._substeps += 1
        return None, None

    def record_post_step(self):
        decimation = int(self._env.cfg.decimation)
        if self._substeps != decimation:
            raise RuntimeError(
                f"Expected {decimation} diagnostic substeps, observed {self._substeps}"
            )
        history = _physics_contact_sensor(self._env).data.net_forces_w_history
        if history.shape[1] < decimation:
            raise RuntimeError(
                f"Contact history {history.shape[1]} is shorter than decimation {decimation}"
            )
        impulse_w = history[:, :decimation, self._foot_ids].sum(dim=1) * self._env.physics_dt
        return "physics_transition", {
            "applied_joint_torque_mean": self._torque_sum / decimation,
            "foot_contact_impulse": impulse_w,
        }


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        return None
    digest = hashlib.sha256()
    with open(resolved, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actuator_metadata(robot) -> dict[str, dict]:
    result = {}
    for name, actuator in robot.actuators.items():
        result[str(name)] = {
            "type": type(actuator).__name__,
            "joint_names": list(actuator.joint_names),
            "min_delay": int(getattr(actuator.cfg, "min_delay", 0)),
            "max_delay": int(getattr(actuator.cfg, "max_delay", 0)),
        }
    return result


class PhysicsMetadataRecorderTerm(recorder_manager.RecorderTerm):
    """Write the runtime-resolved raw schema consumed by the v3 consolidator."""

    cfg: PhysicsMetadataRecorderCfg

    def close(self, file_path: str):
        term = _action_term(self._env)
        robot = self._env.scene["robot"]
        command = self._env.command_manager.get_term("motion")
        joint_ids = getattr(term, "_joint_ids")
        joint_names = list(getattr(term, "_joint_names"))
        nominal = _physics_nominal_joint_pos(self._env, joint_ids).detach().cpu()
        clip = getattr(term, "_clip", None)
        wrapper = getattr(self._env, "wrapper", None)
        wrapper_clip = None if wrapper is None else wrapper.config.get("action_clip_value", None)
        terrain_cfg = getattr(getattr(self._env.cfg, "scene", None), "terrain", None)
        terrain_material = getattr(terrain_cfg, "physics_material", None)
        spawn_cfg = getattr(robot.cfg, "spawn", None)
        asset_path = getattr(spawn_cfg, "asset_path", None)
        articulation_props = getattr(spawn_cfg, "articulation_props", None)
        motion_keys = getattr(command.motion_lib, "_motion_data_keys", [])
        mapping = getattr(self._env.cfg, "isaaclab_to_mujoco_mapping", None)
        metadata = {
            "schema_version": self.cfg.schema_version,
            "raw_dataset_file": os.path.basename(f"{file_path}.hdf5"),
            "dimensions": {"state": PHYSICS_STATE_DIM, "action": MINIMAL_ACTION_DIM},
            "state_field_order": [
                "joint_pos_canonical", "joint_vel", "base_lin_vel_robot",
                "base_ang_vel_robot", "gravity_robot", "base_height", "foot_contact",
            ],
            "state_field_dimensions": [29, 29, 3, 3, 3, 1, 2],
            "action_field": "action_target_canonical",
            "joint_names": joint_names,
            "body_names": list(robot.body_names),
            "foot_body_names": list(PHYSICS_FOOT_BODY_NAMES),
            "isaaclab_to_mujoco_mapping": mapping,
            "nominal_default_joint_pos": _json_parameter(
                nominal, self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "nominal_default_joint_vel": _json_parameter(
                robot.data.default_joint_vel[:, joint_ids].detach().cpu(),
                self._env.num_envs,
                MINIMAL_ACTION_DIM,
            ),
            "action_scale": _json_parameter(
                getattr(term, "_scale", 1.0), self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "action_offset": _json_parameter(
                getattr(term, "_offset", 0.0), self._env.num_envs, MINIMAL_ACTION_DIM
            ),
            "action_clip": (
                _json_parameter(clip, self._env.num_envs, MINIMAL_ACTION_DIM, 2)
                if clip is not None else None
            ),
            "wrapper_action_clip": wrapper_clip,
            "wrapper_action_transform_enabled": bool(
                wrapper is not None and wrapper.action_transform_module is not None
            ),
            "action_term_type": type(term).__name__,
            "action_mapping": "processed_abs = clip(raw * action_scale + action_offset)",
            "runtime_joint": {
                "default_joint_pos": _json_parameter(
                    robot.data.default_joint_pos[:, joint_ids], self._env.num_envs, MINIMAL_ACTION_DIM
                ),
                "position_limits": _json_parameter(
                    robot.data.joint_pos_limits[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM, 2,
                ),
                "velocity_limits": _json_parameter(
                    robot.data.joint_vel_limits[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
                "effort_limits": _json_parameter(
                    robot.data.joint_effort_limits[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
                "stiffness": _json_parameter(
                    robot.data.joint_stiffness[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
                "damping": _json_parameter(
                    robot.data.joint_damping[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
                "armature": _json_parameter(
                    robot.data.joint_armature[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
                "friction": _json_parameter(
                    robot.data.joint_friction_coeff[:, joint_ids], self._env.num_envs,
                    MINIMAL_ACTION_DIM,
                ),
            },
            "actuator_groups": _actuator_metadata(robot),
            "runtime_rigid_body": {
                "mass": _json_runtime_tensor(robot.root_physx_view.get_masses(), self._env.num_envs),
                "inertia": _json_runtime_tensor(robot.root_physx_view.get_inertias(), self._env.num_envs),
                "com": _json_runtime_tensor(robot.root_physx_view.get_coms(), self._env.num_envs),
                "material_columns": ["static_friction", "dynamic_friction", "restitution"],
                "material": _json_runtime_tensor(
                    robot.root_physx_view.get_material_properties(), self._env.num_envs
                ),
            },
            "terrain": {
                "type": getattr(terrain_cfg, "terrain_type", None),
                "height_reference": "root_pos_w.z - env_origin.z",
                "static_friction": getattr(terrain_material, "static_friction", None),
                "dynamic_friction": getattr(terrain_material, "dynamic_friction", None),
                "restitution": getattr(terrain_material, "restitution", None),
                "friction_combine_mode": getattr(terrain_material, "friction_combine_mode", None),
                "restitution_combine_mode": getattr(
                    terrain_material, "restitution_combine_mode", None
                ),
            },
            "simulation": {
                "sim_dt": self._env.physics_dt,
                "control_dt": self._env.step_dt,
                "decimation": int(self._env.cfg.decimation),
                "sample_rate_hz": 1.0 / self._env.step_dt,
                "gravity_w": list(getattr(self._env.cfg.sim, "gravity", (0.0, 0.0, -9.81))),
                "solver_position_iteration_count": getattr(
                    articulation_props, "solver_position_iteration_count", None
                ),
                "solver_velocity_iteration_count": getattr(
                    articulation_props, "solver_velocity_iteration_count", None
                ),
            },
            "asset": {
                "path": asset_path,
                "sha256": _file_sha256(asset_path),
            },
            "frames": {
                "quaternion_order": "wxyz",
                "base_velocity": "pelvis/body",
                "gravity": "world-down unit vector in pelvis frame",
                "foot_impulse": "world",
                "units": "SI, joint angles in rad",
            },
            "contact": {
                "threshold_n": PHYSICS_CONTACT_THRESHOLD_N,
                "history_length": int(_physics_contact_sensor(self._env).cfg.history_length),
            },
            "motion_id_to_key": {
                str(index): str(key) for index, key in enumerate(motion_keys)
            },
            "motion_collection": {
                "eval_motion_repeat": int(command.cfg.eval_motion_repeat),
                "eval_require_full_batch": bool(command.cfg.eval_require_full_batch),
                "variant_offset": int(command.cfg.eval_variant_offset),
                "randomization_profile": str(command.cfg.collection_randomization_profile),
                "seed": int(command.cfg.collection_seed),
                "motion_manifest": command.cfg.collection_motion_manifest,
                "env_to_variant": [
                    int(command.cfg.eval_variant_offset)
                    + env_id % int(command.cfg.eval_motion_repeat)
                    for env_id in range(self._env.num_envs)
                ],
            },
            "reset_randomization": {
                "enabled": bool(command.cfg.randomize_eval_resets),
                "pose_range": {str(name): list(bounds) for name, bounds in command.cfg.pose_range.items()},
                "velocity_range": {
                    str(name): list(bounds) for name, bounds in command.cfg.velocity_range.items()
                },
                "joint_position_range": list(command.cfg.joint_position_range),
                "joint_velocity_range": list(command.cfg.joint_velocity_range),
            },
            "external_wrench_recording": "sparse; baseline collection forbids random pushes",
            "active_events": {
                str(mode): list(names)
                for mode, names in self._env.event_manager.active_terms.items()
            },
            "termination_terms": list(self._env.termination_manager.active_terms),
            "termination_term_mapping": _minimal_termination_term_mapping(
                self._env.termination_manager
            ),
        }
        output_path = self.cfg.schema_output_path or f"{file_path}.schema.json"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)


@configclass
class MinimalStateTRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalStateTRecorderTerm


@configclass
class MinimalGoalTRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalGoalTRecorderTerm


@configclass
class MinimalActionsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalActionsRecorderTerm


@configclass
class MinimalStateTp1RecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalStateTp1RecorderTerm


@configclass
class MinimalOutcomeRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalOutcomeRecorderTerm


@configclass
class MinimalMotionRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalMotionRecorderTerm


@configclass
class MinimalContextTRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalContextTRecorderTerm


@configclass
class MinimalMetadataRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = MinimalMetadataRecorderTerm
    schema_version: str = "sonic_minimal_sa_v2"
    schema_output_path: str | None = None


@configclass
class PhysicsStateTRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = PhysicsStateTRecorderTerm


@configclass
class PhysicsActionsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = PhysicsActionsRecorderTerm


@configclass
class PhysicsStateTp1RecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = PhysicsStateTp1RecorderTerm


@configclass
class PhysicsDiagnosticsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = PhysicsDiagnosticsRecorderTerm


@configclass
class PhysicsMetadataRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = PhysicsMetadataRecorderTerm
    schema_version: str = "sonic_physics_sa_raw_v3"
    schema_output_path: str | None = None
