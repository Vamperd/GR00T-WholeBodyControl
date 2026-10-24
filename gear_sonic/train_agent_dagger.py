#!/usr/bin/env python3
"""Run standalone DAgger with a frozen SONIC release teacher."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# Match train_agent_trl.py's import-path guard when this file is executed
# directly from the gear_sonic directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "Isaac Lab is required for DAgger. Activate the SONIC Isaac environment "
        "before running this entrypoint."
    )
    raise SystemExit(1)

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch

from gear_sonic.trl.trainer.dagger_trainer import DaggerTrainer
from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def _build_app_launcher(config: DictConfig):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Standalone SONIC DAgger trainer")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    args_cli.num_envs = int(config.num_envs)
    args_cli.seed = int(config.seed)
    args_cli.env_spacing = config.manager_env.config.env_spacing
    args_cli.output_dir = str(config.get("output_dir", config.experiment_dir))
    args_cli.enable_cameras = bool(
        config.manager_env.config.get("enable_cameras", False)
        or config.manager_env.config.get("render_results", False)
        or config.manager_env.config.get("render_ego", False)
        or config.manager_env.config.get("overview_camera", False)
    )
    args_cli.headless = bool(config.headless)
    args_cli.multi_gpu = False
    args_cli.distributed = False
    args_cli.device = str(config.get("device", "cuda:0"))
    args_cli.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    )

    from filelock import FileLock

    with FileLock("/tmp/isaaclab_app_launcher.lock"):
        app_launcher = AppLauncher(args_cli)
    return app_launcher, args_cli


def _prepare_model_dimensions(env, config: DictConfig) -> None:
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in ("policy", "critic"):
            continue
        group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
            example_obs, key
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    if config.manager_env.config.get("meta_action_dim", None) is not None:
        raise ValueError("DAgger requires direct actions; meta_action_dim is not supported")
    env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]


@hydra.main(
    config_path="config",
    config_name="exp/manager/universal_token/all_modes/sonic_release_dagger_mlp",
    version_base="1.1",
)
def main(config: DictConfig) -> None:
    from gear_sonic import train_agent_trl

    os.chdir(hydra.utils.get_original_cwd())
    seeding(int(config.seed))
    app_launcher, args_cli = _build_app_launcher(config)
    device = torch.device(str(args_cli.device))

    try:
        config.manager_env.config.headless = bool(config.headless)
        config.manager_env.config.experiment_dir = str(config.experiment_dir)
        config.manager_env.config.save_rendering_dir = str(
            Path(config.experiment_dir) / "renderings_training"
        )
        Path(config.experiment_dir).mkdir(parents=True, exist_ok=True)
        Path(config.experiment_dir, "manifests").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment_dir) / "config.yaml")

        env = train_agent_trl.create_manager_env(config, str(device), args_cli)
        _prepare_model_dimensions(env, config)

        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        student = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _resolve=False,
        ).to(device)
        teacher = custom_instantiate(
            config.algo.config.teacher_actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            input_key="actor_obs",
            _resolve=False,
        ).to(device)

        materialize_lazy_params(student, env)
        materialize_lazy_params(teacher, env)

        trainer = DaggerTrainer(
            env=env,
            student=student,
            teacher=teacher,
            config=config.dagger,
            run_dir=config.experiment_dir,
            device=device,
            teacher_checkpoint=config.dagger.teacher_checkpoint,
            student_checkpoint=config.dagger.get("student_checkpoint", None),
        )
        metrics = trainer.run()
        logger.info("DAgger finished: {}", metrics)
    finally:
        try:
            env.close()
        except (NameError, AttributeError):
            pass
        app_launcher.app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
