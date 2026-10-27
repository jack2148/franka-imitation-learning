"""LeRobot policy adapter using the checkpoint's saved processors and queues."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


LEROBOT_ROOT = Path(os.environ.get("MIMIC_LEROBOT_ROOT", str(Path(__file__).resolve().parents[2] / "vendor/lerobot")))
LEROBOT_SRC = LEROBOT_ROOT / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


POLICY_TYPES = {"act": "act", "diffusion_policy": "diffusion"}


@dataclass(frozen=True)
class PolicyAction:
    raw: np.ndarray
    postprocessed: np.ndarray


class LeRobotPolicyAdapter:
    """Small ACT/DP-neutral wrapper around current LeRobot inference APIs."""

    def __init__(self, policy_name: str, checkpoint: Path, device: str) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        if policy_name not in POLICY_TYPES:
            raise ValueError(f"unsupported policy: {policy_name}")
        checkpoint = checkpoint.resolve()
        self._validate_checkpoint(checkpoint)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        expected_type = POLICY_TYPES[policy_name]
        if config.type != expected_type:
            raise ValueError(f"selected {policy_name}, but checkpoint policy type is {config.type}")
        config.device = device
        config.pretrained_path = str(checkpoint)
        # All backbone weights are restored from model.safetensors. Avoid an
        # unnecessary network/cache lookup while constructing the local model.
        if hasattr(config, "pretrained_backbone_weights"):
            config.pretrained_backbone_weights = None

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.policy_name = policy_name
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.config = config

    @staticmethod
    def _validate_checkpoint(path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"checkpoint is incomplete; missing: {', '.join(missing)}")

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def select_action(self, observation: dict[str, np.ndarray], *, task: str | None = None) -> PolicyAction:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        prepared: dict[str, Any] = {key: value for key, value in observation.items()}
        autocast = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and bool(self.config.use_amp)
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            prepared = prepare_observation_for_inference(
                prepared,
                self.device,
                task=task or getattr(self, "task", "pick_and_place"),
                robot_type="fr3",
            )
            prepared = self.preprocessor(prepared)
            raw_tensor = self.policy.select_action(prepared)
            raw = raw_tensor.detach().cpu().numpy().copy()
            final_tensor = self.postprocessor(raw_tensor)

        final = final_tensor.detach().cpu().numpy()
        if final.shape == (1, 8):
            final = final[0]
        if raw.shape == (1, 8):
            raw = raw[0]
        return PolicyAction(raw=np.asarray(raw), postprocessed=np.asarray(final))

    @property
    def queue_semantics(self) -> dict[str, Any]:
        if self.policy_name == "act":
            return {
                "chunk_size": int(self.config.chunk_size),
                "actions_consumed_before_replan": int(self.config.n_action_steps),
                "internal_queue": self.config.temporal_ensemble_coeff is None,
                "reset_between_episodes": True,
            }
        return {
            "observation_history": int(self.config.n_obs_steps),
            "action_horizon": int(self.config.horizon),
            "actions_consumed_before_replan": int(self.config.n_action_steps),
            "internal_queue": True,
            "reset_between_episodes": True,
        }
