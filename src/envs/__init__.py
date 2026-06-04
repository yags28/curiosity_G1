import gymnasium as gym
from isaaclab.scene import InteractiveSceneCfg

from .tasks.distant_target import DistantTargetEnv, DistantTargetEnvCfg
from .tasks.elevated_button import ElevatedButtonEnv, ElevatedButtonEnvCfg
from .tasks.occluded_retrieval import OccludedRetrievalEnv, OccludedRetrievalEnvCfg
from .tasks.weight_lever import WeightLeverEnv, WeightLeverEnvCfg
from .tasks.composite import CompositeEnv, CompositeEnvCfg

_ENV_MAP = {
    "distant_target":    (DistantTargetEnv,    DistantTargetEnvCfg),
    "elevated_button":   (ElevatedButtonEnv,   ElevatedButtonEnvCfg),
    "occluded_retrieval":(OccludedRetrievalEnv, OccludedRetrievalEnvCfg),
    "weight_lever":      (WeightLeverEnv,       WeightLeverEnvCfg),
    "composite":         (CompositeEnv,         CompositeEnvCfg),
}


def make_env(cfg: dict):
    """Instantiate the correct Isaac Lab env from a config dict."""
    task      = cfg["task"]
    env_cfg_d = cfg.get("env", {})
    num_envs  = env_cfg_d.get("num_envs", 32)
    control_dt = 0.02  # decimation=4, physics_dt=0.005
    ep_steps   = env_cfg_d.get("episode_length", 1000)

    EnvClass, CfgClass = _ENV_MAP[task]
    env_cfg = CfgClass()
    env_cfg.scene = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_cfg.scene.env_spacing,
        replicate_physics=True,
    )
    env_cfg.episode_length_s = ep_steps * control_dt
    env_cfg.seed = cfg.get("seed", 42)
    return EnvClass(env_cfg)

gym.register(
    id="CDL-DistantTarget-v0",
    entry_point="src.envs.tasks.distant_target:DistantTargetEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": DistantTargetEnvCfg},
)

gym.register(
    id="CDL-ElevatedButton-v0",
    entry_point="src.envs.tasks.elevated_button:ElevatedButtonEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": ElevatedButtonEnvCfg},
)

gym.register(
    id="CDL-OccludedRetrieval-v0",
    entry_point="src.envs.tasks.occluded_retrieval:OccludedRetrievalEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": OccludedRetrievalEnvCfg},
)

gym.register(
    id="CDL-WeightLever-v0",
    entry_point="src.envs.tasks.weight_lever:WeightLeverEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": WeightLeverEnvCfg},
)

gym.register(
    id="CDL-Composite-v0",
    entry_point="src.envs.tasks.composite:CompositeEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CompositeEnvCfg},
)
