"""Additive HUD04 registrations; leave every upstream G1 task unchanged."""
import gymnasium as gym

_ENTRY = "instinctlab.tasks.shadowing.perceptive.config.hud04"
for _suffix, _config in (("", "HUD04PerceptivePlaneEnvCfg"), ("-Play", "HUD04PerceptivePlaneEnvCfg_PLAY")):
    gym.register(
        id="Instinct-Perceptive-Shadowing-HUD04-Plane" + _suffix + "-v0",
        entry_point="instinctlab.envs:InstinctRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _ENTRY + ".perceptive_shadowing_cfg:" + _config,
            "instinct_rl_cfg_entry_point": _ENTRY + ".agents.instinct_rl_ppo_cfg:HUD04PerceptivePlanePPORunnerCfg",
        },
    )

for _suffix, _config in (("", "HUD04PerceptiveTerrainEnvCfg"), ("-Play", "HUD04PerceptiveTerrainEnvCfg_PLAY")):
    gym.register(
        id="Instinct-Perceptive-Shadowing-HUD04-Terrain" + _suffix + "-v0",
        entry_point="instinctlab.envs:InstinctRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _ENTRY + ".terrain_shadowing_cfg:" + _config,
            "instinct_rl_cfg_entry_point": _ENTRY + ".agents.instinct_rl_ppo_cfg:HUD04PerceptiveTerrainPPORunnerCfg",
        },
    )