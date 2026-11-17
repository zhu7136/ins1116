"""Conservative learning-smoke defaults; not permission to run training.

Architecture follows pinned upstream G1 perceptive PPO, but no G1 policy/weights
are imported. Hyperparameters are starting points, not HUD04-tuned results.
"""
from isaaclab.utils import configclass
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg, InstinctRlEncoderActorCriticCfg,
    InstinctRlNormalizerCfg, InstinctRlOnPolicyRunnerCfg, InstinctRlPpoAlgorithmCfg)


@configclass
class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
    channels = [32, 32]
    kernel_sizes = [3, 3]
    strides = [1, 1]
    paddings = [1, 1]
    hidden_sizes = [32]
    nonlinearity = "ReLU"
    use_maxpool = False
    output_size = 32
    component_names = ["depth_image"]
    takeout_input_components = True


@configclass
class EncoderCfg:
    depth_image: DepthImageEncoderCfg = DepthImageEncoderCfg()


@configclass
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    encoder_configs = EncoderCfg()
    critic_encoder_configs = None


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 1
    num_mini_batches = 1
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class HUD04PerceptivePlanePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 1
    log_interval = 1
    experiment_name = "hud04_perceptive_plane"
    resume = False
    load_run = None

    def __post_init__(self):
        super().__post_init__()
        # No silent resume into a differently ordered/semantically different G1 model.
        if self.resume or self.load_run is not None:
            raise ValueError("HUD04 integration candidate does not accept checkpoint resume")


@configclass
class HUD04PerceptiveTerrainPPORunnerCfg(HUD04PerceptivePlanePPORunnerCfg):
    experiment_name = "hud04_perceptive_terrain"

    def __post_init__(self):
        super().__post_init__()
        # This remains only the bounded one-update plumbing configuration.
        # Construction itself does not launch PPO or assert dynamic readiness.
        if self.max_iterations != 1 or self.num_steps_per_env != 24:
            raise ValueError("Terrain integration smoke is bounded to 24 steps / one update")
        if self.algorithm.num_learning_epochs != 1 or self.algorithm.num_mini_batches != 1:
            raise ValueError("Terrain smoke requires one epoch and one minibatch")