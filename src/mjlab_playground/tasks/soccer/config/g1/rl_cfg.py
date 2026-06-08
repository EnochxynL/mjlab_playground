"""RL configurations for Unitree G1 soccer tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _base_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Shared PPO hyperparameters for both soccer stages."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        save_interval=500,
        num_steps_per_env=24,
    )


# Both stages share the same experiment_name so that Stage 2 can naturally
# resume from a Stage 1 checkpoint within the same log directory.
_SOCCER_EXPERIMENT_NAME = "g1_soccer"


def g1_soccer_destination_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 2: kick-to-destination on flat ground (30k iterations)."""
    cfg = _base_ppo_runner_cfg()
    cfg.experiment_name = _SOCCER_EXPERIMENT_NAME
    cfg.max_iterations = 30_000
    return cfg


def g1_soccer_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Stage 1: motion-skill acquisition on gravel terrain (4k iterations)."""
    cfg = _base_ppo_runner_cfg()
    cfg.experiment_name = _SOCCER_EXPERIMENT_NAME
    cfg.max_iterations = 4_000
    return cfg
