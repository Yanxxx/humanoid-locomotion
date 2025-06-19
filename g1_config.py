# g1/g1_config.py
# This file centralizes all hyperparameters for the training process.

import flax.struct as struct

@struct.dataclass
class EnvConfig:
    xml_path: str = "./g1/g1.xml"
    # The number of physics steps to take per environment step.
    # Corresponds to Isaac Gym's `decimation`.
    action_repeat: int = 4
    # The duration of one physics step.
    physics_dt: float = 0.005 # Simulation frequency: 1 / 0.005 = 200 Hz
    
@struct.dataclass
class TrainConfig:
    num_envs: int = 4096
    num_iterations: int = 5000
    # Number of env steps to roll out for each update.
    rollout_length: int = 24
    learning_rate: float = 1e-3
    # PPO-specific hyperparameters
    ppo_epochs: int = 5
    num_minibatches: int = 4
    clip_eps: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95 # GAE lambda
    log_interval: int = 10
    
@struct.dataclass
class PolicyConfig:
    actor_hidden_dims: tuple = (256, 128)
    critic_hidden_dims: tuple = (256, 128)

@struct.dataclass
class Config:
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    policy: PolicyConfig = PolicyConfig()


