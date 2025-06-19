# g1/g1_config.py
# This file centralizes all hyperparameters, mirroring the original LeggedRobotCfg.

import flax.struct as struct
import jax.numpy as jnp

@struct.dataclass
class EnvConfig:
    xml_path: str = "./g1/g1.xml"
    action_repeat: int = 4  # Corresponds to control.decimation
    physics_dt: float = 0.005  # Simulation frequency: 200 Hz
    max_episode_length_s: float = 20.0  # Episode length in seconds

@struct.dataclass
class AssetConfig:
    foot_name: str = "foot"
    penalize_contacts_on: tuple = ("thigh", "shank")
    terminate_after_contacts_on: tuple = ("base",)

@struct.dataclass
class RewardScales:
    # Mirrored from the original LeggedRobotCfg.rewards.scales
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -5.0
    torques: float = -0.00001
    action_rate: float = -0.01
    dof_acc: float = -2.5e-7
    collision: float = -1.0
    # Custom rewards
    tracking_lin_vel: float = 4.0
    tracking_ang_vel: float = 2.0
    alive: float = 2.0 # Corresponds to a positive constant reward

@struct.dataclass
class RewardsConfig:
    scales: RewardScales = RewardScales()
    base_height_target: float = 0.9
    tracking_sigma: float = 0.25 # for exp reward shaping

@struct.dataclass
class CommandsConfig:
    # Command ranges for curriculum learning
    initial_lin_vel_x: tuple = (-1.0, 1.0)
    initial_lin_vel_y: tuple = (-0.5, 0.5)
    initial_ang_vel_yaw: tuple = (-1.0, 1.0)
    
    max_lin_vel_x: float = 2.0
    max_lin_vel_y: float = 0.5
    max_ang_vel_yaw: float = 2.0
    
    resampling_time: float = 10.0 # Time between command resampling in seconds

@struct.dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: tuple = (0.5, 1.25)
    randomize_base_mass: bool = True
    added_mass_range: tuple = (-1.0, 3.0)
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0
    
@struct.dataclass
class ControlConfig:
    # PD controller gains
    stiffness: dict = {'joint': 50.0} # Simplified, one value for all joints
    damping: dict = {'joint': 1.0}
    action_scale: float = 0.5
    
@struct.dataclass
class TrainConfig:
    num_envs: int = 4096
    num_iterations: int = 5000
    rollout_length: int = 24
    learning_rate: float = 1e-3
    ppo_epochs: int = 5
    num_minibatches: int = 4
    clip_eps: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    log_interval: int = 20
    save_interval: int = 500

@struct.dataclass
class PolicyConfig:
    actor_hidden_dims: tuple = (256, 128)
    critic_hidden_dims: tuple = (256, 128)

@struct.dataclass
class FullConfig:
    env: EnvConfig = EnvConfig()
    asset: AssetConfig = AssetConfig()
    rewards: RewardsConfig = RewardsConfig()
    commands: CommandsConfig = CommandsConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    control: ControlConfig = ControlConfig()
    train: TrainConfig = TrainConfig()
    policy: PolicyConfig = PolicyConfig()

