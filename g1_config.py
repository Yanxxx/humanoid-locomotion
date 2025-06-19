# g1/g1_config.py
# This file centralizes all hyperparameters, mirroring the provided LeggedRobotCfg.
# The structure is designed to be JAX-friendly using flax.struct.dataclass.

import flax.struct as struct
from typing import Tuple, List, Dict

# --- Environment and Simulation Configs ---
@struct.dataclass
class SimConfig:
    dt: float = 0.005
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0., 0., -9.81)

@struct.dataclass
class EnvConfig:
    num_envs: int = 4096
    env_spacing: float = 3.0
    episode_length_s: float = 20.0
    send_timeouts: bool = True
    decimation: int = 4 # Corresponds to control.decimation

@struct.dataclass
class TerrainConfig:
    mesh_type: str = 'plane'
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0

@struct.dataclass
class AssetConfig:
    file: str = "./g1/g1.xml" # Placeholder, should be updated if needed
    foot_name: str = "foot"
    penalize_contacts_on: Tuple[str, ...] = ("thigh", "shank")
    terminate_after_contacts_on: Tuple[str, ...] = ("base",)
    fix_base_link: bool = False
    self_collisions: int = 0

# --- Robot and Control Configs ---

@struct.dataclass
class InitState:
    pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Note: JAX version handles default angles differently, often centered at 0
    # This dictionary is kept for reference.
    # default_joint_angles: Dict[str, float] = {"joint_a": 0., "joint_b": 0.}

@struct.dataclass
class ControlConfig:
    control_type: str = 'P'
    # NOTE: JAX version simplifies this. A full implementation would parse this dict.
    stiffness: float = 10.0  # Simplified from {'joint_a': 10.0, 'joint_b': 15.}
    damping: float = 1.0     # Simplified from {'joint_a': 1.0, 'joint_b': 1.5}
    action_scale: float = 0.5

# --- Training, Rewards, and Randomization ---

@struct.dataclass
class CommandsConfig:
    curriculum: bool = False
    max_curriculum: float = 1.0
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True
    class ranges:
        lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
        heading: Tuple[float, float] = (-3.14, 3.14)
    
    ranges: ranges = ranges()

@struct.dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.25)
    randomize_base_mass: bool = False
    added_mass_range: Tuple[float, float] = (-1.0, 1.0)
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0

@struct.dataclass
class RewardScales:
    termination: float = -0.0
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.0
    torques: float = -0.00001
    dof_vel: float = -0.0
    dof_acc: float = -2.5e-7
    base_height: float = -0.0
    feet_air_time: float = 1.0
    collision: float = -1.0
    feet_stumble: float = -0.0
    action_rate: float = -0.01
    stand_still: float = -0.0

@struct.dataclass
class RewardsConfig:
    scales: RewardScales = RewardScales()
    only_positive_rewards: bool = True
    tracking_sigma: float = 0.25
    base_height_target: float = 1.0
    max_contact_force: float = 100.0

# --- PPO Algorithm Specific Configs ---

@struct.dataclass
class PolicyPPOConfig:
    init_noise_std: float = 1.0
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = 'elu'

@struct.dataclass
class AlgorithmPPOConfig:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.e-3
    schedule: str = 'adaptive'
    gamma: float = 0.99
    lam: float = 0.95 # Corresponds to gae_lambda
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

@struct.dataclass
class RunnerPPOConfig:
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    # Load/resume params are handled by the training script logic, not stored in state
    
# --- Top-Level Config ---
@struct.dataclass
class FullConfig:
    env: EnvConfig = EnvConfig()
    terrain: TerrainConfig = TerrainConfig()
    commands: CommandsConfig = CommandsConfig()
    init_state: InitState = InitState()
    control: ControlConfig = ControlConfig()
    asset: AssetConfig = AssetConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardsConfig = RewardsConfig()
    # PPO-specific training configs
    policy_ppo: PolicyPPOConfig = PolicyPPOConfig()
    algorithm_ppo: AlgorithmPPOConfig = AlgorithmPPOConfig()
    runner_ppo: RunnerPPOConfig = RunnerPPOConfig()

