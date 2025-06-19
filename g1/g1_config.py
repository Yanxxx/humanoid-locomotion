# g1/g1_config.py
# 最终优化版配置文件
# 本文件将所有超参数集中管理，结构上镜像了原始的LeggedRobotCfg，
# 并为JAX的函数式和JIT编译环境做了优化。

import flax.struct as struct
from typing import Tuple, Dict

# --- 环境与仿真配置 (Environment and Simulation Configs) ---
@struct.dataclass
class SimConfig:
    dt: float = 0.005  # 物理仿真步长 (s), 对应 200 Hz
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0., 0., -9.81)

@struct.dataclass
class EnvConfig:
    num_envs: int = 4096
    episode_length_s: float = 20.0
    decimation: int = 4  # 动作降频，每4个仿真步执行1次决策

# --- 地形与机器人资源配置 (Terrain and Asset Configs) ---
@struct.dataclass
class AssetConfig:
    file: str = "./g1/g1.xml"
    foot_name: str = "foot"
    penalize_contacts_on: Tuple[str, ...] = ("thigh", "shank")
    terminate_after_contacts_on: Tuple[str, ...] = ("base",)
    fix_base_link: bool = False
    self_collisions: int = 0

# --- 机器人控制与初始状态 (Robot Control and Initial State) ---
@struct.dataclass
class InitState:
    pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0) # w, x, y, z
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@struct.dataclass
class ControlConfig:
    control_type: str = 'P'
    stiffness: Dict[str, float] = struct.field(default_factory=lambda: {'joint_a': 10.0, 'joint_b': 15.})
    damping: Dict[str, float] = struct.field(default_factory=lambda: {'joint_a': 1.0, 'joint_b': 1.5})
    action_scale: float = 0.5

# --- 训练、奖励与随机化 (Training, Rewards, and Randomization) ---
@struct.dataclass
class CommandsConfig:
    curriculum: bool = False
    max_curriculum: float = 1.0
    num_commands: int = 3
    resampling_time: float = 10.0
    @struct.dataclass
    class Ranges:
        lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
    ranges: Ranges = Ranges()

@struct.dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.25)
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0

@struct.dataclass
class RewardScales:
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.0
    torques: float = -0.00001
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    collision: float = -1.0
    # Added constant alive reward
    alive: float = 2.0

@struct.dataclass
class RewardsConfig:
    scales: RewardScales = RewardScales()
    tracking_sigma: float = 0.25
    base_height_target: float = 1.0

# --- PPO算法专属配置 (PPO Algorithm Specific Configs) ---
@struct.dataclass
class PolicyPPOConfig:
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = 'elu'

@struct.dataclass
class AlgorithmPPOConfig:
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.e-3
    schedule: str = 'adaptive'
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    max_grad_norm: float = 1.0

@struct.dataclass
class RunnerPPOConfig:
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50

# --- 顶级配置 (Top-Level Config) ---
@struct.dataclass
class FullConfig:
    sim: SimConfig = SimConfig()
    env: EnvConfig = EnvConfig()
    asset: AssetConfig = AssetConfig()
    control: ControlConfig = ControlConfig()
    commands: CommandsConfig = CommandsConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardsConfig = RewardsConfig()
    policy_ppo: PolicyPPOConfig = PolicyPPOConfig()
    algorithm_ppo: AlgorithmPPOConfig = AlgorithmPPOConfig()
    runner_ppo: RunnerPPOConfig = RunnerPPOConfig()
