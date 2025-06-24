import torch
import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 1. 参数定义 (根据 agent.yaml 和 unitree.py) ---

# G1机器人的总关节数
# 注意：这个列表的顺序必须与你的MuJoCo模型中关节的顺序严格一致！
# 你需要根据你的 a_g1_mujoco.xml 文件来确定这个顺序。
# 下面是一个基于 unitree.py 推断的示例顺序：
JOINT_ORDER = [
    # Legs (9)
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "torso_joint",
    # Feet (4)
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Arms & Hands (24)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "left_five_joint", "left_three_joint", "left_six_joint", "left_four_joint",
    "left_zero_joint", "left_one_joint", "left_two_joint",
    "right_five_joint", "right_three_joint", "right_six_joint", "right_four_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint"
]
NUM_ACTIONS = len(JOINT_ORDER)  # 应该是 37

# 高度扫描维度 (来自 env.yaml)
HEIGHT_SCAN_DIMS = 160 

# 总观测维度
NUM_OBS = 3 + 3 + 3 + 3 + NUM_ACTIONS + NUM_ACTIONS + NUM_ACTIONS + HEIGHT_SCAN_DIMS # = 281

# 动作缩放因子 (来自 env.yaml)
ACTION_SCALE = 0.5

# 默认关节姿态 (来自 unitree.py, 顺序必须与 JOINT_ORDER 一致！)
# 你需要手动将 unitree.py 中的字典映射到 JOINT_ORDER 列表
default_joint_pos_map = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
} # ... 其他关节默认为0

# 创建一个有序的默认姿态数组
# (这是一个示例实现，你需要完善它)
DEFAULT_JOINT_POS = np.zeros(NUM_ACTIONS)
# for i, name in enumerate(JOINT_ORDER):
#     # ... 根据 default_joint_pos_map 填充 ...


# --- 2. 加载模型 (与之前相同，但使用确认后的维度) ---
policy = DeployedActor(NUM_OBS, NUM_ACTIONS)
# ... 加载权重的代码 ...


# --- 3. MuJoCo观测函数 (与之前相同，但使用确认后的参数) ---
# def get_observation_from_mujoco(...):
# ...


# --- 4. 主循环 (与之前相同，但使用确认后的参数) ---
# ...
# # 在循环内部：
# control_signal = DEFAULT_JOINT_POS + last_action * ACTION_SCALE
# mj_data.ctrl[:NUM_ACTIONS] = control_signal
# ...
