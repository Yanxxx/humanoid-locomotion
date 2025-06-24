import numpy as np
import mujoco

# 假设 num_actions = 12
# 假设 height_scan_dims = 160

def get_observation_from_mujoco(model, data, command, last_action, default_joint_pos):
    """
    从MuJoCo的data对象中收集数据并构建精确的观测向量。
    """
    # --- 获取物理状态 ---
    root_rot_mat = data.sensordata[...].reshape(3,3) # 假设有传感器获取旋转矩阵
    
    # 1. 机身线速度 (body frame)
    body_lin_vel = root_rot_mat.T @ data.qvel[:3]
    
    # 2. 机身角速度 (body frame)
    body_ang_vel = root_rot_mat.T @ data.qvel[3:6]
    
    # 3. 投影重力 (body frame)
    gravity_vec = np.array([0, 0, -9.81])
    projected_gravity = root_rot_mat.T @ gravity_vec

    # 4. 关节数据
    dof_pos = data.qpos[7:] # 浮动基座后的关节
    dof_vel = data.qvel[6:]

    # --- 处理观测项 ---
    # 5. 速度指令
    velocity_commands = np.array(command)
    
    # 6. 相对关节位置 (joint_pos_rel)
    # 网络学习的是相对于默认姿态的偏移
    joint_pos_rel = dof_pos - default_joint_pos

    # 7. 相对关节速度 (joint_vel_rel)
    joint_vel_rel = dof_vel # 在Isaac Lab中，相对速度通常就是关节速度本身

    # 8. 上一步的动作
    # last_action 是网络直接输出的，未经过缩放
    
    # 9. 高度扫描 (需要一个辅助函数来实现)
    # 这个函数需要模拟从机器人躯干中心向下投射的网格光线
    # 并返回每个点的离地高度
    height_scan = get_height_scan(model, data, root_rot_mat) # 这是一个占位符
    
    # --- 拼接与缩放 ---
    # 根据 env.yaml，目前没有定义 scale，但定义了 noise。部署时我们忽略 noise。
    # 确保拼接顺序与 env.yaml 中定义的顺序完全一致！
    obs_vec = np.concatenate([
        body_lin_vel,
        body_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
        height_scan.flatten() # 展平高度图
    ])
    
    return obs_vec

def get_height_scan(model, data, root_rot_mat):
    """
    占位符函数：在MuJoCo中模拟Isaac Lab的高度扫描传感器。
    这是一个复杂的任务，需要使用MuJoCo的 mj_ray* 函数族。
    
    1. 定义一个 16x10 的网格点，其坐标相对于机器人躯干。
    2. 将这些点的坐标从机身坐标系转换到世界坐标系。
    3. 从一个较高的位置（如躯干上方20米处）沿着-Z方向（机身坐标系）为每个点投射一条射线。
    4. 计算每个射线与地面的交点，得到离地高度。
    5. 对结果进行裁剪，范围为[-1, 1]。
    
    返回一个 (16, 10) 的numpy数组。
    """
    # 此处为伪代码实现
    scan_points = np.zeros((16, 10))
    # ... 实现射线投射逻辑 ...
    return scan_points
