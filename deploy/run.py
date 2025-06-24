import mujoco.viewer
import time
import torch

# --- 初始化 ---
# 加载MuJoCo模型
mj_model = mujoco.MjModel.from_xml_path("path_to_your_g1_model.xml")
mj_data = mujoco.MjData(mj_model)

# 加载PyTorch策略网络 (如之前所述)
# policy = ...

# --- 参数确认 ---
# 从 env.yaml 确认
ACTION_SCALE = 0.5 
# 获取默认关节姿态，用于计算相对关节位置 (obs) 和作为动作偏移 (action)
default_joint_pos = mj_model.qpos0[7:] 

# 动作和观测维度
num_actions = len(default_joint_pos)
# num_obs = 3 + 3 + 3 + 3 + num_actions + num_actions + num_actions + 160
# 实例化网络...

# --- 变量初始化 ---
last_action = np.zeros(num_actions) 
command = [1.0, 0.0, 0.0]  # 向前走

# --- 启动仿真 ---
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        step_start_time = time.time()

        # 1. 从MuJoCo获取观测
        obs_np = get_observation_from_mujoco(mj_model, mj_data, command, last_action, default_joint_pos)
        
        # 2. 模型推理
        obs_tensor = torch.from_numpy(obs_np).float()
        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        
        last_action = action_tensor.cpu().numpy()
        
        # 3. 将网络输出转换为MuJoCo的控制信号
        # action = default_joint_pos + network_output * scale
        control_signal = default_joint_pos + last_action * ACTION_SCALE
        
        # 4. 应用控制信号
        mj_data.ctrl[:num_actions] = control_signal
        
        # 5. 执行一步仿真
        mujoco.mj_step(mj_model, mj_data)
        
        # 6. 渲染
        viewer.sync()
        
        # 保持仿真频率
        time.sleep(max(0, 0.02 - (time.time() - step_start_time))) # 假设50Hz
