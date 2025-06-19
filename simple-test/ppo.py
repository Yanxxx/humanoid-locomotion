import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

class UnitreeG1EnvAdvanced(gym.Env):
    """
    为Unitree G1设计的、具有高级奖励函数的强化学习环境。
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 33}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- MuJoCo 初始化 ---
        # !!! 修改为你的模型路径 !!!
        xml_path = 'unitree_g1/scene.xml'
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except FileNotFoundError:
            print(f"错误: 在 '{xml_path}' 找不到XML文件")
            raise

        self.render_mode = render_mode
        self.viewer = None
        
        # 存储初始状态用于重置
        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)

        # --- 定义动作和观测空间 ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        
        # 观测空间: [机身高度, 机身四元数, 所有关节角度, 机身速度, 所有关节速度]
        obs_size = 1 + 4 + (self.model.nq - 7) + 6 + (self.model.nv - 6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # --- 奖励函数权重 (超参数) ---
        self.W_FORWARD = 1.5      # 前进速度
        self.W_UPRIGHT = 0.5      # 身体姿态
        self.W_HEIGHT = 0.3       # 身体高度
        self.W_ALIVE = 0.05       # 存活
        self.C_ACTION = 0.005     # 动作成本
        self.C_JOINT_VEL = 0.001  # 关节速度成本

        self.TARGET_HEIGHT = 0.80 # 期望的机身高度 (米), 你需要根据模型进行调整

    def _get_obs(self):
        """构建观测向量"""
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        torso_height = np.array([qpos[2]])
        torso_orientation = qpos[3:7] # w, x, y, z
        joint_positions = qpos[7:]
        torso_velocities = qvel[:6] # 3-dim linear, 3-dim angular
        joint_velocities = qvel[6:]
        
        return np.concatenate([
            torso_height, torso_orientation, joint_positions, 
            torso_velocities, joint_velocities
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)
        
        # --- DEBUG ---
        # print(f"Reset: Initial Height: {self.data.qpos[2]:.2f}")
        # # 计算初始姿态的up_vec_z
        # initial_orientation_quat = self.data.qpos[3:7]
        # initial_up_vec_z = 2 * (initial_orientation_quat[0] * initial_orientation_quat[2] - initial_orientation_quat[1] * initial_orientation_quat[3])
        # print(f"Reset: Initial Upright Vec Z: {initial_up_vec_z:.2f}")
        # --- END DEBUG ---

        return self._get_obs(), {}

    def step(self, action):
        """执行一个仿真步"""
        x_pos_before = self.data.qpos[0]

        # 应用动作并步进仿真
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=4) # 控制频率 = 仿真频率 / nstep

        x_pos_after = self.data.qpos[0]
        
        # --- 计算高级奖励 ---
        # 1. 前进奖励
        forward_velocity = (x_pos_after - x_pos_before) / (self.model.opt.timestep * 4)
        forward_reward = self.W_FORWARD * min(forward_velocity, 1.5) # 速度上限，防止奖励爆炸

        # 2. 姿态奖励
        orientation_quat = self.data.qpos[3:7] # [w, x, y, z]
        # 计算世界坐标系的Z轴在机器人自身坐标系下的投影 (Z-Z dot product)
        up_vec_z = 2 * (orientation_quat[0] * orientation_quat[2] - orientation_quat[1] * orientation_quat[3])
        upright_reward = self.W_UPRIGHT * up_vec_z

        # 3. 高度奖励
        current_height = self.data.qpos[2]
        height_error = (self.TARGET_HEIGHT - current_height) ** 2
        height_reward = self.W_HEIGHT * np.exp(-5.0 * height_error) # 使用指数函数，越接近目标高度奖励越高

        # 4. 存活奖励
        alive_bonus = self.W_ALIVE
        
        # 5. 成本惩罚 (惩罚项为负)
        action_cost = self.C_ACTION * np.sum(np.square(action))
        joint_vel_cost = self.C_JOINT_VEL * np.sum(np.square(self.data.qvel[6:]))
        
        # 总奖励
        reward = forward_reward + upright_reward + height_reward + alive_bonus - action_cost - joint_vel_cost

        # --- 判断是否结束 ---
        is_fallen = current_height < 0.5# or up_vec_z < 0.8
        terminated = is_fallen
        if terminated:
            reward = -50.0 # 摔倒给予巨大惩罚
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
    
    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == '__main__':
    # --- 训练主流程 ---
    log_dir = "ppo_g1_logs/"
    model_path = "ppo_g1_model.zip"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 创建环境
    # 训练时可以关闭渲染以提速: render_mode=None
    env = UnitreeG1EnvAdvanced(render_mode=None)

    # # 2. PPO 超参数 (针对MuJoCo任务优化)
    # ppo_params = {
    #     "n_steps": 2048,           # 每次更新前收集的样本数
    #     "batch_size": 64,          # 每个minibatch的大小
    #     "n_epochs": 10,            # 每次更新时数据被重复使用的次数
    #     "gamma": 0.99,             # 折扣因子
    #     "gae_lambda": 0.95,        # GAE lambda参数
    #     "clip_range": 0.2,         # PPO裁剪范围
    #     "ent_coef": 0.0,           # 熵系数，鼓励探索
    #     "vf_coef": 0.5,            # 值函数系数
    #     "learning_rate": 3e-4,     # 学习率
    #     "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])) # 策略和值网络的结构
    # }
    
    # 3. 创建PPO模型
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **ppo_params)
    
    # 4. 创建评估回调函数，在训练过程中定期评估并保存最佳模型
    # eval_env = UnitreeG1EnvAdvanced(render_mode=None)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                              log_path=log_dir, eval_freq=10000,
                                #  deterministic=True)#, render=False)

    # 5. 开始训练 (这是一个漫长的过程)
    # print("--- 开始训练 ---")
    # 至少需要几百万个时间步才能看到效果
    # model.learn(total_timesteps=5_000_000, callback=eval_callback)
    
    # 6. 保存最终模型
    # model.save(model_path)
    # env.close()

    # --- 评估训练好的模型 ---
    print("\n--- 开始评估 ---")
    # 加载表现最好的模型进行评估
    best_model = PPO.load(os.path.join(log_dir, "best_model.zip"))
    eval_env = UnitreeG1EnvAdvanced(render_mode="human")
    
    obs, _ = eval_env.reset()
    for i in range(5000):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            print("评估结束，重置环境。")
            obs, _ = eval_env.reset()
    eval_env.close()