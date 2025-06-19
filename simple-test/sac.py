import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
import os

class UnitreeG1Env(gym.Env):
    """
    Custom Gymnasium Environment for Unitree G1 Humanoid.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- MuJoCo Initialization ---
        # MODIFY THIS PATH
        xml_path = '../g1/scene.xml'
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except FileNotFoundError:
            print(f"Error: XML file not found at '{xml_path}'")
            raise

        self.render_mode = render_mode
        self.viewer = None
        
        # Store initial state for resetting
        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)

        # --- Define Spaces ---
        # Action Space: Control for all 29 actuators, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Observation Space: [torso_height, torso_quat, all_joint_pos, torso_vel, all_joint_vel]
        # 1 (z_height) + 4 (orientation) + (nq-7) joints + 6 (torso vel/ang_vel) + (nv-6) joint_vel
        obs_size = 1 + 4 + (self.model.nq - 7) + 6 + (self.model.nv - 6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        print(f"Environment Initialized:")
        print(f"  - Action Space Size: {self.model.nu}")
        print(f"  - Observation Space Size: {obs_size}")


    def _get_obs(self):
        """Constructs the observation vector."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # Torso height (z-pos) and orientation (quaternion)
        torso_height = np.array([qpos[2]])
        torso_orientation = qpos[3:7]

        # Joint positions and velocities (excluding root joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]

        # Torso linear and angular velocities
        torso_velocities = qvel[:6]
        
        return np.concatenate([
            torso_height,
            torso_orientation,
            joint_positions,
            torso_velocities,
            joint_velocities
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # Set state to initial position and velocity
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        
        # Add slight random noise to initial joint positions for robustness
        qpos_noise = self.np_random.uniform(low=-0.02, high=0.02, size=self.model.nq)
        # Do not add noise to the root joint position/orientation
        qpos_noise[:7] = 0 
        self.data.qpos[:] += qpos_noise
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        # Get x-position before the step
        x_pos_before = self.data.qpos[0]

        # Apply action to actuators
        # Scale action by actuator gain (often helpful)
        action_scaled = action * self.model.actuator_gainprm[:, 0]
        self.data.ctrl[:] = action_scaled
        
        # Step the simulation forward
        # nstep > 1 allows the simulation to run at a higher frequency than the control frequency
        mujoco.mj_step(self.model, self.data, nstep=5)
        
        # Get x-position after the step
        x_pos_after = self.data.qpos[0]

        # --- Calculate Reward ---
        # 1. Forward Velocity Reward: Encourage moving forward
        forward_velocity = (x_pos_after - x_pos_before) / (self.model.opt.timestep * 5)
        forward_reward = 2.0 * forward_velocity

        # 2. Alive Bonus: Encourage staying upright
        alive_bonus = 1.0

        # 3. Control Cost: Penalize large actions to encourage efficiency
        control_cost = 0.1 * np.square(action).mean()
        
        # 4. Torso Orientation Cost: Penalize falling over
        torso_orientation = self.data.qpos[3:7] # [w, x, y, z]
        # Project world z-axis vector (0,0,1) into the torso's frame
        # See: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        up_vec_in_torso_frame_z = 1.0 - 2.0 * (torso_orientation[1]**2 + torso_orientation[2]**2)
        orientation_cost = 0.5 * (1.0 - up_vec_in_torso_frame_z)**2

        reward = forward_reward + alive_bonus - control_cost - orientation_cost

        # --- Check for Termination ---
        torso_height = self.data.qpos[2]
        is_fallen = torso_height < 0.6 or up_vec_in_torso_frame_z < 0.7
        
        terminated = is_fallen
        
        if terminated:
            reward = -100.0 # Heavy penalty for falling

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    # --- Training ---
    log_dir = "sac_g1_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Create the environment
    env = UnitreeG1Env(render_mode="human") # Use 'human' to watch, None for faster headless training
    
    # 2. Instantiate the SAC agent
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # 3. Train the model
    print("--- Starting Training ---")
    model.learn(total_timesteps=1_000_000)
    
    # 4. Save the trained model
    model.save("sac_unitree_g1")
    
    env.close()

    # --- Evaluation ---
    print("\n--- Starting Evaluation ---")
    model = SAC.load("sac_unitree_g1")
    env = UnitreeG1Env(render_mode="human")
    
    obs, _ = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished.")
            obs, _ = env.reset()

    env.close()
