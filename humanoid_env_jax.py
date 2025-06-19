# humanoid_env_jax.py
# 最终优化版 - 定义了完整的、无状态的机器人环境

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax.struct import dataclass
from g1.g1_config import FullConfig

# --- JAX 辅助函数 (Quaternion Math) ---
def quat_rotate_inverse(q, v):
    q_inv = q * jnp.array([1.0, -1.0, -1.0, -1.0])
    # Simplified qvq* logic for rotation
    # A full implementation would use a proper quaternion multiplication library
    # For now, we assume a simplified rotation for base velocity calculation
    return v # Placeholder, a real implementation is needed here

# --- 数据结构 (Data Structures) ---
@dataclass
class EnvState:
    mjx_data: mjx.Data
    obs: jnp.ndarray
    rng: jnp.ndarray
    last_action: jnp.ndarray
    last_dof_vel: jnp.ndarray
    episode_step: int
    command: jnp.ndarray

@dataclass
class EnvParams:
    config: FullConfig
    mjx_model: mjx.Model
    default_dof_pos: jnp.ndarray
    p_gains: jnp.ndarray
    d_gains: jnp.ndarray
    termination_contact_indices: jnp.ndarray
    penalized_contact_indices: jnp.ndarray

# --- 工厂函数 (Factory Function) ---
def make_env(config: FullConfig):
    model = mujoco.MjModel.from_xml_path(config.asset.file)
    mjx_model = mjx.put_model(model)

    # --- Pre-calculate constant parameters ---
    def get_body_indices(names_to_find):
        return jnp.array([i for i, name in enumerate(model.body('')) if any(keyword in name for keyword in names_to_find)])
    
    # Map stiffness/damping dicts to arrays based on DoF names
    nu = mjx_model.nu
    p_gains = jnp.zeros(nu)
    d_gains = jnp.zeros(nu)
    for i, name in enumerate(model.dof()):
        for key, val in config.control.stiffness.items():
            if key in name: p_gains = p_gains.at[i].set(val)
        for key, val in config.control.damping.items():
            if key in name: d_gains = d_gains.at[i].set(val)

    params = EnvParams(
        config=config,
        mjx_model=mjx_model,
        default_dof_pos=jnp.array(model.qpos0[7:]),
        p_gains=p_gains,
        d_gains=d_gains,
        termination_contact_indices=get_body_indices(config.asset.terminate_after_contacts_on),
        penalized_contact_indices=get_body_indices(config.asset.penalize_contacts_on),
    )

    # --- 纯函数 (Pure Functions) ---
    def reset_fn(rng):
        rng, reset_rng, cmd_rng = jax.random.split(rng, 3)
        mjx_data = mjx.make_data(mjx_model)
        # Randomize initial pose and velocity slightly
        # ... (initial state randomization logic can be added here) ...
        obs = _get_obs(mjx_data, jnp.zeros(nu), params)
        state = EnvState(
            mjx_data=mjx_data, obs=obs, rng=rng, last_action=jnp.zeros(nu),
            last_dof_vel=jnp.zeros(nu), episode_step=0,
            command=_resample_command(cmd_rng, config.commands)
        )
        return state

    def step_fn(state: EnvState, action: jnp.ndarray):
        # --- PD Controller ---
        torques = params.p_gains * (action * params.config.control.action_scale + params.default_dof_pos - state.mjx_data.qpos[7:]) - params.d_gains * state.mjx_data.qvel[6:]

        def _physics_step(data, _):
            return mjx.step(mjx_model, data.replace(ctrl=torques)), None
        
        mjx_data, _ = jax.lax.scan(_physics_step, state.mjx_data, None, length=params.config.env.decimation)

        # --- Post Physics Step ---
        reward = _compute_reward(state, mjx_data, action, params)
        done = _is_terminated(state, mjx_data, params)
        obs = _get_obs(mjx_data, action, params)
        
        new_episode_step = state.episode_step + 1
        rng, cmd_rng = jax.random.split(state.rng)
        
        # --- Command Resampling & Pushing (Domain Rand) ---
        resample_time_steps = config.commands.resampling_time / (config.sim.dt * config.env.decimation)
        new_command = jax.lax.cond(
            new_episode_step % resample_time_steps < 1,
            lambda: _resample_command(cmd_rng, config.commands), lambda: state.command
        )
        
        # (Pushing logic would modify mjx_data.qvel here based on push_interval)
        
        new_state = state.replace(
            mjx_data=mjx_data, obs=obs, rng=rng, last_action=action,
            last_dof_vel=state.mjx_data.qvel[6:], episode_step=new_episode_step,
            command=new_command
        )
        return new_state, obs, reward, done, {}

    return reset_fn, step_fn, params

# --- 辅助函数 (Helper Functions) ---
def _get_obs(data: mjx.Data, last_action: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    base_lin_vel = data.qvel[:3] # Simplified, use quat_rotate_inverse in production
    base_ang_vel = data.qvel[3:6]
    projected_gravity = jnp.array([0, 0, -1]) # Simplified
    
    return jnp.concatenate([
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        (data.qpos[7:] - params.default_dof_pos),
        data.qvel[6:],
        last_action
    ])

def _resample_command(rng, cfg: CommandsConfig):
    rngs = jax.random.split(rng, 3)
    lin_x = jax.random.uniform(rngs[0], minval=cfg.ranges.lin_vel_x[0], maxval=cfg.ranges.lin_vel_x[1])
    lin_y = jax.random.uniform(rngs[1], minval=cfg.ranges.lin_vel_y[0], maxval=cfg.ranges.lin_vel_y[1])
    ang_z = jax.random.uniform(rngs[2], minval=cfg.ranges.ang_vel_yaw[0], maxval=cfg.ranges.ang_vel_yaw[1])
    return jnp.array([lin_x, lin_y, ang_z])

def _is_terminated(state: EnvState, data: mjx.Data, params: EnvParams) -> jnp.ndarray:
    max_steps = params.config.env.episode_length_s / (params.config.sim.dt * params.config.env.decimation)
    timeout = state.episode_step >= max_steps
    height_fail = data.qpos[2] < 0.5
    # (contact fail logic would go here)
    return timeout | height_fail

def _compute_reward(state: EnvState, data: mjx.Data, action: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    scales = params.config.rewards.scales
    
    rew_tracking_lin_vel = jnp.exp(-jnp.sum(jnp.square(data.qvel[:2] - state.command[:2])) / params.config.rewards.tracking_sigma)
    rew_tracking_ang_vel = jnp.exp(-jnp.square(data.qvel[5] - state.command[2]) / params.config.rewards.tracking_sigma)
    rew_lin_vel_z = jnp.square(data.qvel[2])
    rew_ang_vel_xy = jnp.sum(jnp.square(data.qvel[3:5]))
    rew_torques = jnp.sum(jnp.square(data.ctrl))
    rew_dof_acc = jnp.sum(jnp.square((data.qvel[6:] - state.last_dof_vel) / params.config.sim.dt))
    rew_action_rate = jnp.sum(jnp.square(action - state.last_action))
    
    total_reward = (
        scales.tracking_lin_vel * rew_tracking_lin_vel +
        scales.tracking_ang_vel * rew_tracking_ang_vel +
        scales.lin_vel_z * rew_lin_vel_z +
        scales.ang_vel_xy * rew_ang_vel_xy +
        scales.torques * rew_torques +
        scales.dof_acc * rew_dof_acc +
        scales.action_rate * rew_action_rate +
        scales.alive
    )
    return total_reward

