# humanoid_env_jax.py
# This file defines the stateless environment logic using JAX and MuJoCo MJX.

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flax.struct import dataclass

from g1.g1_config import EnvConfig

# A simple dataclass to hold the state of one environment instance.
@dataclass
class EnvState:
    mjx_data: mjx.Data
    obs: jnp.ndarray
    rng: jnp.ndarray

# Another dataclass to hold the constant parameters of the environment.
@dataclass
class EnvParams:
    config: EnvConfig

def make_env(config: EnvConfig):
    """Factory function to create the environment's step and reset functions."""
    model = mujoco.MjModel.from_xml_path(config.xml_path)
    # Convert the CPU model to a JIT-compatible JAX model
    mjx_model = mjx.put_model(model)
    params = EnvParams(config=config)

    def reset_fn(rng):
        """Pure function to reset the environment."""
        rng, reset_rng = jax.random.split(rng)
        
        # Initialize physics state
        mjx_data = mjx.make_data(mjx_model)
        
        # TODO: Add noise to initial DOF state like in the original code
        # Example: mjx_data = mjx_data.replace(qpos=initial_qpos_with_noise)
        
        obs = _get_obs(mjx_data)
        return EnvState(mjx_data=mjx_data, obs=obs, rng=rng)

    def step_fn(state: EnvState, action: jnp.ndarray):
        """Pure function to step the environment."""
        
        def _physics_step(data, _):
            """A single physics step, to be repeated `action_repeat` times."""
            # In JAX, we replace PD controller logic with direct torque application for simplicity here.
            # A full implementation would compute torques based on PD control as in the original code.
            data = data.replace(ctrl=action)
            return mjx.step(mjx_model, data), None

        # Repeat the physics step `action_repeat` times using jax.lax.scan
        # This is the JAX equivalent of the `for _ in range(decimation)` loop.
        mjx_data, _ = jax.lax.scan(_physics_step, state.mjx_data, None, length=params.config.action_repeat)

        obs = _get_obs(mjx_data)
        reward = _compute_reward(mjx_data)
        done = _is_terminated(mjx_data)
        
        new_state = state.replace(mjx_data=mjx_data, obs=obs)
        return new_state, obs, reward, done, {}

    return reset_fn, step_fn

# --- Helper functions that mirror the logic in the original LeggedRobot class ---

def _get_obs(data: mjx.Data) -> jnp.ndarray:
    """Computes observations from the physics state."""
    # This is a simplified observation space. A full implementation would match
    # the original code's observation structure (base vel, gravity, dof pos/vel, etc.)
    # qpos[0] is free root joint x, qpos[1] is y, etc.
    return jnp.concatenate([
        data.qpos,
        data.qvel,
        # TODO: Add projected gravity, commands, last actions etc.
    ])

def _compute_reward(data: mjx.Data) -> jnp.ndarray:
    """Computes the reward for the current state."""
    # Simplified reward: move forward and stay alive.
    
    # 1. Tracking linear velocity (x-axis)
    target_velocity = 2.0
    forward_velocity = data.qvel[0]
    tracking_reward = jnp.exp(-jnp.square(forward_velocity - target_velocity))

    # 2. Staying alive reward
    alive_reward = 2.0
    
    # 3. Penalize high torques/actions (energy penalty)
    energy_penalty = -0.01 * jnp.sum(jnp.square(data.ctrl))

    return tracking_reward + alive_reward + energy_penalty

def _is_terminated(data: mjx.Data) -> jnp.ndarray:
    """Checks for termination conditions."""
    # Terminate if torso height is too low or has fallen over (base z-axis rotation)
    height = data.qpos[2]
    # Check orientation (simplified: look at z-component of orientation quaternion)
    # A more robust check would use rotation matrices like the original code.
    upright = data.qpos[6] > 0.5 # qpos[3:7] is the quaternion w,x,y,z
    
    return jnp.where((height < 0.8) | (upright < 0), True, False)
