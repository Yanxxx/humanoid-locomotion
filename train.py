# train_jax.py
# The main script to launch training, including curriculum learning logic.

import jax
import jax.numpy as jnp
import optax
import mujoco
from flax.training.train_state import TrainState

from g1.g1_config import FullConfig, CommandsConfig
from humanoid_env_jax import make_env
from networks_jax import ActorCritic

def main():
    config = FullConfig()
    
    # --- 1. Environment Setup ---
    env_reset, env_step, env_params = make_env(config)
    num_envs = config.train.num_envs
    vmapped_reset = jax.vmap(env_reset)
    vmapped_step = jax.vmap(env_step, in_axes=(0, 0))

    # --- 2. Network and Optimizer Setup ---
    rng = jax.random.PRNGKey(0)
    _rng, temp_rng = jax.random.split(rng)
    _temp_state = env_reset(temp_rng)
    obs_dim = _temp_state.obs.shape[0]
    action_dim = env_params.mjx_model.nu

    network = ActorCritic(
        action_dim=action_dim,
        actor_hidden_dims=config.policy.actor_hidden_dims,
        critic_hidden_dims=config.policy.critic_hidden_dims
    )
    
    rng, net_rng = jax.random.split(_rng)
    init_obs = jnp.zeros((1, obs_dim))
    params = network.init(net_rng, init_obs)["params"]
    optimizer = optax.adam(learning_rate=config.train.learning_rate)

    # --- 3. JAX TrainState with Curriculum ---
    # We now include curriculum state within our main TrainState
    class CurriculumTrainState(TrainState):
        env_state: jax.Array
        last_obs: jax.Array
        rng: jax.Array
        command_cfg: CommandsConfig # Curriculum state
        mean_episode_reward: float = 0.0

    train_state = CurriculumTrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
        env_state=None, # Will be initialized in the loop
        last_obs=None,
        rng=rng,
        command_cfg=config.commands,
        mean_episode_reward=jnp.array(0.0)
    )

    # --- 4. JIT-compiled Training Step ---
    @jax.jit
    def train_step(state, _):
        # --- A. Rollout Phase ---
        def _env_step_for_rollout(carry, _):
            _train_state, _obs, _rng = carry
            _rng, action_rng = jax.random.split(_rng)
            
            dist, value = _train_state.apply_fn({"params": _train_state.params}, _obs)
            action = dist.sample(seed=action_rng)
            log_prob = dist.log_prob(action)

            next_env_state, next_obs, reward, done, info = vmapped_step(_train_state.env_state, action)
            
            # Reset logic
            _rng, reset_rngs = jax.random.split(_rng)
            reset_rngs = jax.random.split(reset_rngs, num_envs)
            new_env_state = jax.vmap(lambda r: env_reset(r)[0])(reset_rngs)
            
            # If an env is done, we use the new state, otherwise keep the old one.
            # This is a key pattern in JAX RL.
            final_env_state = jax.tree_map(
                lambda x, y: jnp.where(done[:, None], x, y), new_env_state, next_env_state
            )
            final_obs = jnp.where(done[:, None], new_env_state.obs, next_obs)

            transition = {"obs": _obs, "reward": reward, "done": done, "action": action, "log_prob": log_prob, "value": value}
            return (_train_state.replace(env_state=final_env_state), final_obs, _rng), transition

        (final_state, _, _), transitions = jax.lax.scan(
            _env_step_for_rollout,
            (state, state.last_obs, state.rng),
            None,
            length=config.train.rollout_length
        )

        # --- B. PPO Update Phase (simplified) ---
        def _update_epoch(carry, _):
            _train_state, _transitions = carry
            # (A full PPO update with GAE, minibatches, and shuffling would be here)
            # This simplified version just shows the gradient update part.
            def _loss_fn(params):
                # ... (Loss calculation as in previous example) ...
                return 0.0 # Placeholder
            
            grads = jax.grad(_loss_fn)(_train_state.params)
            _train_state = _train_state.apply_gradients(grads=grads)
            return (_train_state, _transitions), None

        (updated_train_state, _), _ = jax.lax.scan(
            _update_epoch, (final_state, transitions), None, length=config.train.ppo_epochs
        )

        # --- C. Curriculum Learning Update ---
        mean_reward = jnp.mean(transitions["reward"])
        
        # This pure function updates the command curriculum based on performance.
        def _update_curriculum(cmd_cfg, mean_reward):
            # If tracking reward is high, increase command range
            # The original code's logic is translated here.
            reward_threshold = 0.8 * config.rewards.scales.tracking_lin_vel
            
            def _increase_ranges(cfg):
                new_range_x = cfg.initial_lin_vel_x[1] + 0.1
                # ... update other ranges ...
                return cfg.replace(initial_lin_vel_x=(-new_range_x, new_range_x))
            
            def _do_nothing(cfg):
                return cfg
            
            # Use jax.lax.cond for conditional logic inside a JIT'd function.
            return jax.lax.cond(mean_reward > reward_threshold, _increase_ranges, _do_nothing, cmd_cfg)

        new_command_cfg = _update_curriculum(updated_train_state.command_cfg, mean_reward)

        # Return the new state for the next iteration
        next_state = updated_train_state.replace(
            env_state=final_state.env_state,
            last_obs=final_state.env_state.obs,
            rng=final_state.rng,
            command_cfg=new_command_cfg,
            mean_episode_reward=mean_reward
        )
        return next_state, {"mean_reward": mean_reward}

    # --- 5. Main Training Loop ---
    print("Starting JAX training...")
    rng, initial_rngs = jax.random.split(train_state.rng)
    initial_rngs = jax.random.split(initial_rngs, num_envs)
    initial_env_state = jax.vmap(lambda r: env_reset(r)[0])(initial_rngs)
    
    # Initialize the main state object for the loop
    current_state = train_state.replace(
        env_state=initial_env_state,
        last_obs=initial_env_state.obs,
        rng=rng
    )

    for it in range(config.train.num_iterations):
        current_state, metrics = train_step(current_state, None)
        
        if it % config.train.log_interval == 0:
            reward = metrics["mean_reward"].block_until_ready()
            print(f"Iteration: {it}, Mean Reward: {reward:.4f}")

if __name__ == "__main__":
    main()
