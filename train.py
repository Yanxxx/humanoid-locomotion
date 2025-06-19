# train_jax.py
# The main script to launch training.

import jax
import jax.numpy as jnp
import optax
import mujoco
from flax.training.train_state import TrainState

from g1.g1_config import Config
from humanoid_env_jax import make_env
from networks_jax import ActorCritic

def main():
    config = Config()
    
    # --- 1. Environment Setup ---
    env_reset, env_step = make_env(config.env)

    # Automatically vectorize the environment functions using jax.vmap
    num_envs = config.train.num_envs
    vmapped_reset = jax.vmap(env_reset)
    vmapped_step = jax.vmap(env_step, in_axes=(0, 0)) # vmap over state and action

    # --- 2. Network and Optimizer Setup ---
    # Get observation and action dimensions
    _rng, temp_rng = jax.random.split(jax.random.PRNGKey(0))
    _temp_state, _ = env_reset(temp_rng)
    obs_dim = _temp_state.obs.shape[0]
    action_dim = mujoco.MjModel.from_xml_path(config.env.xml_path).nu

    network = ActorCritic(
        action_dim=action_dim,
        actor_hidden_dims=config.policy.actor_hidden_dims,
        critic_hidden_dims=config.policy.critic_hidden_dims
    )
    
    # Initialize network parameters
    rng, net_rng = jax.random.split(_rng)
    init_obs = jnp.zeros((1, obs_dim))
    params = network.init(net_rng, init_obs)["params"]

    # Setup optimizer
    optimizer = optax.adam(learning_rate=config.train.learning_rate)

    # Create TrainState to manage all states
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)

    # --- 3. JIT-compiled Training Step ---
    @jax.jit
    def train_step(state, _):
        # Unpack states
        train_state, env_state, last_obs, rng = state

        # --- A. Rollout Phase (Data Collection) ---
        def _env_step_for_rollout(carry, _):
            # Collects one step of experience
            _train_state, _env_state, _obs, _rng = carry
            _rng, action_rng = jax.random.split(_rng)
            
            dist, value = _train_state.apply_fn({"params": _train_state.params}, _obs)
            action = dist.sample(seed=action_rng)
            log_prob = dist.log_prob(action)

            # Parallel step in all environments
            next_env_state, next_obs, reward, done, info = vmapped_step(_env_state, action)
            
            # Reset environments that are 'done'
            _rng, reset_rngs = jax.random.split(_rng)
            reset_rngs = jax.random.split(reset_rngs, num_envs)
            new_env_state, new_obs = vmapped_reset(reset_rngs)
            
            # If done, use new state, otherwise keep old state
            next_env_state = jax.tree_map(
                lambda x, y: jnp.where(done[:, None], x, y), new_env_state, next_env_state
            )
            next_obs = jnp.where(done[:, None], new_obs, next_obs)

            transition = {
                "obs": _obs, "action": action, "reward": reward,
                "done": done, "value": value, "log_prob": log_prob
            }
            return (_train_state, next_env_state, next_obs, _rng), transition

        # Use jax.lax.scan for an efficient, JIT-able rollout loop
        (final_train_state, final_env_state, final_obs, final_rng), transitions = jax.lax.scan(
            _env_step_for_rollout,
            (train_state, env_state, last_obs, rng),
            None,
            length=config.train.rollout_length
        )

        # --- B. PPO Update Phase ---
        def _update_epoch(update_state, _):
            _train_state, _transitions, _rng = update_state
            
            def _ppo_loss_fn(params):
                # This is a simplified PPO loss. A full implementation includes GAE calculation.
                dist, value = _train_state.apply_fn({"params": params}, _transitions["obs"])
                log_prob = dist.log_prob(_transitions["action"])
                
                # Ratio of new to old policy probabilities
                ratio = jnp.exp(log_prob - _transitions["log_prob"])
                
                # Simplified advantage (in a real implementation, this comes from GAE)
                advantage = _transitions["reward"] - _transitions["value"] # Highly simplified!
                
                # Clipped surrogate objective
                loss1 = advantage * ratio
                loss2 = advantage * jnp.clip(ratio, 1.0 - config.train.clip_eps, 1.0 + config.train.clip_eps)
                policy_loss = -jnp.minimum(loss1, loss2).mean()

                # Value loss
                value_loss = jnp.square(value - _transitions["reward"]).mean() # Simplified target

                # Entropy loss
                entropy_loss = -dist.entropy().mean()

                total_loss = (policy_loss + 
                              config.train.value_loss_coef * value_loss +
                              config.train.entropy_coef * entropy_loss)
                return total_loss

            # Calculate gradients and update the model
            grads = jax.grad(_ppo_loss_fn)(_train_state.params)
            _train_state = _train_state.apply_gradients(grads=grads)
            
            return (_train_state, _transitions, _rng), None

        # Run PPO update epochs
        # Note: A full implementation would shuffle data and use minibatches here.
        (updated_train_state, _, _), _ = jax.lax.scan(
            _update_epoch, (train_state, transitions, rng), None, length=config.train.ppo_epochs
        )

        # Prepare state for the next iteration
        next_state = (updated_train_state, final_env_state, final_obs, final_rng)
        metrics = {"total_loss": 0.0} # TODO: actually return loss from update
        
        return next_state, metrics

    # --- 4. Main Training Loop ---
    print("Starting JAX training...")
    rng, initial_rngs = jax.random.split(rng)
    initial_rngs = jax.random.split(initial_rngs, num_envs)
    
    # Initial reset to get the first state and observation
    initial_env_state, initial_obs = vmapped_reset(initial_rngs)
    
    current_state = (train_state, initial_env_state, initial_obs, rng)

    for it in range(config.train.num_iterations):
        current_state, metrics = train_step(current_state)
        
        if it % config.train.log_interval == 0:
            # We can't print from within a JIT'd function, so we do it here.
            # `block_until_ready` ensures the computation is finished before printing.
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            print(f"Iteration: {it}, Metrics: {metrics}")

if __name__ == "__main__":
    main()
