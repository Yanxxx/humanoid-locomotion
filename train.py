# train_jax.py
# 最终优化版 - 完整、可JIT编译的PPO训练循环

import jax
import jax.numpy as jnp
import optax
import mujoco
from flax.training.train_state import TrainState
import time

from g1.g1_config import FullConfig
from humanoid_env_jax import make_env
from networks_jax import ActorCritic

def main():
    config = FullConfig()
    
    # --- 1. 环境与网络设置 ---
    env_reset, env_step, env_params = make_env(config)
    num_envs = config.env.num_envs
    vmapped_reset = jax.vmap(env_reset)
    vmapped_step = jax.vmap(env_step, in_axes=(0, 0))

    rng = jax.random.PRNGKey(42)
    _rng, temp_rng = jax.random.split(rng)
    _temp_state = env_reset(temp_rng)
    obs_dim = _temp_state.obs.shape[0]
    action_dim = env_params.mjx_model.nu

    network = ActorCritic(
        action_dim=action_dim,
        actor_hidden_dims=config.policy_ppo.actor_hidden_dims,
        critic_hidden_dims=config.policy_ppo.critic_hidden_dims,
        activation=config.policy_ppo.activation
    )
    
    rng, net_rng = jax.random.split(_rng)
    init_obs = jnp.zeros((1, obs_dim))
    params = network.init(net_rng, init_obs)["params"]
    optimizer = optax.adam(learning_rate=config.algorithm_ppo.learning_rate)

    # --- 2. JAX训练状态管理 ---
    class JaxRLTrainState(TrainState):
        env_state: jax.Array
        last_obs: jax.Array
        rng: jax.Array

    train_state = JaxRLTrainState.create(
        apply_fn=network.apply, params=params, tx=optimizer,
        env_state=None, last_obs=None, rng=rng
    )

    # --- 3. JIT编译的核心训练步骤 ---
    @jax.jit
    def train_step(state):
        train_state, last_obs, env_state, rng = state

        # --- A. 数据采集 (Rollout) ---
        def _env_step_rollout(carry, _):
            _env_state, _last_obs, _rng = carry
            _rng, action_rng = jax.random.split(_rng)
            
            dist, value = train_state.apply_fn({"params": train_state.params}, _last_obs)
            action = dist.sample(seed=action_rng)
            log_prob = dist.log_prob(action)

            next_env_state, next_obs, reward, done, info = vmapped_step(_env_state, action)
            
            _rng, reset_rngs = jax.random.split(_rng)
            reset_rngs = jax.random.split(reset_rngs, num_envs)
            new_env_state = jax.vmap(env_reset)(reset_rngs)
            
            final_env_state = jax.tree_map(lambda x, y: jnp.where(done[:, None], x, y), new_env_state, next_env_state)
            final_obs = jnp.where(done[:, None], new_env_state.obs, next_obs)

            transition = {"obs": _last_obs, "reward": reward, "done": done, "action": action, "log_prob": log_prob, "value": value}
            return (final_env_state, final_obs, _rng), transition

        (final_env_state, final_obs, final_rng), transitions = jax.lax.scan(
            _env_step_rollout, (env_state, last_obs, rng), None, length=config.runner_ppo.num_steps_per_env
        )

        # --- B. 计算优势 (GAE) ---
        _, last_val = train_state.apply_fn({"params": train_state.params}, final_obs)
        
        def _gae_step(carry, transition):
            gae, next_value = carry
            delta = transition["reward"] + config.algorithm_ppo.gamma * next_value * (1 - transition["done"]) - transition["value"]
            gae = delta + config.algorithm_ppo.gamma * config.algorithm_ppo.lam * (1 - transition["done"]) * gae
            return (gae, transition["value"]), gae
        
        _, advantages = jax.lax.scan(_gae_step, (jnp.zeros(num_envs), last_val), transitions, reverse=True)
        returns = advantages + transitions["value"]
        
        # --- C. PPO 更新 ---
        def _update_epoch(carry, _):
            _train_state, _rng, _transitions, _advantages, _returns = carry
            
            # 打乱数据
            _rng, shuffle_rng = jax.random.split(_rng)
            batch_size = num_envs * config.runner_ppo.num_steps_per_env
            indices = jax.random.permutation(shuffle_rng, batch_size)
            
            # 展平数据
            flat_transitions = jax.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), _transitions)
            flat_advantages = _advantages.flatten()
            flat_returns = _returns.flatten()

            def _update_minibatch(minibatch_state, batch_indices):
                _ppo_train_state = minibatch_state
                mb_adv = flat_advantages[batch_indices]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                def _loss_fn(params):
                    dist, value = _ppo_train_state.apply_fn({"params": params}, flat_transitions["obs"][batch_indices])
                    log_prob = dist.log_prob(flat_transitions["action"][batch_indices])
                    
                    ratio = jnp.exp(log_prob - flat_transitions["log_prob"][batch_indices])
                    loss1 = mb_adv * ratio
                    loss2 = mb_adv * jnp.clip(ratio, 1.0 - config.algorithm_ppo.clip_param, 1.0 + config.algorithm_ppo.clip_param)
                    policy_loss = -jnp.minimum(loss1, loss2).mean()

                    value_loss = jnp.square(value - flat_returns[batch_indices]).mean()
                    entropy_loss = -dist.entropy().mean()

                    total_loss = policy_loss + config.algorithm_ppo.value_loss_coef * value_loss + config.algorithm_ppo.entropy_coef * entropy_loss
                    return total_loss

                grads = jax.grad(_loss_fn)(_ppo_train_state.params)
                _ppo_train_state = _ppo_train_state.apply_gradients(grads=grads)
                return _ppo_train_state
            
            minibatch_size = batch_size // config.algorithm_ppo.num_mini_batches
            shuffled_train_state = jax.lax.fori_loop(0, config.algorithm_ppo.num_mini_batches, 
                lambda i, s: _update_minibatch(s, indices[i*minibatch_size:(i+1)*minibatch_size]), _train_state)
            
            return (shuffled_train_state, _rng, _transitions, _advantages, _returns), None
            
        (updated_train_state, _, _, _, _), _ = jax.lax.scan(
            _update_epoch, (train_state, rng, transitions, advantages, returns), None, length=config.algorithm_ppo.num_learning_epochs
        )

        metrics = {"mean_reward": jnp.mean(transitions["reward"])}
        return (updated_train_state, final_obs, final_env_state, final_rng), metrics

    # --- 4. 主训练循环 ---
    print("开始 JAX 训练...")
    rng, initial_rngs = jax.random.split(train_state.rng)
    initial_rngs = jax.random.split(initial_rngs, num_envs)
    initial_env_state = vmapped_reset(initial_rngs)
    
    current_state = (train_state, initial_env_state.obs, initial_env_state, rng)

    for it in range(config.runner_ppo.max_iterations):
        start_time = time.time()
        current_state, metrics = train_step(current_state)
        jax.block_until_ready(current_state) # 等待JIT计算完成
        end_time = time.time()
        
        if it % config.runner_ppo.log_interval == 0:
            steps_per_sec = (num_envs * config.runner_ppo.num_steps_per_env) / (end_time - start_time)
            print(f"迭代: {it}, 平均奖励: {metrics['mean_reward']:.4f}, SPS: {int(steps_per_sec)}")

if __name__ == "__main__":
    main()
