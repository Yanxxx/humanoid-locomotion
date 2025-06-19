# networks_jax.py
# 最终优化版 - 定义了Actor-Critic神经网络结构

import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# 将字符串激活函数名映射到实际的JAX函数
ACTIVATION_FN_MAP = {
    'elu': nn.elu,
    'relu': nn.relu,
    'selu': nn.selu,
    'tanh': nn.tanh,
    'sigmoid': nn.sigmoid,
}

class ActorCritic(nn.Module):
    action_dim: int
    actor_hidden_dims: Sequence[int]
    critic_hidden_dims: Sequence[int]
    activation: str = 'elu'

    @nn.compact
    def __call__(self, x):
        activation_fn = ACTIVATION_FN_MAP[self.activation]
        
        # --- Actor Network ---
        actor_net = x
        for hidden_dim in self.actor_hidden_dims:
            actor_net = nn.Dense(features=hidden_dim)(actor_net)
            actor_net = activation_fn(actor_net)
        mu = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor_net)
        
        # 可学习的动作分布标准差
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)

        # --- Critic Network ---
        critic_net = x
        for hidden_dim in self.critic_hidden_dims:
            critic_net = nn.Dense(features=hidden_dim)(critic_net)
            critic_net = activation_fn(critic_net)
        value = nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0))(critic_net)

        return dist, jnp.squeeze(value, axis=-1)

