# networks_jax.py
# Defines the policy and value networks using Flax.

import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

class ActorCritic(nn.Module):
    action_dim: int
    actor_hidden_dims: Sequence[int]
    critic_hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Actor network
        actor_net = x
        for hidden_dim in self.actor_hidden_dims:
            actor_net = nn.Dense(features=hidden_dim)(actor_net)
            actor_net = nn.tanh(actor_net)
        mu = nn.Dense(features=self.action_dim)(actor_net)
        
        # Learnable log standard deviation for action distribution
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)

        # Critic network
        critic_net = x
        for hidden_dim in self.critic_hidden_dims:
            critic_net = nn.Dense(features=hidden_dim)(critic_net)
            critic_net = nn.tanh(critic_net)
        value = nn.Dense(features=1)(critic_net)

        return dist, jnp.squeeze(value, axis=-1)


