import flax.linen as nn
from typing import Any
from networks.net_modules import *
import jax.numpy as jnp
import distrax
from config import cfg


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        final_dim: Final dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
    """

    hidden_dims: tuple[int, ...]
    final_dim: int
    log_std_min: int = -5
    log_std_max: int = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims)
        self.mean_net = nn.Dense(self.final_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.final_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.final_dim,))

    def __call__(
            self,
            observations,
            goals=None,
            goal_encoded=False,
            temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)

        return distribution


# class LowPolicy(nn.Module): # state, subgoal => action
#     hidden_dims: tuple[int, ...]
#     activation: Any = nn.relu
#     layer_norm: bool = True
#     goal_rep: bool = False  # will experiment; keeping this vs removing this
#
#     def setup(self):
#         self.low_policy = MLP(self.hidden_dims, self.activation, self.layer_norm)
#
#     def __call__(self, obs, subgoals):
#         logits = self.low_policy(jnp.append(obs, subgoals, axis=1))
#         dist = distrax.Categorical(logits=logits/cfg.eval_temp)
#         return dist
#
# class HighPolicy(nn.Module): # state, goal => subgoal
#     hidden_dims: tuple[int, ...]
#     activation: Any = nn.relu
#     layer_norm: bool = True
#     goal_rep: bool = False
#
#     def setup(self):
#         self.high_policy = MLP(self.hidden_dims, self.activation, self.layer_norm)
#
#     def __call__(self, obs, goals):
#         logits = self.high_policy(jnp.append(obs, goals, axis=1))
#         dist = distrax.Categorical(logits=logits/cfg.eval_temp)
#         return dist
#

class ValueNet(nn.Module): # state, (sub)goal (rep) => value
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True
    goal_rep: Any = False

    def setup(self):
        mlp_ensemble = ensemblize(MLP, cfg.num_ensemble)
        self.value_net = mlp_ensemble(self.hidden_dims + (1,), self.activation, self.layer_norm)

    def __call__(self, obs, goals):
        if self.goal_rep:
            pass

        return self.value_net(jnp.append(obs, goals, axis=1))

# class GoalRep(nn.Module): # (sub)goal => (sub)goal rep
#     hidden_dims: tuple[int, ...]
#     activation: Any = nn.relu
#     layer_norm: bool = True
#
#     def setup(self):
#         self.goal_rep = MLP(self.hidden_dims, self.activation, self.layer_norm)
#
#     def __call__(self, goals):
#         return self.goal_rep(goals)
