import flax.linen as nn
from typing import Any
from networks.mods import MLP
import jax.numpy as jnp


class LowPolicy(nn.Module): # state, subgoal => action
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True
    goal_rep: bool = False  # will experiment; keeping this vs removing this

    def setup(self):
        self.low_policy = MLP(self.hidden_dims, self.activation, self.layer_norm)
    
    def __call__(self, obs, subgoals):
        return self.low_policy(jnp.append(obs, subgoals, axis=1))

class HighPolicy(nn.Module): # state, goal => subgoal
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True
    goal_rep: bool = False

    def setup(self):
        self.high_policy = MLP(self.hidden_dims, self.activation, self.layer_norm)
        
    def __call__(self, obs, goals):
        return self.high_policy(jnp.append(obs, goals, axis=1))


class ValueNet(nn.Module): # state, (sub)goal (rep) => value
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True
    goal_rep: Any = False

    def setup(self):
        self.value_net = MLP(self.hidden_dims, self.activation, self.layer_norm)

    def __call__(self, obs, goals):
        if self.goal_rep:
            pass

        return self.value_net(jnp.append(obs, goals, axis=1))


class GoalRep(nn.Module): # (sub)goal => (sub)goal rep
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True

    def setup(self):
        self.goal_rep = MLP(self.hidden_dims, self.activation, self.layer_norm)
        
    def __call__(self, goals):
        return self.goal_rep(goals)