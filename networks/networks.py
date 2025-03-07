from flax import linen as nn
from typing import Any
from modules import MLP


class LowPolicy(nn.Module): # state, subgoal => action
    layer_dims: list
    layers: list = []
    goal_rep: bool = True  # will experiment; keeping this vs removing this

    def setup(self):
        self.low_policy = MLP(self.layer_dims, self.layers)

class HighPolicy(nn.Module): # state, goal => subgoal
    layer_dims: list
    layers: list = []
    goal_rep: bool = True

    def setup(self):
        self.high_policy = MLP(self.layer_dims, self.layers)


class ValueNet(nn.Module): # state, (sub)goal (rep) => value
    layer_dims: list
    layers: list = []
    goal_rep: Any = None

    def setup(self):
        self.value_net = MLP(self.layer_dims, self.layers)

    def __call__(self, obs, goals):
        if self.goal_rep:
            goals = self.goal_rep(goals) # if we choose to have a latent goal rep layer


class GoalRep(nn.Module): # (sub)goal => (sub)goal rep
    layer_dims: list
    layers: list = []

    def setup(self):
        self.goal_rep = MLP(self.layer_dims, self.layers)