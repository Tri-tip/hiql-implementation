from collections import namedtuple
import jax, jax.numpy as jnp
import ogbench
from flax.core.frozen_dict import FrozenDict
import numpy as np

dataset_name = 'antmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name,  # Dataset name.
    dataset_dir='~/.ogbench/data',  # Directory to save datasets (optional).
    compact_dataset=False,  # Whether to use a compact dataset (optional; see below).
)

Batch = namedtuple("Batch",
                   "obs "
                   "next_obs "
                   "rewards "
                   "value_goals "
                   "low_actor_goals "
                   "high_actor_goals "
                   "high_actor_targets "
                   "actions")


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


# using non-compact dataset
class Dataset(FrozenDict):
    @classmethod
    def create(cls, **kwargs):
        assert 'observations' in kwargs and 'next_observations' in kwargs
        jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), kwargs)
        return cls(kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = kwargs  # the dict with all the actual data
        self.size = get_size(self.data)
        self.terminal_locations = np.where(
            self.data["terminals"] == 1)  # indices of terminals; to avoid bugs in goal select

    # random indices in range
    def get_random_idx(self, amt):
        return np.random.randint(0, self.size, size=amt)

    # indices => batch of transitions
    def get_subset(self, idxs):
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.data)

    # amt => batch of size amt of random transitions
    def sampleTransitions(self, amt):
        return self.get_subset(self.get_random_idx(amt))

    # indices => obs
    def get_observations(self, idxs):
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.data['observations'])

    def sample_goals(self, amt, ):
        pass

    def sample(self, amt, idxs=None) -> Batch:
        if not idxs:
            idxs = self.get_random_idx(amt)


        return Batch(

        )