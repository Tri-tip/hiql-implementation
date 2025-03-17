from collections import namedtuple

import flax.struct
import jax, jax.numpy as jnp
import ogbench
from flax.core.frozen_dict import FrozenDict
import numpy as np
from config import cfg

Batch = namedtuple("Batch",
                   "obs "
                   "next_obs "
                   "masks "
                   "terminals "
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
        return cls(**kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = kwargs  # the dict with all the actual data
        self.size = get_size(self.data)
        (self.terminal_idx,) = np.where(
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

    # these methods are mostly copied from the original HIQL, as there is not much to change
    def sample_goals(self, amt, idxs, p_curgoal, p_trajgoal, geo_sampling: bool = False):
        random_g_idx = self.get_random_idx(amt)

        # goals from current trajectory
        final_state_idxs = self.terminal_idx[np.searchsorted(self.terminal_idx, idxs)]
        # geometric:
        if geo_sampling:
            offsets = np.random.geometric(p=1 - cfg.discount, size=amt)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # uniform:
            distances = np.random.rand(amt)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)

        goal_idxs = np.where(
            np.random.rand(amt) < p_trajgoal / (1.0 - p_curgoal + 1e-6), traj_goal_idxs, random_g_idx
        )
        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(amt) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def sample(self, amt, idxs=None) -> Batch:
        if not idxs:
            idxs = self.get_random_idx(amt)

        raw_batch = self.sampleTransitions(amt)
        batch = {
            "obs": raw_batch["observations"],
            "next_obs": raw_batch["next_observations"],
            "actions": raw_batch["actions"],
            "terminals": raw_batch["terminals"]
        }

        # VF goals
        value_goal_idxs = self.sample_goals(
            amt,
            idxs,
            cfg.V_p_curgoal,
            cfg.V_p_trajgoal,
            cfg.V_geom_sample
        )
        batch["value_goals"] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if cfg.gc_negative else 0.0)

        # low level actor goals
        final_state_idxs = self.terminal_idx[np.searchsorted(self.terminal_idx, idxs)]
        low_goal_idxs = np.minimum(idxs + cfg.subgoal_steps, final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        if cfg.A_geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - cfg.discount, size=amt)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(amt)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)

        high_traj_target_idxs = np.minimum(idxs + cfg.subgoal_steps, high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.get_random_idx(amt)
        high_random_target_idxs = np.minimum(idxs + cfg.subgoal_steps, final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(amt) < cfg.A_p_randomgoal
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch["high_actor_goals"] = self.get_observations(high_goal_idxs)
        batch["high_actor_targets"] = self.get_observations(high_target_idxs)

        return Batch(
            obs=batch["obs"],
            next_obs=batch["next_obs"],
            masks=batch["masks"],
            terminals=batch["terminals"],
            rewards=batch["rewards"],
            value_goals=batch["value_goals"],
            low_actor_goals=batch["low_actor_goals"],
            high_actor_goals=batch["high_actor_goals"],
            high_actor_targets=batch["high_actor_targets"],
            actions=batch["actions"]
        )
