from collections import namedtuple
import jax.numpy as jnp

Batch = namedtuple("Batch",
                   "obs "
                   "next_obs "
                   "rewards "
                   "value_goals "
                   "low_actor_goals "
                   "high_actor_goals "
                   "high_actor_targets")

def gen_batch() -> Batch:
    return Batch()

def sample_goals(batch : Batch, idx):
    return batch, idx

