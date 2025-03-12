import flax
import jax
import jax.numpy as jnp
from batch_gen import Batch, Dataset
from config import cfg
from networks.nets import LowPolicy, HighPolicy, ValueNet
import util as utils
import optax
import functools as ft
from typing import Any

class HIQLAgent(flax.struct.PyTreeNode):
    train_state: utils.TrainState
    dataset: Dataset

    # expectile loss: MSE, but weight adv >= 0 differently than adv < 0
    def expectile_loss(self, ME, adv, expectile = cfg.expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (ME ** 2)

    @jax.jit
    def high_actor_loss(self, params, batch: Batch): # standard A2C loss method
        high_actor_net = ft.partial(self.train_state.select("high_lvl_policy"), params=params)

    @jax.jit
    def low_actor_loss(self, params, batch: Batch): # standard A2C loss method
        low_actor_net = ft.partial(self.train_state.select("low_lvl_policy"), params=params)

    @jax.jit
    def value_loss(self, params, batch: Batch):
        value_net = ft.partial(self.train_state.select("value_net"), params=params)
        target_net = ft.partial(self.train_state.select("target_net"), params=params)


    @jax.jit
    def total_loss(self, params, batch: Batch):
        pass

    @classmethod
    def create(cls, dataset):
        rng = jax.random.PRNGKey(cfg.seed)
        rng, init_rng = jax.random.split(rng, 2)

        networks = {
            "value_net": ValueNet(hidden_dims=cfg.value_HD),
            "target_net": ValueNet(hidden_dims=cfg.value_HD),
            "low_lvl_policy": LowPolicy(hidden_dims=cfg.low_actor_HD),
            "high_lvl_policy": HighPolicy(hidden_dims=cfg.high_actor_HD),
        }

        dataset_gen = Dataset.create(**dataset)
        setup_batch = dataset_gen.sample(1)

        network_args = {
            "value_net": (setup_batch.obs, setup_batch.value_goals),
            "target_net": (setup_batch.obs, setup_batch.value_goals),
            "low_lvl_policy": (setup_batch.obs, setup_batch.low_actor_goals),
            "high_lvl_policy": (setup_batch.obs, setup_batch.high_actor_goals)
        }

        networks = utils.ModuleDict(networks)
        network_optim = optax.adam(learning_rate=cfg.lr)

        params = networks.init(init_rng, **network_args)["params"]

        network = utils.TrainState.create(networks, params, network_optim)
        return cls(train_state=network, dataset=dataset_gen)


    def sample_actions(self, obs):
        pass

    def update(self):
        # every 50 steps, update the target network
        pass
    
