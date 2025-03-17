import flax
import jax
import jax.numpy as jnp
from batch_gen import Batch, Dataset
from config import cfg
from networks.networks import GCActor, ValueNet
import util as utils
import optax
import functools as ft
from typing import Any


class HIQLAgent(flax.struct.PyTreeNode):
    train_state: utils.TrainState
    action_space: int
    state_space: int

    # expectile loss: MSE, but weight adv >= 0 differently than adv < 0
    def expectile_loss(self, ME, adv, expectile=cfg.expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (ME ** 2)

    def high_actor_loss(self, params, batch: Batch):  # standard A2C loss method
        high_actor_net = ft.partial(self.train_state.select("high_lvl_policy"), params=params)
        value_net = ft.partial(self.train_state.select("value_net"))

        vns = jnp.array(value_net(batch.obs, batch.high_actor_goals)).squeeze()
        next_vns = jnp.array(value_net(batch.next_obs, batch.high_actor_goals)).squeeze()

        v, next_v = jnp.average(vns), jnp.average(next_vns)  # assume scalar
        adv = next_v - v

        exp_a = jnp.minimum(jnp.exp(cfg.high_alpha * adv), 100.00)

        # goal rep here:

        # standard AWR loss
        curr_dist = high_actor_net(batch.obs, batch.high_actor_goals, params=params)
        log_loss = curr_dist.log_prob(batch.high_actor_targets)

        actor_loss = -(exp_a * log_loss).mean()
        return actor_loss

    def low_actor_loss(self, params, batch: Batch):  # standard A2C loss method
        low_actor_net = ft.partial(self.train_state.select("low_lvl_policy"), params=params)
        value_net = ft.partial(self.train_state.select("value_net"))

        vns = jnp.array(value_net(batch.obs, batch.low_actor_goals)).squeeze()
        next_vns = jnp.array(value_net(batch.next_obs, batch.low_actor_goals)).squeeze()

        v, next_v = jnp.average(vns), jnp.average(next_vns)  # assume scalar
        adv = next_v - v

        exp_a = jnp.minimum(jnp.exp(cfg.low_alpha * adv), 100.00)

        # goal representation system goes here, if needed

        # standard AWR loss
        curr_dist = low_actor_net(batch.obs, batch.low_actor_goals,
                                  params=params)  # distribution over action space, manually set
        log_loss = curr_dist.log_prob(
            batch.actions)  # log probs of every action in the batch, according to the distribution

        actor_loss = -(exp_a * log_loss).mean()
        return actor_loss

    def value_loss(self, params, batch: Batch):
        value_net = ft.partial(self.train_state.select("value_net"), params=params)
        target_net = ft.partial(self.train_state.select("target_net"), params=params)

        next_vts = jnp.array(target_net(batch.next_obs, batch.value_goals)).squeeze()
        next_vt = jnp.min(next_vts, axis=0)  # expect scalar
        q = batch.rewards + cfg.discount * batch.masks * next_vt

        vts = jnp.array(target_net(batch.obs, batch.value_goals)).squeeze()
        vt = jnp.average(vts, axis=0)
        adv = q - vt

        vs = jnp.array(value_net(batch.obs, batch.value_goals, params=params)).squeeze()
        v = jnp.average(vs, axis=0)

        value_losses = []
        for i, next_vt_i in enumerate(next_vts):
            qn = batch.rewards + cfg.discount * batch.masks * next_vt_i
            value_loss = self.expectile_loss(adv, qn - vs[i], cfg.expectile).mean()
            value_losses.append(value_loss)

        value_loss = sum(value_losses)

        return value_loss

    @jax.jit
    def total_loss(self, params, batch: Batch):
        return self.value_loss(params, batch) + self.high_actor_loss(params, batch) + self.low_actor_loss(params, batch)

    @classmethod
    def create(cls, init_batch, action_space, state_space):
        rng1, rng2 = jax.random.split(jax.random.key(0))
        rng, init_rng = jax.random.split(rng1, 2)

        networks = {
            "value_net": ValueNet(hidden_dims=cfg.value_HD),
            "target_net": ValueNet(hidden_dims=cfg.value_HD),
            "low_lvl_policy": GCActor(hidden_dims=cfg.value_HD, final_dim=action_space, state_dependent_std=False,
                                      const_std=cfg.const_std),
            "high_lvl_policy": GCActor(hidden_dims=cfg.value_HD, final_dim=state_space, state_dependent_std=False,
                                       const_std=cfg.const_std),
        }

        network_args = {
            "value_net": (init_batch.obs, init_batch.value_goals),
            "target_net": (init_batch.obs, init_batch.value_goals),
            "low_lvl_policy": (init_batch.obs, init_batch.low_actor_goals),
            "high_lvl_policy": (init_batch.obs, init_batch.high_actor_goals)
        }

        networks = utils.ModuleDict(networks)
        network_optim = optax.adam(learning_rate=cfg.lr)

        params = networks.init(init_rng, **network_args)["params"]

        network = utils.TrainState.create(networks, params, network_optim)
        return cls(train_state=network, action_space=action_space, state_space=state_space)

    def sample_actions(self, obs):
        pass

    @jax.jit
    def update(self, batch):
        def loss_fn(params):
            return self.total_loss(params, batch)

        new_train_state = self.train_state.apply_loss_and_grad(loss_fn)
        self.train_state.fix_target()

        return self.replace(train_state=new_train_state)
