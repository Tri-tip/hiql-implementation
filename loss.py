import jax, jax.numpy as jnp
from batch_gen import Batch
import config
from HIQL import HIQLAgent

# expectile loss: MSE, but weight adv >= 0 differently than adv < 0
def expectile_loss(ME, adv, expectile = config.expectile):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (ME ** 2)

class LossCalculator:
    # to use the networks of such agent
    def __init__(self, agent: HIQLAgent):
        self.agent = agent

    def high_actor_loss(self, batch: Batch): # standard A2C loss method
        print(batch.high_actor_goals)

    def low_actor_loss(self, batch: Batch): # standard A2C loss method
        pass

    def value_loss(self, batch: Batch):
        pass


@jax.jit
def total_loss(batch: Batch):
    pass