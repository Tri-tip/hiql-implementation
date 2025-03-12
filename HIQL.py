import jax
import jax.numpy as jnp
from batch_gen import Batch
import config
from nets import LowPolicy, HighPolicy, ValueNet
import util as utils
import optax 

class HIQLAgent:
    # expectile loss: MSE, but weight adv >= 0 differently than adv < 0
    def expectile_loss(ME, adv, expectile = config.expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (ME ** 2)

    @jax.jit
    def high_actor_loss(self, batch: Batch): # standard A2C loss method
        print(batch.high_actor_goals)

    @jax.jit
    def low_actor_loss(self, batch: Batch): # standard A2C loss method
        pass

    @jax.jit
    def value_loss(self, batch: Batch):
        pass


    @jax.jit
    def total_loss(self, batch: Batch):
        pass
        

    def create(self):
        rng = jax.random.PRNGKey(config.seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        self.networks = {
            "value_net": ValueNet(hidden_dims=(5, 6, 7)),
            "target_net": ValueNet(hidden_dims=(5, 6, 7)),
            "low_lvl_policy": LowPolicy(hidden_dims=(5, 6, 7)),
            "high_lvl_policy": HighPolicy(hidden_dims=(5, 6, 7)),
        }
        network_args = {
            "value_net": (jnp.ones((5, 7)), jnp.ones((5, 7))),
            "target_net": (jnp.ones((7, 9)), jnp.ones((7, 9))),
            "low_lvl_policy": (jnp.ones((8, 10)), jnp.ones((8, 10))),
            "high_lvl_policy": (jnp.ones((9, 11)), jnp.ones((9, 11)))
        }
        
        networks = utils.ModuleDict(self.networks)
        network_optim = optax.adam(learning_rate=config.lr)

        params = networks.init(init_rng, **network_args)["params"]

        network = utils.TrainState.create(networks, params, network_optim, {})
        print(network.select("value_net")(jnp.ones((5, 7)), jnp.ones((5, 7))))
        network.select("low_lvl_policy")(jnp.ones((8, 10)), jnp.ones((8, 10)))
    
    def sample_actions(self, obs): 
        pass
    
    def update(self):
        pass
    
    
hiql_agent = HIQLAgent()
hiql_agent.create()