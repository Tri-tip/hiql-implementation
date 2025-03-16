from HIQL import HIQLAgent
import ogbench
import jax.numpy as jnp
from batch_gen import Dataset



dataset_name = 'antmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name,
    dataset_dir='datasets',
    compact_dataset=False
)

dataset = Dataset.create(**train_dataset)
hiql_agent = HIQLAgent.create(dataset.sample(1), env.action_space.shape[0], env.observation_space.shape[0])

batch = dataset.sample(5)

a, b = hiql_agent.train_state.select("value_net")(batch.obs, batch.value_goals)
d = hiql_agent.train_state.select("low_lvl_policy")(batch.obs, batch.low_actor_goals)
print(hiql_agent.value_loss(hiql_agent.train_state.params, batch))
print(hiql_agent.low_actor_loss(hiql_agent.train_state.params, batch))
print(hiql_agent.high_actor_loss(hiql_agent.train_state.params, batch))
print(hiql_agent.total_loss(hiql_agent.train_state.params, batch))
