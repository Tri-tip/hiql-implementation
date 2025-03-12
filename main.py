from HIQL import HIQLAgent
import ogbench
import jax.numpy as jnp

dataset_name = 'antmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name,
    dataset_dir='datasets',
    compact_dataset=False
)


hiql_agent = HIQLAgent.create(train_dataset)
batch =  hiql_agent.dataset.sample(15)
a, b = hiql_agent.train_state.select("value_net")(batch.obs, batch.value_goals)
print(a.shape, b.shape)