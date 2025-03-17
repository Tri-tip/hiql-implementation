from HIQL import HIQLAgent
import ogbench
import jax.numpy as jnp
from batch_gen import Dataset
import matplotlib.pyplot as plt
from config import cfg

dataset_name = 'antmaze-medium-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name,
    dataset_dir='datasets',
    compact_dataset=False
)

dataset = Dataset.create(**train_dataset)
hiql_agent = HIQLAgent.create(dataset.sample(1), env.action_space.shape[0], env.observation_space.shape[0])

losses = []

for i in range(cfg.episodes):
    batch = dataset.sample(cfg.batch_size)
    hiql_agent = hiql_agent.update(batch)
    if i % 10 == 0:
        test_batch = dataset.sample(cfg.batch_size)
        loss = hiql_agent.total_loss(hiql_agent.train_state.params, test_batch)
        losses.append(loss)
        print(f"episode {i}: loss = {loss}")

plt.plot(losses)
plt.show()


def evaluate():
    pass
