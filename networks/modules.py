from flax import nnx
import jax.numpy as jnp
from typing import Any

# for now, all networks will be basic MLPs
# However, will add all the bells and whistles once I get the basic implementation done

class MLP(nnx.Module):
    activation: Any = nnx.relu

    def __init__(self, hidden_dims: tuple[int, ...], in_size: int, out_size: int, rngs: nnx.Rngs, layer_norm: bool = True):
        self.layers = []
        self.in_size = in_size
        self.out_size = out_size
        self.layer_norm = layer_norm
        self.hidden_dims = hidden_dims
        self.rngs = rngs

        self.layer_dims = hidden_dims + tuple([out_size])

        self.layers.append(nnx.Linear(in_size, hidden_dims[0], rngs=rngs))
        for i, layer_dim in enumerate(self.hidden_dims): 
            if i == len(hidden_dims) - 1:
                self.layers.append(nnx.Linear(layer_dim, out_size, rngs=rngs))
            else: 
                self.layers.append(nnx.Linear(layer_dim, hidden_dims[i+1], rngs=rngs))

    # have let activating last always be false; feel like that's never used 
    # no layernorm at end right? normalizing doesnt make sense there
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                if self.layer_norm:
                    x = nnx.LayerNorm(self.layer_dims[i], rngs=self.rngs)(x)
        return x

testModel = MLP((200, 400, 600, 800), 3, 2, nnx.Rngs(0))
nnx.display(testModel)