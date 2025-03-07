from flax import linen as nn
from typing import Any

# for now, all networks will be basic MLPs
# However, will add all the bells and whistles once I get the basic implementation done

class MLP(nn.Module):
    layer_dims: list
    layers: list = []
    layer_norm: bool = True
    activation: Any = nn.relu
    activate_last: bool = False

    def setup(self):
        self.layers = []
        for layer_dim in self.layer_dims:
            self.layers.append(nn.Dense(layer_dim))

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)

        if self.activate_last:
            x = self.activation(x)

        return x
