import flax.linen as nn
from typing import Any

# for now, all networks will be basic MLPs
# However, will add all the bells and whistles once I get the basic implementation done

# ensemblize; not sure how to implement
def ensemblize():
    pass

class MLP(nn.Module):
    
    hidden_dims: tuple[int, ...]
    activation: Any = nn.relu
    layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x):
        for i, layer_dim in enumerate(self.hidden_dims):
            x = nn.Dense(layer_dim)(x)
            if i < len(self.hidden_dims) - 1:
                x = self.activation(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x
