import flax.linen as nn
from typing import Any
from config import cfg

# for now, all networks will be basic MLPs
# However, will add all the bells and whistles (ie mean nets) once I get the basic implementation done

# ensemblize; not sure how to implement
def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )

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
