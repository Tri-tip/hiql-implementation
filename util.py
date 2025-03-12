import flax, flax.linen as nn
from typing import Any
import functools

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# collection of all 5 neural networks together
class ModuleDict(nn.Module):
    modules: dict[str, nn.Module]
    
    def __call__(self, *args, module=None, **kwargs):
        assert module is None or module in self.modules.keys()
        
        # initial pass is to set it up, set up all the parameters 
        if not module:
            assert kwargs.keys() == self.modules.keys()
            outs = {}
            for k in kwargs.keys():
                v = kwargs[k] 
                if isinstance(v, tuple):
                    outs[k] = self.modules[k](*v)
                else: 
                    outs[k] = self.modules[k](v)
                
            return outs

        # otherwise, just call whatever network we were asked to cool
        return self.modules[module](*args)
        
    
class TrainState(flax.struct.PyTreeNode):
    step: int
    config: dict
    params: Any
    apply_fn: Any = nonpytree_field()
    model_arch: Any = nonpytree_field()
    tx: Any = nonpytree_field()
    opt_state: Any = None

    
    # custom trainstate, wraps Flax's builtin TrainState
    @classmethod
    def create(cls,
        model_arch, # will be the grand moduledict
        params, # will be set of params initialized on moduledict
        tx, # optax optim
        config # dict from config.py
    ): 
        if tx:
            opt_state = tx.init(params)
            
            
        return cls(
            step=0,
            apply_fn=model_arch.apply,
            model_arch=model_arch,
            tx=tx,
            opt_state = opt_state,
            config = config,
            params = params
        )
    
    def __call__(self, *args, params=None, module=None):
        # if no parameters specified, then use the stored ones
        # else, if we explicitly provide parameters, then can gradient-optimize params 
        if not params:
            params = self.params

        return self.apply_fn({"params": params}, *args, module=module)
    
    def select(self, network_name):
        return functools.partial(self, module=network_name)