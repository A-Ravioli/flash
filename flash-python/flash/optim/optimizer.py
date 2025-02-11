"""
Base class for all optimizers.
"""

from typing import Dict, Iterable, Optional, Union

from .. import Variable, Tensor

class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, params: Iterable[Variable], defaults: Dict):
        self.defaults = defaults
        self.state: Dict[Variable, Dict] = {}
        self.param_groups: list[Dict] = []
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
            
        for param_group in param_groups:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group: Dict) -> None:
        """Add a param group to the optimizer's param groups."""
        if not isinstance(param_group, dict):
            param_group = {'params': param_group}
            
        params = param_group['params']
        if isinstance(params, Variable):
            params = [params]
            
        param_group['params'] = list(params)
        
        for param in param_group['params']:
            if not isinstance(param, Variable):
                raise TypeError("optimizer can only optimize Variables")
            if not param.requires_grad:
                raise ValueError("optimizing a parameter that doesn't require gradients")
            
        for name, default in self.defaults.items():
            if name != 'params':
                param_group.setdefault(name, default)
                
        self.param_groups.append(param_group)
    
    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.zero_grad()
    
    def step(self) -> None:
        """Performs a single optimization step (parameter update)."""
        raise NotImplementedError
    
    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a dict."""
        state_dict = {
            'state': self.state,
            'param_groups': self.param_groups,
        }
        return state_dict
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state."""
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups'] 