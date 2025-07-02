"""
Defines a simple autograd engine and uses it to classify points in the plane to 3 classes  
(red, green, blue) using a simple multilayer perceptron (MLP).   
"""

import math 
from utils import RNG, gen_data_yinyang, draw_dot, vis_color 
random = RNG(42) 

# ---------------------------------------------
# Value. Similar to PyTorch's Tensor but only of size 1 element 

class Value:
    """ stores a single scalar value and its gradient """ 
    
    def __init__(self, data, _prev=(), _op=''):
        self.data = data 
        self.grad = 0 
        # internal variables used for autograd graph construction 
        self._backward = lambda: None 
        self._prev = _prev 
        self._op = _op # the op that produced this node, for graphviz / debugging / etc 
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward 
        
        return out 