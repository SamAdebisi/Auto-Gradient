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
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward 
        
        return out 
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only suppoprting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad 
        out._backward = _backward 
        
        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad 
        out._backward = _backward 
        
        return out 
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out.data**2) * out.grad 
        out._backward = _backward 
        
        return out 
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward 
        
        return out 
    
    def log(self):
        # this is the natural log 
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1/self.data) * out.grad 
        out._backward = _backward 