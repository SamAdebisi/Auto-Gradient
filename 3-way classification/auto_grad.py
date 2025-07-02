"""
Defines a simple autograd engine and uses it to classify points in the plane to 3 classes  
(red, green, blue) using a simple multilayer perceptron (MLP).   
"""

import math 
from utils import RNG, gen_data_yinyang, draw_dot, vis_color 
random = RNG(42) 