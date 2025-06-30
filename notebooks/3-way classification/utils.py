# rng related 

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc. 

class RNG: 
    
    def __init__(self, seed):
        self.state = seed 
        
    