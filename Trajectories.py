import numpy as np

class Lemniscate:
    def __init__(self, the_T):
        self.T = the_T
    
    def setDuration(self, the_duration):
        self.T = the_duration
    
    def evaluate(self, the_time):
        x = np.sin( the_time/self.T*2*np.pi)
        y = np.sin( the_time/self.T*2*np.pi)*np.cos( the_time/self.T*2*np.pi)
        return np.array([x, y])