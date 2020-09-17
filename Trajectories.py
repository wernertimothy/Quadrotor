import numpy as np

class Lemniscate:

    def __init__(self, the_T):
        self.T = the_T

        self.__x = lambda the_time : np.cos( the_time/self.T*2*np.pi)
        self.__y = lambda the_time : np.sin( the_time/self.T*2*np.pi)*np.cos( the_time/self.T*2*np.pi)
    
    def setDuration(self, the_duration):
        self.T = the_duration

    def visualize(self):
        Lem = np.zeros( (2,1000) )
        for pos, val in enumerate(np.linspace(0,1,1000)):
            Lem[0,pos] = np.cos(val*2*np.pi)
            Lem[1,pos] = np.sin(val*2*np.pi)*np.cos(val*2*np.pi)
        return Lem

    def evaluate(self, the_time):
        return np.array([self.__x(the_time), self.__y(the_time)])