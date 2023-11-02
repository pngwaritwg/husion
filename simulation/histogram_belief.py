import numpy as np
import math


class HistogramBelief:
    def __init__(self, state_boundary):
        self.state_boundary = state_boundary
        self.initial_belief = np.ones(self.state_boundary) / np.sum(np.ones(self.state_boundary))
        self.belief = self.initial_belief
    
    def precalculate_random_walk_gaussian(self):
        pad_x = math.ceil(self.state_boundary[1]/2)
        pad_y = math.ceil(self.state_boundary[0]/2)
        center_x = math.ceil(self.state_boundary[1]/2)
        center_y = math.ceil(self.state_boundary[0]/2)
        x, y = np.meshgrid(np.linspace(-center_x-pad_x,pad_x+self.state_boundary[1]-center_x,self.state_boundary[1]+2*pad_x,endpoint=False), np.linspace(-center_y-pad_y,pad_y+self.state_boundary[0]-center_y,self.state_boundary[0]+2*pad_y,endpoint=False))
        dist = np.sqrt(x*x + y*y)
        sigma = 2.0
        mean = 0
        likelihood = np.exp(-((dist-mean)**2 / (2.0*sigma**2)))
        likelihood = likelihood/np.sum(likelihood)
        self.precalculated_gaussian_at_center = likelihood

    def propagate_belief_random_walk_motion_model(self):
        posterior = np.zeros_like(self.belief)
        for x in range(self.state_boundary[1]):
            for y in range(self.plan_dim[0]):
                pad_x = math.ceil(self.state_boundary[1]/2)
                pad_y = math.ceil(self.state_boundary[0]/2)
                center_x = math.ceil(self.state_boundary[1]/2)
                center_y = math.ceil(self.state_boundary[0]/2)
                dx = center_x-x
                dy = center_y-y
                likelihood = self.precalculated_gaussian_at_center[pad_y+dy : pad_y+dy+self.state_boundary[0], pad_x+dx : pad_x+dx+self.state_boundary[1]]
                likelihood = likelihood/np.sum(likelihood)
                posterior[y,x] = np.sum(self.belief*likelihood)
        self.belief = posterior
    
    def update_belief(self, likelihood):
        self.belief = self.belief*likelihood
        self.belief = self.belief/np.sum(self.belief)