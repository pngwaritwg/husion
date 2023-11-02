import numpy as np


class ParticleFilter:
    def __init__(self, state_boundary, robot_sensor_model):
        self.state_boundary = state_boundary
        self.robot_sensor_model = robot_sensor_model
        self.Q = np.diag([1e-1, 1e-1])  # Motion model noise covariance (random walk).
        self.LQ = np.linalg.cholesky(self.Q)
        self.n = 1000 # Number of particle filter 
        wu = 1 / self.n  # Initial uniform weights.
        self.px = np.random.uniform(low=(0,0), high=(self.state_boundary[1]-1,self.state_boundary[0]-1), size=(self.n,2))  # particle state
        self.pw = [[wu]] * self.n  # particle weight
        self.px = np.array(self.px).reshape(-1, len(self.state_boundary))
        self.pw = np.array(self.pw).reshape(-1, 1)
        if np.max(self.px[:,0]) > self.state_boundary[1]-1 or np.max(self.px[:,1]) > self.state_boundary[0]-1:
            raise ValueError('initialized state exceeds boundary')
    
    def propagate_particle_random_walk_motion_model(self,x, w):
        F = np.array([[1, 0], [0, 1]])
        f = np.dot(F, x) + w.reshape(-1)
        return f.reshape([-1, 1])
        
    def sample_particle_motion_model(self):
        """Sample noise for random walk model""" 
        for i in range(self.n):
            w = np.dot(self.LQ, np.random.randn(len(self.state_boundary), 1))
            self.px[i, :] = np.clip(self.propagate_particle_random_walk_motion_model(self.px[i, :], w).reshape(-1), np.array([0,0]), np.array([self.state_boundary[1]-1, self.state_boundary[0]-1]))
    
    def update_importance_weight_robot_sensor_observation(self, target_position, robot_position):
        w = np.zeros([self.n, 1]) 
        position_in_fov_list = self.robot_sensor_model.compute_position_in_fov_list(robot_position)
        for i in range(self.n):
            w[i] = self.robot_sensor_model.compute_particle_likelihood(target_position, (self.px[i,0],self.px[i,1]), position_in_fov_list)
        self.pw = np.multiply(self.pw, w)  
        self.pw = self.pw / np.sum(self.pw)
        self.wtot = np.sum(self.pw)
    
    def update_importance_weight_human_observation(self, human_likelihood):
        w = np.zeros([self.n, 1])
        for i in range(self.n):
            w[i] = human_likelihood[round(self.px[i,1]),round(self.px[i,0])]
        self.pw = np.multiply(self.pw, w)  
        self.pw = self.pw / np.sum(self.pw)
        self.wtot = np.sum(self.pw)

    def low_variance_resample(self):
        W = np.cumsum(self.pw)
        r = np.random.rand(1) / self.n
        j = 1
        for i in range(self.n):
            u = r + (i - 1) / self.n
            while u > W[j]:
                j = j + 1
            self.px[i, :] = self.px[j, :]
            self.pw[i] = 1 / self.n
    
    def clip_particle_weight(self):
        self.pw = np.clip(self.pw, 1e-15, 1e15)
