import numpy as  np
import math


class RobotSensorModel:
    SIN_VALUES = [math.sin(i* (180/math.pi)) for i in range(0,361,1)]
    COS_VALUES = [math.cos(i* (180/math.pi)) for i in range(0,361,1)]

    def __init__(self, state_boundary, occupancy_gridmap):
        self.state_boundary = state_boundary
        self.occupancy_gridmap = occupancy_gridmap
        self.radius_fov = 25 
        self.rays = 360
        self.step = math.floor(self.radius_fov/10)
        self.false_positive = 0.2
        self.false_negative = 0.2
    
    def compute_in_fov_bool_grid(self, robot_position):
        in_fov_bool_grid = np.zeros(self.state_boundary)
        in_fov_bool_grid[robot_position[1],robot_position[0]] = 1
        for i in range(0, self.rays + 1, self.step): 
            ax = self.COS_VALUES[i] 
            ay = self.SIN_VALUES[i] 
            x = robot_position[0]
            y = robot_position[1]
            for z in range(0,self.radius_fov): # Cast the ray
                x += ax
                y += ay
                if x < 0 or y < 0 or x > self.state_boundary[0]-1 or y > self.state_boundary[1]-1: # If ray is out of range
                    break
                if self.occupancy_gridmap[int(round(y)),int(round(x))] == 0:  # Stop ray if it hit a wall.
                    break
                in_fov_bool_grid[int(round(y)),int(round(x))] = 1
        return in_fov_bool_grid

    def compute_position_in_fov_list(self, robot_position):
        position_in_fov_list = [] 
        for i in range(0, self.rays + 1, self.step): 
            ax = self.COS_VALUES[i] 
            ay = self.SIN_VALUES[i] 
            x = robot_position[0]
            y = robot_position[1]
            for z in range(0,self.radius_fov): # Cast the ray
                x += ax
                y += ay
                if x < 0 or y < 0 or x > self.state_boundary[0]-1 or y > self.state_boundary[1]-1: # If ray is out of range
                    break
                if self.occupancy_gridmap[int(round(y)),int(round(x))] == 0:  # Stop ray if it hit a wall.
                    break
                position_in_fov_list.append((int(round(x)),int(round(y))))     
        return position_in_fov_list       
    
    def compute_grid_likelihood(self, target_position, robot_position):
        in_fov_bool_grid = self.compute_in_fov_bool_grid(robot_position)
        if np.linalg.norm(np.array(target_position)-np.array(robot_position)) <= self.radius_fov:
            in_fov_cell_likelihood = 1-self.false_positive
            out_fov_cell_likelihood = self.false_positive
        else:
            out_fov_cell_likelihood = 1-self.false_negative
            in_fov_cell_likelihood = self.false_negative

        grid_likelihood = 1.0 * np.ones_like(in_fov_bool_grid)
        for i in range(in_fov_bool_grid.shape[0]):
            for j in range(in_fov_bool_grid.shape[1]):
                if in_fov_bool_grid[j,i] == 0:
                    grid_likelihood[j,i] = out_fov_cell_likelihood
                elif in_fov_bool_grid[j,i] == 1:
                    grid_likelihood[j,i] = in_fov_cell_likelihood
                else:
                    raise ValueError('fov value error')           
        return grid_likelihood
    
    def compute_particle_likelihood(self, target_position, robot_position, position_in_fov_list):
        if (round(robot_position[0]),round(robot_position[1])) in position_in_fov_list:
            if np.linalg.norm(np.array(target_position)-np.array(robot_position)) <= self.radius_fov:
                likelihood = 1-self.false_positive
            else:
                likelihood = self.false_positive
        else:
            if np.linalg.norm(np.array(target_position)-np.array(robot_position)) > self.radius_fov:
                likelihood = 1-self.false_negative
            else:
                likelihood = self.false_negative
        return likelihood
        