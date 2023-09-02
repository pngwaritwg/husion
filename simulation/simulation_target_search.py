from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.ttk import Combobox 
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import scipy
from scipy import stats
import datetime
import os
import json

def load_map_from_test_dataset__and_re_ratio():
    print('loading')
    path = '../building_entrance_street_dataset/chula_engineering/overview_map/map_subdistrict_1_only_building.png'
    map_with_building_resolution_4_to_5 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    map_with_building_resolution_1_to_1 = cv2.resize(map_with_building_resolution_4_to_5, (500,500))
    path = '../building_entrance_street_dataset/chula_engineering/overview_map/map_with_building_street_and_all_entrances.png'
    map_with_building_street_and_all_entrances_4_to_5 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    map_with_building_street_and_all_entrances_resolution_1_to_1 = cv2.resize(map_with_building_street_and_all_entrances_4_to_5, (500,500))
    path = 'map/chula_engineering/map_without_entrance_and_label.png'
    cv2.imwrite(path, map_with_building_resolution_1_to_1)
    path = 'map/chula_engineering/map_with_entrance_and_label.png'
    cv2.imwrite(path, map_with_building_street_and_all_entrances_resolution_1_to_1)

class SimulationTargetSearch():
    def __init__(self,is_histogram_belief_system=True,map_name='demo',is_static_target=True) -> None:
        self.is_histogram_belief_system = is_histogram_belief_system
        if is_histogram_belief_system:
            self.system_type = 'histogram_belief_system'
        else:
            self.system_type = 'particle_filter_system'
        self.is_static_target = is_static_target
        self.map_name = map_name
        self.read_setup_image_for_plot_plan()
        self.initialize_start_robot_target_position()
        self.radius_reach = 25 # default value
        self.radius_fov = 25 # default value
        self.SIN_VALUES = [math.sin(i* (180/math.pi)) for i in range(0,361,1)]
        self.COS_VALUES = [math.cos(i* (180/math.pi)) for i in range(0,361,1)]
        self.setup_create_occupancy_gridmap()
        self.initial_belief = np.ones_like(self.is_no_obstacle_plan) / np.sum(np.ones_like(self.is_no_obstacle_plan) )
        self.is_language_input = False
        self.t = 0
        self.is_sim_running = False
        self.plan = True
        self.snapshot = False
        self.is_robot_observation = False
        self.debug()

    def read_setup_image_for_plot_plan(self):
        self.map_plot_original = cv2.imread(f'map/{self.map_name}/map_with_entrance_and_label.png', cv2.COLOR_BGRA2BGR)
        self.map_plan_original = cv2.imread(f'map/{self.map_name}/map_without_entrance_and_label.png', cv2.COLOR_BGRA2BGR)
        self.padding_size = 100
        self.resize_ratio = 1
        self.plot_dim = (self.map_plot_original.shape[0] - self.padding_size*2, self.map_plot_original.shape[1] - self.padding_size*2)
        self.plan_dim = (self.map_plan_original.shape[0] - (self.padding_size//self.resize_ratio)*2, self.map_plan_original.shape[1] - (self.padding_size//self.resize_ratio)*2)
        self.map_plot = self.map_plot_original[self.padding_size:self.map_plot_original.shape[0]-self.padding_size,self.padding_size:self.map_plot_original.shape[1]-self.padding_size]
        self.map_plan = self.map_plan_original[self.padding_size//self.resize_ratio:self.map_plan_original.shape[0]-self.padding_size//self.resize_ratio, self.padding_size//self.resize_ratio:self.map_plan_original.shape[1]-self.padding_size//self.resize_ratio]
        cv2.imwrite(f'map/{self.map_name}/map_plot.png',self.map_plot)
        # self.map_plot = cv2.merge((self.map_plot,self.map_plot,self.map_plot))

    def debug(self):
        cropped_img = self.map_plan_original[100:400,100:400]
        cv2.imwrite(f'map/{self.map_name}/debug.png',cropped_img)

    def initialize_start_robot_target_position(self):
        self.start_position_robot_plan = (16,16) # initial value
        self.start_position_target_plan = (18,72) # initial value
        self.start_position_robot_plot = (min(self.start_position_robot_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_robot_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        self.start_position_target_plot = (min(self.start_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        
    def setup_create_occupancy_gridmap(self): 
        # background_value = 255
        gray_img = cv2.cvtColor(self.map_plan, cv2.COLOR_BGR2GRAY)
        (_, black_and_white_img) = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'map/{self.map_name}/map_plan.png', black_and_white_img)
        self.is_no_obstacle_plan = (black_and_white_img/255).astype(int) 
        cv2.imwrite(f'map/{self.map_name}/map_plan.png', self.is_no_obstacle_plan*255)
        # is_no_building_plan = self.map_plan_original[self.padding_size//self.resize_ratio:self.map_plan_original.shape[0]-self.padding_size//self.resize_ratio, self.padding_size//self.resize_ratio:self.map_plan_original.shape[1]-self.padding_size//self.resize_ratio]  != building_value
        # cv2.imwrite(f'map/{self.map_name}/map_plan.png', is_no_building_plan*255)
        # self.is_no_obstacle_plan = is_no_building_plan.astype(int)
        self.grid = Grid(matrix=self.is_no_obstacle_plan)
        self.grid_for_target_path = Grid(matrix=self.is_no_obstacle_plan)
        print(f'plan_dim {self.plan_dim}, plot_dim {self.plot_dim}')
   
    def multivariate_gaussian(self, mu, Sigma = np.array([[ 1. , 0.], [0.,  1.]]),figure_size=(100,100)):
        """Return the multivariate Gaussian distribution on array pos."""
        X = np.linspace(0, figure_size[0], figure_size[0])
        Y = np.linspace(0, figure_size[1], figure_size[1])
        X, Y = np.meshgrid(X, Y)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N
        
    def robot_observation_model(self,target_position, robot_position, false_positive = 0.2, false_negative = 0.2, compute_likelihood = True):
        figure_size = self.plan_dim
        rays = 360
        step = math.floor(self.radius_fov/10)
        fov = np.zeros_like(self.is_no_obstacle_plan)
        fov[robot_position[1],robot_position[0]] = 1
        if not compute_likelihood:
            position_in_fov_list = [] 
        for i in range(0, rays + 1, step): 
            ax = self.SIN_VALUES[i] # Get precalculated value sin(x / (180 / pi))
            ay = self.COS_VALUES[i] # cos(x / (180 / pi))
            x = robot_position[0]
            y = robot_position[1]
            for z in range(0,self.radius_fov): # Cast the ray
                x += ax
                y += ay
                if x < 0 or y < 0 or x > figure_size[0]-1 or y > figure_size[1]-1: # If ray is out of range
                    break
                if self.is_no_obstacle_plan[int(round(y)),int(round(x))] == 0:  # Stop ray if it hit a wall.
                    break
                if compute_likelihood:                                    
                    fov[int(round(y)),int(round(x))] = 1
                else:
                    position_in_fov_list.append((int(round(x)),int(round(y))))
        if np.linalg.norm(np.array(target_position)-np.array(robot_position)) <= self.radius_fov:
            fov_cell = 1-false_positive
            outer_cell = false_positive
        else:
            outer_cell = 1-false_negative
            fov_cell = false_negative

        if compute_likelihood:
            # compute the likelihood for a whole space array 
            likelihood = 1.0 * np.ones_like(self.is_no_obstacle_plan)
            for i in range(fov.shape[0]):
                for j in range(fov.shape[1]):
                    if fov[j,i] == 0:
                        likelihood[j,i] = outer_cell
                    elif fov[j,i] == 1:
                        likelihood[j,i] = fov_cell
                    else:
                        raise ValueError('fov value error')           
            return likelihood
        else:
            return position_in_fov_list
    
    def human_observation_model(self, landmark, spatial_relation):
        path = f'../precalculated_human_obs/ld81/{landmark}/{spatial_relation}.csv'
        csv = pd.read_csv(path)
        csv = csv.iloc[:, 1:]
        likelihood = np.array(csv)
        if self.is_negative_input:
            return 1 - likelihood
        else:
            return likelihood

    def dummy_human_observation_model(self,mode,center):
        mu = np.array(center)
        human_observation = np.array(self.multivariate_gaussian(mu,Sigma = np.array([[ 50. , 0.], [0.,  50.]]),figure_size=(100,100)))
        if mode == 'negative':
            human_observation = np.max(human_observation) - human_observation
            human_observation = human_observation / np.sum(human_observation)
        human_observation = human_observation + np.max(human_observation)/(0.1*self.plan_dim[0]*self.plan_dim[1]) 
        human_observation = human_observation / np.sum(human_observation)
        return human_observation

    def create_heatmap(self,distribution):
        distribution_plot = np.repeat(distribution, self.resize_ratio, axis=1).repeat(self.resize_ratio, axis=0)
        distribution_plot = distribution_plot[:self.plot_dim[0],:self.plot_dim[1]]
        heatmapshow = cv2.normalize(distribution_plot, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        return heatmap

    def sample_particle_constant_velocity_motion_model(self,x, w):
        dt = 1
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        f = np.dot(F, x) + w.reshape(-1)
        return f.reshape([-1, 1])
    
    def sample_particle_random_walk_motion_model(self,x, w):
        F = np.array([[1, 0], [0, 1]])
        f = np.dot(F, x) + w.reshape(-1)
        return f.reshape([-1, 1])
    
    # def move_target_path_random_walk_gaussian_step(self, target_position):
    #     sigma = np.diag(np.power([2, 2], 2))
    #     L = np.linalg.cholesky(sigma)
    #     check_wall = True
    #     while check_wall:
    #         noise = np.dot(L, np.random.randn(2, 1)).reshape(-1)
    #         x, y = np.clip(target_position + noise,[0,0],[self.plan_dim[0]-1,self.plan_dim[1]-1])
    #         if self.is_no_obstacle_plan[int(round(y)),int(round(x))] != 0:
    #             check_wall = False
    #     return  [int(round(x)),int(round(y))]

    def move_target_path_with_goal(self):
        if self.t == 0:
            self.is_find_new_target_path = True
            self.path_target_step = 0
            self.target_position_plan_history = []
        if self.is_find_new_target_path:
            path_start = self.grid_for_target_path.node(self.current_true_target_position_plan[0], self.current_true_target_position_plan[1])
            while True:
                self.end_position_target_plan = self.random_position_no_obstacle()
                if self.is_no_obstacle_plan[self.end_position_target_plan[1],self.end_position_target_plan[0]] == 0:
                    continue
                finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                path_end = self.grid_for_target_path.node(self.end_position_target_plan[0],self.end_position_target_plan[1])
                self.target_path, _ = finder.find_path(path_start, path_end, self.grid_for_target_path) # the path also include the start position
                self.grid_for_target_path.cleanup()
                if len(self.target_path) > 1:
                    next_target_position_plan = self.target_path[1] # select the next position in the path
                    self.is_find_new_target_path = False # always prevent finding a new path unless further condition for planning exists
                    self.path_target_step += 1
                    break
                elif len(self.target_path) == 1: # the start and stop position is the same resulting in a path include only a point
                    next_target_position_plan = self.target_path[0] # target repeat its position 
                    self.is_find_new_target_path = False # always prevent finding a new path unless further condition for planning exists
                    break
                else:
                    pass
        else:
            next_target_position_plan = self.target_path[1+self.path_target_step] # if there is no finding a new path, select the next position in the path
            self.path_target_step += 1
        dist_target_plan = np.linalg.norm(np.array(self.end_position_target_plan)-np.array(self.current_true_target_position_plan))
        if dist_target_plan<self.radius_reach:
            # replan the path 
            self.is_find_new_target_path = True
            self.path_target_step = 0
        self.target_position_plan_history.append(self.current_true_target_position_plan)
        self.current_true_target_position_plan = next_target_position_plan
    
    def precalculate_random_walk_gaussian(self):
        pad_x = math.ceil(self.plan_dim[1]/2)
        pad_y = math.ceil(self.plan_dim[0]/2)
        center_x = math.ceil(self.plan_dim[1]/2)
        center_y = math.ceil(self.plan_dim[0]/2)
        x, y = np.meshgrid(np.linspace(-center_x-pad_x,pad_x+self.plan_dim[1]-center_x,self.plan_dim[1]+2*pad_x,endpoint=False), np.linspace(-center_y-pad_y,pad_y+self.plan_dim[0]-center_y,self.plan_dim[0]+2*pad_y,endpoint=False))
        dist = np.sqrt(x*x+y*y)
        sigma = 2.0
        mean = 0
        likelihood = np.exp(-( (dist-mean)**2 / ( 2.0 * sigma**2 ) ) )
        likelihood = likelihood / np.sum(likelihood)
        self.precalculated_gaussian_at_center = likelihood

    def predict_dynamic_random_walk_motion_model(self,prior):
        path = f'histogram_belief_system/debug'
        if self.t == 5:
            likelihood_heatmap = self.create_heatmap(prior)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path+f'/prior_{self.t}.png',likelihood_heatmap)
        posterior = np.zeros_like(prior)
        for x in range(self.plan_dim[1]):
            for y in range(self.plan_dim[0]):
                pad_x = math.ceil(self.plan_dim[1]/2)
                pad_y = math.ceil(self.plan_dim[0]/2)
                center_x = math.ceil(self.plan_dim[1]/2)
                center_y = math.ceil(self.plan_dim[0]/2)
                dx = center_x - x
                dy = center_y - y
                likelihood = self.precalculated_gaussian_at_center[pad_y+dy:pad_y+dy+self.plan_dim[0],pad_x+dx:pad_x+dx+self.plan_dim[1]]
                likelihood = likelihood / np.sum(likelihood)
                posterior[y,x] = np.sum(prior * likelihood)
        if self.t == 5:
            likelihood_heatmap = self.create_heatmap(posterior)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path+f'/posterior_{self.t}.png',likelihood_heatmap)
        return posterior
            
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
    
    def step_particle_filter_system(self,update_display=True):
        print(f'timestep: {self.t}') 
        if self.t == 0:
            self.datetime = datetime.datetime.now()
            # initialize history for further plot
            self.dist_from_target_list = []
            self.add_lang_input_timestep_list = []
            self.robot_position_plan_history = []
            self.start_time = time.time()
            # initialize robot position (x,y)
            self.current_robot_position_plan = self.start_position_robot_plan
            # initialize target position (x,y)
            self.current_true_target_position_plan = self.start_position_target_plan
            # initialize particles and weights
            posterior = self.initial_belief
            # define particle filter dynamic, noise covariance and parameters
            # self.Q = np.diag([1e-1, 1e-1, 1e-2, 1e-2]) # motion model noise covariance (constant velocity)
            self.Q = np.diag([1e-1, 1e-1]) # motion model noise covariance (random walk)
            self.LQ = np.linalg.cholesky(self.Q)
            self.n = 1000 # number of particle filter 
            # self.x = np.zeros([4, 1]) # state of a particle (x,y,dx,dy) (constant velocity)
            # self.x = np.zeros([2, 1]) # state of a particle (x,y) (random walk)
            self.x = np.clip(np.dot(3* self.plan_dim[0] * np.eye(2), np.random.randn(2, 1)), np.array([0,0]).reshape(-1,1), np.array([self.plan_dim[0]-1, self.plan_dim[1]-1]).reshape(-1,1)) # state of a particle (x,y) (random walk)
            Sigma = 5* self.plan_dim[0] * np.eye(len(self.x)) # noise covariance of gaussain for initial sampling
            wu = 1 / self.n  # initial uniform weights
            L_init = np.linalg.cholesky(Sigma)
            # self.px = []
            # self.pw = []
            # for i in range(self.n):
            #     # self.px.append(np.clip(np.dot(L_init, np.random.randn(len(self.x), 1)) + np.array([self.plan_dim[0]/2,self.plan_dim[1]/2,0,0]).reshape(-1,1), np.array([0,0,-100,-100]).reshape(-1,1), np.array([self.plan_dim[0]-1, self.plan_dim[1]-1,100,100]).reshape(-1,1)))
            #     self.px.append(np.clip(np.dot(L_init, np.random.randn(len(self.x), 1)) + np.array([self.plan_dim[0]/2,self.plan_dim[1]/2]).reshape(-1,1), np.array([0,0]).reshape(-1,1), np.array([self.plan_dim[0]-1, self.plan_dim[1]-1]).reshape(-1,1)))
            #     self.pw.append(wu)
            self.px = np.random.uniform(low=(0,0), high=(self.plan_dim[1]-1,self.plan_dim[0]-1), size=(self.n,2))
            self.pw = [[wu]] * self.n
            self.px = np.array(self.px).reshape(-1, len(self.x))
            self.pw = np.array(self.pw).reshape(-1, 1)
            if np.max(self.px[:,0]) > self.plan_dim[1]-1 or np.max(self.px[:,1]) > self.plan_dim[0]-1:
                raise ValueError('initialized state exceed boundary')
        else:
            # sample from motion model
            if not self.is_static_target:
                for i in range(self.n):
                    # sample noise for constant velocity model
                    # w = np.dot(self.LQ, np.random.randn(4, 1))
                    # sample noise for random walk model
                    w = np.dot(self.LQ, np.random.randn(len(self.x), 1))
                    # self.px[i, :] = self.constant_velocity_motion_model(self.px[i, :], w).reshape(-1)
                    # self.px[i, :] = np.clip(self.constant_velocity_motion_model(self.px[i, :], w).reshape(-1), np.array([0,0,-100,-100]), np.array([self.plan_dim[0]-1, self.plan_dim[1]-1,100,100]))
                    # self.px[i, :] = np.clip(self.random_walk_motion_model(self.px[i, :], w).reshape(-1), np.array([0,0,-100,-100]), np.array([self.plan_dim[0]-1, self.plan_dim[1]-1,100,100]))
                    self.px[i, :] = np.clip(self.sample_particle_random_walk_motion_model(self.px[i, :], w).reshape(-1), np.array([0,0]), np.array([self.plan_dim[1]-1, self.plan_dim[0]-1]))
            else:
                # keep the same particles
                pass
            # update importance weights with measurements
            w = np.zeros([self.n, 1]) 
            position_in_fov_list = self.robot_observation_model(self.current_true_target_position_plan, self.current_robot_position_plan,compute_likelihood=False)
            for i in range(self.n):
                if (round(self.px[i,0]),round(self.px[i,1])) in position_in_fov_list:
                    if np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan)) <= self.radius_fov:
                        w[i] = 0.95
                    else:
                        w[i] = 0.05
                else:
                    if np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan)) > self.radius_fov:
                        w[i] = 0.95
                    else:
                        w[i] = 0.05
            # print(f'check max of 1st state: {np.max(self.px[:,0])}')
            # print(f'check max of 2nd state: {np.max(self.px[:,1])}')
            # print(f'check min of particle weight: {np.min(self.pw)}')
            # print(f'check max of particle weight: {np.max(self.pw)}')
            # update and normalize weights
            self.pw = np.multiply(self.pw, w)  
            self.pw = self.pw / np.sum(self.pw)
            wtot = np.sum(self.pw)
            if wtot <= 0:
                raise ValueError('The total weight of particles is less than or equal 0')    

            # check boolean value from the insert button of language input
            if self.is_language_input:
                print('fusing language input')
                # posterior = posterior * self.dummy_human_observation_model(mode='positive',center=current_true_target_position_plan)
                human_likelihood = self.human_observation_model(spatial_relation=self.spatial_relation,landmark=self.landmark)
                for i in range(self.n):
                    w[i] = human_likelihood[round(self.px[i,1]),round(self.px[i,0])]
                # update and normalize weights
                self.pw = np.multiply(self.pw, w)  # since we used motion model to sample
                self.pw = self.pw / np.sum(self.pw)
                # compute total weight
                wtot = np.sum(self.pw)
                if wtot <= 0:
                    raise ValueError('The total weight of particles is less than or equal 0') 
                self.is_language_input = False
                # if insert the language input, replan the path of a robot
                self.plan = True
                posterior = posterior/np.sum(posterior)
                self.add_lang_input_timestep_list.append((self.t,self.is_negative_input))
            # retrieve estimate target position via MAP
            self.x = self.px[np.argmax(self.pw.reshape(-1)),:] 
            # compute effective number of particles
            self.Neff = 1 / np.sum(np.power(self.pw, 2)) 
            print(f'Neff: {self.Neff}')   
            # check if resampling is needed via the effective numner of particles
            if self.Neff < self.n /2:
                self.low_variance_resample()
        
        if self.plan == True:
            # self.current_estimated_target_position_plan = (int(self.x[0][0]),int(self.x[1][0])) # for a weighted mean estimation of particles and their weights
            # self.current_estimated_target_position_plan = (int(round(self.x.reshape(-1)[0])),int(round(self.x.reshape(-1)[1]))) # for a MAP of particles from their weights
            self.current_estimated_target_position_plan_sorted = self.px.copy()[np.argsort(self.pw.reshape(-1))[::-1]] # sort the particle in descending order of its weight
            print(f'shape of test: {self.current_estimated_target_position_plan_sorted.shape}')
            print(f'test: {self.current_estimated_target_position_plan_sorted}')
            self.plan = False
            self.find_new_path = True # when replan, find a new path
            self.path_step = 0
            self.path = None

        # step and path planning
        if self.find_new_path:
            path_start = self.grid.node(self.current_robot_position_plan[0], self.current_robot_position_plan[1])
            for current_estimated_target_position_plan in self.current_estimated_target_position_plan_sorted:
                current_estimated_target_position_plan = np.rint(current_estimated_target_position_plan).astype(int)
                if self.is_no_obstacle_plan[current_estimated_target_position_plan[1],current_estimated_target_position_plan[0]] == 0:
                    continue
                finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                path_end = self.grid.node(current_estimated_target_position_plan[0],current_estimated_target_position_plan[1])
                self.path, _ = finder.find_path(path_start, path_end, self.grid) # the path also include the start position
                self.grid.cleanup()
                next_robot_position_plan = None
                if len(self.path) > 1:
                    next_robot_position_plan = self.path[1] # select the next position in the path
                    self.find_new_path = False # always prevent finding a new path unless further condition for planning exists
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.path_step += 1
                    break
                elif len(self.path) == 1: # the start and stop position is the same resulting in a path include only a point
                    next_robot_position_plan = self.path[0] # robot repeat its position 
                    self.find_new_path = False # always prevent finding a new path unless further condition for planning exists
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    break
                else:
                    pass
        else:
            next_robot_position_plan = self.path[1+self.path_step] # if there is no finding a new path, select the next position in the path
            self.path_step += 1
        
        # create image for the graphic interface
        map_plot_with_particle = self.map_plot.copy()
        cv2.circle(map_plot_with_particle,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(255, 200, 0),thickness=-1)
        cv2.circle(map_plot_with_particle,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=self.radius_fov*self.resize_ratio,color=(100, 100, 100),thickness=1)
        cv2.circle(map_plot_with_particle,(min(self.current_estimated_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_estimated_target_position_plan[1]*self.resize_ratio,self.plot_dim[1])),radius=2,color=(100, 255, 0),thickness=-1)
        cv2.circle(map_plot_with_particle,(min(self.current_true_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_true_target_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(0, 0, 0),thickness=-1)
        for i in range(self.n):
            # print(f'check radius: {6 - min((round(math.log10(1e-3/self.pw[i])),4))}')
            # particle_radius = 6 - min((round(math.log10(1e-3/self.pw[i])),4))
            particle_color = 25 * min((round(math.log10(1e-3/self.pw[i])),10)) # the bigger the weight, the more color intensity 
            # print(f'particle position: {(min(round(self.px[i][0])*self.resize_ratio,self.plot_dim[0]), min(round(self.px[i][1])*self.resize_ratio,self.plot_dim[1]-1))}')
            cv2.circle(map_plot_with_particle,(min(round(self.px[i][0])*self.resize_ratio,self.plot_dim[0]), min(round(self.px[i][1])*self.resize_ratio,self.plot_dim[1]-1)),radius=1,color=(particle_color, particle_color, 255),thickness=cv2.FILLED)
        # check if the robot reach the target 
        dist = np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan))
        self.dist_from_target_list.append(dist*4) #1 plan grid = 4 m
        if dist<self.radius_reach:
            print('             reach target')
            self.reach = True
            self.is_sim_running = False
            # plot distance vs time and robot path 
            self.plot_distance_time()
            self.plot_map_plan_with_path()
            if not self.is_static_target:
                self.save_target_path()
        
        # check if the robot reach the estimated target 
        dist_plan = np.linalg.norm(np.array(self.current_estimated_target_position_plan)-np.array(self.current_robot_position_plan))
        if dist_plan<self.radius_reach:
            print('             reach plan')
            # replan the path 
            self.plan = True

        # update the true target position 
        if not self.is_static_target:
            self.move_target_path_with_goal()
        # update robot position 
        self.current_robot_position_plan = next_robot_position_plan
        # save robot position in history list
        self.robot_position_plan_history.append(self.current_robot_position_plan)
        self.t += 1
        elapsed_time = time.time() - self.start_time
        # calculate velocity of the robot in the simulation for debug
        sim_velocity = self.t * 4 /elapsed_time
        print(f'sim velocity {sim_velocity}')

        # update image in the graphic interface 
        if update_display:
            B_hm,G_hm,R_hm = cv2.split(map_plot_with_particle)
            img_cv_tk = cv2.merge((R_hm,G_hm,B_hm))
            im = Image.fromarray(img_cv_tk)
            self.img_tk = ImageTk.PhotoImage(image=im)
            self.img_label.config(image=self.img_tk)
            
    def step_histogram_belief_system(self,update_display=True): 
        print(f'timestep: {self.t}') 
        self.start_time = time.time()
        # current_true_target_position_plan = self.target_motion_model(1,self.t)
        if self.t == 0:
            self.datetime = datetime.datetime.now()
            # initialize history for further plot
            self.dist_from_target_list = []
            self.add_lang_input_timestep_list = []
            self.robot_position_plan_history = []
            # initialize robot position (x,y)
            self.current_robot_position_plan = self.start_position_robot_plan
            # initialize target position (x,y)
            self.current_true_target_position_plan = self.start_position_target_plan
            # initialize posterior
            posterior = self.initial_belief
            self.random_walk_covariance = np.diag([1e-1, 1e-1])
            if not self.is_static_target:
                self.precalculate_random_walk_gaussian()
        else:
            # propagate belief with dynamic prediction
            if not self.is_static_target:
                posterior = self.predict_dynamic_random_walk_motion_model(self.prior)
            else:
                posterior = self.prior
            # measurement updates
            # collect boolean value from the tickbox button of enable robot observation
            self.is_robot_observation = int(self.is_robot_observation_temp.get())
            if self.is_robot_observation:
                posterior = posterior * self.robot_observation_model(self.current_true_target_position_plan, self.current_robot_position_plan)
            else:
                pass
            posterior = posterior/np.sum(posterior)
            # check boolean value from the insert button of language input
            if self.is_language_input:
                print('fusing language input')
                # posterior = posterior * self.dummy_human_observation_model(mode='positive',center=current_true_target_position_plan)
                posterior = posterior * self.human_observation_model(spatial_relation=self.spatial_relation,landmark=self.landmark)
                self.is_language_input = False
                # if insert the language input, replan the path of a robot
                self.plan = True
                # normalize the posterior
                posterior = posterior/np.sum(posterior)
                self.add_lang_input_timestep_list.append((self.t,self.is_negative_input))
                # capture language input likelihood for debug
                if self.snapshot:
                    language_input_likelihood_heatmap = self.create_heatmap(self.human_observation_model(spatial_relation=self.spatial_relation,landmark=self.landmark))
                    path = f'histogram_belief_system/snapshot/language_input_likelihood'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(path+f'/{self.spatial_relation}_{self.landmark}.png',language_input_likelihood_heatmap)
        
        # retrieve estimate target position via MAP. In order to handle point inside buidling, sort all of the point in descending value order
        if self.plan == True:
            top_sorted_estimated_target_number = self.plan_dim[0]*self.plan_dim[1]
            current_estimated_target_position_plan_sorted = np.argsort(posterior, axis=None)[::-1][:top_sorted_estimated_target_number] 
            self.current_estimated_target_position_plan_sorted = [(np.unravel_index(p, posterior.shape)[1], np.unravel_index(p, posterior.shape)[0]) for p in current_estimated_target_position_plan_sorted]
            self.plan = False
            self.find_new_path = True
            self.path_step = 0
            self.path = None

        # step and path planning
        if self.find_new_path:
            print('find new path')
            equal_groups = [[]]
            k = 0
            for i in range(len(self.current_estimated_target_position_plan_sorted)-1):
                ind = self.current_estimated_target_position_plan_sorted[i]
                val = posterior[ind[1],ind[0]]
                next_ind = self.current_estimated_target_position_plan_sorted[i+1]
                next_val = posterior[next_ind[1],next_ind[0]]
                equal_groups[k].append(ind)
                if val == next_val:
                    pass
                else:
                    k += 1
                    equal_groups.append([])
            last_ind = self.current_estimated_target_position_plan_sorted[len(self.current_estimated_target_position_plan_sorted)-1]
            equal_groups[k].append(last_ind)
            print(f'number of equal groups:{len(equal_groups)}')
            for i in range(len(equal_groups)):
                print(f'# positions in the equal group {i}: {len(equal_groups[i])}')
                if i< 2:
                    print(f'before shuffle equal group 10 samples:{equal_groups[i][:10]}')
                equal_groups[i] = random.sample(equal_groups[i],len(equal_groups[i]))
                if i< 2:
                    print(f'after shuffle equal group 10 samples:{equal_groups[i][:10]}')
            self.current_estimated_target_position_plan_sorted = [ind for equal_group in equal_groups for ind in equal_group]
            path_start = self.grid.node(self.current_robot_position_plan[0], self.current_robot_position_plan[1])
            for current_estimated_target_position_plan in self.current_estimated_target_position_plan_sorted:
                if self.is_no_obstacle_plan[current_estimated_target_position_plan[1],current_estimated_target_position_plan[0]] == 0:
                    continue
                finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                path_end = self.grid.node(current_estimated_target_position_plan[0],current_estimated_target_position_plan[1])
                self.path, _ = finder.find_path(path_start, path_end, self.grid)
                self.grid.cleanup()
                next_robot_position_plan = None
                if len(self.path) > 1:
                    next_robot_position_plan = self.path[1]
                    self.find_new_path = False
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.path_step += 1
                    self.timestep_distance_robot = 1 
                    break
                elif len(self.path) == 1:
                    next_robot_position_plan = self.path[0]
                    self.find_new_path = False
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.timestep_distance_robot = 0
                    break
                else:
                    pass
        else:
            next_robot_position_plan = self.path[1+self.path_step]
            self.path_step += 1
            self.timestep_distance_robot = 1 
        
        # create image for the graphic interface
        posterior_heatmap = self.create_heatmap(posterior)
        map_plot_with_heatmap = cv2.addWeighted(posterior_heatmap, 0.7, self.map_plot.copy(), 0.3, 0)
        cv2.circle(map_plot_with_heatmap,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(255, 200, 0),thickness=-1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=self.radius_fov*self.resize_ratio,color=(100, 100, 100),thickness=1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_estimated_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_estimated_target_position_plan[1]*self.resize_ratio,self.plot_dim[1])),radius=2,color=(255, 0, 255),thickness=-1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_true_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_true_target_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(0, 0, 0),thickness=-1)
        
        # check if the robot reach the target 
        dist = np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan))
        self.dist_from_target_list.append(dist*4) #1 plan grid = 4 m
        if dist<self.radius_reach:
            print('reach target')
            self.reach = True
            self.is_sim_running = False
            # plot distance vs time and robot path 
            self.plot_distance_time()
            self.plot_map_plan_with_path()
            if not self.is_static_target:
                self.save_target_path()
        
        # check if the robot reach the estimated target (plan target)
        dist_plan = np.linalg.norm(np.array(self.current_estimated_target_position_plan)-np.array(self.current_robot_position_plan))
        if dist_plan<self.radius_reach:
            print('reach plan')
            # replan the path 
            self.plan = True

        # update the true target position
        if not self.is_static_target:
            self.move_target_path_with_goal()
        # update robot position 
        self.current_robot_position_plan = next_robot_position_plan
        # save robot position in history list
        self.robot_position_plan_history.append(self.current_robot_position_plan)
        # update next prior
        self.prior = posterior
        self.t += 1
 
        # update image in the graphic interface
        if update_display:
            B_hm,G_hm,R_hm = cv2.split(map_plot_with_heatmap)
            img_cv_tk = cv2.merge((R_hm,G_hm,B_hm))
            im = Image.fromarray(img_cv_tk)
            self.img_tk = ImageTk.PhotoImage(image=im)
            self.img_label.config(image=self.img_tk)
        self.timestep_time = time.time() - self.start_time
        if self.snapshot:
            path = f'histogram_belief_system/snapshot/posterior/{self.datetime}'
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path+f'/{self.start_position_robot_plan}_{self.start_position_target_plan}_{len(self.add_lang_input_timestep_list)}_{self.is_robot_observation}_{self.t}.png',map_plot_with_heatmap)

    def plot_distance_time(self):
        fig, ax = plt.subplots()
        ax.plot(self.dist_from_target_list)
        ax.set_title('Distance of the robot from the target vs Timestep')
        ax.set_xlabel('timestep')
        ax.set_ylabel('distance of the robot from the target (m)')
        for added_t in self.add_lang_input_timestep_list:
            if added_t[1]:
                ax.axvline(x=added_t[0], color='r')
            else:
                ax.axvline(x=added_t[0], color='g')
        path = f'{self.system_type}/plot/distance_time/{self.datetime}'
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path+f'/{self.start_position_robot_plan}_{self.start_position_target_plan}_{len(self.add_lang_input_timestep_list)}_{self.is_robot_observation}')
        canvas=FigureCanvasTkAgg(fig,master=self.window)
        canvas.get_tk_widget().grid(row=1,column=3)
        canvas.draw()
        fig.clf()
    
    def plot_map_plan_with_path(self):
        map_plan_with_path = self.map_plot.copy()
        robot_position_history_plot = []
        for pos in self.robot_position_plan_history:
            robot_position_history_plot.append((min(pos[0]*self.resize_ratio,self.plot_dim[0]-1),min(pos[1]*self.resize_ratio,self.plot_dim[1]-1)))
        robot_position_history_plot = np.array(robot_position_history_plot)
        cv2.polylines(map_plan_with_path, [robot_position_history_plot], isClosed=False, color=(0,255,0), thickness=1)
        cv2.circle(map_plan_with_path, self.start_position_robot_plot, radius=2, color=(255,200,0),thickness=-1)
        cv2.circle(map_plan_with_path, self.start_position_target_plot, radius=2, color=(0,0,255),thickness=-1)
        path = f'{self.system_type}/plot/path/{self.datetime}'
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+f'/{self.start_position_robot_plan}_{self.start_position_target_plan}_{len(self.add_lang_input_timestep_list)}_{self.is_robot_observation}.png',map_plan_with_path)

    def save_target_path(self):
        path = f'{self.system_type}/target_path_plan_history/{self.datetime}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+f'/{self.start_position_robot_plan}_{self.start_position_target_plan}_{len(self.add_lang_input_timestep_list)}_{self.is_robot_observation}.json', "w") as file:
            json.dump(self.target_position_plan_history, file)

    def reset_img_plot(self,first_display):
        map_plot_with_target_robot_start = cv2.circle(self.map_plot.copy(),self.start_position_robot_plot,radius=2,color=(255, 200, 0),thickness=-1)
        cv2.circle(map_plot_with_target_robot_start,self.start_position_target_plot,radius=2,color=(0, 0, 255),thickness=-1)
        B_hm,G_hm,R_hm = cv2.split(map_plot_with_target_robot_start)
        img_cv_tk = cv2.merge((R_hm,G_hm,B_hm))
        im = Image.fromarray(img_cv_tk)
        self.img_tk = ImageTk.PhotoImage(image=im)
        if first_display:
            pass
        else:
            self.img_label.config(image=self.img_tk)
    
    def random_position_initial_condition(self):
        while True:
            start_robot_position_plan = self.random_position_no_obstacle()
            start_target_position_plan = self.random_position_no_obstacle()
            dist = ((start_robot_position_plan[0] - start_target_position_plan[0])**2 + (start_robot_position_plan[1] - start_target_position_plan[1])**2)**0.5
            if dist < 50:
                continue
            self.start_position_robot_plan = start_robot_position_plan
            self.start_position_target_plan = start_target_position_plan
            self.end_position_target_plan = start_target_position_plan
            self.start_position_robot_plot = (min(self.start_position_robot_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_robot_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
            self.start_position_target_plot = (min(self.start_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
            self.end_position_target_plot = (min(self.end_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.end_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
            print(f'random positon start_robot_position_plan:{start_robot_position_plan} start_position_target_plan:{start_target_position_plan}')
            break

    def random_position_no_obstacle(self):
        # need rework 
        positions_wih_no_obstacle = np.argwhere(self.is_no_obstacle_plan)
        random_idx = random.randint(0,positions_wih_no_obstacle.shape[0]-1)
        random_position = positions_wih_no_obstacle[random_idx]
        return (random_position[1],random_position[0])
        # while True:
        #     p = random.randint(0,self.is_no_obstacle_plan.shape[0]-1)
        #     q = random.randint(0,self.is_no_obstacle_plan.shape[1]-1)
        #     if self.is_no_obstacle_plan[q, p] == 1:
        #         return (p,q)

    def random_position_reset_plot(self):
        self.random_position_initial_condition()
        self.reset_img_plot(first_display=False)

    def set_position(self):
        self.start_position_robot_plan = (int(self.start_position_x_robot_textbox.get()), int(self.start_position_y_robot_textbox.get()))
        self.start_position_target_plan = (int(self.start_position_x_target_textbox.get()), int(self.start_position_y_target_textbox.get()))
        self.end_position_target_plan = (int(self.start_position_x_target_textbox.get()), int(self.start_position_y_target_textbox.get()))
        self.start_position_robot_plot = (min(self.start_position_robot_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_robot_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        self.start_position_target_plot = (min(self.start_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        self.end_position_target_plot = (min(self.end_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.end_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
    
    def set_position_reset_plot(self):
        self.set_position()
        self.reset_img_plot(first_display=False)

    def insert_language_input(self):
        self.landmark = int(self.landmark_box.get())
        self.is_negative_input = int(self.is_negative_input_temp.get())
        self.spatial_relation = self.spatial_relation_box.get()
        self.is_language_input = True
        
    def start(self):
         if not self.is_sim_running:
            self.is_sim_running = True
            self.simulate()

    def simulate(self):
        """Step the simulation"""
        if self.is_sim_running:
            if self.is_histogram_belief_system:
                self.step_histogram_belief_system() 
            else:
                self.step_particle_filter_system()
            desired_velocity = 42
            base_delay = 10 # delay for visualization in gui
            if self.timestep_distance_robot > 0:
                desired_period = self.timestep_distance_robot  * 1000 / desired_velocity
                if desired_period - self.timestep_time <= base_delay:
                    print('computation bottlenck the velocity of robot')
                delay =  round(max((desired_period - self.timestep_time),base_delay))
            else:
                delay = max(self.timestep_time,base_delay)
            self.window.after(delay, self.simulate)

    def pause(self):
        """Pause the simulation"""
        self.is_sim_running = False 

    def reset(self):
        self.is_sim_running = False 
        self.plan = True
        self.t = 0
        self.reset_img_plot(first_display=False)

    def set_snapshot(self):
        self.snapshot = True

    def display(self):
        self.window = tk.Tk()
        self.is_sim_running = False
        self.reset_img_plot(first_display=True)
        frame_img_label = tk.Frame(
            master=self.window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_img_label.grid(row=1, column=1, padx=5, pady=5)

        frame_control = tk.Frame(
            master=self.window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_control.grid(row=1, column=2, padx=5, pady=5)

        frame_start_position_robot_label = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_robot_label.grid(row=1, column=1, padx=5, pady=5)

        frame_start_position_x_robot_textbox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_x_robot_textbox.grid(row=1, column=2, padx=5, pady=5)

        frame_start_position_y_robot_textbox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_y_robot_textbox.grid(row=1, column=3, padx=5, pady=5)

        frame_start_position_target_label = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_target_label.grid(row=2, column=1, padx=5, pady=5)

        frame_start_position_x_target_textbox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_x_target_textbox.grid(row=2, column=2, padx=5, pady=5)

        frame_start_position_y_target_textbox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_position_y_target_textbox.grid(row=2, column=3, padx=5, pady=5)
        
        frame_apply_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_apply_button.grid(row=3, column=2, padx=5, pady=5)

        frame_random_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_random_button.grid(row=3, column=3, padx=5, pady=5)

        frame_enable_robot_observation = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_enable_robot_observation.grid(row=3, column=4, padx=5, pady=5)

        frame_start_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_start_button.grid(row=4, column=1, padx=5, pady=5)

        frame_pause_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_pause_button.grid(row=4, column=2, padx=5, pady=5)

        frame_reset_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_reset_button.grid(row=4, column=3, padx=5, pady=5)

        frame_snapshot_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_snapshot_button.grid(row=4, column=4, padx=5, pady=5)

        frame_negative_input_tickbox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_negative_input_tickbox.grid(row=5, column=1, padx=5, pady=5)

        frame_sr_label = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_sr_label.grid(row=5, column=2, padx=5, pady=5)

        frame_sr_combobox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_sr_combobox.grid(row=5, column=3, padx=5, pady=5)

        frame_ld_label = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_ld_label.grid(row=6, column=2, padx=5, pady=5)

        frame_ld_combobox = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_ld_combobox.grid(row=6, column=3, padx=5, pady=5)

        frame_insert_button = tk.Frame(
            master=frame_control,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame_insert_button.grid(row=7, column=2, padx=5, pady=5)
        
        self.img_label = tk.Label(frame_img_label, image=self.img_tk)
        start_position_robot_label = tk.Label(frame_start_position_robot_label, text='start position of robot (x,y)')
        self.start_position_x_robot_textbox = tk.Entry(frame_start_position_x_robot_textbox)
        self.start_position_y_robot_textbox = tk.Entry(frame_start_position_y_robot_textbox)
        start_position_target_label = tk.Label(frame_start_position_target_label, text='static position of target (x,y)')
        self.start_position_x_target_textbox = tk.Entry(frame_start_position_x_target_textbox)
        self.start_position_y_target_textbox = tk.Entry(frame_start_position_y_target_textbox)
        apply_button = tk.Button(frame_apply_button, text="Apply", command=self.set_position_reset_plot)
        random_button = tk.Button(frame_random_button, text="Random", command=self.random_position_reset_plot)
        self.is_robot_observation_temp = tk.IntVar()
        robot_observation_tickbox = tk.Checkbutton(frame_enable_robot_observation, text='enable robot sensor', variable=self.is_robot_observation_temp)
        start_button = tk.Button(frame_start_button, text="Start/Unpause", command=self.start)
        pause_button = tk.Button(frame_pause_button, text="Pause", command=self.pause)
        reset_button = tk.Button(frame_reset_button, text="Reset", command=self.reset)
        snapshot_button = tk.Button(frame_snapshot_button, text="Snapshot", command=self.set_snapshot)
        self.is_negative_input_temp = tk.IntVar()
        negative_input_tickbox = tk.Checkbutton(frame_negative_input_tickbox, text='not', variable=self.is_negative_input_temp)
        spatail_relation_label = tk.Label(frame_sr_label, text='spatial relation')
        self.spatial_relation_box = tk.StringVar()
        sr_combobox = Combobox(frame_sr_combobox, width = 27, textvariable = self.spatial_relation_box)
        sr_combobox['values'] = ('front','inside','near','far')
        landmark_label = tk.Label(frame_ld_label, text='landmark')
        self.landmark_box = tk.StringVar()
        ld_combobox = Combobox(frame_ld_combobox, width = 27, textvariable = self.landmark_box)
        ld_combobox['values'] = ('1','2','3','4','5','6','7')
        insert_button = tk.Button(frame_insert_button, text="Insert", command=self.insert_language_input)
        self.img_label.pack()
        start_position_robot_label.pack()
        self.start_position_x_robot_textbox.pack()
        self.start_position_y_robot_textbox.pack()
        start_position_target_label.pack()
        self.start_position_x_target_textbox.pack()
        self.start_position_y_target_textbox.pack()
        apply_button.pack()
        random_button.pack()
        robot_observation_tickbox.pack()
        start_button.pack()
        pause_button.pack()
        reset_button.pack()
        snapshot_button.pack()
        negative_input_tickbox.pack()
        spatail_relation_label.pack()
        sr_combobox.pack()
        landmark_label.pack()
        ld_combobox.pack()
        insert_button.pack()
        self.window.mainloop()

if __name__ == "__main__":
    load_map_from_test_dataset__and_re_ratio()
    scenario = SimulationTargetSearch(is_histogram_belief_system=True,map_name='chula_engineering',is_static_target=True)
    scenario.display()
    scenario.start()