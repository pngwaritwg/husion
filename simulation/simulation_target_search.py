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
from particle_filter import ParticleFilter
from historgram_belief import HistogramBelief
from robot_sensor_model import RobotSensorModel
import datetime
import os
import json

def load_map_from_test_dataset_and_change_resolution():
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
        self.setup_image_for_plot_plan()
        self.setup_start_robot_target_position()
        self.setup_occupancy_gridmap()
        self.radius_reach = 25 
        self.is_language_input = False
        self.t = 0
        self.is_sim_running = False
        self.is_plan = True
        self.is_snapshot = False
        self.is_robot_observation = False

    def setup_image_for_plot_plan(self):
        self.map_plot_original = cv2.imread(f'map/{self.map_name}/map_with_entrance_and_label.png', cv2.COLOR_BGRA2BGR)
        self.map_plan_original = cv2.imread(f'map/{self.map_name}/map_without_entrance_and_label.png', cv2.COLOR_BGRA2BGR)
        self.padding_size = 100
        self.resize_ratio = 1
        self.plot_dim = (self.map_plot_original.shape[0] - self.padding_size*2, self.map_plot_original.shape[1] - self.padding_size*2)
        self.plan_dim = (self.map_plan_original.shape[0] - (self.padding_size//self.resize_ratio)*2, self.map_plan_original.shape[1] - (self.padding_size//self.resize_ratio)*2)
        self.map_plot = self.map_plot_original[self.padding_size:self.map_plot_original.shape[0]-self.padding_size,self.padding_size:self.map_plot_original.shape[1]-self.padding_size]
        self.map_plan = self.map_plan_original[self.padding_size//self.resize_ratio:self.map_plan_original.shape[0]-self.padding_size//self.resize_ratio, self.padding_size//self.resize_ratio:self.map_plan_original.shape[1]-self.padding_size//self.resize_ratio]
        cv2.imwrite(f'map/{self.map_name}/map_plot.png',self.map_plot)

    def setup_start_robot_target_position(self):
        self.start_position_robot_plan = (16,16)  # Default value
        self.start_position_target_plan = (18,72)  # Default value
        self.start_position_robot_plot = (min(self.start_position_robot_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_robot_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        self.start_position_target_plot = (min(self.start_position_target_plan[0]*self.resize_ratio, self.plot_dim[0]-1), min(self.start_position_target_plan[1]*self.resize_ratio, self.plot_dim[1]-1))
        
    def setup_occupancy_gridmap(self): 
        # background_value = 255
        gray_img = cv2.cvtColor(self.map_plan, cv2.COLOR_BGR2GRAY)
        (_, black_and_white_img) = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'map/{self.map_name}/map_plan.png', black_and_white_img)
        self.is_no_obstacle_plan = (black_and_white_img/255).astype(int) 
        cv2.imwrite(f'map/{self.map_name}/map_plan.png', self.is_no_obstacle_plan*255)
        self.grid = Grid(matrix=self.is_no_obstacle_plan)
        self.grid_for_target_path = Grid(matrix=self.is_no_obstacle_plan)
        print(f'plan_dim {self.plan_dim}, plot_dim {self.plot_dim}')
    
    def load_human_observation_model(self, landmark, spatial_relation):
        path = f'../precalculated_human_obs/ld81/{landmark}/{spatial_relation}.csv'
        csv = pd.read_csv(path)
        csv = csv.iloc[:, 1:]
        likelihood = np.array(csv)
        if self.is_negative_input:
            return 1 - likelihood
        else:
            return likelihood

    def create_heatmap(self,distribution):
        distribution_plot = np.repeat(distribution, self.resize_ratio, axis=1).repeat(self.resize_ratio, axis=0)
        distribution_plot = distribution_plot[:self.plot_dim[0],:self.plot_dim[1]]
        heatmapshow = cv2.normalize(distribution_plot, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        return heatmap

    def move_target_along_path_to_goal(self):
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
                self.target_path, _ = finder.find_path(path_start, path_end, self.grid_for_target_path)  # The path also include the start position.
                self.grid_for_target_path.cleanup()
                if len(self.target_path) > 1:
                    next_target_position_plan = self.target_path[1]  # Select the next position in the path.
                    self.is_find_new_target_path = False  # Always prevent finding a new path unless further condition for planning exists.
                    self.path_target_step += 1
                    break
                elif len(self.target_path) == 1:  # The start and stop position is the same resulting in a path include only a point.
                    next_target_position_plan = self.target_path[0]  # The target repeat its position. 
                    self.is_find_new_target_path = False  # Always prevent finding a new path unless further condition for planning exists.
                    break
                else:
                    pass
        else:
            next_target_position_plan = self.target_path[1+self.path_target_step]  # If there is no finding a new path, select the next position in the path.
            self.path_target_step += 1
        dist_target_plan = np.linalg.norm(np.array(self.end_position_target_plan)-np.array(self.current_true_target_position_plan))
        if dist_target_plan < self.radius_reach:
            # Replan the path.
            self.is_find_new_target_path = True
            self.path_target_step = 0
        self.target_position_plan_history.append(self.current_true_target_position_plan)
        self.current_true_target_position_plan = next_target_position_plan
    
    def step_particle_filter_system(self, is_update_display=True):
        print(f'timestep: {self.t}') 
        if self.t == 0:
            self.datetime = datetime.datetime.now()
            # Initialize history for further plot.
            self.dist_from_target_list = []
            self.add_lang_input_timestep_list = []
            self.robot_position_plan_history = []
            self.start_time = time.time()
            # Initialize robot position (x,y).
            self.current_robot_position_plan = self.start_position_robot_plan
            # Initialize target position (x,y).
            self.current_true_target_position_plan = self.start_position_target_plan
            # Initialize robot sensor model.
            self.robot_sensor_model = RobotSensorModel(self.plan_dim, self.is_no_obstacle_plan)
            # Initialize particle filter.
            self.particle_filter = ParticleFilter(self.plan_dim, self.robot_sensor_model) 
        else:
            # Sample particles from motion model.
            if not self.is_static_target:
                self.particle_filter.sample_particle_motion_model()

            # Update importance weights with measurements.
            self.particle_filter.update_importance_weight_robot_sensor_observation(self.current_true_target_position_plan, self.current_robot_position_plan)

            # Check boolean value from the insert button of language input.
            if self.is_language_input:
                print('fusing language input')
                human_likelihood = self.human_observation_model(spatial_relation=self.spatial_relation,landmark=self.landmark)
                self.particle_filter.update_importance_weight_human_observation(human_likelihood)
                self.is_language_input = False
                # If insert the language input, replan the path of a robot.
                self.is_plan = True
                posterior = posterior/np.sum(posterior)
                self.add_lang_input_timestep_list.append((self.t,self.is_negative_input))
            self.particle_filter.clip_particle_weight()  # Clip particle weights to prevent numerical error.

            # Compute effective number of particles
            effective_particle_number = 1 / np.sum(np.power(self.particle_filter.pw, 2)) 
            print(f'effective particle number: {effective_particle_number}')   
            # Check if resampling is needed via the effective numner of particles.
            if effective_particle_number < self.particle_filter.n /2:
                self.particle_filter.low_variance_resample()
        
        if self.is_plan == True:
            self.current_estimated_target_position_plan_sorted = self.particle_filter.px.copy()[np.argsort(self.particle_filter.pw.reshape(-1))[::-1]]  # Sort the particle in descending order of their weights.
            self.is_plan = False
            self.is_find_new_path = True  # When replan, find a new path.
            self.path_step = 0
            self.path = None

        # Step and path planning
        if self.is_find_new_path:
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
                    self.is_find_new_path = False # always prevent finding a new path unless further condition for planning exists
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.path_step += 1
                    self.timestep_distance_robot = 1 
                    break
                elif len(self.path) == 1: # the start and stop position is the same resulting in a path include only a point
                    next_robot_position_plan = self.path[0] # robot repeat its position 
                    self.is_find_new_path = False # always prevent finding a new path unless further condition for planning exists
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.timestep_distance_robot = 0
                    break
                else:
                    pass
        else:
            next_robot_position_plan = self.path[1+self.path_step] # if there is no finding a new path, select the next position in the path
            self.path_step += 1
            self.timestep_distance_robot = 1 
        
        # Create image for the graphic interface.
        map_plot_with_particle = self.map_plot.copy()
        cv2.circle(map_plot_with_particle,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(255, 200, 0),thickness=-1)
        cv2.circle(map_plot_with_particle,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=self.robot_sensor_model.radius_fov*self.resize_ratio,color=(100, 100, 100),thickness=1)
        cv2.circle(map_plot_with_particle,(min(self.current_estimated_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_estimated_target_position_plan[1]*self.resize_ratio,self.plot_dim[1])),radius=2,color=(100, 255, 0),thickness=-1)
        cv2.circle(map_plot_with_particle,(min(self.current_true_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_true_target_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(0, 0, 0),thickness=-1)
        # Visualize each particle with a color intensity according to its weight.
        for i in range(self.particle_filter.n):
            particle_color = 25 * min((round(math.log10(1e-3/self.particle_filter.pw[i])),10))  # The bigger the weight, the more color intensity 
            cv2.circle(map_plot_with_particle,(min(round(self.particle_filter.px[i][0])*self.resize_ratio,self.plot_dim[0]), min(round(self.particle_filter.px[i][1])*self.resize_ratio,self.plot_dim[1]-1)),radius=1,color=(particle_color, particle_color, 255),thickness=cv2.FILLED)
        # Check if the robot reach the target 
        dist = np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan))
        self.dist_from_target_list.append(dist*4) #1 plan grid = 4 m
        if dist < self.radius_reach:
            print('reach target')
            self.reach = True
            self.is_sim_running = False
            self.plot_distance_time()
            self.plot_map_plan_with_path()
            if not self.is_static_target:
                self.save_target_path()
        
        # Check if the robot reach the estimated target.
        dist_plan = np.linalg.norm(np.array(self.current_estimated_target_position_plan)-np.array(self.current_robot_position_plan))
        if dist_plan < self.radius_reach:
            print('reach plan')
            # Replan the path.
            self.is_plan = True

        # Update the true target position. 
        if not self.is_static_target:
            self.move_target_along_path_to_goal()
        # Update robot position. 
        self.current_robot_position_plan = next_robot_position_plan
        # Save robot position in history list.
        self.robot_position_plan_history.append(self.current_robot_position_plan)
        self.t += 1

        # Update image in the graphic interface 
        if is_update_display:
            B_hm,G_hm,R_hm = cv2.split(map_plot_with_particle)
            img_cv_tk = cv2.merge((R_hm,G_hm,B_hm))
            im = Image.fromarray(img_cv_tk)
            self.img_tk = ImageTk.PhotoImage(image=im)
            self.img_label.config(image=self.img_tk)
        self.timestep_time = time.time()-self.start_time
            
    def step_histogram_belief_system(self, is_update_display=True): 
        print(f'timestep: {self.t}') 
        self.start_time = time.time()
        if self.t == 0:
            self.datetime = datetime.datetime.now()
            # Initialize history for further plot.
            self.dist_from_target_list = []
            self.add_lang_input_timestep_list = []
            self.robot_position_plan_history = []
            # Initialize robot position (x,y).
            self.current_robot_position_plan = self.start_position_robot_plan
            # Initialize target position (x,y).
            self.current_true_target_position_plan = self.start_position_target_plan
            # Initialize robot sensor model.
            self.robot_sensor_model = RobotSensorModel(self.plan_dim, self.is_no_obstacle_plan)
            # Initialize histogram belief.
            self.histogram_belief = HistogramBelief(self.plan_dim)
            if not self.is_static_target:
                self.histogram_belief.precalculate_random_walk_gaussian()
        else:
            # Propagate belief with dynamic prediction.
            if not self.is_static_target:
                self.histogram_belief.propagate_belief_random_walk_motion_model()
            
            # Measurement update step.
            # Collect boolean value from the interface button from 'enable robot observation' button.
            self.is_robot_observation = int(self.is_robot_observation_temp.get())
            if self.is_robot_observation:
                robot_sensor_likelihood = self.robot_sensor_model.compute_grid_likelihood(self.current_true_target_position_plan, self.current_robot_position_plan)
                self.histogram_belief.update_belief(robot_sensor_likelihood)
            
            # Check boolean value from the interface from 'language input' button.
            if self.is_language_input:
                print('fusing language input')
                language_input_likelihood = self.load_human_observation_model(spatial_relation=self.spatial_relation, landmark=self.landmark)
                self.histogram_belief.update_belief(language_input_likelihood)
                self.is_language_input = False
                # If insert the language input, replan the path of a robot.
                self.is_plan = True
                self.add_lang_input_timestep_list.append((self.t,self.is_negative_input))
                # Capture language input likelihood for debug.
                if self.is_snapshot:
                    language_input_likelihood_heatmap = self.create_heatmap(language_input_likelihood)
                    path = f'histogram_belief_system/snapshot/language_input_likelihood'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(path+f'/{self.spatial_relation}_{self.landmark}.png', language_input_likelihood_heatmap)
        belief = self.histogram_belief.belief

        # Retrieve estimated target position via MAP. In order to handle point inside buidling, sort all of the point in descending value order.
        if self.is_plan == True:
            top_sorted_estimated_target_number = self.plan_dim[0]*self.plan_dim[1]
            belief = self.histogram_belief.belief
            current_estimated_target_position_plan_sorted = np.argsort(belief, axis=None)[::-1][:top_sorted_estimated_target_number] 
            self.current_estimated_target_position_plan_sorted = [(np.unravel_index(p, belief.shape)[1], np.unravel_index(p, belief.shape)[0]) for p in current_estimated_target_position_plan_sorted]
            self.is_plan = False
            self.is_find_new_path = True
            self.path_step = 0
            self.path = None

        # Step and path planning.
        if self.is_find_new_path:
            print('find new path')
            equal_groups = [[]]  # Equal groups of belief, in histogram belief, it is common for some grid position to have equal belief.
            k = 0
            for i in range(len(self.current_estimated_target_position_plan_sorted)-1):
                ind = self.current_estimated_target_position_plan_sorted[i]
                val = belief[ind[1],ind[0]]
                next_ind = self.current_estimated_target_position_plan_sorted[i+1]
                next_val = belief[next_ind[1],next_ind[0]]
                equal_groups[k].append(ind)
                if val == next_val:
                    pass
                else:
                    k += 1
                    equal_groups.append([])
            last_ind = self.current_estimated_target_position_plan_sorted[len(self.current_estimated_target_position_plan_sorted)-1]
            equal_groups[k].append(last_ind)
            for i in range(len(equal_groups)):
                equal_groups[i] = random.sample(equal_groups[i],len(equal_groups[i]))  # Randomize the order of grid position within each group
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
                    self.is_find_new_path = False
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.path_step += 1
                    self.timestep_distance_robot = 1 
                    break
                elif len(self.path) == 1:
                    next_robot_position_plan = self.path[0]
                    self.is_find_new_path = False
                    self.current_estimated_target_position_plan = current_estimated_target_position_plan
                    self.timestep_distance_robot = 0
                    break
                else:
                    pass
        else:
            next_robot_position_plan = self.path[1+self.path_step]
            self.path_step += 1
            self.timestep_distance_robot = 1 
        
        # Create image for the graphic interface.
        posterior_heatmap = self.create_heatmap(belief)
        map_plot_with_heatmap = cv2.addWeighted(posterior_heatmap, 0.7, self.map_plot.copy(), 0.3, 0)
        cv2.circle(map_plot_with_heatmap,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(255, 200, 0),thickness=-1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_robot_position_plan[0]*self.resize_ratio,self.plot_dim[0]), min(self.current_robot_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=self.robot_sensor_model.radius_fov*self.resize_ratio,color=(100, 100, 100),thickness=1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_estimated_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_estimated_target_position_plan[1]*self.resize_ratio,self.plot_dim[1])),radius=2,color=(255, 0, 255),thickness=-1)
        cv2.circle(map_plot_with_heatmap,(min(self.current_true_target_position_plan[0]*self.resize_ratio,self.plot_dim[0]-1), min(self.current_true_target_position_plan[1]*self.resize_ratio,self.plot_dim[1]-1)),radius=2,color=(0, 0, 0),thickness=-1)
        
        # Check if the robot reach the target 
        dist = np.linalg.norm(np.array(self.current_true_target_position_plan)-np.array(self.current_robot_position_plan))
        self.dist_from_target_list.append(dist)
        position_in_fov_list = self.robot_sensor_model.compute_position_in_fov_list(self.current_robot_position_plan)
        if dist < self.radius_reach:
            if (self.current_true_target_position_plan[0],self.current_true_target_position_plan[1]) in position_in_fov_list:
                print('reach target')
                self.reach = True
                self.is_sim_running = False
                self.plot_distance_time()
                self.plot_map_plan_with_path()
                if not self.is_static_target:
                    self.save_target_path()
                self.save_robot_path()
        
        # Check if the robot reach the estimated target (plan target).
        dist_plan = np.linalg.norm(np.array(self.current_estimated_target_position_plan)-np.array(self.current_robot_position_plan))
        if dist_plan < self.radius_reach:
            if (self.current_estimated_target_position_plan[0],self.current_estimated_target_position_plan[1]) in position_in_fov_list:
                print('reach plan')
                self.is_plan = True

        # Update the true target position.
        if not self.is_static_target:
            self.move_target_along_path_to_goal()
        # Update the robot position.
        self.current_robot_position_plan = next_robot_position_plan
        # Save robot position in history list
        self.robot_position_plan_history.append(self.current_robot_position_plan)
        self.t += 1
 
        # Update image in the graphic interface.
        if is_update_display:
            B_hm,G_hm,R_hm = cv2.split(map_plot_with_heatmap)
            img_cv_tk = cv2.merge((R_hm,G_hm,B_hm))
            im = Image.fromarray(img_cv_tk)
            self.img_tk = ImageTk.PhotoImage(image=im)
            self.img_label.config(image=self.img_tk)
        self.timestep_time = time.time()-self.start_time
        if self.is_snapshot:
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
    
    def save_robot_path(self):
        path = f'{self.system_type}/robot_path_plan_history/{self.datetime}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+f'/{self.start_position_robot_plan}_{self.start_position_target_plan}_{len(self.add_lang_input_timestep_list)}_{self.is_robot_observation}.json', "w") as file:
            json.dump(self.robot_position_plan_history, file)

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
        positions_wih_no_obstacle = np.argwhere(self.is_no_obstacle_plan)
        random_idx = random.randint(0,positions_wih_no_obstacle.shape[0]-1)
        random_position = positions_wih_no_obstacle[random_idx]
        return (random_position[1],random_position[0])

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
        if self.is_sim_running:
            if self.is_histogram_belief_system:
                self.step_histogram_belief_system() 
            else:
                self.step_particle_filter_system()
            desired_velocity = 42
            base_delay = 10  # Delay for visualization in graphic interface.
            if self.timestep_distance_robot > 0:
                desired_period = self.timestep_distance_robot  * 1000 / desired_velocity
                if desired_period - self.timestep_time <= base_delay:
                    print('computation limits the velocity of the robot')
                delay =  round(max((desired_period - self.timestep_time),base_delay))
            else:
                delay = max(self.timestep_time,base_delay)
            self.window.after(delay, self.simulate)

    def pause(self):
        self.is_sim_running = False 

    def reset(self):
        self.is_sim_running = False 
        self.is_plan = True
        self.t = 0
        self.reset_img_plot(first_display=False)

    def set_snapshot(self):
        self.is_snapshot = True

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
    load_map_from_test_dataset_and_change_resolution()
    scenario = SimulationTargetSearch(is_histogram_belief_system=True, map_name='chula_engineering', is_static_target=True)
    scenario.display()
    scenario.start()