import cv2
import numpy as np
import math
import random

class RuleBasedModel:
    def __init__(self,spatial_relation,**kwargs):
        self._spatial_relation = spatial_relation
        rules = {'front':FrontRule(),'near':NearRule(),'inside':InsideRule(),'far':FarRule()}
        self.rule = rules[self._spatial_relation]
        self.params = kwargs
    
    def plot_likelihood(self,building_img, building_with_entrance_img=None, building_entrance_coordinate=None):
        likelihood = self.rule.compute(building_img, building_with_entrance_img, **self.params)
        heatmapshow = cv2.normalize(likelihood, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        building_with_entrance_img = cv2.cvtColor(building_with_entrance_img, cv2.IMREAD_COLOR)
        print(building_with_entrance_img.shape)
        print(heatmap.shape)
        img = cv2.addWeighted(heatmap,0.6,building_with_entrance_img,0.4,0)
        seed = random.randint(0,999999)
        cv2.imwrite(f'results/{seed}.png',img)


class NearRule():
    def compute(self, building_img, building_with_entrance_img=None, type='expert_border', field_coeff=1.0):
        (_, black_white_building_img) = cv2.threshold(building_img, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        dilated_edges = cv2.dilate(cv2.Canny(black_white_building_img,0,255),kernel)
        cv2.imshow('edges',dilated_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        (building_cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        moment_x = 0
        moment_y = 0
        area = 0
        for i in range(len(building_cnts)):
            if hier[0][i,3] == -1:
                outer_cnt_idx = i
                area += cv2.contourArea(building_cnts[i])
                moment = cv2.moments(building_cnts[i])
                moment_x += moment["m10"] 
                moment_y += moment["m01"]
                print(f'adding area {area} and moment x {moment_x} moment y {moment_y}')
            elif hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                area -= cv2.contourArea(building_cnts[i])
                moment = cv2.moments(building_cnts[i])
                moment_x -= moment["m10"] 
                moment_y -= moment["m01"]
                print(f'subtracting area {area} and moment x {moment_x} moment y {moment_y}')
        center_x = int(moment_x / area)
        center_y = int(moment_y / area)
        building_cnt_center = (center_x,center_y)
        for i in range(len(building_cnts)):
            if hier[0][i,3] == -1:
                _, cnt_dim, _ = cv2.minAreaRect(building_cnts[i])
                field_width = min(cnt_dim)
                break
        if  type != 'baseline':
            for i in range(len(building_cnts)):
                if hier[0][i,3] == -1:
                    if type == 'expert_border':
                        likelihood = np.zeros(building_img.shape)
                        for x in range(building_img.shape[0]):
                            for y in range(building_img.shape[1]):
                                dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                                if dist_border <= 0:            
                                    likelihood[y,x] = math.exp(-(dist_border**2) / (field_coeff*field_width**2)) 
                                else:
                                    likelihood[y,x] = 0
                        return likelihood
                    if type == 'expert_convexhull':
                        convexhull_cnt = cv2.convexHull(building_cnts[i])
                        likelihood = np.zeros(building_img.shape)
                        for x in range(building_img.shape[0]):
                            for y in range(building_img.shape[1]):
                                dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                                dist_convexhull_border = cv2.pointPolygonTest(convexhull_cnt,(x,y),True)
                                dist_center = np.linalg.norm(np.array([x,y])-np.array(building_cnt_center))
                                if dist_border <= 0 and dist_convexhull_border <= 0:            
                                    likelihood[y,x] = math.exp(-(dist_convexhull_border**2) / (field_coeff*field_width**2))
                                elif dist_border <= 0 and dist_convexhull_border > 0:
                                    likelihood[y,x] = 1
                                else:
                                    likelihood[y,x] = 0
                        return likelihood
        elif type == 'baseline':
            likelihood = np.zeros(building_img.shape)
            for y in range(building_img.shape[0]):
                for x in range(building_img.shape[1]):
                    for i in range(len(building_cnts)):
                        if hier[0][i,3] == -1 :
                            dist_outer_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)  
                        dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                        dist_center = np.linalg.norm(np.array([x,y])-np.array(building_cnt_center))
                        if dist_outer_border <= 0 and  hier[0][i,3] == -1:            
                            likelihood[y,x] = math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        elif dist_outer_border > 0 and dist_border <= 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:            
                            likelihood[y,x] = 0
                        elif dist_outer_border > 0 and dist_border > 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                            likelihood[y,x] = math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        else:
                            pass
            return likelihood
        else:
            raise ValueError('invalid type')
        # if type == 'baseline':
        #     field_width - min(cnt_dim)
        #     building_idxs = np.argwhere(black_white_building_img == 255)
        #     likelihood = np.zeros(building_img.shape)
        #     for y in range(building_img.shape[0]):
        #         for x in range(building_img.shape[1]):
        #             print((x,y))
        #             closest_building_point = min(building_idxs, key=lambda p: np.linalg.norm(np.array([x,y])-np.array([p[1],p[0]])))
        #             dist_border = np.linalg.norm(np.array([x,y])-np.array([closest_building_point[1],closest_building_point[0]]))
        #             dist_center = np.linalg.norm(np.array([x,y])-np.array(building_cnt_center))
        #             if dist_border > 0:
        #                 likelihood[y,x] = math.exp(-(dist_center**2) / (field_coeff*field_width**2)) 
        #             else:
        #                 likelihood[y,x] = 0
        #     return likelihood    
class FarRule():
    def compute(self, building_img, building_with_entrance_img=None, type='expert_border', field_coeff=1.0):
        (_, black_white_building_img) = cv2.threshold(building_img, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        dilated_edges = cv2.dilate(cv2.Canny(black_white_building_img,0,255),kernel)
        cv2.imshow('edges',dilated_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        (building_cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        moment_x = 0
        moment_y = 0
        area = 0
        for i in range(len(building_cnts)):
            if hier[0][i,3] == -1:
                outer_cnt_idx = i
                area += cv2.contourArea(building_cnts[i])
                moment = cv2.moments(building_cnts[i])
                moment_x += moment["m10"] 
                moment_y += moment["m01"]
                print(f'adding area {area} and moment x {moment_x} moment y {moment_y}')
            elif hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                area -= cv2.contourArea(building_cnts[i])
                moment = cv2.moments(building_cnts[i])
                moment_x -= moment["m10"] 
                moment_y -= moment["m01"]
                print(f'subtracting area {area} and moment x {moment_x} moment y {moment_y}')
        center_x = int(moment_x / area)
        center_y = int(moment_y / area)
        building_cnt_center = (center_x,center_y)
        for i in range(len(building_cnts)):
            if hier[0][i,3] == -1:
                _, cnt_dim, _ = cv2.minAreaRect(building_cnts[i])
                field_width = min(cnt_dim)
                break
        if  type != 'baseline':
            for i in range(len(building_cnts)):
                if hier[0][i,3] == -1:
                    if type == 'expert_border':
                        likelihood = np.zeros(building_img.shape)
                        for x in range(building_img.shape[0]):
                            for y in range(building_img.shape[1]):
                                dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                                if dist_border <= 0:            
                                    likelihood[y,x] = 1 - math.exp(-(dist_border**2) / (field_coeff*field_width**2)) 
                                else:
                                    likelihood[y,x] = 0
                        return likelihood
                    if type == 'expert_convexhull':
                        convexhull_cnt = cv2.convexHull(building_cnts[i])
                        likelihood = np.zeros(building_img.shape)
                        for x in range(building_img.shape[0]):
                            for y in range(building_img.shape[1]):
                                dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                                dist_convexhull_border = cv2.pointPolygonTest(convexhull_cnt,(x,y),True)
                                dist_center = np.linalg.norm(np.array([x,y])-np.array(building_cnt_center))
                                if dist_border <= 0 and dist_convexhull_border <= 0:            
                                    likelihood[y,x] = 1 - math.exp(-(dist_convexhull_border**2) / (field_coeff*field_width**2))
                                elif dist_border <= 0 and dist_convexhull_border > 0:
                                    likelihood[y,x] = 0
                                else:
                                    likelihood[y,x] = 0
                        return likelihood
        elif type == 'baseline':
            likelihood = np.zeros(building_img.shape)
            for y in range(building_img.shape[0]):
                for x in range(building_img.shape[1]):
                    for i in range(len(building_cnts)):
                        if hier[0][i,3] == -1 :
                            dist_outer_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)  
                        dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                        dist_center = np.linalg.norm(np.array([x,y])-np.array(building_cnt_center))
                        if dist_outer_border <= 0 and  hier[0][i,3] == -1:            
                            likelihood[y,x] = 1 - math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        elif dist_outer_border > 0 and dist_border <= 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:            
                            likelihood[y,x] = 0
                        elif dist_outer_border > 0 and dist_border > 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                            likelihood[y,x] = 1 - math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        else:
                            pass
            return likelihood
        else:
            raise ValueError('invalid type')
class InsideRule():
    def compute(self, building_img, building_with_entrance_img=None):
        (_, black_white_building_img) = cv2.threshold(building_img, 200, 255, cv2.THRESH_BINARY)
        likelihood = (black_white_building_img == 0).astype(np.uint8)
        return likelihood

class FrontRule():
    def compute(self, building_img, building_with_entrance_img=None, building_entrance_coordinate=None, type='expert', field_coeff=0.5, entrance_coeff=0.3):
        (_, black_white_building_img) = cv2.threshold(building_img, 200, 255, cv2.THRESH_BINARY)
        if building_with_entrance_img is None:
            if building_entrance_coordinate is None:
                raise ValueError('cant compute entrance center coordinate')
        else:
            black_white_entrance_only_img = building_with_entrance_img != 128
            black_white_entrance_only_img = black_white_entrance_only_img.astype(np.uint8) *255
            resized_ratio = 2
            reszied_black_white_entrance_only_img = cv2.resize(black_white_entrance_only_img, (black_white_entrance_only_img.shape[0]*resized_ratio,black_white_entrance_only_img.shape[1]*resized_ratio))
            kernel = np.ones((2,2),np.uint8)
            dilated_edges = cv2.dilate(cv2.Canny(reszied_black_white_entrance_only_img,0,255),kernel)
            (entrance_cnts, _) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            building_entrance_coordinate = []
            for i in range(0,len(entrance_cnts),2):
                dr_cnt = cv2.drawContours(255*np.ones_like(reszied_black_white_entrance_only_img),[entrance_cnts[i]],-1, 0,-1)
                moments = cv2.moments(entrance_cnts[i])
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                img = cv2.circle(dr_cnt,(center_x,center_y),5,(128,128,128),-1)
                building_entrance_coordinate.append((int(center_x/resized_ratio),int(center_y/resized_ratio)))
        dilated_edges = cv2.dilate(cv2.Canny(black_white_building_img,0,255),kernel)
        (building_cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if type == 'baseline_sloop':
            moment_x = 0
            moment_y = 0
            area = 0
            for i in range(len(building_cnts)):
                if hier[0][i,3] == -1:
                    outer_cnt_idx = i
                    area += cv2.contourArea(building_cnts[i])
                    moment = cv2.moments(building_cnts[i])
                    moment_x += moment["m10"] 
                    moment_y += moment["m01"]
                    print(f'adding area {area} and moment x {moment_x} moment y {moment_y}')
                elif hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                    area -= cv2.contourArea(building_cnts[i])
                    moment = cv2.moments(building_cnts[i])
                    moment_x -= moment["m10"] 
                    moment_y -= moment["m01"]
                    print(f'subtracting area {area} and moment x {moment_x} moment y {moment_y}')
            center_x = int(moment_x / area)
            center_y = int(moment_y / area)
            building_cnt_center = (center_x,center_y)
        # calculate field width and entrance width
        for i in range(len(building_cnts)):
            if hier[0][i,3] == -1:
                _, cnt_dim, _ = cv2.minAreaRect(building_cnts[i])
                field_width = min(cnt_dim)
                entrance_width = max(cnt_dim)
                break
        # calculate direction vectors
        if type == 'expert' or type == 'baseline_sloop':
            for i in range(len(building_cnts)):
                if hier[0][i,3] == -1:
                    direction_vectors = []
                    for e in range(len(building_entrance_coordinate)):
                        dr_cnt = cv2.drawContours(255*np.ones_like(building_img),[building_cnts[i]],-1,0,-1)
                        local_view_width = 20
                        white_border_width = 1
                        area_around_entrance = dr_cnt[building_entrance_coordinate[e][1]-int(local_view_width/2)+white_border_width:building_entrance_coordinate[e][1]+int(local_view_width/2)-white_border_width,building_entrance_coordinate[e][0]-int(local_view_width/2)+white_border_width:building_entrance_coordinate[e][0]+int(local_view_width/2)-white_border_width]
                        bordered_area_around_entrance = cv2.copyMakeBorder(area_around_entrance,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
                        dilated_edges = cv2.dilate(cv2.Canny(bordered_area_around_entrance,0,255),kernel)
                        (cnts_around_entrance, _) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        cnts_around_entrance_center_coordinates = []
                        for j in range(0,len(cnts_around_entrance),2):
                            moments = cv2.moments(cnts_around_entrance[i])
                            local_center_x = int(moments["m10"] / moments["m00"])
                            local_center_y = int(moments["m01"] / moments["m00"])
                            center_x = building_entrance_coordinate[e][0] - int(local_view_width/2) + local_center_x
                            center_y = building_entrance_coordinate[e][1] - int(local_view_width/2) + local_center_y
                            cnts_around_entrance_center_coordinates.append((center_x,center_y))
                        mean_cnts_around_entrance_center_coordinate = np.mean(cnts_around_entrance_center_coordinates,axis=0)
                        direction_vector = np.array(building_entrance_coordinate[e]) - mean_cnts_around_entrance_center_coordinate
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)
                        direction_vectors.append(direction_vector)
        # calculate likelihood using the calculated components aboved
        if type == 'expert':
            # loop for only the outest cnt then return likelihood
            for i in range(len(building_cnts)):
                if hier[0][i,3] == -1:            
                    likelihood = np.zeros(building_img.shape)
                    for y in range(building_img.shape[0]):
                        for x in range(building_img.shape[1]):
                            dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                            if dist_border <= 0:
                                dist_nearest_entrance = []
                                for e in range(len(building_entrance_coordinate)):
                                    dist_nearest_entrance.append(np.linalg.norm(np.array((x,y))-building_entrance_coordinate[e]))
                                min_dist_nearest_entrance = np.min(dist_nearest_entrance)
                                nearest_entrance_idx = np.argmin(dist_nearest_entrance)
                                position_vector = np.array([x,y]) - np.array(building_entrance_coordinate[nearest_entrance_idx])
                                position_vector = position_vector / max(1e-8,np.linalg.norm(position_vector))  
                                scaling_factor =  (math.pi/2) / (math.pi/2 + math.pi/2)
                                dot_product = math.cos(scaling_factor *  math.acos(np.clip(np.dot(position_vector,direction_vectors[nearest_entrance_idx]),-1.0,1.0)))
                                if dot_product < 0:
                                    dot_product = 0
                                likelihood[y,x] =  dot_product * math.exp(-(dist_border**2) / (field_coeff*field_width**2)) * math.exp(-(min_dist_nearest_entrance**2) / (entrance_coeff*entrance_width**2))
                            else:
                                likelihood[y,x] = 0
        elif type == 'baseline':
            likelihood = np.zeros(building_img.shape)
            chosen_entrance_idx = 0
            for y in range(building_img.shape[0]):
                for x in range(building_img.shape[1]):
                    # for each (x,y), consider every contour of the building (the outest contour, the first layer of inner contours (if multiple hole in a building))
                    for i in range(len(building_cnts)):
                        if hier[0][i,3] == -1 :
                            dist_outer_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)  
                        dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                        dist_chosen_entrance = np.linalg.norm(np.array((x,y))-building_entrance_coordinate[chosen_entrance_idx])
                        if dist_outer_border <= 0 and  hier[0][i,3] == -1:            
                            likelihood[y,x] =  math.exp(-(dist_chosen_entrance**2) / (entrance_coeff*entrance_width**2))
                            break
                        elif dist_outer_border > 0 and dist_border <= 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:            
                            likelihood[y,x] = 0
                        elif dist_outer_border > 0 and dist_border > 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                            likelihood[y,x] =  math.exp(-(dist_chosen_entrance**2) / (entrance_coeff*entrance_width**2))
                            break
                        else:
                            likelihood[y,x] = 0
        elif type == 'baseline_sloop':
            chosen_entrance_idx = 0
            likelihood = np.zeros(building_img.shape)
            chosen_entrance_idx = 0
            for y in range(building_img.shape[0]):
                for x in range(building_img.shape[1]):
                    position_vector = np.array([x,y]) - np.array(building_cnt_center)
                    position_vector = position_vector / max(1e-8,np.linalg.norm(position_vector))
                    dot_product = math.cos(math.acos(np.clip(np.dot(position_vector,direction_vectors[chosen_entrance_idx]),-1.0,1.0))) 
                    if dot_product < 0:
                        dot_product = 0
                    # for each (x,y), consider every contour of the building (the outest contour, the first layer of inner contours (if multiple hole in a building))
                    for i in range(len(building_cnts)):
                        if hier[0][i,3] == -1 :
                            dist_outer_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)  
                        dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
                        dist_chosen_entrance = np.linalg.norm(np.array((x,y))-building_entrance_coordinate[chosen_entrance_idx])
                        if dist_outer_border <= 0 and  hier[0][i,3] == -1:            
                            likelihood[y,x] =  dot_product * math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        elif dist_outer_border > 0 and dist_border <= 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:            
                            likelihood[y,x] = 0
                        elif dist_outer_border > 0 and dist_border > 0 and hier[0][i,2] == -1 and hier[0][i,3] != outer_cnt_idx:
                            likelihood[y,x] =  dot_product * math.exp(-(dist_border**2) / (field_coeff*field_width**2))
                            break
                        else:
                            likelihood[y,x] = 0
        else:
            raise ValueError('invalid type')
    
        return likelihood
        # for i in range(len(building_cnts)):
        #     if hier[0][i,3] == -1:
        #         _, cnt_dim, _ = cv2.minAreaRect(building_cnts[i])
        #         if type == 'expert' or type == 'baseline_sloop':
        #             field_width = min(cnt_dim)
        #             entrance_width = max(cnt_dim)
        #             direction_vectors = []
        #             for e in range(len(building_entrance_coordinate)):
        #                 dr_cnt = cv2.drawContours(255*np.ones_like(building_img),[building_cnts[i]],-1,0,-1)
        #                 local_view_width = 20
        #                 white_border_width = 1
        #                 area_around_entrance = dr_cnt[building_entrance_coordinate[e][1]-int(local_view_width/2)+white_border_width:building_entrance_coordinate[e][1]+int(local_view_width/2)-white_border_width,building_entrance_coordinate[e][0]-int(local_view_width/2)+white_border_width:building_entrance_coordinate[e][0]+int(local_view_width/2)-white_border_width]
        #                 bordered_area_around_entrance = cv2.copyMakeBorder(area_around_entrance,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
        #                 dilated_edges = cv2.dilate(cv2.Canny(bordered_area_around_entrance,0,255),kernel)
        #                 (cnts_around_entrance, _) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #                 cnts_around_entrance_center_coordinates = []
        #                 for j in range(0,len(cnts_around_entrance),2):
        #                     moments = cv2.moments(cnts_around_entrance[i])
        #                     local_center_x = int(moments["m10"] / moments["m00"])
        #                     local_center_y = int(moments["m01"] / moments["m00"])
        #                     center_x = building_entrance_coordinate[e][0] - int(local_view_width/2) + local_center_x
        #                     center_y = building_entrance_coordinate[e][1] - int(local_view_width/2) + local_center_y
        #                     cnts_around_entrance_center_coordinates.append((center_x,center_y))
        #                 mean_cnts_around_entrance_center_coordinate = np.mean(cnts_around_entrance_center_coordinates,axis=0)
        #                 direction_vector = np.array(building_entrance_coordinate[e]) - mean_cnts_around_entrance_center_coordinate
        #                 direction_vector = direction_vector / np.linalg.norm(direction_vector)
        #                 direction_vectors.append(direction_vector)
        #         elif type == 'baseline':
        #             entrance_width = 100
        #         elif type == 'baseline_sloop':
        #             field_width = 100
        #         else:
        #             raise ValueError('invalid type')
        #         likelihood = np.zeros(building_img.shape)
        #         for y in range(building_img.shape[0]):
        #             for x in range(building_img.shape[1]):
        #                 dist_border = cv2.pointPolygonTest(building_cnts[i],(x,y),True)
        #                 if dist_border <= 0:
        #                     if type == 'expert':
        #                         dist_nearest_entrance = []
        #                         for e in range(len(building_entrance_coordinate)):
        #                             dist_nearest_entrance.append(np.linalg.norm(np.array((x,y))-building_entrance_coordinate[e]))
        #                         min_dist_nearest_entrance = np.min(dist_nearest_entrance)
        #                         nearest_entrance_idx = np.argmin(dist_nearest_entrance)
        #                         position_vector = np.array([x,y]) - np.array(building_entrance_coordinate[nearest_entrance_idx])
        #                         position_vector = position_vector / max(1e-8,np.linalg.norm(position_vector))  
        #                         scaling_factor =  (math.pi/2) / (math.pi/2 + math.pi/2)
        #                         dot_product = math.cos(scaling_factor *  math.acos(np.clip(np.dot(position_vector,direction_vectors[nearest_entrance_idx]),-1.0,1.0)))
        #                         if dot_product < 0:
        #                             dot_product = 0
        #                         likelihood[y,x] =  dot_product * math.exp(-(dist_border**2) / (field_coeff*field_width**2)) * math.exp(-(min_dist_nearest_entrance**2) / (entrance_coeff*entrance_width**2))
        #                     elif type == 'baseline':
        #                         chosen_entrance_idx = 0
        #                         dist_chosen_entrance = np.linalg.norm(np.array((x,y))-building_entrance_coordinate[chosen_entrance_idx])
        #                         likelihood[y,x] =  math.exp(-(dist_chosen_entrance**2) / (entrance_coeff*entrance_width**2)) 
        #                     elif type == 'baseline_sloop':
        #                         chosen_entrance_idx = 0
        #                         position_vector = np.array([x,y]) - np.array(building_cnt_center)
        #                         position_vector = position_vector / max(1e-8,np.linalg.norm(position_vector))
        #                         dot_product = math.cos(math.acos(np.clip(np.dot(position_vector,direction_vectors[chosen_entrance_idx]),-1.0,1.0))) 
        #                         if dot_product < 0:
        #                             dot_product = 0
        #                         likelihood[y,x] =  dot_product * math.exp(-(dist_border**2) / (field_coeff*field_width**2))
        #                     else:
        #                         raise ValueError('invalid type')
        #                 else:
        #                     likelihood[y,x] = 0
        #         return likelihood    
        
def test():
    district_name = 'sanamluang'
    building_number = 4
    path = f'../building_entrance_street_dataset/{district_name}/building/{building_number}.png'
    building_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    path = f'../building_entrance_street_dataset/{district_name}/building_with_entrance/{building_number}.png'
    building_with_entrance_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    params = {'type':'expert', 'field_coeff':3.0, 'entrance_coeff':1.0}
    model = RuleBasedModel('front', **params)
    # params = {'type':'expert_convexhull', 'field_coeff':0.5}
    # model = RuleBasedModel('far', **params)
    model.plot_likelihood(building_img, building_with_entrance_img, building_entrance_coordinate=None)
if __name__ == "__main__":
    test()