import osmnx as ox
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import os

download_map = False
district_name = 'chula_engineering'
if download_map:
    # Specify the name that is used to seach for the data
    # place_name = "Engineering 3 Building"
    # graph = ox.graph_from_place(place_name)

    # Fetch OSM street network from the center point (lat,long)
    # ladprao_centaragrand
    # center_point = (13.8178049, 100.5605714)
    # center_point = (13.81738,100.55982) # selected
    # center_point = (13.81653,100.56043) 
    # ladprao_pttenco
    # center_point = (13.8184062, 100.5572552)
    # center_point = (13.81921,100.55685)
    # center_point = (13.81790,100.55774) 
    # langsuan area
    # center_point = [13.7373183,100.5426757]
    # victory monument 
    # center_point = [13.7649689,100.5382802]
    # ratchaprarop rd.
    # center_point = [13.7612467,100.5425805]
    # central_ratchadamnoen rd.
    # center_point = [13.7565415,100.5035276] # selected
    # northern_sanamluang
    # center_point = [13.7578625,100.4949353]
    # center_point = [13.75877,100.49241]
    # sanamluang
    # center_point = [13.75290,100.49568] # selected
    # lumphini_sathorn
    # center_point = [13.72710,100.54115] # selected
    # center_point = [13.72499,100.54384] 
    # chongnonsri_sathorn
    # center_point = [13.72172,100.53029]
    # center_point = [13.72185,100.53031] # selected
    # center_point = [13.71975,100.52933]
    # chidlom_ploenchit
    # center_point = [13.74422,100.54568]
    # triamudom
    # center_point = [13.74020,100.53233] # selected
    # southern_sanamluang
    # center_point = [13.75252,100.49640]
    # center_point = [13.75368,100.49431]
    # center_point = [13.75184,100.49531]
    # center_point = [13.75055,100.49728]
    # siam
    # center_point = [13.74646,100.53088]
    # center_point = [13.74460,100.52980] # selected
    # center_point = [13.74493,100.53416]
    # pratunam
    # center_point = [13.74998,100.54122] # selected
    # rama3_ratchada
    # center_point = [13.69614,100.53824] # selected
    # sathorn_naratiwat
    # center_point = [13.71165,100.53714] # selected
    # prakanong
    # center_point = [13.72591,100.59850] # selected
    # bangken
    # center_point = [13.82534,100.56214] #selected
    # chula engineering
    center_point = [13.73631,100.53301]
    # cf = '["entrance"~"yes|secondary|main|service|exit|entrance"]'
    cf = '["highway"~"primary|secondary|tertiary|service|service_link|trunk|residential|unclassified|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
    graph = ox.graph_from_point(center_point, dist=1600, network_type='all_private',retain_all=False, truncate_by_edge=False, custom_filter=cf)
    buildings = ox.geometries_from_point(center_point, tags={"building":True}, dist=1600)
    print(buildings)

    fig = plt.figure(figsize=(32,32))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    # ax.set_facecolor('#f2efe9') # rgba(242,239,233,255)
    
    path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
   
    nodes, edges = ox.graph_to_gdfs(graph)
    
    edges.plot(ax=ax, linewidth=2, edgecolor='#fcd6a4') # rgba(252,214,164,255)
    buildings.plot(ax=ax, color='#b2abae') # rgba(178,171,174,255)
    plt.savefig(path+f'original_map.png')

    fig = plt.figure(figsize=(32,32))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    edges.plot(ax=ax, linewidth=0, edgecolor='white') # rgba(252,214,164,255)
    buildings.plot(ax=ax, color='#b2abae') # rgba(178,171,174,255)
    plt.savefig(path+f'original_map_only_building.png')
    
check_original = False   
if check_original:
    path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
    img = cv2.imread(path+f'original_map.png', cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    resized_img = cv2.resize(img, (800,800))
    grayImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    building_only_img = cv2.imread(path+f'original_map_only_building.png', cv2.IMREAD_UNCHANGED)
    print('building_only_img Dimensions : ',building_only_img.shape)
    resized_img = cv2.resize(building_only_img, (800,800))
    grayImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

crop_original = False   
if crop_original:
    path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
    img = cv2.imread(path+f'original_map.png', cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    cropped_img =img[int(img.shape[1]/2) - 800:int(img.shape[1]/2) + 800, int(img.shape[0]/2) - 800:int(img.shape[0]/2) + 800]
    cv2.imwrite(path+f'original_map.png', cropped_img)
    resized_img = cv2.resize(cropped_img, (800,800))
    grayImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    building_only_img = cv2.imread(path+f'original_map_only_building.png', cv2.IMREAD_UNCHANGED)
    print('building_only_img Dimensions : ',building_only_img.shape)
    cropped_img =building_only_img[int(img.shape[1]/2) - 800:int(img.shape[1]/2) + 800, int(img.shape[0]/2) - 800:int(img.shape[0]/2) + 800]
    cv2.imwrite(path+f'original_map_only_building.png', cropped_img)
    resized_img = cv2.resize(cropped_img, (800,800))
    grayImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

crop_center = {'ladprao_centaragrand':[(560,685),(860,830)],
               'central_ratchadamnoen_rd':[(806,826)],
               'sanamluang':[(746,676),(712,910),(968,1012),(780,420)],
               'lumphini_sathorn':[(700,742)],
               'chongnonsri_sathorn':[(690,930),(806,790)],
               'triamudom':[(814,744)],
               'pratunam':[(862,806)],
               'rama3_ratchada':[(756,614),(622,910),(1156,482)],
               'sathorn_naratiwat':[(912,600),(580,1024)],
               'prakanong':[(850,1150),(980,390)],
               'bangken':[(983,590),(1312,754)],
               'chula_engineering':[(784,828)]}
crop_subdistrict = False
subdistrict_number = 1
if crop_subdistrict:
    path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
    img = cv2.imread(path+f'original_map.png', cv2.IMREAD_UNCHANGED)
    min_width = min(img.shape[0],img.shape[1])
    crop_center_map = crop_center[district_name][subdistrict_number-1]
    cropped_img = img[int(crop_center_map[1]) - 200:int(crop_center_map[1]) + 200, int(crop_center_map[0]) - 200:int(crop_center_map[0]) + 200]
    print('Cropped Dimensions : ',cropped_img.shape)
    cv2.imshow("Cropped image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(path+f'map_subdistrict_{subdistrict_number}.png', cropped_img)
    grayImage = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_only_building = cv2.imread(path+f'original_map_only_building.png', cv2.IMREAD_UNCHANGED)
    # if the building only contain shift error from the map image
    shift_x = 0
    shift_y = 0
    cropped_img_only_building = img_only_building[int(crop_center_map[1])- shift_y - 200:int(crop_center_map[1])- shift_y + 200, int(crop_center_map[0]) - shift_x - 200:int(crop_center_map[0]) - shift_x + 200]
    # cropped_img_only_building = img_only_building[int(crop_center_map[1]) - 200:int(crop_center_map[1]) + 200, int(crop_center_map[0]) - 200:int(crop_center_map[0]) + 200]
    cv2.imwrite(path+f'map_subdistrict_{subdistrict_number}_only_building.png', cropped_img_only_building)
    grayImage = cv2.cvtColor(cropped_img_only_building, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayImage", grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # dim = (400,400)
    # resized_img = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_NEAREST)
    # cv2.imwrite(path+f'400X400_map_subregion_{subregion_number}.png', resized_img)
    # print('Resize Dimensions : ',resized_img.shape)

# correct_resize_subdistrict = False
# # resize to correct the resolution of the map to be 1px:1m, originally the 400 px x 400 px map download posses an area of 500 m X 500 m
# if correct_resize_subdistrict:
#     path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
#     img = cv2.imread(path+f'map_subdistrict_{subdistrict_number}.png', cv2.IMREAD_UNCHANGED)
#     resized_img = cv2.resize(img, (500,500))
#     cv2.imwrite(path+f'map_subdistrict_{subdistrict_number}.png', resized_img)
#     path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
#     img = cv2.imread(path+f'map_subdistrict_{subdistrict_number}_only_building.png', cv2.IMREAD_UNCHANGED)
#     resized_img = cv2.resize(img, (500,500))
#     cv2.imwrite(path+f'map_subdistrict_{subdistrict_number}_only_building.png', resized_img)

# extract each building and plot an entrance
extract_building = True
if extract_building:
    path = f'../building_entrance_street_dataset/{district_name}/overview_map/'
    map_with_building = cv2.imread(path+f'map_subdistrict_{subdistrict_number}_only_building.png', cv2.IMREAD_UNCHANGED)
    map_with_building_and_street = cv2.imread(path+f'map_subdistrict_{subdistrict_number}.png', cv2.IMREAD_UNCHANGED)
    gray_scaled_map_with_building = cv2.cvtColor(map_with_building, cv2.COLOR_BGR2GRAY)

    # resize for better edges and contour from edges (only for chula engineering map)
    resized_dim = (map_with_building.shape[0]*2,map_with_building.shape[1]*2)
    resized_gray_scaled_map_with_building = cv2.resize(gray_scaled_map_with_building, resized_dim)
    resized_map_with_building_and_street = cv2.resize(map_with_building_and_street, resized_dim)
    resized_map_with_building_and_street = cv2.cvtColor(resized_map_with_building_and_street, cv2.IMREAD_COLOR)

    map_with_building_and_street = cv2.cvtColor(map_with_building_and_street, cv2.IMREAD_COLOR)

    (thresh, black_and_white_map_with_building) = cv2.threshold(resized_gray_scaled_map_with_building, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('Resized gray scaled map with building', resized_gray_scaled_map_with_building)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    canny_edges = cv2.Canny(black_and_white_map_with_building,0,255)
    cv2.imshow('Canny edges',canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((2,2),np.uint8)
    dilated_edges = cv2.dilate(canny_edges,kernel)
    cv2.imshow('Dilated edges',dilated_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # (cnts, hier) = cv2.findContours(canny_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(f'length of contours: {len(cnts)}')
    cnt_list = []
    dr_cnt_list = []
    selected_cnts = {'ladprao_centaragrand':[[41,51],[41]],
                     'central_ratchadamnoen_rd':[[106,116,122]],
                     'sanamluang':[[67],[88],[256],[50]],
                     'lumphini_sathorn':[[165]],
                     'chongnonsri_sathorn':[[19,25],[9,18,43]],
                     'triamudom':[[10,22]],
                     'pratunam':[[56]],
                     'rama3_ratchada':[[12],[0],[3]],
                     'sathorn_naratiwat':[[15,20],[9]],
                     'prakanong':[[49],[6,9]],
                     'bangken':[[63],[23]],
                     'chula_engineering':[[31,35,36,38,44,50,51,53,55,57,60,63,68,73,78,81,82,83,90,96]]}
    # for i in range(len(cnts)):
    for i in selected_cnts[district_name][subdistrict_number-1]:
        area = cv2.contourArea(cnts[i])
        if area < 300:
            continue
        print(f'index: {i}, area: {area}')
        dr_cnt = cv2.drawContours(255*np.ones_like(black_and_white_map_with_building),[cnts[i]],-1, 0,-1)
        dr_cnt = cv2.resize(dr_cnt, (map_with_building.shape[0],map_with_building.shape[1]))
        _, dr_cnt = cv2.threshold(dr_cnt, 200, 255, cv2.THRESH_BINARY)
        # dr_cnt_hole = cv2.drawContours(255*np.ones_like(blackAndWhiteImage),[cnts[258]],-1, 0,-1)
        # dr_cnt_inv = cv2.bitwise_not(dr_cnt_hole)
        # dr_cnt_hole2 = cv2.drawContours(255*np.ones_like(blackAndWhiteImage),[cnts[92]],-1, 0,-1)
        # dr_cnt_inv2 = cv2.bitwise_not(dr_cnt_hole2)
        cv2.imshow('dr_cnt',dr_cnt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imshow('dr_cnt_inv',dr_cnt_inv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('dr_cnt_hollow',dr_cnt)   
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        dr_cnt_list.append(dr_cnt)
        cnt_list.append(cnts[i])
        # if i == 0:
        #     dr_cnt_list.append(dr_cnt+dr_cnt_inv)
        # else:
        #     dr_cnt_list.append(dr_cnt)
    black_and_white_map_with_building = cv2.resize(black_and_white_map_with_building, (map_with_building.shape[0],map_with_building.shape[1]))
    _, black_and_white_map_with_building = cv2.threshold(black_and_white_map_with_building, 200, 255, cv2.THRESH_BINARY)
    custom_entrance = True
    if custom_entrance:
        entrances = {'ladprao_centaragrand':[[[[(188,142),(183,137)],[(169,161),(163,158)]],[[(137,99),(140,94)]]],[[[(92,58),(97,58)],[(87,90),(87,95)],[(181,114),(178,119)]]]],
                     'central_ratchadamnoen_rd':[[[[(90,211),(95,212)],[(119,217),(124,218)],[(150,224),(155,225)]],[[(260,187),(266,189)],[(292,194),(296,190)]],[[(104,152),(109,153)]]]],
                     'sanamluang':[[[[(151,267),(156,267)],[(104,194),(105,189)]]],[[[(115,180),(115,185)],[(286,177),(286,182)]]],[[[(98,177),(98,182)],[(110,222),(115,222)]]],[[[(175,170),(177,165)]]]],
                     'lumphini_sathorn':[[[[(230,174),(236,176)],[(296,193),(290,196)],[(308,205),(306,200)]]]],
                     'chongnonsri_sathorn':[[[[(101,184),(107,182)]],[[(280,132),(285,132)],[(316,194),(318,200)]]],[[[(243,244),(249,242)],[(237,255),(240,261)]],[[(95,153),(101,155)],[(124,192),(127,198)]],[[(245,178),(251,176)]]]],
                     'triamudom':[[[[(154,311),(160,312)]],[[(249,278),(254,279)],[(214,277),(219,277)],[(211,262),(214,258)]]]],
                     'pratunam':[[[[(136,137),(141,134)],[(137,179),(132,174)],[(216,195),(221,196)],[(277,199),(282,195)]]]],
                     'rama3_ratchada':[[[[(240,223),(245,221)]]],[[[(208,216),(214,217)]]],[[[(140,255),(145,256)]]]],
                     'sathorn_naratiwat':[[[[(183,126),(188,127)],[(176,131),(175,136)],[(212,207),(217,209)],[(225,206),(226,201)]],[[(219,105),(223,103)],[(250,137),(252,142)],[(260,183),(265,181)]]],[[[(308,136),(310,141)]]]],
                     'prakanong':[[[[(119,170),(119,175)]]],[[[(236,242),(241,242)]],[[(140,181),(145,181)]]]],
                     'bangken':[[[[(131,197),(136,199)],[(218,152),(223,148)]]],[[[(117,204),(119,199)]]]],
                     'chula_engineering':[[[[(139,272),(143,272)]],[[(244,273),(249,274)],[(271,266),(271,271)]],[[(216,246),(220,247)]],[[(184,243),(188,244)]],[[(129,245),(129,250)]],[[(205,231),(209,232)]],[[(186,229),(190,230)]],[[(270,213),(275,214)],[(264,243),(269,244)]],[[(131,207),(131,211)]],[[(228,203),(232,204)]],[[(190,199),(194,200)]],[[(158,193),(162,194)]],[[(279,179),(283,180)],[(274,202),(278,203)]],[[(223,177),(227,177)],[(210,185),(214,186)],[(232,189),(236,189)]],[[(156,165),(160,166)],[(141,174),(145,174)],[(163,178),(167,178)]],[[(125,188),(129,188)]],[[(115,155),(119,156)]],[[(285,169),(289,170)],[(286,152),(290,152)]],[[(291,148),(295,148)]],[[(203,138),(208,139)],[(233,151),(232,155)],[(174,137),(173,141)],[(199,158),(203,159)],[(261,139),(265,139)]]]]
                     }
        entrance = entrances[district_name][subdistrict_number-1]
        # door_position_list = [[(209,161)]]
        # demo_map_with_door_400X400 = grayImage.copy()
        map_with_building_street_and_all_entrances = map_with_building_and_street.copy()
        for i in range(len(dr_cnt_list)):
            path = f'../building_entrance_street_dataset/{district_name}/'
            if not os.path.exists(path+'building'):
                os.makedirs(path+'building')
                print("The new directory is created!")
            if not os.path.exists(path+'building_with_entrance'):
                os.makedirs(path+'building_with_entrance')
                print("The new directory is created!")
            if not os.path.exists(path+'building_with_entrance_street_highlight'):
                os.makedirs(path+'building_with_entrance_street_highlight')
                print("The new directory is created!")    
            # (thresh, blackAndWhiteBuilding) = cv2.threshold(dr_cnt_list[i] + blackAndWhiteImage, 200, 255, cv2.THRESH_BINARY) 
            (_, black_and_white_building) = cv2.threshold(dr_cnt_list[i] + black_and_white_map_with_building, 200, 255, cv2.THRESH_BINARY) # retrieve the lost border of the building from tho original black and white map with building
            # building_wo_door_list.append(blackAndWhiteBuilding)
            starting_index = 1 # manually set to handle starting building index when there are multiple subdistricts in a district
            cv2.imwrite(path+f'building/{i+starting_index}.png',black_and_white_building)
            cv2.imshow(f'building {i+starting_index}', black_and_white_building)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            map_with_building_highlight_and_street = cv2.drawContours(resized_map_with_building_and_street.copy(),[cnt_list[i]],-1,(203,174,158),-1)
            map_with_building_highlight_and_street = cv2.resize(map_with_building_highlight_and_street, (map_with_building.shape[0],map_with_building.shape[1]))
            for j in range(len(entrance[i])):
                print(f'entrance node: {entrance[i][j]}')
                circle_center = np.mean(entrance[i][j],axis=0).astype(int)
                print(f'entrance center: {circle_center}')
                cv2.circle(black_and_white_building, circle_center, 5, 128, -1)
                cv2.line(map_with_building_highlight_and_street, entrance[i][j][0], entrance[i][j][1], (50,50,50), 2)
                cv2.line(map_with_building_street_and_all_entrances, entrance[i][j][0], entrance[i][j][1], (50,50,50), 2)
                # cv2.circle(building_street_with_highlight, entrance[i][j], 5, (128,128,128), -1)
            # cv2.putText(demo_map_with_door_400X400, f'{i+1}', (door_position_list[i][0]+20, door_position_list[i][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 75, 1, cv2.LINE_AA)
            cv2.imwrite(path+f'building_with_entrance/{i+starting_index}.png',black_and_white_building)
            cv2.imshow(f'building_with_entrance_{i+starting_index}', black_and_white_building)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            cv2.imwrite(path+f'building_with_entrance_street_highlight/{i+starting_index}.png',map_with_building_highlight_and_street)
            cv2.imshow(f'building_with_entrance_street_highlight{i+starting_index}', map_with_building_highlight_and_street)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            # building_with_door_list.append(building_with_door)
        cv2.imwrite(path+f'overview_map/map_with_building_street_and_all_entrances.png',map_with_building_street_and_all_entrances)
        cv2.imshow('map_with_building_street_and_all_entrances', map_with_building_street_and_all_entrances)
            

add_coordinate = False
rule_version  = 'around'
district_names = ['ladprao_centaragrand',
                     'central_ratchadamnoen_rd',
                     'sanamluang',
                     'lumphini_sathorn',
                     'chongnonsri_sathorn',
                     'triamudom',
                     'pratunam',
                     'rama3_ratchada',
                     'sathorn_naratiwat',
                     'prakanong',
                     'bangken']

selected_cnts = {'ladprao_centaragrand':[[41,51],[41]],
                     'central_ratchadamnoen_rd':[[106,116,122]],
                     'sanamluang':[[67],[88],[256],[50]],
                     'lumphini_sathorn':[[165]],
                     'chongnonsri_sathorn':[[19,25],[9,18,43]],
                     'triamudom':[[10,22]],
                     'pratunam':[[56]],
                     'rama3_ratchada':[[12],[0],[3]],
                     'sathorn_naratiwat':[[15,20],[9]],
                     'prakanong':[[49],[6,9]],
                     'bangken':[[63],[23]]}
entrances = {'ladprao_centaragrand':[[[[(188,142),(183,137)],[(169,161),(163,158)]],[[(137,99),(140,94)]]],[[[(92,58),(97,58)],[(87,90),(87,95)],[(181,114),(178,119)]]]],
                'central_ratchadamnoen_rd':[[[[(90,211),(95,212)],[(119,217),(124,218)],[(150,224),(155,225)]],[[(260,187),(266,189)],[(292,194),(296,190)]],[[(104,152),(109,153)]]]],
                'sanamluang':[[[[(151,267),(156,267)],[(104,194),(105,189)]]],[[[(115,180),(115,185)],[(286,177),(286,182)]]],[[[(98,177),(98,182)],[(110,222),(115,222)]]],[[[(175,170),(177,165)]]]],
                'lumphini_sathorn':[[[[(230,174),(236,176)],[(296,193),(290,196)],[(308,205),(306,200)]]]],
                'chongnonsri_sathorn':[[[[(101,184),(107,182)]],[[(280,132),(285,132)],[(316,194),(318,200)]]],[[[(243,244),(249,242)],[(237,255),(240,261)]],[[(95,153),(101,155)],[(124,192),(127,198)]],[[(245,178),(251,176)]]]],
                'triamudom':[[[[(154,311),(160,312)]],[[(249,278),(254,279)],[(214,277),(219,277)],[(211,262),(214,258)]]]],
                'pratunam':[[[[(136,137),(141,134)],[(137,179),(132,174)],[(216,195),(221,196)],[(277,199),(282,195)]]]],
                'rama3_ratchada':[[[[(240,223),(245,221)]]],[[[(208,216),(214,217)]]],[[[(140,255),(145,256)]]]],
                'sathorn_naratiwat':[[[[(183,126),(188,127)],[(176,131),(175,136)],[(212,207),(217,209)],[(225,206),(226,201)]],[[(219,105),(223,103)],[(250,137),(252,142)],[(260,183),(265,181)]]],[[[(308,136),(310,141)]]]],
                'prakanong':[[[[(119,170),(119,175)]]],[[[(236,242),(241,242)]],[[(140,181),(145,181)]]]],
                'bangken':[[[[(131,197),(136,199)],[(218,152),(223,148)]]],[[[(117,204),(119,199)]]]],
                'chula_engineering':[[[[(139,272),(143,272)]],[[(244,273),(249,274)],[(271,266),(271,271)]],[[(216,246),(220,247)]],[[(184,243),(188,244)]],[[(129,245),(129,250)]],[[(205,231),(209,232)]],[[(186,229),(190,230)]],[[(270,213),(275,214)],[(264,243),(269,244)]],[[(131,207),(131,211)]],[[(228,203),(232,204)]],[[(190,199),(194,200)]],[[(158,193),(162,194)]],[[(279,179),(283,180)],[(274,202),(278,203)]],[[(223,177),(227,177)],[(210,185),(214,186)],[(232,189),(236,189)]],[[(156,165),(160,166)],[(141,174),(145,174)],[(163,178),(167,178)]],[[(125,188),(129,188)]],[[(115,155),(119,156)]],[[(285,169),(289,170)],[(286,152),(290,152)]],[[(291,148),(295,148)]],[[(203,138),(208,139)],[(233,151),(232,155)],[(174,137),(173,141)],[(199,158),(203,159)],[(261,139),(265,139)]]]]
                }
read_csv = False
if read_csv:
    path = '../building_entrance_street_dataset/coordinate_sr.csv'
    df = pd.read_csv(path)
    print(df.head())
    
if add_coordinate:
    if rule_version == 'stratified':
        df = pd.DataFrame(columns = ['near', 'front', 'far','inside'])
        for district_name in district_names:
            print(f'in district name {district_name}')
            building_number = 0
            for subdistrict_number in range(len(selected_cnts[district_name])):
                subdistrict_cnts = selected_cnts[district_name][subdistrict_number]
                for building_cnt_number in range(len(subdistrict_cnts)):
                    building_number += 1
                    print(f'    in building number {building_number}')
                    possible_near = []
                    possible_front = []
                    possible_far = []
                    possible_inside = []
                    building_img = cv2.imread(f'../building_entrance_street_dataset/{district_name}/building/{building_number}.png', cv2.IMREAD_UNCHANGED)
                    # cv2.imshow('building_img',building_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # building_with_entrance = cv2.imread(f'building_entrance_dataset/{district_name}/building_with_entrance/{building_number}.png', cv2.IMREAD_UNCHANGED)
                    # building_with_entrance = cv2.cvtColor(building_with_entrance,cv2.COLOR_GRAY2RGB)
                    entrance_coordinate = entrances[district_name][subdistrict_number][building_cnt_number]
                    grayImage = building_img
                    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
                    dilated_edges = cv2.dilate(cv2.Canny(blackAndWhiteImage,0,255),None)
                    (cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    for j in range(len(cnts)):
                        area = cv2.contourArea(cnts[j])
                        if area < 500:
                            raise ValueError('invalid contour area')
                        else:
                            # dr_cnt = cv2.drawContours(255*np.ones_like(blackAndWhiteImage),[cnts[j]],-1, 0,-1)
                            # cv2.imshow('dr_cnt',dr_cnt)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            center, dim, angle = cv2.minAreaRect(cnts[j])
                            print(f'        building shape: {building_img.shape}, contour dim: {dim}')
                            front_radius_threshhold = max(dim)
                            for x in range(50,350):
                                for y in range(50,350):
                                    dist = cv2.pointPolygonTest(cnts[j],(x,y),True)
                                    dist_entrance = []
                                    for e in range(len(entrance_coordinate)):
                                        dist_entrance.append(np.linalg.norm(np.array((x,y))-np.mean(entrance_coordinate[e],axis=0).astype(int)))
                                    if dist >= -50 and dist < 0  :
                                        possible_near.append({'x':x,'y':y})
                                    if np.min(dist_entrance) <= front_radius_threshhold and dist < 0 and dist > -70:
                                        possible_front.append({'x':x,'y':y})
                                    if dist < -50:
                                        possible_far.append({'x':x,'y':y})
                                    if dist > 0:
                                        possible_inside.append({'x':x,'y':y})
                            break
                    # sample 30 points from possible points
                    possible_near_arr = np.array(possible_near)
                    possible_front_arr = np.array(possible_front)
                    possible_far_arr = np.array(possible_far)
                    possible_inside_arr = np.array(possible_inside)
                    print(f'            check shape:{possible_near_arr.shape}')
                    print(f'            check shape:{possible_front_arr.shape}')
                    print(f'            check shape:{possible_far_arr.shape}')
                    print(f'            check shape:{possible_inside_arr.shape}')
                    sampled_near = np.random.choice(possible_near_arr,30,replace=False)
                    sampled_front = np.random.choice(possible_front_arr,30,replace=False)
                    sampled_far = np.random.choice(possible_far_arr,30,replace=False)
                    sampled_inside = np.random.choice(possible_inside_arr,30,replace=False)
                    # serie = pd.Series({'near':sampled_near,'front':sampled_front,'far':sampled_far,'inside':sampled_inside})
                    # serie.name = f'{district_name}/{building_number+1}'
                    df.loc[f'{district_name}/{building_number}'] = {'near':sampled_near,'front':sampled_front,'far':sampled_far,'inside':sampled_inside}              
            print(df.head())
        path = '../building_entrance_street_dataset/stratified_coordinate_sr.csv'
        df.to_csv(path)  
    elif rule_version == 'around':
        df = pd.DataFrame(columns = ['around'])
        for district_name in district_names:
            print(f'in district name {district_name}')
            building_number = 0
            for subdistrict_number in range(len(selected_cnts[district_name])):
                subdistrict_cnts = selected_cnts[district_name][subdistrict_number]
                for building_cnt_number in range(len(subdistrict_cnts)):
                    building_number += 1
                    print(f'    in building number {building_number}')
                    possible_around = []
                    building_img = cv2.imread(f'../building_entrance_street_dataset/{district_name}/building/{building_number}.png', cv2.IMREAD_UNCHANGED)
                    entrance_coordinate = entrances[district_name][subdistrict_number][building_cnt_number]
                    grayImage = building_img
                    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
                    dilated_edges = cv2.dilate(cv2.Canny(blackAndWhiteImage,0,255),None)
                    (cnts, hier) = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    for j in range(len(cnts)):
                        area = cv2.contourArea(cnts[j])
                        if area < 500:
                            raise ValueError('invalid contour area')
                        else:
                            center, dim, angle = cv2.minAreaRect(cnts[j])
                            print(f'        building shape: {building_img.shape}, contour dim: {dim}')
                            front_radius_threshhold = max(dim)
                            for x in range(50,350):
                                for y in range(50,350):
                                    dist = cv2.pointPolygonTest(cnts[j],(x,y),True)
                                    if dist >= -100 and dist < 0  :
                                        possible_around.append({'x':x,'y':y})
                            break
                    # sample 30 points from possible points
                    possible_around_arr = np.array(possible_around)
                    print(f'            check shape:{possible_around_arr.shape}')
                    sampled_around = np.random.choice(possible_around_arr,120,replace=False)
                    df.loc[f'{district_name}/{building_number}'] = {'around':sampled_around}              
            print(df.head())
        path = '../building_entrance_street_dataset/around_coordinate_sr.csv'
        df.to_csv(path)  

# # plot coordinate on building_with_entrance
# for sr in ['near','front','far','inside']:
#     path = f'building_entrance_dataset/{region_name}/building_with_entrance_coordinate/{building_number}/{sr}'
#     if not os.path.exists(path):
#         os.makedirs(path)
#         print("The new directory is created!")

# for k in range(len(sampled_near)):
#     building_with_entrance_coordinate = cv2.circle(building_with_entrance.copy(),(sampled_near[k]['x'],sampled_near[k]['y']),2,(255,255,0),-1)
#     cv2.imwrite(f'building_entrance_dataset/{region_name}/building_with_entrance_coordinate/{building_number}/near/{sampled_near[k]}.png',building_with_entrance_coordinate)
# for k in range(len(sampled_front)):
#     building_with_entrance_coordinate = cv2.circle(building_with_entrance.copy(),(sampled_front[k]['x'],sampled_front[k]['y']),2,(255,255,0),-1)
#     cv2.imwrite(f'building_entrance_dataset/{region_name}/building_with_entrance_coordinate/{building_number}/front/{sampled_front[k]}.png',building_with_entrance_coordinate)
# for k in range(len(sampled_far)):
#     building_with_entrance_coordinate = cv2.circle(building_with_entrance.copy(),(sampled_far[k]['x'],sampled_far[k]['y']),2,(255,255,0),-1)
#     cv2.imwrite(f'building_entrance_dataset/{region_name}/building_with_entrance_coordinate/{building_number}/far/{sampled_far[k]}.png',building_with_entrance_coordinate)
# for k in range(len(sampled_inside)):
#     building_with_entrance_coordinate = cv2.circle(building_with_entrance.copy(),(sampled_inside[k]['x'],sampled_inside[k]['y']),2,(255,255,0),-1)
#     cv2.imwrite(f'building_entrance_dataset/{region_name}/building_with_entrance_coordinate/{building_number}/inside/{sampled_inside[k]}.png',building_with_entrance_coordinate)
    
        



