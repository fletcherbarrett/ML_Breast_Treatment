# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:28:50 2021

@author: fletcherbarrett
"""

from CodeforFletcher import *
from Polygon import *
import numpy as np
import itertools
from tabulate import tabulate
import glob

#choice = ['FB','DIBH']
choice = ['FB']


# choice = ['test']
for s in choice:
    files = 0
    for filepath in glob.iglob('H:/Patients/' + s + '/*'):
        batch_anonymize(filepath, save_dir='H:/Anonymized_DATA/AnonymizedDICOM_' + s)
        files += 1
    
    #Define organs of interest    
    OOI = ['LUNG_R','CTVN_IMC_R','HEART']
    
    #Pairs of organs
    OOI_pairs = list(itertools.combinations(OOI, 2))
    
    data = []
    
    for i in np.arange(1,files+1):
        struct,dose,plan = load_dcm(i, data_dir='H:/Anonymized_DATA/AnonymizedDICOM_' + s)
        structures = read_structure(struct,[dose],plan,['LUNG_R','CTVN_IMC_R','CLAVICLE_R','HEART','LIVER','LUNG_ANT'],[])
        
        dic = {}
        
        clav_point = get_crit_points(structures['CLAVICLE_R']['voxels'])
        liv_point = get_crit_points(structures['LIVER']['voxels'])
        lung_point = get_crit_points(structures['LUNG_ANT']['voxels']) 

        dimension = ['_Med-Lat_(cm)','_Ant-Post_(cm)','_Sup-Inf_(cm)']
        position = ['_Lat', '_Med', '_Ant', '_Post', '_Inf', '_Sup']
        
        for organ in OOI:
            
            temp = get_crit_points(structures[organ]['voxels'])
                                               
            dic[organ + dimension[0]] = (temp[0][0][0]-temp[1][0][0])/10.0
            dic[organ + dimension[1]] = (temp[2][0][1]-temp[3][0][1])/10.0
            dic[organ + dimension[2]] = (temp[4][0][2]-temp[5][0][2])/10.0
            
            dic['Clav' + position[1] + '-' + organ + position[0]] = (temp[0][0][0]-clav_point[1][0][0])/10.0
            dic['Clav' + position[2] + '-' + organ + position[3]] = (temp[3][0][1]-clav_point[2][0][1])/10.0
            dic['Clav' + position[4] + '-' + organ + position[5]] = (temp[5][0][2]-clav_point[4][0][2])/10.0
            
            if organ == 'LUNG_R':
                dic[organ+'_Volume_(cm^3)'] = structures[organ]['volume (cc)']
                dic[organ+'_Max_Length_(cm)'] = get_max_chord(structures[organ])/10.0
                dic[organ+'_Surface_Area_(cm^2)'] = get_surface_area(structures[organ])
                dic['Liver' + position[5] + '-' + organ + position[5]] = (temp[5][0][2]-liv_point[5][0][2])/10.0
                dic['Lung' + position[2] + '-' + organ + position[5]] = (temp[5][0][2]-lung_point[4][0][2])/10.0
                  
        for pairs in OOI_pairs:
            dic[pairs[0] + '-' + pairs[1] + '_Min_(cm)'] = closest_OAR_proximity(pairs[0],pairs[1],structures)/10.0
            dic[pairs[0] + '-' + pairs[1] + '_Centroid_(cm)'] = distances3d(centroid(structures[pairs[0]]['voxels']),centroid(structures[pairs[1]]['voxels']))/10.0
        
        data_array = [vals for vals in list(dic.values())]
        data.append(data_array)
    
    f = 'H:/Feat_importances_txt_files/Feature_Data_' + s + ".txt"
    f = open(f, "a")
    
    name_array = [keys for keys in list(dic.keys())]
    txt_data = {"Features": name_array}
    
    for pat_num in range(len(data)):
        txt_data['Patient%s' %(pat_num+1)] = data[pat_num]
    
    f.write(tabulate(txt_data, headers = "keys"))
    
    f.close()    

    
    
    