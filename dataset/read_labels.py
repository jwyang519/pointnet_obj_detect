# This scrip shows how to load the table polygon labels
# The file ./labels/tabletop_labels.dat contains a python 
# 3D list where indexes correspond to [frame][table_instance][coordinate]:
# coordinates x,y represent polygon vertices around a table instance

import os
import pickle
import matplotlib.pyplot as plt

path = "./data_reduced/harvard_c5/hv_c5_1/"
# path = "./data_reduced/harvard_c6/hv_c6_1/"
# path = "./data_reduced/mit_76_studyroom/76-1studyroom2/"
# path = "./data_reduced/mit_32_d507/d507_2/"
# path = "./data_reduced/harvard_c11/hv_c11_2/"
# path = "./data_reduced/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/"
# path = "./data_reduced/mit_76_459/76-459b/"

label_path = "labels/tabletop_labels.dat"
img_path   = "image/"

with open(path+label_path, 'rb') as label_file:
    tabletop_labels = pickle.load(label_file)
    label_file.close()

img_list   = os.listdir(path+img_path)

for polygon_list,img_name in zip(tabletop_labels,img_list): 
    img = plt.imread(path+img_path+img_name)
    plt.imshow(img)
    for polygon in polygon_list:
        plt.plot(polygon[0]+polygon[0][0:1],polygon[1]+polygon[1][0:1],'r')
    plt.axis('off')
    plt.show()