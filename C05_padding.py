# Pose estimation approach (PEA)
# Jorrit van Gils
# Padding to allow obtaining key-point coordinates
# 12/01/2022

#import libraries
from fastai.data.all import *
from fastai.vision.all import *
import cv2 #load images
import fastbook #resize
from fastbook import *
import matplotlib.pyplot as plt #visualise images
import matplotlib.image as mpimg
from PIL import Image

# RUN C05_padding.py EITHER LOCALLY OR ON THE GPU PC!

#set path to videos and final padding folder
path = "/Thesis/02DeepLabCut/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/images/"
#path = '/opt/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'
path_padding = path+"padding/"

#file_paths[0] = Path('C:/Users/jorri/PycharmProjects/Thesis/02DeepLabCut/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/images/0a559cbf-7832-4d0b-b5e1-3237f234c9e3.jpg')
file_paths = get_image_files(path)
# python object, method properties
# file_paths[0].name
# file_paths[0].parent

strings =[]
tensors = []
# range(0, 4)
for i in range(len(file_paths)):
    # str(file_paths[0]) = 'C:\\Users\\jorri\\PycharmProjects\\Thesis\\02DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30\\images\\0a559cbf-7832-4d0b-b5e1-3237f234c9e3.jpg'
    string = str(file_paths[i])
    # print(string)
    if string is not None:
        strings.append(string)

    # cv2.imread(str(file_paths[0])) = array([[[53, 53, 53],
    #                                          [40, 40, 40],
    tensor = cv2.imread(string)
    #print(tensor.shape)
    if tensor is not None:
        tensors.append(tensor)

############## 2 take the width and hight from image with largest dimentions #############
# h,w,c = images[0].shape
#max height and max width.
max_height = 0
max_width = 0
for i in range(len(tensors)):
    #Go through each image and store the shape with largest value in max_dimension
    #print("Original dimension image", i, " : ",images[i].shape)
    w,h,c = tensors[i].shape
    if (h > max_height):
        max_height = h
    if (w > max_width):
        max_width = w
# indexing with a : slice keeps the axis before 2 (axis 0 and 1) of (1427, 2048, 3)
print("The max_width = ",max_width, "and the max_heigth = ", max_height)

############## 3 apply padding and convert all images to the image with the largest dimentions ###########
images = []
for i in range(len(strings)):
    img = Image.open(strings[i])
    #print(img)
    images.append(img)

if not os.path.exists(path_padding):
    os.makedirs(path_padding)

#Apply padding
for i in range(len(images)):
    img = images[i]
    #Blue rectangle (mode, size, color=0)
    result = Image.new(img.mode, (max_height, max_width), (0, 0, 255))
    #copy image on rectangle
    result.paste(img)
    result.save(path_padding + file_paths[i].name)
