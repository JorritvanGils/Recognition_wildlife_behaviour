# Pose estimation approach (PEA)
# Jorrit van Gils
# Create training dataset by manual train and test split
# 1/12/2021

import deeplabcut
import pandas as pd
import os #for path split
import json

# RUN C03_create_training_dataset.py ON THE GPU PC! (but uncomment the 3 lines with notion LOCAL allows to run this script locally)

# 1 Load the csv file with for every image labelled key-point data into pandas + add /opt/jorrit_model_rd/ to server folder
#df Local
#df = pd.read_csv('C:/Users/jorri/PycharmProjects/Thesis/02DeepLabCut/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/training-datasets/iteration-0/UnaugmentedDataSet_Reddeer_DLC_JvGSep30/CollectedData_Jorrit van Gils.csv')
#df GPU
df = pd.read_csv('/opt/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/training-datasets/iteration-0/UnaugmentedDataSet_Reddeer_DLC_JvGSep30/CollectedData_Jorrit van Gils.csv')
#remove the first two rows of the df as these are not needed to iterate over the values.
df_values = df.drop([0,1])

# 2 create two arrays trainIndices = [], testIndices = []
trainIndices = []
testIndices = []

# 3 go through the csv file line by line
# 4 keep track of i (starting with i=0), Every line i = i + 1
# 5 assign the row to either train or test

#path Local
#path = "c:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/images/processed/train/"
#path GPU
path = "/opt/jorrit_model_rd/train/"

for i in range(len(df_values)):
    # iloc iterate over the rows, ['scorer'] takes the columns of interest
    path_df = df_values.iloc[i]['scorer']
    #split local
    split_df = path_df.split('\\')
    #split GPU
    #split_df = path_df.split('/')
    file = path + split_df[1] + ".jpg"
    if os.path.isfile(file):
        trainIndices.append(i)
    else:
        testIndices.append(i)

trainIndices
testIndices
# 6
config = '/opt/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'
deeplabcut.create_training_dataset(config,Shuffles=[0] ,trainIndices=[trainIndices],testIndices=[testIndices], net_type='resnet_50', augmenter_type='imgaug')

#Shuffles -> # Shuffles let you create different test/train splits (randomly or manually), but each shuffle would
# include all of the same images - just allocated differently (shuffles). If you decided later that you wanted a different
# allocation of test/train images, or to try the random test/train allocation that is the default, with the same
# base set of images, you can just change it to Shuffles=[1], and it will create a new training dataset for training
# without overwriting shuffle 0.


