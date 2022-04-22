# Pose estimation approach (PEA)
# Jorrit van Gils
# Random forest based on extracted features
# 23/02/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import numpy
from matplotlib import pyplot

# step 1 - load feature extraction output csv file as features
filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/01feature_extraction_output.csv"
features = pd.read_csv(filepath).drop(["Unnamed: 0", "file_name"], axis=1)
cols = ['muzzle_toe_LF','muzzle_toe_RF', 'muzzle_toe_RB', 'muzzle_toe_LB', 'head_heel_LF','head_heel_RF', 'head_heel_RB', 'head_heel_LB'] # one or more

upper_quantiles = features[cols].quantile(0.97)
down_quantiles = features[cols].quantile(0.03)
outliers_high = (features[cols] > upper_quantiles)
outliers_low = (features[cols] < down_quantiles)
features[outliers_high] = np.nan
features.fillna(upper_quantiles, inplace=True)
features[outliers_low] = np.nan
features.fillna(down_quantiles, inplace=True)

# step 2 - scale the data.
to_scale = ['angle_LF_K', 'angle_RF_K', 'angle_RB_K', 'angle_LB_K', 'angle_LF_H',
       'angle_RF_H', 'angle_RB_H', 'angle_LB_H', 'muzzle_toe_LF',
       'muzzle_toe_RF', 'muzzle_toe_RB', 'muzzle_toe_LB', 'head_heel_LF',
       'head_heel_RF', 'head_heel_RB', 'head_heel_LB']
ss = StandardScaler()
_ = ss.fit(features[to_scale])
features[to_scale] = pd.DataFrame(ss.transform(features[to_scale]), columns=to_scale)
features.to_csv("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/02outliers_and_scaling_output.csv")

# visualise + describing removing the outliers and scaling
# features[to_scale].hist(figsize=(14, 14));
# features['y_angle_LFw'].hist(figsize=(14, 14), by = features["behaviour"]);
# features.hist(figsize=(16, 12));
# describe = features.describe().T.round(3)

# step 3 - Creating the histograms and save them locally
moving_all = features[features['behaviour']=='m']
foraging_all = features[features['behaviour']=='f']
# other_all = features[features['behaviour']=='o']
columns = features.columns
columns= columns[0:-3]
for column in columns:
    moving = list(moving_all[column])
    foraging = list(foraging_all[column])
    # other = list(other_all[column])
    bins = numpy.linspace(-3, 3, 25)
    pyplot.hist(moving, bins, alpha=0.5, label='moving')
    pyplot.hist(foraging, bins, alpha=0.5, label='foraging')
    pyplot.legend(loc='upper right')
    plt.title(column)
    pyplot.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/histogram_" + column)
    pyplot.clf()

four_features = features[["angle_LB_K", "angle_LB_H", "muzzle_toe_LB", "head_heel_LB", "behaviour"]]
four_features.head()

conditions = [
    (four_features['behaviour'] == "f"),
    (four_features['behaviour'] == "m"),
    (four_features['behaviour'] == "o")
    ]
values = ['foraging', 'moving', 'other']
four_features['behaviour'] = np.select(conditions, values)

four_features.rename(columns = {'behaviour':'label', 'angle_LB_K':'feature1', 'angle_LB_H':'feature2', 'muzzle_toe_LB':'feature3', 'head_heel_LB':'feature4'}, inplace = True)
four_features["prediction"] = four_features["label"]