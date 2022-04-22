# Thesis
# Jorrit van Gils
# Create confusion matrix
# 18/03/2022

######################### part 1 parent behaviour #################################

import pandas as pd
import numpy as np
import re
import pandas
from sklearn import metrics
import json

# 1 Load the object detection (Yolo) prediction data
filepath = "C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/Results/parent_behaviour_predictions.csv"
preds = pd.read_csv(filepath).drop(["id", "asset_id", "model_id", "id-2", "validated", "meta", "in_validation_set", "unclear", "project_id", "deleted", "liked", "original_category"], axis=1)

# 2 manually filter the prediction column for values with the highest confidence (perhaps manually)
# check!

# 3 Add ground truth labels. Merge based on communal attribute 'path'. Replace nummerical values for behaviours e.g. 'foraging'.
filepath = "C:/Users/jorri/PycharmProjects/BOX21/import_BOX21/required_files/labels_Reddeer_JG.csv"
df_agouti = pd.read_csv(filepath)
df_agouti = df_agouti[["behaviour", "path"]]
preds = pd.merge(preds, df_agouti)

# go over the rows and obtain the number of the class
all_preds = []
for index, row in preds.iterrows():
    pred = row["preds"]
    tail = pred.rsplit(",", 1)[1]
    tail_num = re.findall(r'\d+', tail)
    all_preds.append(tail_num[0])

preds["preds_num"] = all_preds
preds['preds_num'] = preds['preds_num'].astype(int)

# #translate pred_num into pred_word
# converted_values = []
# for i in range(len(preds)):
#     row = preds.iloc[i]
#     value = row["preds_num"]
#     converter_value = -1
#     if value ==0:
#         converter_value = 'other'
#         print(converter_value)
#     elif value == 1:
#         converter_value = "foraging"
#     elif value == 2:
#         converter_value = "moving"
#     elif value == 99:
#         converter_value = "missing"
#     converted_values.append(converter_value)
# preds["pred"] = converted_values

#translate pred_num into pred_word
conditions = [
    (preds['preds_num'] == 0),
    (preds['preds_num'] == 1),
    (preds['preds_num'] == 2),
    (preds['preds_num'] == 99)
    ]
values = ['other', 'foraging', 'moving', 'missing']
preds['pred'] = np.select(conditions, values)

# #translate behaviour into words
# conditions = [
#     (preds['behaviour'] == "f"),
#     (preds['behaviour'] == "m"),
#     (preds['behaviour'] == "o")
#     ]
# values = ['foraging', 'moving', 'other']
# preds['behaviour'] = np.select(conditions, values)

#translate behaviour into numbers
conditions = [
    (preds['behaviour'] == "f"),
    (preds['behaviour'] == "m"),
    (preds['behaviour'] == "o")
    ]
values = ['0', '1', '2']
preds['behaviour_num'] = np.select(conditions, values)
preds['behaviour_num'] = preds['behaviour_num'].astype(int)

#translate preds_num from the csv into the real order 0=f, 1=m, 2 =0
conditions = [
    (preds['preds_num'] == 0),
    (preds['preds_num'] == 1),
    (preds['preds_num'] == 2),
    (preds['preds_num'] == 99)
    ]
values = [2, 0, 1, 99]
preds['preds_num'] = np.select(conditions, values)
preds['preds_num'] = preds['preds_num'].astype(int)

#translate behaviour into numbers
conditions = [
    (preds['pred'] == "foraging"),
    (preds['pred'] == "moving"),
    (preds['pred'] == "other")
    ]
values = ['f', 'm', 'o']
preds['pred'] = np.select(conditions, values)

#reorder preds dataframe
del preds["preds"]
preds = preds[['path', 'behaviour', 'behaviour_num', 'pred','preds_num']]

# delete all rows that the YOLOv5 network was not able to detect any categoriescontain missing values.
preds = preds.drop(preds[preds.preds_num == 99].index)
# or let the model create a random guess

# 4 prepare confusion matrix: bring all test ground truth (y_actu) and predicted values (y_pred) seperately in a list
# y_actu = preds["behaviour_num"]
# y_pred = preds["preds_num"]
# y_pred.value_counts()
y_actu = preds["behaviour"]
y_pred = preds["pred"]
y_actu = pd.Series(y_actu, name='True')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=0)

import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap , vmin=0, vmax=0.8) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    #print(tick_marks)
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/results/03_confusion_matrix_parent")

plot_confusion_matrix(df_conf_norm, title='YOLOv5')

# 6 print statistics precision and recall
print(metrics.classification_report(y_actu, y_pred, labels=["f","m","o"]))
report = metrics.classification_report(y_actu, y_pred, labels=["f","m","o"], output_dict=True)
# print(metrics.classification_report(y_actu, y_pred, labels=["foraging","moving","other", "missing"]))
# report = metrics.classification_report(y_actu, y_pred, labels=["foraging","moving","other", "missing"], output_dict=True)


######################### part 2 behaviour #######################################
import pandas as pd
import numpy as np
import re
import pandas

# 1 Load the object detection (Yolo) prediction data
filepath = "C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/Results/behaviour_predictions.csv"
preds = pd.read_csv(filepath).drop(["id", "asset_id", "model_id", "id-2", "validated", "meta", "in_validation_set", "unclear", "project_id", "deleted", "liked", "original_category"], axis=1)

# 2 manually filter the prediction column for values with the highest confidence (perhaps manually)
# check!

# 3 Add ground truth labels. Merge based on communal attribute 'path'. Replace nummerical values for behaviours e.g. 'foraging'.
filepath = "C:/Users/jorri/PycharmProjects/BOX21/import_BOX21/required_files/labels_Reddeer_JG.csv"
df_agouti = pd.read_csv(filepath)
df_agouti = df_agouti[["behaviour_sub", "path"]]
preds = pd.merge(preds, df_agouti)

# go over the rows and obtain the number of the class
all_preds = []
for index, row in preds.iterrows():
    pred = row["preds"]
    tail = pred.rsplit(",", 1)[1]
    tail_num = re.findall(r'\d+', tail)
    all_preds.append(tail_num[0])

preds["preds_num"] = all_preds
preds['preds_num'] = preds['preds_num'].astype(int)

#PUT THESE CONDITIONS IN A FUNCTION!!

#translate pred_num into pred_word
conditions = [
    (preds['preds_num'] == 0),
    (preds['preds_num'] == 1),
    (preds['preds_num'] == 2),
    (preds['preds_num'] == 3),
    (preds['preds_num'] == 4),
    (preds['preds_num'] == 5),
    (preds['preds_num'] == 6),
    (preds['preds_num'] == 7),
    (preds['preds_num'] == 8),
    (preds['preds_num'] == 9),
    (preds['preds_num'] == 10),
    (preds['preds_num'] == 99)
    ]
values = ['camera watching', 'grooming', 'roaring', 'sitting', 'standing', 'vigilance', 'grazing', 'browsing', 'scanning', 'running', 'walking', 'missing']
preds['pred'] = np.select(conditions, values)

#translate behaviour into words
conditions = [
    (preds['behaviour_sub'] == "ru"),
    (preds['behaviour_sub'] == "w"),
    (preds['behaviour_sub'] == "sc"),
    (preds['behaviour_sub'] == "b"),
    (preds['behaviour_sub'] == "gra"),
    (preds['behaviour_sub'] == "ro"),
    (preds['behaviour_sub'] == "si"),
    (preds['behaviour_sub'] == "gro"),
    (preds['behaviour_sub'] == "st"),
    (preds['behaviour_sub'] == "v"),
    (preds['behaviour_sub'] == "c")
    ]
values = ['running', 'walking', 'scanning', 'browsing', 'grazing', 'roaring', 'sitting', 'grooming', 'standing', 'vigilance', 'camera watching']
preds['behaviour'] = np.select(conditions, values)

#translate behaviour into numbers
conditions = [
    (preds['behaviour'] == "browsing"),
    (preds['behaviour'] == "grazing"),
    (preds['behaviour'] == "scanning"),
    (preds['behaviour'] == "walking"),
    (preds['behaviour'] == "running"),
    (preds['behaviour'] == "vigilance"),
    (preds['behaviour'] == "standing"),
    (preds['behaviour'] == "sitting"),
    (preds['behaviour'] == "roaring"),
    (preds['behaviour'] == "grooming"),
    (preds['behaviour'] == "camera watching")
    ]
values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
preds['behaviour_num'] = np.select(conditions, values)
preds['behaviour_num'] = preds['behaviour_num'].astype(int)

#translate preds_num from the csv into the real order 0=f, 1=m, 2 =0
conditions = [
    (preds['preds_num'] == 0),
    (preds['preds_num'] == 1),
    (preds['preds_num'] == 2),
    (preds['preds_num'] == 3),
    (preds['preds_num'] == 4),
    (preds['preds_num'] == 5),
    (preds['preds_num'] == 6),
    (preds['preds_num'] == 7),
    (preds['preds_num'] == 8),
    (preds['preds_num'] == 9),
    (preds['preds_num'] == 10),
    (preds['preds_num'] == 99)
    ]
values = [10,9,8,7,6,5,1,0,2,4,3,99]
#9,8,7,0
preds['preds_num'] = np.select(conditions, values)
preds['preds_num'] = preds['preds_num'].astype(int)

#translate preds into abbreviations
conditions = [
    (preds['pred'] == "running"),
    (preds['pred'] == "walking"),
    (preds['pred'] == "scanning"),
    (preds['pred'] == "browsing"),
    (preds['pred'] == "grazing"),
    (preds['pred'] == "roaring"),
    (preds['pred'] == "sitting"),
    (preds['pred'] == "grooming"),
    (preds['pred'] == "standing"),
    (preds['pred'] == "vigilance"),
    (preds['pred'] == "camera watching")
    ]
values = ['ru', 'w', 'sc', 'b', 'gra', 'ro', 'si', 'gro', 'st', 'v', 'c']
preds['pred'] = np.select(conditions, values)

#reorder preds dataframe
del preds["preds"]
preds = preds[['path', 'behaviour_sub', "behaviour_num", 'pred', "preds_num"]]

# delete all rows that the YOLOv5 network was not able to detect any categoriescontain missing values.
preds = preds.drop(preds[preds.preds_num == 99].index)
# or let the model create a random guess

# 4 prepare confusion matrix: bring all test ground truth (y_actu) and predicted values (y_pred) seperately in a list
# y_actu = preds["behaviour_num"]
# y_pred = preds["preds_num"]
# y_pred.value_counts()
y_test = preds["behaviour_sub"]
y_pred = preds["pred"]
y_test = pd.Series(y_actu, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

# 5 plot confusion matrix,
import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap , vmin=0, vmax=0.5) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    # plt.ylabel(df_confusion.index.name)
    # plt.xlabel(df_confusion.columns.name)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/results/03_confusion_matrix")

plot_confusion_matrix(df_conf_norm, title='ODA')

# 6 print statistics precision and recall
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))


