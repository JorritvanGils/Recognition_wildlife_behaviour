# Pose estimation approach (PEA)
# Jorrit van Gils
# Random forest
# 13/03/2022

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
  )
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import joblib

######################### part 1 parent behaviour #################################

filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/02outliers_and_scaling_output.csv"

# obtain the manually generated train and test split created in C03_create_training_dataset.py
features = pd.read_csv(filepath).drop(["Unnamed: 0", "behaviour_sub"], axis=1)
features_train = features.loc[features['in_validation_set'] == False]
X_train, y_train = features_train.drop(["behaviour", "in_validation_set"], axis=1), features_train["behaviour"].values.flatten()
features_test = features.loc[features['in_validation_set'] == True]
X_test, y_test = features_test.drop(["behaviour", "in_validation_set"], axis=1), features_test["behaviour"].values.flatten()
feature_names = X_train.columns

# when a new random split is preferred (not in this study) use the code below:
# features = pd.read_csv(filepath).drop(["Unnamed: 0", "behaviour_sub", "in_validation_set"], axis=1)
# # Build feature/target arrays
# X, y = features.drop("behaviour", axis=1), features["behaviour"].values.flatten()
# feature_names = X.columns
# # Create train/test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=1121218, test_size=0.22, stratify=y
# )

# Build categorical preprocessor
categorical_cols = X_train.select_dtypes(include="object").columns.to_list()
categorical_pipe = make_pipeline(
    OneHotEncoder(sparse=False, handle_unknown="ignore")
  )

# Build numeric processor
to_log = []
to_scale = []
numeric_pipe_1 = make_pipeline(PowerTransformer())
numeric_pipe_2 = make_pipeline(StandardScaler())

# Full processor
full = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipe, categorical_cols),
        ("power_transform", numeric_pipe_1, to_log),
        ("standardization", numeric_pipe_2, to_scale),
    ]
)

# Final pipeline combined with RandomForest
forest = RandomForestClassifier(max_depth=13)
pipeline = Pipeline(
    steps=[
        #("preprocess", full),
        ("base",forest),
    ]
)
# Fit
_ = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Save the model
joblib.dump(pipeline, "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/models/C01_PEA_RF_model_parent.joblib")
# Load the model and compute predictions
# loaded_rf = joblib.load("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/models/RF_model_parent.joblib")
# loaded_rf.predict(X_test) #notice they are exactly the same as y_pred

############### Evaluation of the RF ######################
# 1) save feature importance plot in directory 'results'
# 2) create classification report (precision, recall, F1-score)
# 3) create a dataframe rf_pred for 03decision_tree
# 4) create a confidence_matrix with matplotlib

# 1) Feature importance
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance")
ax.set_ylabel("Feature importance (MDI)")
fig.tight_layout()
# uncomment this line for behaviour
fig.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03random_forest_feature_importance_parent")
# uncomment this line for behaviour_sub
# fig.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/02random_forest_feature_importance")

# 2) classification report
#Precision, recall and F1-score
print(classification_report(y_test, y_pred))

#ROC_AUC score
# Generate membership scores with .predict_proba
y_pred_probs = pipeline.predict_proba(X_test)
# Calculate ROC_AUC
roc_auc_score = roc_auc_score(
    y_test, y_pred_probs, multi_class="ovr", average="weighted"
  )
print(roc_auc_score)

# 3) create a dataframe rf_pred for 03decision_tree
rf_pred = X_test
# save predictions as csv
rf_pred["behaviour_pred"] = y_pred.tolist()
rf_pred.to_csv("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/03random_forest_output_parent.csv")

# 4) create a confidence_matrix by matplotlib
# y_actu = preds["behaviour_num"]
# y_pred = preds["preds_num"]
# y_pred.value_counts()
# y_actu = y_train
# y_pred = y_pred
y_test = pd.Series(y_test, name='True')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_test, y_pred)
df_confusion = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap, vmin=0, vmax=0.8) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03_confusion_matrix_parent")

plot_confusion_matrix(df_conf_norm, title='Confusion matrix PEA')


######################### part 2 behaviour #######################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
  )
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import joblib

filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/02outliers_and_scaling_output.csv"

# obtain the manually generated train and test split created in C03_create_training_dataset.py
features = pd.read_csv(filepath).drop(["Unnamed: 0", "behaviour"], axis=1)
features_train = features.loc[features['in_validation_set'] == False]
X_train, y_train = features_train.drop(["behaviour_sub", "in_validation_set"], axis=1), features_train["behaviour_sub"].values.flatten()
features_test = features.loc[features['in_validation_set'] == True]
X_test, y_test = features_test.drop(["behaviour_sub", "in_validation_set"], axis=1), features_test["behaviour_sub"].values.flatten()
feature_names = X_train.columns

# when a new random split is preferred (not in this study) use the code below:
# features = pd.read_csv(filepath).drop(["Unnamed: 0", "behaviour", "in_validation_set"], axis=1)
# # Build feature/target arrays
# X, y = features.drop("behaviour_sub", axis=1), features["behaviour_sub"].values.flatten()
# feature_names = X.columns
# # Create train/test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=1121218, test_size=0.22, stratify=y
# )

# Build categorical preprocessor
categorical_cols = X_train.select_dtypes(include="object").columns.to_list()
categorical_pipe = make_pipeline(
    OneHotEncoder(sparse=False, handle_unknown="ignore")
  )

# Build numeric processor
to_log = []
to_scale = []
numeric_pipe_1 = make_pipeline(PowerTransformer())
numeric_pipe_2 = make_pipeline(StandardScaler())

# Full processor
full = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipe, categorical_cols),
        ("power_transform", numeric_pipe_1, to_log),
        ("standardization", numeric_pipe_2, to_scale),
    ]
)

# Final pipeline combined with RandomForest
forest = RandomForestClassifier(max_depth=13)
pipeline = Pipeline(
    steps=[
        #("preprocess", full),
        ("base",forest),
    ]
)
# Fit
_ = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Save the model
joblib.dump(pipeline, "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/models/C02_PEA_RF_model.joblib")
# Load the model and compute predictions
# loaded_rf = joblib.load("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/models/RF_model.joblib")
# loaded_rf.predict(X_test) #notice they are exactly the same as y_pred

############### Evaluation of the RF ######################
# 1) save feature importance plot in directory 'results'
# 2) create classification report (precision, recall, F1-score)
# 3) create a dataframe rf_pred for 03decision_tree
# 4) create a confidence_matrix with matplotlib

# 1) Feature importance
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance")
ax.set_ylabel("Feature importance (MDI)")
fig.tight_layout()
fig.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03random_forest_feature_importance")

# 2) classification report
#Precision, recall and F1-score
print(classification_report(y_test, y_pred))

#ROC_AUC score
# Generate membership scores with .predict_proba
y_pred_probs = pipeline.predict_proba(X_test)
# Calculate ROC_AUC  !!!does not work because classes are missing(?)!!!
# roc_auc_score = roc_auc_score(
#     y_test, y_pred_probs, multi_class="ovr", average="weighted"
#   )
# print(roc_auc_score)

# 3) create a dataframe rf_pred for 04decision_tree
rf_pred = X_test
# save predictions as csv
rf_pred["behaviour_sub_pred"] = y_pred.tolist()
rf_pred.to_csv("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/03random_forest_output.csv")

# 4) create a confidence_matrix by matplotlib
# y_actu = preds["behaviour_num"]
# y_pred = preds["preds_num"]
# y_pred.value_counts()
y_test = pd.Series(y_test, name='True')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_test, y_pred)
df_confusion = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap, vmin=0, vmax=0.5) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel("True")
    # plt.xlabel(df_confusion.columns.name)
    plt.xlabel("Predicted")
    plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03_confusion_matrix")

plot_confusion_matrix(df_conf_norm, title='Confusion matrix PEA')

