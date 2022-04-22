# Pose estimation approach (PEA)
# Jorrit van Gils
# Decision tree based on predicted random forest labels
# 03/03/2022

######################### part 1 parent behaviour #################################
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
  )
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt

# step 1 - load feature extraction output csv file as features
filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/03random_forest_output_parent.csv"
rf_pred = pd.read_csv(filepath)

# Build feature/target arrays
# uncomment this line for behaviour
X, y = rf_pred.drop("behaviour_pred", axis=1), rf_pred["behaviour_pred"].values.flatten()
feature_names = X.columns

# Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, test_size=0.20, stratify=y
)

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
forest = RandomForestClassifier(max_depth=3, n_estimators=1)
pipeline = Pipeline(
    steps=[
        #("preprocess", full),
        ("base",forest),
    ]
)
# Fit
_ = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

############### Create dendrogram ######################
print(len(forest.estimators_))
plt.figure(figsize=(20,20))
_ = tree.plot_tree(forest.estimators_[0], feature_names=X.columns, filled=True)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03decision_tree_dendogram_parent")



######################### part 2 behaviour #######################################
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
  )
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt

# step 1 - load feature extraction output csv file as features
filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/03random_forest_output.csv"
rf_pred = pd.read_csv(filepath)

# Build feature/target arrays
X, y = rf_pred.drop("behaviour_sub_pred", axis=1), rf_pred["behaviour_sub_pred"].values.flatten()
feature_names = X.columns

# Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, test_size=0.20, stratify=y
)

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
forest = RandomForestClassifier(max_depth=3, n_estimators=1)
pipeline = Pipeline(
    steps=[
        #("preprocess", full),
        ("base",forest),
    ]
)
# Fit
_ = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

############### Create dendrogram ######################
print(len(forest.estimators_))
plt.figure(figsize=(20,20))
_ = tree.plot_tree(forest.estimators_[0], feature_names=X.columns, filled=True)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/results/03decision_tree_dendogram")
