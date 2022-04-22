# Pose estimation approach (PEA)
# Jorrit van Gils
# feature extraction key-points for behaviour detection
# 16/02/2022

import pandas as pd
import numpy as np
import math

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

# step 1 - load df with predicted key-point coordinates from DLC
filepath = "C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/01feature_extraction_input_paddingDLC_resnet50_Reddeer_DLC_JvGSep30shuffle0_250000.csv"
df = pd.read_csv(filepath)
bodyparts, coords = list(df.iloc[0]), list(df.iloc[1])

bodyparts_coords = []
counter = 0
for bodypart in bodyparts:
       bodyparts_coords.append(bodypart + "_" + coords[counter])
       counter +=1

df.columns = bodyparts_coords
df = df.iloc[2::]
df = df[df.columns.drop(list(df.filter(regex='likelihood')))]

cols = df.columns.drop('bodyparts_coords')
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df = df.rename(columns={"bodyparts_coords": "file_name"})


all_angles_L_Fw_y,all_angles_R_Fw_y, all_angles_R_B_y, all_angles_L_B_y = [], [], [], []
all_angles_L_Fw_h,all_angles_R_Fw_h, all_angles_R_B_h, all_angles_L_B_h = [], [], [], []
all_vertical_m_t_scaled_L_Fw, all_vertical_m_t_scaled_R_Fw, all_vertical_m_t_scaled_R_B, all_vertical_m_t_scaled_L_B = [], [], [], []
all_vertical_th_h_scaled_L_Fw, all_vertical_th_h_scaled_R_Fw, all_vertical_th_h_scaled_R_B, all_vertical_th_h_scaled_L_B = [], [], [], []

angle_names = [('L_leg_Fw_T_x','L_leg_Fw_T_y', 'L_leg_Fw_H_x', 'L_leg_Fw_H_y', 'L_leg_Fw_K_x', 'L_leg_Fw_K_y', 'Muzzle_y', 'Top_head_y'),
               ('R_leg_Fw_T_x','R_leg_Fw_T_y','R_leg_Fw_H_x', 'R_leg_Fw_H_y','R_leg_Fw_K_x','R_leg_Fw_K_y', 'Muzzle_y', 'Top_head_y'),
               ('R_leg_B_T_x', 'R_leg_B_T_y', 'R_leg_B_H_x', 'R_leg_B_H_y','R_leg_B_K_x','R_leg_B_K_y', 'Muzzle_y', 'Top_head_y'),
               ('L_leg_B_T_x','L_leg_B_T_y','L_leg_B_H_x','L_leg_B_H_y','L_leg_B_K_x','L_leg_B_K_y', 'Muzzle_y', 'Top_head_y')
               ]
counter = 0
for index, row in df.iterrows():
    for angle_name in angle_names:
        t_x = row[angle_name[0]]
        t_y = row[angle_name[1]]
        h_x = row[angle_name[2]]
        h_y = row[angle_name[3]]
        k_x = row[angle_name[4]]
        k_y = row[angle_name[5]]
        m_y = row[angle_name[6]]
        th_y = row[angle_name[7]]

        angle_y = round(getAngle((h_x,h_y),(k_x,k_y),(k_x,h_y)), 3)
        if angle_y > 180:
            angle_y = 180 - (angle_y - 180)
        angle_k = round(getAngle((t_x, t_y), (h_x, h_y), (k_x, k_y)),3)
        if angle_k > 180:
            angle_k = 180 - (angle_k - 180)
        vertical_m_t_scaled = round((m_y - t_y) / (h_y - t_y), 3)
        vertical_th_h_scaled = round((th_y - h_y) / (k_y - h_y), 3)

        if counter == 0:
            all_angles_L_Fw_y.append(angle_y)
            all_angles_L_Fw_h.append(angle_k)
            all_vertical_m_t_scaled_L_Fw.append(vertical_m_t_scaled)
            all_vertical_th_h_scaled_L_Fw.append(vertical_th_h_scaled)
        elif counter == 1:
            all_angles_R_Fw_y.append(angle_y)
            all_angles_R_Fw_h.append(angle_k)
            all_vertical_m_t_scaled_R_Fw.append(vertical_m_t_scaled)
            all_vertical_th_h_scaled_R_Fw.append(vertical_th_h_scaled)
        elif counter == 2:
            all_angles_R_B_y.append(angle_y)
            all_angles_R_B_h.append(angle_k)
            all_vertical_m_t_scaled_R_B.append(vertical_m_t_scaled)
            all_vertical_th_h_scaled_R_B.append(vertical_th_h_scaled)
        elif counter == 3:
            all_angles_L_B_y.append(angle_y)
            all_angles_L_B_h.append(angle_k)
            all_vertical_m_t_scaled_L_B.append(vertical_m_t_scaled)
            all_vertical_th_h_scaled_L_B.append(vertical_th_h_scaled)
        else:
            continue

        if counter != len(angle_names)-1:
            counter += 1
        else:
            counter = 0

df["angle_LF_K"],df["angle_RF_K"],df["angle_RB_K"], df["angle_LB_K"] = all_angles_L_Fw_y, all_angles_R_Fw_y, all_angles_R_B_y, all_angles_L_B_y
df["angle_LF_H"],df["angle_RF_H"],df["angle_RB_H"], df["angle_LB_H"] = all_angles_L_Fw_h, all_angles_R_Fw_h, all_angles_R_B_h, all_angles_L_B_h
df["muzzle_toe_LF"], df["muzzle_toe_RF"], df["muzzle_toe_RB"], df["muzzle_toe_LB"] = all_vertical_m_t_scaled_L_Fw, all_vertical_m_t_scaled_R_Fw, all_vertical_m_t_scaled_R_B, all_vertical_m_t_scaled_L_B
df["head_heel_LF"], df["head_heel_RF"], df["head_heel_RB"], df["head_heel_LB"] = all_vertical_th_h_scaled_L_Fw, all_vertical_th_h_scaled_R_Fw, all_vertical_th_h_scaled_R_B, all_vertical_th_h_scaled_L_B

#df_features = df[["file_name", "angle_LF_K", "angle_R_Fw_y", "angle_R_B_y", "angle_L_B_y", "angle_L_Fw_k", "angle_R_Fw_k", "angle_R_B_k", "angle_L_B_k", "m_t_y_rel_L_Fw", "m_t_y_rel_R_Fw","m_t_y_rel_R_B","m_t_y_rel_L_B", "th_h_y_rel_L_Fw","th_h_y_rel_R_Fw","th_h_y_rel_R_B","th_h_y_rel_L_B"]]

filepath = "C:/Users/jorri/PycharmProjects/BOX21/import_BOX21/required_files/labels_Reddeer_JG.csv"
df_agouti = pd.read_csv(filepath)
df_agouti = df_agouti[["file_name", "behaviour", "behaviour_sub", "in_validation_set"]]
df_features = pd.merge(df, df_agouti)
df_features = df_features.iloc[:, [0] + list(range(37,56,1))]

# df_features
# df = df_features.rename(columns={"Category":"Pet"})

df_features.to_csv("C:/Users/jorri/PycharmProjects/Thesis/03BehaviourClassification/files/01feature_extraction_output.csv")

df_features.hist(figsize=(16, 12));

#Feature 1: shortest angle of the heel with the toe and the knee (x4)
#   Angle between 180 and 170 degrees this might indicate standing.
#   Angle lower than 170 decrees (bent knee) is likely to fall within range of moving
#   foraging can either happen while moving or while standing still. Not a key feature.

#Feature 2: shortest angle of the knee with heel, and a hypothetical point with x value of heel and y value of knee. (x4)
#   Angle within the range of 0 to 10 degrees might indicate standing
#   Angle higher than 10 degrees (diagonal leg) is likely to fall within range of moving
#   foraging can either happen while moving or while standing still. Not a key feature.

#Feature 3: distance muzzle_y to toe_y, relative to heel_y to toe_y (x4)
#   distance range from 0 to 2 might indicate foraging
#   distance range from 2 - 10 might indicate standing/other
#   distance range 10+ are probably errors

#Feature 4: distance top_head_y to heel_y, relative to knee_y to heel_y (x4)
#   distance range from 0 to 2 might indicate foraging
#   distance range from 2 - 10 might indicate standing/other
#   distance range 10+ are probably errors


