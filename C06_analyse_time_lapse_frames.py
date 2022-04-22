# Pose estimation approach (PEA)
# Jorrit van Gils
# Analyze time lapse frames
# 13/12/2021

import deeplabcut

# RUN C06_analyse_time_lapse_frames.py ON THE GPU PC!

#config_path = 'C:/Users/jorri/PycharmProjects/Thesis/02DeepLabCut/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'
config_path = '/opt/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'
#apply padding to change some images from shape 971,1920,3 to 1427,2048,3
path = '/opt/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/videos/padding/'

#GPU
deeplabcut.analyze_time_lapse_frames(config_path, path, frametype='.jpg', shuffle=0, save_as_csv=True)
#local (results in an error tensor flow not updated)
#deeplabcut.analyze_time_lapse_frames(config_path, 'C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gilts-2021-09-30\\videos\\', frametype='.jpg', shuffle=0, save_as_csv=True)
