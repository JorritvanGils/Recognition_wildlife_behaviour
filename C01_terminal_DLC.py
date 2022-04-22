# Pose estimation approach (PEA)
# Jorrit van Gils
# How to use DeepLabCut from the terminal
# 30/09/2021

# How to install DeepLabCut
# more info check https://deeplabcut.github.io/DeepLabCut/docs/installation.html
# 1) install anaconda

# From here, all commands are performed in the Terminal like in windows the command prompt
# 2) Download DLC conda file
# git clone https://github.com/DeepLabCut/DeepLabCut.git

# navigate to the folder where you downloaded e.g. cd C:\Users\YourUserName\Downloads and type in the terminal
# conda env create -f DEEPLABCUT.yaml
# activate DEEPLABCUT
# by now you should see (DEEPLABCUT) on the left of your terminal screen

#How to start DeepLabCut?
# activate DEEPLABCUT
# ipython
# import deeplabcut

#From here the user can decide to continue via the terminal or via the graphical user interfase
# deeplabcut.launch_dlc()

# Perform this step in Python to obtain the URLs for manually importing images
import os
#image_directory = 'C:\\Users\\jorri\\DeepLabCut\\DLC_test_train\\'
image_directory = 'C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30\\videos'
image_paths = []
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        path = os.path.join(image_directory, filename)
        image_paths.append(path)
        #or run it in one line
        # image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if
        #                file.endswith('.jpg')]
    else:
        continue
image_paths[0]

# so that you can find back your config_path
# config_path = 'C:/Users/jorri/DeepLabCut/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'

#(A) Create a new project
# copy the array form image path into image_path
# config_path=deeplabcut.create_new_project('Reddeer_DLC_JvG', 'Jorrit van Gils', image_paths, working_directory='C:\\Users\\jorri\\DeepLabCut', copy_videos=True, multianimal=False)

# Optionally add more Videos
# deeplabcut.add_new_videos(config_path, ['C:\\Users\\jorri\\DeepLabCut\\DLC_test_train\\00b5636b-9c22-4249-bc85-d1b76934df3c.jpg'], copy_videos=True)

#(B) Configure the Project
#performed manually

#(C) Data selection (extract frames)
# deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop=False)

#(D) Labeling frames
# deeplabcut.label_frames(config_path)

# (E) Check annotated frames
#use check_labels before proceding
# deeplabcut.check_labels(config_path, visualizeindividuals=True)

# If needed return to step D

#(F) Create training dataset GPU!
# Continue on this later via: https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html

#(G) Train network GPU!

#(H) Evaluate network GPU!

#(i) Analyze video
# deeplabcut.analyze_videos(config_path,['C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30\\videos\\00add8fd-bd07-4d43-a11d-ae8c30808c7e.jpg'], shuffle=0, save_as_csv=True, videotype=‘.avi’)
# deeplabcut.analyze_time_lapse_frames(config_path, 'C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gilts-2021-09-30\\videos', frametype='.jpg', shuffle=0, save_as_csv=True)
# deeplabcut.analyze_time_lapse_frames(config_path, 'C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gilts-2021-09-30\\videos\\00add8fd-bd07-4d43-a11d-ae8c30808c7e.jpg', frametype='.jpg', shuffle=0, save_as_csv=True)
# deeplabcut.analyze_time_lapse_frames(config_path, ['C:\\Users\\jorri\\DeepLabCut\\Reddeer_DLC_JvG-Jorrit van Gilts-2021-09-30\\videos\\00add8fd-bd07-4d43-a11d-ae8c30808c7e.jpg'], frametype='.jpg', shuffle=0, save_as_csv=True)