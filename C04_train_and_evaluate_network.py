# Pose estimation approach (PEA)
# Jorrit van Gils
# Train and evaluate network
# 1/12/2021

# RUN C04_train_and_evaluate_network.py ON THE GPU PC!
# Use nohup (see C02_GPU_PC_commands)

import deeplabcut
import tensorflow as tf

config = '/data3/jorrit_model_rd/Reddeer_DLC_JvG-Jorrit van Gils-2021-09-30/config.yaml'

# Adjust memory usage
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.50)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#Edditing the config file in pycharm
trainposeconfigfile, testposeconfigfile, snapshotfolder = deeplabcut.return_train_network_path(config, shuffle=0)
cfg_dlc = deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
# Change the batch_size -> learning from 1 image at the time specifcfs of that image, but 4 at the time, shoulder leg, trying to find pattern works across all images.
cfg_dlc['batch_size'] = 4
deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile, cfg_dlc)

#train and evaluate the network
# allow_growth=False -> a lot of memory GPU, allow_growth=True allows to exceed the allocated GPU memory
deeplabcut.train_network(config, shuffle=0, displayiters=100, saveiters=5000, maxiters=350000, allow_growth=True, gputouse=0, max_snapshots_to_keep = 70)
#deeplabcut.train_network(config, shuffle=1, displayiters=100, saveiters=100)
deeplabcut.evaluate_network(config,plotting=True, Shuffles = [0], gputouse=0, show_errors=True)
