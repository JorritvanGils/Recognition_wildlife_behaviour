# Pose estimation approach (PEA)
# Jorrit van Gils
# How to work remote on the WUR GPU PC
# 30/09/2021

# all this code is used for the terminal
# autofill
# type first letters + tab

# terminal commands to connect with WEC GPU PC
# first activate tunneling connection from local pc to GPU PC via AnyDesk
# ssh wec@localhost -p 2022
# sudo -s (passw: wec123)
# cd /opt
# cd jorrit_model_rd/
# ls

# create a virtual evironment
# cd to a specific folder
# mkdir python-virtual-environments && cd python-virtual-environments
# python -m venv env
# env\Scripts\activate
# deactivate

# activate the virtual environment
# source venv38/bin/activate

# run the script
# python C03_create_training_dataset.py
# nohup python C04_train_and_evaluate_network.py &
# tail -f nohup.out
# error? use python3 instead of python

# remove files
# rm nohup.out
# rm -R evaluation-results_1_12
# rm -R dlc-models

# check how much memory is used
# nvidia-smi
# nvtop (voor de GPU)
# htop (voor de CPU)