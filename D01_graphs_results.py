# Pose estimation approach (PEA)
# Jorrit van Gils
# creating graphs results
# 18/04/2022

######################### part 1 parent behaviour #################################

import numpy as np
import matplotlib.pyplot as plt

# accuracy
method = ["YOLOv5","PEA"]
value = [77,53]
New_Colors = ['dimgray','lightgray']
plt.bar(method, value, color=New_Colors)
# plt.title('Country Vs GDP Per Capita', fontsize=14)
# plt.xlabel('Method', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(top=100)
plt.grid(False)
# plt.legend(loc="upper left")
# plt.show()
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/accuracy_parent")

# Precision
labels = ["foraging", "moving", "other"]
YOLOv5 = [83.33,84.375,60.0]
PEA = [63.0,52.0,35.0]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, YOLOv5, width, label='YOLOv5', color=['dimgray'])
rects2 = ax.bar(x + width/2, PEA, width, label='PEA', color=['lightgray'])
ax.set_ylabel('Precision (%)')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.ylim(top=100)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/precision_parent")

# Recall
labels = ["foraging", "moving", "other"]
YOLOv5 = [88.889,65.854,75.0]
PEA = [76.0, 33.0,46.0]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, YOLOv5, width, label='YOLOv5', color=['dimgray'])
rects2 = ax.bar(x + width/2, PEA, width, label='PEA', color=['lightgray'])
ax.set_ylabel('Recall (%)')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.ylim(top=100)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/recall_parent")


######################### part 2 behaviour #######################################
import numpy as np
import matplotlib.pyplot as plt

# accuracy
method = ["YOLOv5","PEA"]
value = [50,36]
New_Colors = ['dimgray','lightgray']
plt.bar(method, value, color=New_Colors)
# plt.title('Country Vs GDP Per Capita', fontsize=14)
# plt.xlabel('Method', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(top=100)
plt.grid(False)
# plt.legend(loc="upper left")
# plt.show()
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/accuracy")

# Precision
labels = ["gra", "sc", "w","v", "st", "b","ru", "si", "ro","gro", "c"]
YOLOv5 = [79,50,57,26,20,0,0,0,0,0,100]
PEA = [43,7,45,17,100,0,0,0,0,33,0]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, YOLOv5, width, label='YOLOv5', color=['dimgray'])
rects2 = ax.bar(x + width/2, PEA, width, label='PEA', color=['lightgray'])
ax.set_ylabel('Precision (%)')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.ylim(top=100)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/precision")

# Recall
labels = ["gra", "sc", "w","v", "st", "b","ru", "si", "ro","gro", "c"]
YOLOv5 = [87,31,38,88,22,0,0,0,0,0,50]
PEA = [74,8,38,11,11,0,0,0,0,33,0]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, YOLOv5, width, label='YOLOv5', color=['dimgray'])
rects2 = ax.bar(x + width/2, PEA, width, label='PEA', color=['lightgray'])
ax.set_ylabel('Recall (%)')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.ylim(top=100)
plt.savefig("C:/Users/jorri/PycharmProjects/Thesis/results_general/recall")
