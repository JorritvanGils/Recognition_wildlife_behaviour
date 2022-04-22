# Thesis
# Jorrit van Gils
# Object detection inference
# 10/03/2022

import torch
import numpy as np

# This script allows to perform inference on the YOLOv5 model. Applying the model to new unseen frames.

# 1) If you have not done yet, download the model from BOX21 (I changed filename best.pt to B01_YOLOv5_model_behaviour.pt)

# 2) Change the paths to the model files:
path_parent_behaviour = 'C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/model/B01_YOLOv5_model_parent_behaviour.pt'
path_behaviour = 'C:/Users/jorri/PycharmProjects/Thesis/Object_detection_approach/model/B02_YOLOv5_model_behaviour.pt'

# 3) For argument path, choose either the path_parent_behaviour ("foraging", "moving", "other") or path_behaviour (all behaviour)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_parent_behaviour)  # local model

# 4) Choose  your image(s) in url_list. e.g. Red deer (Cervus elaphus) from wildlife camera traps and run function get_predictions
url_list = ["https://multimedia.agouti.eu/assets/6d976508-e30f-480a-84e7-4c3f10c36363/file",
            "https://multimedia.agouti.eu/assets/e7ad9af6-001d-4bdc-961a-a237554be9c0/file",
            "https://image.posterlounge.nl/images/l/1893440.jpg",
            "https://cdn.britannica.com/54/122954-050-83BC7000/Red-deer.jpg",
            "https://www.roblelie.nl/wp-content/uploads/2020/02/hert-en-ree-thumbnail.jpg",
            "https://www.ciwf.nl/media/3960001/Herten_in_het_wild.jpg"]

# url_list = ["https://multimedia.agouti.eu/assets/078e200b-e636-48c8-93ff-136e682f150a/file"]

def get_predictions(url_list):
    all_predictions = []
    counter = 0
    for url in url_list:
        Inferenceresults = model(url)
        # print(Inferenceresults.names)
        Inferenceresults.show()
        if len(Inferenceresults.pred[0]) == 1:
            confidence = Inferenceresults.pred[0][0][4]
            pred_class = Inferenceresults.pred[0][0][5]
            tuple = (counter, confidence, pred_class)
            all_predictions.append(tuple)
        if len(Inferenceresults.pred[0]) > 1:
            for Inferenceresults in Inferenceresults.pred[0]:
                confidence = Inferenceresults[4]
                pred_class = Inferenceresults[5]
                tuple = (counter, confidence, pred_class)
                all_predictions.append(tuple)
        counter = counter + 1
    return all_predictions
predictions = get_predictions(url_list)

# Inspect the first prediction
print(predictions[0])
# each prediction contains a tuple of 3 values: (1) the image it belongs to, (2) the confidence value and (3) the class
#  class (3) has the following order
    # parent behaviour
    # 0 = foraging , 1 = moving, 2 = other

    # all behaviour
    # 0 = camera watching, 1 = grooming, 2 = roaring, 3 = sitting, 4 = standing, 5 = vigilance,
    # 6 = grazing, 7 = browsing, 8 = scanning , 9 = running, 10 = walking, (11 = missing)

# by running predictions, notice that YOLOv5 predicts for some images  multiple bounding boxes
print(predictions) # uncomment



