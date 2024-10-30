
# do: "pip install roboflow supervision opencv-python"
# to get packages needed
# or use "pip3"

# importing libraries
import torch
from roboflow import Roboflow
import supervision as sv
import cv2


# get model with transfer learning
# cited from here: https://universe.roboflow.com/workspace01-ae0oa/fridgify
rf = Roboflow(api_key="CHANGE TO YOUR OWN API KEY")
project = rf.workspace().project("fridgify")
model = project.version(3).model


# feed with our own data
result = model.predict("FridgeVision/sample-images/foodsInFridge.jpg", confidence=40, overlap=30).json()
labels = [item["class"] for item in result["predictions"]]
print(result)
print(labels)





