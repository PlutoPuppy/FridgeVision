# FridgeVision
A machine learning course project that detects food inside fridge with YOLO and give recipes suggestion.

<img width="802" alt="rawdata2" src="https://github.com/user-attachments/assets/2e8eb829-3e95-4371-94ff-d3e937517074">

Figure 1. A manually labelled image using roboflow platform




1) YOLO Model architecture:
<img width="915" alt="Screenshot 2024-10-30 at 12 22 57 AM" src="https://github.com/user-attachments/assets/d547fd24-da8b-4e97-8462-e395c543c784">

Figure 2. Adapted from https://arxiv.org/pdf/1506.02640



2) Recipe recommendation system:

   We used NER-enhanced recipe recommendation model to suggest recipes based on detected food vector.


# How to use this FridgeVision Project

First, open your VScode app on your computer.

Second, open terminal and do "git clone https://github.com/PlutoPuppy/FridgeVision.git"

Then, open src folder and run "APS360_FinalModel.ipynb" [Notice: please add your own API KEY for Roboflow if needed in this project"]

To train our model, the following datasets are used:

For YOLO detection model, please use dataset from our team's self-created open sourced Roboflow dataset, downloadable via this code:


rf = Roboflow(api_key="## USE YOUR OWN API KEY ##")

project = rf.workspace("tiffanyzha").project("oneclassfridgedata")

version = project.version(1)

dataset = version.download("yolov5")



For EfficientNet model, please download the zip file of datasets for cropped images via google drive shared link:

https://drive.google.com/file/d/17ast8UXsUpKC8uYM7XBoEM9iBw0BMThP/view?usp=sharing


For Recipe Recommendation model, we used kaggle dataset from link:

https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images


To test the model on new data, please download this .zip file:

https://drive.google.com/file/d/1-Gpszhwd_JUzfRI0o8VVSRqOdjH4VB4X/view?usp=sharing
