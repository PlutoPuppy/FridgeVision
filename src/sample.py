import torch
from roboflow import Roboflow
import supervision as sv
import cv2
import torch.optim as optim
from yolov5.models.yolo import Model
from yolov5.utils import LoadImagesAndLabels
from yolov5.utils.general import check_dataset

# download datas from roboflow our FrideVision project
rf = Roboflow(api_key="REPLACE WITH YOUR OWN API KEY FOR ROBOFLOW")
project = rf.workspace("steph-r3xmc").project("fridgevision-ms4yf")
version = project.version(1)
dataset = version.download("yolov5")

# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# !pip install -r requirements.txt


# Load dataset:

# Define dataset configuration
data_config = check_dataset('data.yaml')  # Loads and verifies dataset paths and classes

# Load train and val datasets
train_dataset = LoadImagesAndLabels(data_config['train'], img_size=640, batch_size=16)
val_dataset = LoadImagesAndLabels(data_config['val'], img_size=640, batch_size=16)


# modify and load pre-trained model
# Load YOLOv5 model with pre-trained weights
model = Model(cfg='yolov5/models/yolov5s.yaml', ch=3, nc=len(data_config['names']))  # ch=3 for RGB, nc is the number of classes



# Download a compatible weight file, e.g., yolov5s.pt for yolov5s.yaml
# !wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt



# Run this in terminal: !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100

















######## Below is still on progress, don't need to run below ########



model.load_state_dict(torch.load('yolov5s.pt')['model'].state_dict(), strict=False)
# Freeze layers that do feature encoding before classification
# Freeze layers (optional)
for name, param in model.named_parameters():
    if 'model.24' not in name:  # Change layer index as needed
        param.requires_grad = False



# train the fine tuning model with just the last classification layer

# Set up optimizer and criterion
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, targets, paths, _) in enumerate(train_dataset):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    # Validation step (every few epochs)
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(images), targets) for images, targets, _, _ in val_dataset) / len(val_dataset)
        print(f'Validation Loss after Epoch {epoch+1}: {val_loss:.4f}')


torch.save(model.state_dict(), 'fine_tuned_yolov5.pth')
