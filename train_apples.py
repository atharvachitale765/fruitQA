import os
import yaml
import torch
import argparse
import logging
from yolov5 import train
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# Load Configurations from YAML
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Argument Parser for Dynamic Paths
parser = argparse.ArgumentParser(description="Train YOLOv5 and CNN for Apple Classification")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
args = parser.parse_args()
config = load_config(args.config)

# Paths from Config
TRAIN_DIR = config['train_dir']
VAL_DIR = config['val_dir']
CNN_MODEL_PATH = config['cnn_model_path']
YOLO_MODEL_PATH = config['yolo_model_path']
DATA_YAML_PATH = config['data_yaml_path']

# YOLOv5 Training
def train_yolov5():
    logging.basicConfig(filename='training.log', level=logging.INFO)

    args = {
        'img_size': 640,
        'batch_size': 16,
        'epochs': 50,
        'data': DATA_YAML_PATH,
        'weights': 'yolov5s.pt',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_period': 1,
        'project': './runs/train',
        'name': 'apple_quality_detection'
    }

    for epoch in range(args['epochs']):
        logging.info(f'Epoch {epoch + 1}')
        loss, map_50 = train.run(**args)
        logging.info(f'Loss: {loss}, mAP@0.5: {map_50}')
    
    os.rename("./runs/train/apple_quality_detection/weights/best.pt", YOLO_MODEL_PATH)
    print(f"YOLOv5 training complete. Model saved to {YOLO_MODEL_PATH}")

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN Training Function
def train_cnn():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), CNN_MODEL_PATH)
    print(f'CNN model trained and saved at {CNN_MODEL_PATH}')

if __name__ == '__main__':
    train_yolov5()
    train_cnn()
