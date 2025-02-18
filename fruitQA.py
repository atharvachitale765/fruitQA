import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import logging

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


parser = argparse.ArgumentParser(description="Simple CNN for Apple Classification")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
args = parser.parse_args()
config = load_config(args.config)


TRAIN_DIR = config['train_dir']
VAL_DIR = config['val_dir']
MODEL_SAVE_PATH = config['cnn_model_path']


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
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    logging.basicConfig(filename='training.log', level=logging.INFO)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'CNN model trained and saved at {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train_cnn()
