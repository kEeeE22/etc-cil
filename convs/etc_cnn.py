import torch.nn as nn
import torch
import torch.nn.functional as F

class ETC_CNN(nn.Module):
    def __init__(self, out_dim=256):
        super(ETC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=1024, out_features=out_dim)
        self.drop4 = nn.Dropout(0.1)

        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = torch.flatten(x, start_dim=1)
        features = F.relu(self.fc1(x))
        features = self.drop4(features)

        return {
            "features": features,
            "fmaps": [],  
        }
    

class ETC_BN_CNN(nn.Module):
    def __init__(self, out_dim=256):
        super(ETC_CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # Block 3
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        # Fully connected
        self.fc1 = nn.Linear(in_features=1024, out_features=out_dim)
        self.drop4 = nn.Dropout(0.1)

        self.out_dim = out_dim

    def forward(self, x):
        # Block 1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Flatten + FC
        x = torch.flatten(x, start_dim=1)
        features = F.relu(self.fc1(x))
        features = self.drop4(features)

        return {
            "features": features,
            "fmaps": [],  
        }

import torch.nn.functional as F
class ETC_CNN_0_1(nn.Module):
    def __init__(self, out_dim=256):
        super(ETC_CNN_0_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        # self.bn5 = nn.BatchNorm2d(32)
        # self.relu5 = nn.ReLU()

        # self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        # self.bn6 = nn.BatchNorm2d(16)
        # self.relu6 = nn.ReLU()

        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop3 = nn.Dropout(0.25)

        # Define convolutional layers sequence
        self.conv_layers = nn.Sequential(
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2, self.relu2,
            self.pool1, self.drop1,
            self.conv3, self.bn3, self.relu3,
            self.conv4, self.bn4, self.relu4,
            self.pool2, self.drop2,
            # self.conv5, self.bn5, self.relu5,
            # self.conv6, self.bn6, self.relu6,
            # self.pool3, self.drop3
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 20, 256)
            flatten_dim = torch.flatten(self.conv_layers(dummy_input), start_dim=1).shape[1]

        self.fc1 = nn.Linear(in_features=flatten_dim, out_features=out_dim)
        self.bn_fc1 = nn.BatchNorm1d(out_dim)
        # self.drop4 = nn.Dropout(0.1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        features = self.bn_fc1(F.relu(self.fc1(x)))
        # x = self.drop4(x)


        return {
            "features": features,
            "fmaps": [],  
        }