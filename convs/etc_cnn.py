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

        self.fc1 = nn.Linear(in_features=16*2*8, out_features=out_dim)
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