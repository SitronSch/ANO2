import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self):
        print("INIT CNN")
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.AvgPool2d(2, 2)          #AvgPool bude pravdepodobne lepe fungovat v noci
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(2)
        #self.dropout=nn.Dropout(dropoutValue)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x=self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x=self.dropout(x)
        x = self.fc3(x)
        return x


