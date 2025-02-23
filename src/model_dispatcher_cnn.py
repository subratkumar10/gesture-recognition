from numpy import pad
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

import config
# Creating a CNN class
class CustomMotionModel(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=config.EXTRACTED_FEATURES, out_channels=16, kernel_size=3,stride=3,padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=3,padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),stride=3,padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=3,padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size =(2,2), stride = (2,2))
        
        self.fc1 = nn.Linear(64, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, config.NUM_CLASSES)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def dispatch_model(name_of_model):
    model_dict = {
        "CustomMotionModel": CustomMotionModel(),
    }
    return model_dict[name_of_model]