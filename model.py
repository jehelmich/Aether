## define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # Convolution 1 - input: 224x224x1; Output: 112x112x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # 5x5 Convolution Kernel
            nn.BatchNorm2d(32), # Batch Normalisation 
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # Pooling
        )
        
        # Convolution 2 - input: 112x112x32; Output: 56x56x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), # 3x3 Convolution Kernel
            nn.BatchNorm2d(64), # Batch Normalisation 
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # Pooling
        )
        
        # Fully connected - input: 14x14x128; Output: 136
        self.lr = nn.Sequential(            
            nn.Dropout(p=0.6),
            nn.Linear(7*7*128,136),

        )
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):                           
                
                I.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.0001)
                m.bias.data.zero_()
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.lr(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
