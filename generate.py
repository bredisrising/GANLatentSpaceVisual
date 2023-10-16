import torch
from torch import nn
import numpy as np

class Generator:
    def __init__(self):
        self.G = GeneratorModel()
        self.G.load_state_dict(torch.load("./models/1000_epochs_2_dim.pth"))
        pass

    def generate(self, x, y):
        input_tensor = torch.tensor([[[[x]], [[y]]]], dtype = torch.float32)
        image = self.G(input_tensor)
        image = image.squeeze().detach().numpy()
        return image

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
   

        self.a = nn.Sequential(
            nn.ConvTranspose2d(2, 64 * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
        )
        
        self.b = nn.Sequential(
            nn.ConvTranspose2d(64 * 8, 64 * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
        )
        
        self.c = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 1),
            nn.ReLU(True),
        )
        
        self.d = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.a(x)
        #print(x.shape)
        x = self.b(x)
        #print(x.shape)
        x = self.c(x)
        x = self.d(x)
        
        return x