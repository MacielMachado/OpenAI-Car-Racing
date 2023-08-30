from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, input_ch=4, ch=8):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=8*ch, kernel_size=(7,7)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8*ch, out_channels=ch*16, kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*16, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*32, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch*64, out_channels=ch*64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
        )
        self.flat_layer = nn.Sequential(
            nn.Linear(64*ch*1*1, 1024),
            nn.ReLU()
        )
        self.output = nn.Linear(in_features=1024, out_features=3)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv_layer(x.to(torch.float32))
        x = x.view(x.size(0), -1)
        x = self.flat_layer(x)
        x = self.output(x)

        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.sigmoid(x[:,1])
        x[:,2] = torch.sigmoid(x[:,2])

        return x

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
            nn.ReLU()
        )
        self.linearlayers = nn.Sequential(
            nn.Linear(in_features=6400,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=3)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linearlayers(x)

        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.sigmoid(x[:,1])
        x[:,2] = torch.sigmoid(x[:,2])

        return x