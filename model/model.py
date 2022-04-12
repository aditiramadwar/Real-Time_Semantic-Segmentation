import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 

class UNET(nn.Module):
    
    def __init__(self, in_channels = 3, classes = 1):
        super(UNET, self).__init__()
        
        self.in_channels = in_channels
        self.classes = classes
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def sampled(self, in_channels):

        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        self.c1 = self.__double_conv(self.layers[0], self.layers[1])
        self.p1 = self.max_pool_2x2(self.c1)

        self.c2 = self.__double_conv(self.layers[1], self.layers[2])
        self.p2 = self.max_pool_2x2(self.c2)

        self.c3 = self.__double_conv(self.layers[2],self.layers[3])
        self.p3 = self.max_pool_2x2(self.c3)

        self.c4 = self.__double_conv(self.layers[3],self.layers[4])
        self.p4 = self.max_pool_2x2(self.c4)

        self.c5 = self.__double_conv(self.layers[4],self.layers[5])

        self.u6 = nn.ConvTranspose2d(self.layers[5],self.layers[4], kernel_size=2, stride=2)
        self.u6 = torch.cat(self.u6, self.c4)
        self.c6 = self.__double_conv(self.u6, self.layers[4])

        self.u7 = nn.ConvTranspose2d(self.layers[4],self.layers[3], kernel_size=2, stride=2)
        self.u7 = torch.cat(self.u7, self.c3)
        self.c7 = self.__double_conv(self.u7, self.layers[3])

        self.u8 = nn.ConvTranspose2d(self.layers[3],self.layers[2], kernel_size=2, stride=2)
        self.u8 = torch.cat(self.u8, self.c2)
        self.c8 = self.__double_conv(self.u8, self.layers[2])

        self.u9 = nn.ConvTranspose2d(self.layers[2],self.layers[1], kernel_size=2, stride=2)
        self.u9 = torch.cat(self.u9, self.c1)
        self.c9 = self.__double_conv(self.u9, self.layers[1])

        self.final_out = self.final_conv(self.c9)

        return self.final_out

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return conv
    
    def forward(self, x):
        x = self.sampled(x)
        return x 