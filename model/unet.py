import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 

class UNET(nn.Module):
    
    def __init__(self, in_channels = 3, classes = 1):
        super(UNET, self).__init__()
        
        self.in_channels = in_channels
        self.classes = classes

        self.layers = [self.in_channels, 64, 128, 256, 512, 1024]

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def samples(self, input, from_dim, to_dim):
        self.down_conv = self.__double_conv(from_dim, to_dim)
        return self.down_conv(input)

    def conv_trans(self, input, from_dim, to_dim):
        self.up_conv = nn.ConvTranspose2d(from_dim, to_dim, kernel_size=2, stride=2)
        return self.up_conv(input)

    def down_sampled(self,x):
        print("Layer 1 Input:", x.shape)
        self.c1 = self.samples(x, self.layers[0], self.layers[1])
        self.p1 = self.max_pool_2x2(self.c1)

        print("Layer 2 input:", self.p1.shape)
        self.c2 = self.samples(self.p1, self.layers[1], self.layers[2])
        self.p2 = self.max_pool_2x2(self.c2)

        print("Layer 3 input:", self.p2.shape)
        self.c3 = self.samples(self.p2, self.layers[2], self.layers[3])
        self.p3 = self.max_pool_2x2(self.c3)

        print("Layer 4 input:", self.p3.shape)
        self.c4 = self.samples(self.p3, self.layers[3], self.layers[4])
        self.p4 = self.max_pool_2x2(self.c4)

        print("Layer 5 input:", self.p4.shape)
        self.c5 = self.samples(self.p4, self.layers[4], self.layers[5])

    def up_sampled(self):
        print("Layer 6 input", self.c5.shape)
        self.u6 = self.conv_trans(self.c5, self.layers[5], self.layers[4])
        self.u6 = torch.cat((self.u6, self.c4), dim = 1)
        self.c6 = self.samples(self.u6, self.layers[5], self.layers[4])

        print("Layer 7 input", self.c6.shape)
        self.u7 = self.conv_trans(self.c6, self.layers[4], self.layers[3])
        self.u7 = torch.cat((self.u7, self.c3), dim = 1)
        self.c7 = self.samples(self.u7, self.layers[4], self.layers[3])

        print("Layer 8 input", self.c7.shape)
        self.u8 = self.conv_trans(self.c7, self.layers[3], self.layers[2])
        self.u8 = torch.cat((self.u8, self.c2), dim = 1)
        self.c8 = self.samples(self.u8, self.layers[3], self.layers[2])

        print("Layer 9 input", self.c8.shape)
        self.u9 = self.conv_trans(self.c8, self.layers[2], self.layers[1])
        self.u9 = torch.cat((self.u9, self.c1), dim = 1)
        self.c9 = self.samples(self.u9, self.layers[2], self.layers[1])

    def sampled(self, x):
        
        self.down_sampled(x)
        self.up_sampled()

        print("Layer 10 input", self.c9.shape)
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