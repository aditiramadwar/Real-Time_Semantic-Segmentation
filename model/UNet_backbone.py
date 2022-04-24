import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

def get_backbone(name="resnet34", pretrained=True):

    backbone = models.resnet34(pretrained=pretrained)

    assert name == "resnet34", "Only resnet34 Implemented"
    feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
    backbone_output = 'layer4'

    return backbone, feature_names, backbone_output

class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x

class Unet_backbone(nn.Module):

    def __init__(self, backbone_name='resnet34', pretrained=True, encoder_freeze=False, classes=19, decoder_filters=(256, 128, 64, 32, 16), parametric_upsampling=True):
        super(Unet_backbone, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.skip_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        # print(shortcut_chs, bb_out_chs)
        # exit(0)

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.skip_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.skip_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=True))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

    def freeze_encoder(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_backbone(self, x):

        features = {None: None} if None in self.skip_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.skip_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):


        x = torch.zeros(1, 3, 128, 256)
        channels = [0]  
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.skip_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels
    
    def forward_backbone(self, x):

        features = {None: None} if None in self.skip_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.skip_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features
    
    def forward(self, input):

        x, features = self.forward_backbone(input)

        for skip_name, upsample_block in zip(self.skip_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x


if __name__ == "__main__":

    # simple test run
    net = Unet_backbone(backbone_name='resnet34')

    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)





    for _ in range(1):
        with torch.set_grad_enabled(True):
            batch = torch.empty(1, 3, 224, 224).normal_()
            targets = torch.empty(1, 19, 224, 224).normal_()
            start.record()
            out = net(batch)
            print(out.shape, targets.shape)
            # exit(0)
            end.record()
            # torch.cuda.synchronize()
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)

    print(start.elapsed_time(end))