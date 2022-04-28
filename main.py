import utils
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
import numpy as np
from model.UNet_backbone import Unet_backbone
from model.Unet_test import Unet
from model.UNet import UNet
import train
from torchvision import transforms
from PIL import Image
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
def main():
    train_transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    # transform = None
    print("train")
    train_data = get_cityscapes_data(mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 8, transforms = train_transform, shuffle=True)
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    val_data = get_cityscapes_data(mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 8, transforms = val_transform, shuffle=True)
    test_data = get_cityscapes_data(mode='fine', split='test', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform)
    print("Train data loaded successfully")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")
    current_model = Unet_backbone(backbone_name='resnet34')
    # current_model = UNet()
    print("Model loaded")

    optimizer = optim.Adam(current_model.parameters(), lr = LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index = 255) 
    print("Optimizer and Loss defined")

    # print("############### Start Training ################")
    # train.train_model(100, current_model, device, train_data, optimizer, loss_function, val_loader=val_data)
    # evaluate_(test_data, current_model, "finalUNet_res3450.pt", "test_results")
    rt_vid_path = "Cityscapes/leftImg8bit/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00"
    real_time_segmentation(current_model, device, "finalUNet_res3450.pt", rt_vid_path, transform = val_transform)
if __name__ == '__main__':
    main()