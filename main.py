import utils
import torch
import torch.nn as nn
import torch.optim as optim
from utils import utils
from model import unet
from model.UNet import UNet
import train
from torchvision import transforms
from PIL import Image
import multiprocessing

LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
def main():
    transform = transforms.Compose([transforms.Resize((128, 256), interpolation=Image.NEAREST),]) 
    train_data = utils.get_cityscapes_data(mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 8, transforms = transform)
    val_data = utils.get_cityscapes_data(mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 8, transforms = transform)
    print("Train data loaded successfully")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")

    current_model = UNet().to(device)
    print("Model loaded")

    optimizer = optim.Adam(current_model.parameters(), lr = LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index = 255) 
    print("Optimizer and Loss defined")

    print("############### Start Training ################")
    train.train_model(20, current_model, device, train_data, optimizer, loss_function)

if __name__ == '__main__':
    main()