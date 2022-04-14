
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from utils import utils
from model import unet
import train

LEARNING_RATE = 0.0005

def main():
    train_data = utils.get_cityscapes_data(mode='fine', split='train', batch_size = 64)
    print("Train data loaded successfully")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")

    current_model = unet.UNET(in_channels = 3, classes = 19).to(device)
    print("Model loaded")

    optimizer = optim.Adam(current_model.parameters(), lr = LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index = 255) 
    print("Optimizer and Loss defined")

    print("############### Start Training ################")
    train.train_model(1, current_model, device, train_data, optimizer, loss_function)

if __name__ == '__main__':
    main()