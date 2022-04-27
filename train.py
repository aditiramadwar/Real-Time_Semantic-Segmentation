from numpy import Inf
from tqdm import tqdm
from eval_test import mIOU
import torch

from torch.utils.tensorboard import SummaryWriter
# conda install pytorch torchvision -c pytorch

def train(model, device, train_loader, optimizer, loss_function, val_loader):
    running_loss = 0.0
    running_mIOU = 0.0

    for batch in tqdm(train_loader):
        image, labels, _, _ = batch
        image, labels = image.to(device), labels.to(device)

        prediction = model(image)
        # print(prediction.shape)
        # print(labels.shape)
        # exit(0)
        optimizer.zero_grad()
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*image.size(0)
        # print(prediction.shape)
        # exit(0)
        running_mIOU += mIOU(labels, prediction)

    # calculate average loss
    running_loss = running_loss/len(train_loader)
    running_mIOU = running_mIOU/len(train_loader)
        
    # print("Accuracy: {:.2f}".format(calculate_accuracy(image, running_loss)))
    return running_loss, running_mIOU


def train_model(num_epochs, model, device, train_loader, optimizer, loss_function, val_loader = None):
    print("Start training...")
    loss_min = Inf

    writer = SummaryWriter()
    for epoch in range(num_epochs):

        model.train()
        print("Starting Epoch " + str(epoch+1))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function, val_loader)
        val_loss, val_mIOU = evaluate(model, val_loader, device, loss_function)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch+1, num_epochs, train_loss, running_mIOU, val_loss, val_mIOU))
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Validation Loss", val_loss, epoch)

def evaluate(model, data_loader, device, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0
    with torch.no_grad():
        model.eval()
        for image, labels, _, _ in data_loader:
            image, labels = image.to(device), labels.to(device)
            prediction = model(image)
            loss = loss_function(prediction, labels)
            running_loss += loss.item()*image.size(0)
            running_mIOU += mIOU(labels, prediction)
        
        running_loss = running_loss/len(data_loader)
        running_mIOU = running_mIOU/len(data_loader)

    return running_loss, running_mIOU
    