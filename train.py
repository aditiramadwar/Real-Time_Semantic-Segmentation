from numpy import Inf
from tqdm import tqdm
from eval_test import mIOU
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

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
    
    writer = SummaryWriter("runs/UNet_res34")
    images, labels, _, _ = next(iter(train_loader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    model = model.to(device)
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print("Starting Epoch "+str(epoch+1))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function, val_loader)
        val_loss, val_mIOU = evaluate(model, val_loader, device, loss_function)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch+1, num_epochs, train_loss, running_mIOU, val_loss, val_mIOU))
        if epoch%10 == 0:
            save_checkpoint("final"+"UNet_res34"+str(epoch)+".pt", model, optimizer, val_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("mIOU/Train", running_mIOU, epoch)
        writer.add_scalar("mIOU/Validation", val_mIOU, epoch)
        
    # model.load_state_dict(torch.load('model_cifar.pt'))
    # print("Model Saved Successfully")



def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

  
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
    