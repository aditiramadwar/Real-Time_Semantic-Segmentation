from numpy import Inf
import torch
def train(model, device, train_loader, optimizer, loss_function):
    running_loss = 0.0
    for batch in (train_loader):
        image, labels, _ = batch
        image, labels = image.to(device), labels.to(device)
        prediction = model(image)

        optimizer.zero_grad()
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*image.size(0)

    # calculate average loss
    running_loss = running_loss/len(train_loader.dataset)
        
    # print("Accuracy: {:.2f}".format(calculate_accuracy(image, running_loss)))
    return running_loss


def train_model(epoch, model, device, train_loader, optimizer, loss_function, val_loader = None):
    model.train()
    print("Start training...")
    train_loss_min = Inf
    for e in range(epoch):
        print("Epoch:", e)
        train_loss = train(model, device, train_loader, optimizer, loss_function)
        print("Train loss:", train_loss)

        if val_loader != None:
            val_loss = train(model, device, val_loader, optimizer, loss_function)
            print("Validation loss:", val_loss)

        if train_loss <= train_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            train_loss_min = train_loss
        
    model.load_state_dict(torch.load('model_cifar.pt'))
    print("Model Saved Successfully") 
