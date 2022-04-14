from numpy import Inf
from tqdm import tqdm
from eval_test import mIOU

def train(model, device, train_loader, optimizer, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0

    for batch in tqdm(train_loader):
        image, labels, _ = batch
        image, labels = image.to(device), labels.to(device)

        prediction = model(image)
     
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
    for epoch in range(num_epochs):

        model.train()
        print("Starting Epoch "+str(epoch+1))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function)
        print('Epoch [{}/{}],Train Loss: {:.4f}, Train IOU: {:.4f}'.format(epoch+1, num_epochs, train_loss, running_mIOU))

        if val_loader != None:
            model.eval()
            val_loss = train(model, device, val_loader, optimizer, loss_function)
            print("Validation loss:", val_loss)

            # if val_loss <= loss_min:
            #     print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #     loss_min,
            #     val_loss))
            #     torch.save(model.state_dict(), 'model_cifar.pt')
            #     loss_min = val_loss
        
    # model.load_state_dict(torch.load('model_cifar.pt'))
    # print("Model Saved Successfully")
