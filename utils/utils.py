import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
from dataset.CityscapesDataset import CityscapesDataset
from utils.utils import *
import matplotlib.pyplot as plt
import cv2

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

colormap = np.zeros((256, 3), dtype=np.uint8)
colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

def decode_segmap(temp):
    #convert gray scale to color
    # temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = colormap[l][0]
        g[temp == l] = colormap[l][1]
        b[temp == l] = colormap[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
def save_predictions(data, model, save_path):    
    model.eval()
    model.to(device)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data)):

            X, y, s, _ = batch # here 's' is the name of the file stored in the root directory
            X, y = X.to(device), y.to(device)
            predictions = model(X) 
            
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1) 
            pred_labels = pred_labels.float()
            # print("output : " ,s)


            # Resizing predicted images too original size
            pred_labels = transforms.Resize((1024, 2048))(pred_labels)             

            # Configure filename & location to save predictions as images
            s = str(s)
            pos = s.rfind('/', 0, len(s))
            name = s[pos+1:-18]  
            

            save_as_images(pred_labels, save_path, name)                



def save_as_images(pred, folder, image_name):
    # pred = np.transpose(pred.cpu().numpy(), (1,2,0))
    pred = pred.cpu().numpy()[0]
    seg_map = decode_segmap(pred)
    # cv2.imshow("e", seg_map)
    # cv2.waitKey(0)
    # plt.figure()
    # plt.imshow(seg_map)
    # plt.show()
    # exit(0)
    # tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}/{image_name}.png"
    # tensor_pred.save(filename)
    # print(filename)
    plt.imsave(filename, seg_map)
    # exit(0)

def predict(model, image):
    pass

def evaluate(data_loader, model, path, save_path):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'{path} has been loaded and initialized')
    save_predictions(data_loader, model, save_path)


def get_cityscapes_data(
    mode,
    split,
    root_dir='Cityscapes',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
    num_workers=2

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type,transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    return data_loaded