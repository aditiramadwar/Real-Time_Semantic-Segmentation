import numpy as np
import torch
import torch.nn.functional as F

def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1) 
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()
    pred = pred.view(-1)
    label = label.view(-1)

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


# def mIOU(label, pred, num_classes=19):
#     # pred = F.softmax(pred, dim=1)              
#     # pred = np.argmax(pred, axis=1).squeeze(1)
#     iou_list = list()
#     present_iou_list = list()

#     # pred = pred.view(-1)
#     # label = label.view(-1)
#     pred = np.reshape(pred,-1)
#     label = np.reshape(label,-1)
#     # Note: Following for loop goes from 0 to (num_classes-1)
#     # and ignore_index is num_classes, thus ignore_index is
#     # not considered in computation of IoU.
#     for sem_class in range(num_classes):
#         pred_inds = (pred == sem_class)
#         target_inds = (label == sem_class)
#         # print(target_inds)
#         if np.sum(target_inds) == 0:
#             iou_now = float('nan')
#         else: 
#             intersection_now = np.sum(pred_inds[target_inds])
#             union_now =np.sum(pred_inds) + np.sum(target_inds) - intersection_now
#             iou_now = float(intersection_now) / float(union_now)
#             present_iou_list.append(iou_now)
#         iou_list.append(iou_now)
#     return np.mean(present_iou_list)

# pred = Image.open("/home/dev_root/Desktop/Workspace/Robot_Learning/Real-Time_Semantic-Segmentation/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png")
# gt_truth = Image.open("/home/dev_root/Desktop/Workspace/Robot_Learning/Real-Time_Semantic-Segmentation/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png")
# pred = transforms.ToTensor()(pred)
# gt_truth = transforms.ToTensor()(gt_truth)
# print(mIOU(gt_truth, pred))