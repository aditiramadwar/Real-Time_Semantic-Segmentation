import torch
from dataset.CityscapesDataset import CityscapesDataset


def get_cityscapes_data(
    mode,
    split,
    root_dir='datasets/cityscapes',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type,transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return data_loaded