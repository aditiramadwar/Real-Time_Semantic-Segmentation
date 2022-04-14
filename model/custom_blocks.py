import torch.nn as nn
import torch
def max_pool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    return pool

def conv_block_2(in_dim, out_dim, act_fn):

    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim)
    )

    return block

def conv_trans_block(in_dim, out_dim, act_fn):

    block = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )

    return block



