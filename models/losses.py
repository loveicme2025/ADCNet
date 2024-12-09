import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def dice_loss(inputs, targets):
    smooth = 1.0

    # Flatten inputs and targets
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice    
def mixed_loss(input, target, weight=None, reduction='mean', ignore_index=255):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    bce_loss = F.cross_entropy(input=input, target=target, weight=weight,
                              ignore_index=ignore_index, reduction=reduction)
    dice = dice_loss(F.softmax(input, dim=1)[:, 1], (target == 1).float())
    # Combine BCE and Dice losses
    mixed_loss = 0.8*bce_loss +0.2*dice

    return mixed_loss