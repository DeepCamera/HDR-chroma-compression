import torch
import torch.nn as nn
from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab, lab2lch
import numpy as np

"""
Usage during model training: 
from custom_losses import HueL1Loss, HueL2Loss, HueL1Loss2, HueL2Loss2
loss_fn = HueL1Loss()

Last Modified 21 November 2023 by Xenios Milidonis, Copyright (c) 2023 CYENS Centre of Excellence
"""

class HueL1Loss(nn.Module):
    def __init__(self):
        super(HueL1Loss, self).__init__()
    
    def forward(self, input, target):
        # Convert the tensors to numpy arrays. Since tensor shape is [1,c,w,h], take 'first' dim to convert to [w,h,c]
        input = input[0].detach().cpu().float().numpy()
        target = target[0].detach().cpu().float().numpy()

        # Transpose arrays from [c,w,h] to [w,h,c]
        input = np.transpose(input, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        # Normalise the images to [0,1] as required for conversion to HSV. Don't stretch within [0,1] as this is affected by outliers.
        # Pix2pix tensors are within [-1,1] so add 1 and divide by 2.
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Convert the input and target images from RGB to HSV
        input_hsv = rgb_to_hsv(input)
        target_hsv = rgb_to_hsv(target)

        # Extract the hue channel from the HSV images
        input_hue = input_hsv[:, :, 0]
        target_hue = target_hsv[:, :, 0]

        # Calculate the L1 difference of the hue channels
        hue_diff = np.absolute(input_hue - target_hue)

        # Return the mean hue L1 loss
        #return np.mean(hue_diff)
        hue_diff = torch.from_numpy(hue_diff)
        hue_diff = hue_diff.requires_grad_(True)
        return torch.mean(hue_diff.float())

class HueL2Loss(nn.Module):
    def __init__(self):
        super(HueL2Loss, self).__init__()
    
    def forward(self, input, target):
        # Convert the tensors to numpy arrays. Since tensor shape is [1,c,w,h], take 'first' dim to convert to [w,h,c]
        input = input[0].detach().cpu().float().numpy()
        target = target[0].detach().cpu().float().numpy()

        # Transpose arrays from [c,w,h] to [w,h,c]
        input = np.transpose(input, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        # Normalise the images to [0,1] as required for conversion to HSV. Don't stretch within [0,1] as this is affected by outliers.
        # Pix2pix tensors are within [-1,1] so add 1 and divide by 2.
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Convert the input and target images from RGB to HSV
        input_hsv = rgb_to_hsv(input)
        target_hsv = rgb_to_hsv(target)

        # Extract the hue channel from the HSV images
        input_hue = input_hsv[:, :, 0]
        target_hue = target_hsv[:, :, 0]

        # Calculate the L2 difference of the hue channels
        hue_diff = np.power(input_hue - target_hue, 2)

        # Return the mean hue L2 loss
        #return np.mean(hue_diff)
        hue_diff = torch.from_numpy(hue_diff)
        hue_diff = hue_diff.requires_grad_(True)
        return torch.mean(hue_diff.float())

class HueL1Loss2(nn.Module):
    def __init__(self):
        super(HueL1Loss2, self).__init__()
    
    def forward(self, input, target):
        # Convert the tensors to numpy arrays. Since tensor shape is [1,c,w,h], take 'first' dim to convert to [w,h,c]
        input = input[0].detach().cpu().float().numpy()
        target = target[0].detach().cpu().float().numpy()

        # Transpose arrays from [c,w,h] to [w,h,c]
        input = np.transpose(input, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        # Normalise the images to [0,1] as required for conversion to HSV. Don't stretch within [0,1] as this is affected by outliers.
        # Pix2pix tensors are within [-1,1] so add 1 and divide by 2.
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Convert the input and target images from RGB to LCH
        input_lab = rgb2lab(input)
        target_lab = rgb2lab(target)
        input_lch = lab2lch(input_lab)
        target_lch = lab2lch(target_lab)

        # Extract the hue channel from the LCH images
        input_hue = input_lch[:, :, 2]
        target_hue = target_lch[:, :, 2]

        # Calculate the L1 difference of the hue channels
        hue_diff = np.absolute(input_hue - target_hue)

        # Return the mean hue L1 loss
        #return np.mean(hue_diff)
        hue_diff = torch.from_numpy(hue_diff)
        hue_diff = hue_diff.requires_grad_(True)
        return torch.mean(hue_diff.float())

class HueL2Loss2(nn.Module):
    def __init__(self):
        super(HueL2Loss2, self).__init__()
    
    def forward(self, input, target):
        # Convert the tensors to numpy arrays. Since tensor shape is [1,c,w,h], take 'first' dim to convert to [w,h,c]
        input = input[0].detach().cpu().float().numpy()
        target = target[0].detach().cpu().float().numpy()

        # Transpose arrays from [c,w,h] to [w,h,c]
        input = np.transpose(input, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        # Normalise the images to [0,1] as required for conversion to HSV. Don't stretch within [0,1] as this is affected by outliers.
        # Pix2pix tensors are within [-1,1] so add 1 and divide by 2.
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Convert the input and target images from RGB to LCH
        input_lab = rgb2lab(input)
        target_lab = rgb2lab(target)
        input_lch = lab2lch(input_lab)
        target_lch = lab2lch(target_lab)

        # Extract the hue channel from the LCH images
        input_hue = input_lch[:, :, 2]
        target_hue = target_lch[:, :, 2]

        # Calculate the L2 difference of the hue channels
        hue_diff = np.power(input_hue - target_hue, 2)

        # Return the mean hue L2 loss
        #return np.mean(hue_diff)
        hue_diff = torch.from_numpy(hue_diff)
        hue_diff = hue_diff.requires_grad_(True)
        return torch.mean(hue_diff.float())
