"""
Usage during model training: 
from custom_losses import HueL1Loss, HueL2Loss
loss_fn = HueL1Loss()
"""
import torch
import torch.nn as nn
import kornia.color as kc
import math

class HueL1Loss(nn.Module):
    def __init__(self):
        super(HueL1Loss, self).__init__()
    
    def forward(self, input, target):
        # Normalize if needed
        input = (input + 1) / 2
        target = (target + 1) / 2
        input = torch.clamp(input, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Convert the input and target images from RGB to Lab
        input_lab = kc.rgb_to_lab(input)
        target_lab = kc.rgb_to_lab(target)

        # Extract 'a' and 'b' channels
        a_i, b_i = input_lab[:, 1, :, :], input_lab[:, 2, :, :]
        a_t, b_t = target_lab[:, 1, :, :], target_lab[:, 2, :, :]

        # Compute hue (angle) for input and target in radians
        input_hue = torch.atan2(b_i, a_i + 1e-6)
        target_hue = torch.atan2(b_t, a_t + 1e-6)

        # Minimize angular distance properly
        diff = input_hue - target_hue
        hue_diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi

        return torch.mean(torch.abs(hue_diff))

class HueL2Loss(nn.Module):
    def __init__(self):
        super(HueL2Loss, self).__init__()
    
    def forward(self, input, target):
        # Normalize if needed
        input = (input + 1) / 2
        target = (target + 1) / 2
        input = torch.clamp(input, 0, 1)
        target = torch.clamp(target, 0, 1)

        # Convert the input and target images from RGB to Lab
        input_lab = kc.rgb_to_lab(input)
        target_lab = kc.rgb_to_lab(target)

        # Extract 'a' and 'b' channels
        a_i, b_i = input_lab[:, 1, :, :], input_lab[:, 2, :, :]
        a_t, b_t = target_lab[:, 1, :, :], target_lab[:, 2, :, :]

        # Compute hue (angle) for input and target in radians
        input_hue = torch.atan2(b_i, a_i + 1e-6)
        target_hue = torch.atan2(b_t, a_t + 1e-6)

        # Minimize angular distance properly
        diff = input_hue - target_hue
        hue_diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi

        return torch.mean(hue_diff ** 2)
