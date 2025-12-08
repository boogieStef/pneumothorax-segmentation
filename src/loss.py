import torch
import torch.nn as nn

class SegmentationLoss(nn.Module):
    """
    Loss function for binary segmentation.
    Combines Dice Loss and BCEWithLogitsLoss.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. BCE Loss (Pixel-wise accuracy)
        bce_loss = self.bce(logits, targets)
        
        # 2. Dice Loss (Overlap quality)
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        # Dice Coeff = (2 * Intersection) / (Sum of elements)
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Dice Loss = 1 - Dice Coeff
        dice_loss = 1.0 - dice_score
        
        # Combine: 50% BCE + 50% Dice Loss
        return 0.5 * bce_loss + 0.5 * dice_loss