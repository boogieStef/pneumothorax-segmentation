import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Calculates Dice Loss for binary segmentation.
    Loss = 1 - DiceCoefficient
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    """
    Combines Segmentation Loss (Dice + BCE) and Classification Loss (BCE).
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.w_seg = config['train']['weight_segmentation']
        self.w_cls = config['train']['weight_classification']
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, seg_targets, cls_logits, cls_targets):
        # 1. Segmentation Loss
        loss_dice = self.dice_loss(seg_logits, seg_targets)
        loss_bce_seg = self.bce_loss(seg_logits, seg_targets)
        total_seg_loss = 0.5 * loss_dice + 0.5 * loss_bce_seg
        
        # 2. Classification Loss
        loss_cls = self.bce_loss(cls_logits, cls_targets)
        
        # 3. Weighted Sum
        total_loss = (self.w_seg * total_seg_loss) + (self.w_cls * loss_cls)
        
        return total_loss, total_seg_loss, loss_cls