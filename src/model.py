import torch
import torch.nn as nn
import timm

class MultiTaskUNet(nn.Module):
    """
    U-Net architecture with two heads:
    1. Segmentation Head (Decoder) -> Mask
    2. Classification Head (Bottleneck) -> Probability
    Uses 'timm' for the encoder backbone.
    """
    def __init__(self, encoder_name='resnet34', pretrained=True, in_channels=3, classes=1):
        super().__init__()
        
        # 1. Encoder (Backbone)
        # features_only=True returns a list of feature maps from different stages
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        
        # Get channel counts for skip connections
        # Example ResNet34: [64, 64, 128, 256, 512]
        encoder_channels = self.encoder.feature_info.channels()
        
        # 2. Classification Head
        # Attached to the deepest feature map (bottleneck)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(encoder_channels[-1], 1)
        )
        
        # 3. Segmentation Head (Decoder)
        # Building decoder blocks from bottom to top
        self.decoder1 = self._decoder_block(encoder_channels[-1], encoder_channels[-2])
        self.decoder2 = self._decoder_block(encoder_channels[-2], encoder_channels[-3])
        self.decoder3 = self._decoder_block(encoder_channels[-3], encoder_channels[-4])
        self.decoder4 = self._decoder_block(encoder_channels[-4], encoder_channels[-5])
        
        self.final_conv = nn.Conv2d(encoder_channels[-5], classes, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _decoder_block(self, in_channels, out_channels):
        """
        Creates a standard decoder block: Upsample -> Conv -> ReLU -> Conv -> ReLU.
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Pass
        features = self.encoder(x)
        
        # Features for skips (assuming 5 levels)
        # e0=64, e1=64, e2=128, e3=256, e4=512 (Bottleneck)
        e0, e1, e2, e3, e4 = features
        
        # --- Classification Path ---
        cls_logits = self.cls_head(e4)
        
        # --- Segmentation Path ---
        d1 = self.decoder1(e4) + e3
        d2 = self.decoder2(d1) + e2
        d3 = self.decoder3(d2) + e1
        d4 = self.decoder4(d3) + e0
        
        masks = self.final_upsample(d4)
        seg_logits = self.final_conv(masks)
        
        return seg_logits, cls_logits