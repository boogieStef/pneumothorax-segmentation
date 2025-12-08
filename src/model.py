import torch
import torch.nn as nn
import timm
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class ModelFactory:
    """
    Factory class to instantiate segmentation models based on configuration.
    Supports:
    1. 'unet_resnet34': Custom U-Net with ResNet34 backbone (timm).
    2. 'deeplabv3': DeepLabV3 with ResNet50 backbone (torchvision).
    """
    @staticmethod
    def create(config):
        """
        Creates and returns a model instance based on the config.
        
        Args:
            config (dict): Configuration dictionary containing 'train' section.
            
        Returns:
            nn.Module: The requested PyTorch model.
        """
        arch_name = config['train']['architecture']
        pretrained = config['train']['pretrained']
        
        # We output 1 class (binary mask)
        num_classes = 1 
        
        print(f"[INFO] Initializing model architecture: {arch_name}")
        
        if arch_name == "unet_resnet34":
            return SimpleUNet(
                encoder_name="resnet34", 
                pretrained=pretrained, 
                classes=num_classes
            )
        
        elif arch_name == "deeplabv3":
            return DeepLabV3Wrapper(
                pretrained=pretrained, 
                classes=num_classes
            )
        
        else:
            raise ValueError(f"Architecture '{arch_name}' is not implemented in ModelFactory.")


class SimpleUNet(nn.Module):
    """
    Standard U-Net architecture for segmentation.
    Encoder: Pre-trained backbone from 'timm'.
    Decoder: Custom upsampling blocks with skip connections.
    """
    def __init__(self, encoder_name='resnet34', pretrained=True, in_channels=3, classes=1):
        super().__init__()
        
        # 1. Encoder (Backbone)
        # We use 'timm' to get feature maps from different stages
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        
        # Get channel counts (e.g. [64, 64, 128, 256, 512] for resnet34)
        encoder_channels = self.encoder.feature_info.channels()
        
        # 2. Decoder (Segmentation Head)
        # We build decoder blocks from bottom (deepest) to top
        self.decoder1 = self._decoder_block(encoder_channels[-1], encoder_channels[-2])
        self.decoder2 = self._decoder_block(encoder_channels[-2], encoder_channels[-3])
        self.decoder3 = self._decoder_block(encoder_channels[-3], encoder_channels[-4])
        self.decoder4 = self._decoder_block(encoder_channels[-4], encoder_channels[-5])
        
        # Final projection to class mask
        self.final_conv = nn.Conv2d(encoder_channels[-5], classes, kernel_size=1)
        
        # Upsampling to match input resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def _decoder_block(self, in_channels, out_channels):
        """
        Standard decoder block: Upsample -> Conv -> ReLU -> Conv -> ReLU.
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
        # e0..e4 correspond to features at different scales
        e0, e1, e2, e3, e4 = features
        
        # Decoder Pass with Skip Connections
        d1 = self.decoder1(e4) + e3
        d2 = self.decoder2(d1) + e2
        d3 = self.decoder3(d2) + e1
        d4 = self.decoder4(d3) + e0
        
        # Final Output
        masks = self.final_upsample(d4)
        logits = self.final_conv(masks)
        
        return logits


class DeepLabV3Wrapper(nn.Module):
    """
    Wrapper for torchvision's DeepLabV3 model.
    Adapts the output layer to binary segmentation (1 class).
    """
    def __init__(self, pretrained=True, classes=1):
        super().__init__()
        
        # Use default weights if pretrained is requested
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        
        # Load base model
        self.model = deeplabv3_resnet50(weights=weights)
        
        # Modify the Classifier Head
        # Original has 21 classes (COCO), we need 1.
        # DeepLabV3 head structure: classifier -> (0: DeepLabHead -> (4: Conv2d))
        self.model.classifier[4] = nn.Conv2d(256, classes, kernel_size=1)
        
        # Modify the Auxiliary Classifier Head (used for training stability)
        self.model.aux_classifier[4] = nn.Conv2d(256, classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        Returns only the main output tensor, ignoring auxiliary output during inference.
        """
        # torchvision segmentation models return an OrderedDict
        # keys: 'out' (main prediction), 'aux' (auxiliary prediction)
        output = self.model(x)
        return output['out']