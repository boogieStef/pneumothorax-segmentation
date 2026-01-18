import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ModelFactory:
    """
    Factory class to instantiate segmentation models using 
    segmentation-models-pytorch (smp) library.
    """
    @staticmethod
    def create(config):
        """
        Creates and returns a model instance based on the config.
        
        Args:
            config (dict): Configuration dictionary.
            
        Returns:
            nn.Module: The requested PyTorch model.
        """
        arch_name = config['train']['architecture'] # np. "Unet", "UnetPlusPlus", "DeepLabV3Plus"
        encoder_name = config['train'].get('encoder_name', 'resnet34') # np. "resnet34", "efficientnet-b0"
        pretrained = config['train']['pretrained']
        
        # Binary segmentation = 1 class
        num_classes = 1 
        
        # Wybór wag (ImageNet)
        encoder_weights = "imagenet" if pretrained else None
        
        print(f"[INFO] Initializing SMP Model: Arch={arch_name}, Encoder={encoder_name}, Weights={encoder_weights}")
        
        # Dynamiczne tworzenie modelu z biblioteki SMP
        # Obsługuje: Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
        model_fn = getattr(smp, arch_name, None)
        
        if model_fn is None:
            raise ValueError(f"Architecture '{arch_name}' not found in segmentation-models-pytorch library.")
            
        model = model_fn(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None # Zwracamy surowe logity (Loss function ma w sobie sigmoid)
        )
        
        return model