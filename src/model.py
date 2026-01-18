import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ModelFactory:
    @staticmethod
    def create(config):
        arch_name = config['train']['architecture']
        encoder_name = config['train'].get('encoder_name', 'resnet34')
        pretrained = config['train']['pretrained']
        
        encoder_weights = "imagenet" if pretrained else None
        
        # --- ZMIANA KLUCZOWA ---
        # Wracamy do in_channels=1, tak jak w "dobrym kodzie".
        # RTG to grayscale. SMP automatycznie dostosuje wagi pierwszej warstwy.
        model_fn = getattr(smp, arch_name, None)
        model = model_fn(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,  # <--- TU BYŁO 3, ZMIENIAMY NA 1
            classes=1,
            activation=None
        )
        
        return model