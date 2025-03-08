import clip
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from open_clip import create_model_from_pretrained
from torchvision import transforms as tsfm

def get_model_and_transform(model_type, variant, device):
    """Load model and transform based on model type and variant."""
    embedding_dims = {
        'clip': {'B32': 512, 'B16': 512, 'RN50x4': 640},
        'medclip': {'vit': 512, 'rn': 512},
        'biomedclip': {'vit': 512}
    }

    if model_type == 'clip':
        model_name = {'B32': 'ViT-B/32', 'B16': 'ViT-B/16', 'RN50x4': 'RN50x4'}.get(variant, 'RN50x4')
        model, val_transform = clip.load(model_name, device=device)
        model = model.visual
        embedding_dim = embedding_dims['clip'][variant]
    elif model_type == 'medclip':
        if variant == 'vit':
            model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        else:  # 'rn'
            model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        model.from_pretrained()
        model = model.vision_model
        val_transform = tsfm.Compose([
            tsfm.Resize(224),
            tsfm.CenterCrop(224),
            tsfm.Lambda(lambda x: x.convert("RGB")),
            tsfm.ToTensor(),
            tsfm.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304]),
        ])
        embedding_dim = embedding_dims['medclip'][variant]
    elif model_type == 'biomedclip':
        model, val_transform = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model = model.visual
        embedding_dim = embedding_dims['biomedclip']['vit']
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.float()
    return model, val_transform, embedding_dim