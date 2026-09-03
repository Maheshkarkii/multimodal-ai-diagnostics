"""
Image preprocessing and augmentation pipelines for Vision models.
"""

from torchvision import transforms

# ImageNet normalization statistics used by pretrained PyTorch models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Construct training transformations pipeline with mild augmentations.

    Rationale:
    - Grayscale(num_output_channels=3): Expands 1-channel grayscale to 3-channel RGB representation
      matching MobileNetV2 pretrained expectations.
    - Resize: Scales image to model's input spatial resolution (e.g. 224x224).
    - RandomHorizontalFlip(p=0.5): Realistic data augmentation to improve generalization.
    - RandomRotation(degrees=10): Minor rotation robustness without destroying structural semantics.
    - ToTensor: Scales pixel values [0, 255] to float32 [0.0, 1.0] and permutes to (C, H, W).
    - Normalize: Standardizes channels to ImageNet distribution for optimal transfer learning.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Construct deterministic validation/test/inference transformations pipeline.

    Rationale:
    - Grayscale(num_output_channels=3): Replicates 3-channel input format.
    - Resize: Ensures deterministic spatial dimensions matching model input.
    - ToTensor: Standard tensor conversion.
    - Normalize: Identical ImageNet standardization as training pipeline.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
