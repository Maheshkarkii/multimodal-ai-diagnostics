"""
Domain-appropriate image preprocessing and data augmentation pipelines for industrial fault diagnostics.
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_industrial_train_transforms(
    image_size: int = 224,
    horizontal_flip: bool = True,
    rotation_degrees: float = 15.0,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
) -> transforms.Compose:
    """
    Construct training transformations pipeline tailored for industrial component imagery.
    """
    transform_list = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation_degrees > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter_brightness > 0 or color_jitter_contrast > 0:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
            )
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(transform_list)


def get_industrial_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Construct deterministic validation, test, and production inference transformations.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# Aliases for backward compatibility with baseline test suites
get_train_transforms = get_industrial_train_transforms
get_eval_transforms = get_industrial_eval_transforms
