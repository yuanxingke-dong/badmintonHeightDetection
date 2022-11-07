import torch
import matplotlib.pyplot as plt
from typing import Union
from PIL import ImageDraw, Image
from torch import Tensor
from torchvision import transforms


def draw_image(image: Union[Tensor, Image.Image], boxes=None, labels=None, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    if isinstance(image, Tensor):  # 为tensor,则需要转换为Image格式
        if mean is not None and std is not None:
            for i in range(len(mean)):
                mean[i] = -mean[i] / std[i]
                std[i] = 1 / std[i]
        normalize = transforms.Compose([transforms.Normalize(mean, std), transforms.ToPILImage()])
        image = normalize(image)

    if isinstance(boxes, Tensor):
        boxes = boxes.tolist()
    if isinstance(labels, Tensor):
        labels = labels.tolist()
    draw = ImageDraw.Draw(image)
    if boxes is not None and labels is not None:
        for i, target in enumerate(boxes):
            draw.rectangle(target)
            draw.text((target[0], target[1]), str(labels[i]))

    plt.imshow(image)
    plt.show()


def draw_heatmap(image: Union[Tensor, Image.Image], heatmap: Tensor, is_overlap: bool = True, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    if isinstance(image, Tensor):  # 为tensor,则需要转换为Image格式
        if mean is not None and std is not None:
            for i in range(len(mean)):
                mean[i] = -mean[i] / std[i]
                std[i] = 1 / std[i]
        normalize = transforms.Compose([transforms.Normalize(mean, std), transforms.ToPILImage()])
        image = normalize(image)

    assert isinstance(heatmap, Tensor)
    assert len(heatmap.shape) == 2
    H = torch.zeros((3, heatmap.shape[0], heatmap.shape[1]))
    H[0] = heatmap
    H[1] = heatmap
    H[2] = 1 - heatmap
    H = transforms.ToPILImage()(H)

    image = image.convert('RGBA')
    heatmap = H
    # 大小必须相同
    heatmap = heatmap.resize((image.width, image.height), Image.BILINEAR)
    image = Image.blend(image, heatmap, alpha=0.5 if is_overlap else 1)

    draw_image(image)


def draw_heatmap_without_image(heatmap: Tensor, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    assert isinstance(heatmap, Tensor)
    assert len(heatmap.shape) == 2
    H = torch.zeros((3, heatmap.shape[0], heatmap.shape[1]))
    H[0] = heatmap
    H[1] = heatmap
    H[2] = heatmap
    H = transforms.ToPILImage()(H)

    heatmap = H
    # 大小必须相同

    draw_image(heatmap)