from pathlib import Path
from typing import Optional, Callable

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image


class ImageDataset(VisionDataset):
    def __init__(
        self, root: str, transforms: Optional[Callable] = ConvertImageDtype(torch.float)
    ):
        super(ImageDataset, self).__init__(root, transforms=transforms)
        self.dir_path = Path(root)
        self.image_paths = [
            p for p in self.dir_path.iterdir() if not p.name.startswith(".")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx].absolute().as_posix())
        tensor_image = self.transforms(img)
        return tensor_image
