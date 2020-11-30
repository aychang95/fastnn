import logging
from typing import Optional, List, Union, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import ConvertImageDtype

from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np


from fastnn.utils.cv import ImageDataset


from fastnn.processors.base_processing import Processor


logger = logging.getLogger(__name__)


class ObjectDetectionProcessor(Processor):
    """Object Detection processor dealing with image files or 3xHxW formatted images and boxes, scores, labels out processing.
    Since most resizing and padding transforms are done by the object detection models in PyTorch, datasets and dataloaders willl
    generate batches of images as lists.

    Usage:
    ```python
    >>> processor = ObjectDetectionProcessor()
    >>> processor.process(file_paths=["file_path.png"])

    **Parameters:**
    * **label_strings** - List of strings that specify label strings with index as key for this specific processor

    ```
    """

    def __init__(self, label_strings: List[str]):
        self.label_strings = label_strings

    def process(
        self,
        dir_path: str,
        transforms: Optional[Callable] = ConvertImageDtype(torch.float),
    ) -> Dataset:
        """Generate torch `Dataset` object from list of file paths or image Tensors.
        This provides clear tensor input representations for compatible models.

        Returns a Dataset

        * **dir_path** - String path to directory of images you'd like to process
        """

        dataset = ImageDataset(root=dir_path, transforms=transforms)

        return dataset

    def process_batch(
        self,
        dir_path: str,
        transforms: Optional[Callable] = ConvertImageDtype(torch.float),
        mini_batch_size: int = 8,
        use_gpu: bool = False,
    ) -> DataLoader:
        """Generate torch `Dataloader` object from data directory path.
        This provides clear tensor input representations for compatible models.

        Returns a `Dataloader`

        * **dir_path** - String path to directory of images you'd like to process
        * **mini_batch_size** - Batch size for inference
        * **use_gpu** - Bool for using gpu or cpu. If set True but no gpu devices available, model will default to using cpu
        """

        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                logger.info("GPU not available")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        dataset = self.process(dir_path=dir_path, transforms=transforms)

        # Instead of a tensor batch, the lambda collate_fn will provide a list batch
        dataloader = DataLoader(
            dataset,
            batch_size=mini_batch_size,
            collate_fn=lambda x: [[t.to(device) for t in self._od_collate_fn(x)]],
        )

        return dataloader

    def process_output(
        self,
    ):
        pass

    def process_output_batch(
        self, outputs: List[List[torch.Tensor]], dataset: Dataset
    ) -> List[List[Tuple[torch.Tensor, np.array]]]:
        """Process output of object detection model into human legible results.
        Outputs from `FasterRCNNModule`


        Returns batched results of list of list of tuples containing boxed images in tensor and numpy format

        * **outputs** - List of batch output tensors from a model's forward pass
        * **dataset** - Corresponding dataset with originial images matched with model outputs

        """
        # Labeled Images
        results = []

        for idx, out in enumerate(outputs):
            labeled_images = []
            for label_idx in range(1, len(out), 3):
                labels = [self.label_strings[o] for o in out[label_idx]]

                unique_labels = set(labels)
                label_colors_map = {}
                for label in unique_labels:
                    label_colors_map[label] = tuple(
                        np.random.choice(range(256), size=3)
                    )

                label_colors = [label_colors_map[label] for label in labels]

                output_tensor, output_numpy = self.draw_bounding_boxes(
                    ConvertImageDtype(torch.uint8)(
                        dataset[idx * (len(out) // 3) + label_idx // 3]
                    ),
                    out[label_idx - 1],
                    labels=labels,
                    colors=label_colors,
                )
                labeled_images.append((output_tensor, output_numpy))
            results.append(labeled_images)

        return results

    def _od_collate_fn(self, data):
        """Custom collate fn to output dynamic image batches without same-dim requirements via. `stack`.
        This is not technically a "correct" collate_fn for most of torch's vision models. Should be wrapped as a list
        in the lambda collate fn.
        """
        data = [img for img in data]
        return data

    @torch.no_grad()
    def draw_bounding_boxes(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
        width: int = 1,
        font: Optional[str] = "arial.ttf",
        font_size: int = 10,
    ) -> Tuple[torch.Tensor, np.array]:

        """
        Added and modified from TorchVision utils.
        Draws bounding boxes on given image.
        The values of the input image should be uint8 between 0 and 255.
        Args:
            image (Tensor): Tensor of shape (C x H x W)
            bboxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
                the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
                `0 <= ymin < ymax < H`.
            labels (List[str]): List containing the labels of bounding boxes.
            colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
                be represented as `str` or `Tuple[int, int, int]`.
            width (int): Width of bounding box.
            font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
                also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
                `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
            font_size (int): The requested font size in points.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Tensor expected, got {type(image)}")
        elif image.dtype != torch.uint8:
            raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        ndarr = image.permute(1, 2, 0).numpy()
        img_to_draw = Image.fromarray(ndarr)

        img_boxes = boxes.to(torch.int64).tolist()

        draw = ImageDraw.Draw(img_to_draw)

        pixel_ratio = max(1, (max(ndarr.shape[0], ndarr.shape[1]) // 1000))

        for i, bbox in enumerate(img_boxes):
            color = None if colors is None else colors[i]
            draw.rectangle(bbox, width=width * pixel_ratio, outline=color)

            if labels is not None:
                txt_font = (
                    ImageFont.load_default()
                    if font is None
                    else ImageFont.truetype(font=font, size=font_size * pixel_ratio)
                )
                draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1), np.array(
            img_to_draw
        )
