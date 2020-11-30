from typing import Tuple, List, Union

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNNModule(torch.nn.Module):
    """Module for Faster R-CNN Model with ResNet-50 backbone pre-trained on Coco dataset from PyTorch

    Usage:
    ```python
    >>> model = FasterRCNNModule()
    >>> output = model()
    ```

    **Parameters:**

    """

    def __init__(self):
        super(FasterRCNNModule, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(
        self, images: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        """Forward pass on this Module only requires a list of dynamic images of shape [C, H, W].
        To work with a Triton Server which does not encode Python list objects, we enable a
        `torch.Tensor` input that we can trace a model compatible with Triton inference server.

        Both inputs are traceable with this model.

        If input is list of `torch.Tensor` then the output will be a List of torch Tensors moddable by
        3 where every 3 entries corresponds to one image.


        * ** images ** - List of tensor images of shape [C, H, W]
        """
        if isinstance(images, torch.Tensor):
            predictions = self.model([images])
            return (
                predictions[0]["boxes"],
                predictions[0]["labels"],
                predictions[0]["scores"],
            )
        else:
            predictions = self.model(images)
            return self._dict2tensor(predictions)

    def _dict2tensor(self, predictions):
        prediction_outputs = []
        for p in predictions:
            if "masks" in p.keys():
                prediction_outputs += [p["boxes"], p["labels"], p["scores"], p["masks"]]
            prediction_outputs += [p["boxes"], p["labels"], p["scores"]]
        return prediction_outputs
