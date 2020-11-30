import logging
from typing import Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """Exporter Class for turning torch models into torch script models. Requires a torch Module and corresponding `DataLoader` populated with complete data samples.
    This means the `DataLoader` has atleast one full batch sample that can be used as input to the torch Module in order to run `torch.jit.trace()`

    Usage:
    ```python
    >>> exporter = TorchScriptExporter(model=BertQAModule(...), dataloader=dataloader, use_gpu=True)
    >>> exporter.export()
    ```

    **Parameters:**

    * **model** - Torch model/module, preferably from `fastnn.nn`
    * **dataloader** - Torch `DataLoader` that corresponds to the model
    * **use_gpu** - Bool for using gpu or cpu. If set True but no gpu devices available, model will default to using cpu

    """

    def __init__(
        self, model: torch.nn.Module, dataloader: DataLoader, use_gpu: bool = False
    ):
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Torch model set to device {device}")
            else:
                device = torch.device("cpu")
                logger.info(f"GPU not available...device set to {device}")
        else:
            device = torch.device("cpu")
            logger.info(f"Torch model set to device {device}")

        self.model = model
        self.model.eval()
        self.model.to(device)
        self.dataloader = dataloader
        self.torchscript_model = None

    def export(self) -> torch.jit.ScriptModule:
        """Traces pytorch model and returns `ScriptModule` model"""
        batch_input = next(iter(self.dataloader))
        self.torchscript_model = torch.jit.trace(self.model, tuple(batch_input))
        return self.torchscript_model

    def serialize(self, file_path: Union[Path, str]):
        """Serialize and save model

        * **file_path** - String file path to save serialized torchscript model
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if self.torchscript_model:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.torchscript_model.save(str(file_path))
