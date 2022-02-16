from typing import Tuple

import torch
from transformers import AutoModelForTokenClassification


class NERModule(torch.nn.Module):
    """Module for Token Classification models from Transformers

    Usage:
    ```python
    >>> model = NERModule(model_name_or_path="dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> output = model(*batch)
    ```

    **Parameters:**

    * **model_name_or_path** - String key of Transformer's pre-trained model hosted in Hugging Face's Model Repository
    """

    def __init__(
        self,
        model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
    ):
        super(NERModule, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, torchscript=True
        )

    def forward(
        self, input_ids, attention_mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on this NER module requires input_ids and attention_mask.

        * **input_ids** - Tensor generated from FastNN `Processor` class
        * **attention_mask** - Tensor generated from FastNN `Processor` class
        * ** &ast;args ** - args available to make abstraction easier for variant `Processor` classes
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits[0], input_ids
