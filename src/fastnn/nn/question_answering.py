from typing import Tuple

import torch
from transformers import AutoModelForQuestionAnswering


class BertQAModule(torch.nn.Module):
    """Module for Bert-based Question Answering models from Transformers

    Usage:
    ```python
    >>> model = BertQAModule(model_name_or_path="distilbert-base-cased-distilled-squad")
    >>> output = model(*batch)
    ```

    **Parameters:**

    * **model_name_or_path** - String key of Transformer's pre-trained model hosted in Hugging Face's Model Repository
    """

    def __init__(
        self, model_name_or_path: str = "distilbert-base-cased-distilled-squad"
    ):
        super(BertQAModule, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path, torchscript=True
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, example_indices, *args
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on this QA module requires input_ids, attention_mask, and example_indices.
        The `example_indices` parameter is required specifically for span-based extractive question anwering model outputs
        in order to retain the structure of inputs within the final start and end logit calculations.

        * **input_ids** - Tensor generated from FastNN `Processor` class
        * **attention_mask** - Tensor generated from FastNN `Processor` class
        * **token_type_ids** - Tensor generated from FastNN `Processor` class
        * **example_indices** - Tensor generated from FastNN `Processor` class
        * ** &ast;args ** - args available to make abstraction easier for variant `Processor` classes
        """
        start_logits, end_logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return start_logits, end_logits, example_indices
