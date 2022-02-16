import logging
from typing import List, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataloader import default_collate

from transformers import (
    AutoTokenizer,
)

from transformers.data.processors.squad import SquadFeatures, SquadResult

from fastnn.utils.qa.squad_metrics import (
    compute_predictions_logits,
)

from fastnn.processors.base_processing import Processor


logger = logging.getLogger(__name__)


class TransformersTokenTaggingProcessor(Processor):
    """Token Tagging Data Processor. Use this class to generate tensor inputs from human legible text/string data.
    This class can be used with a majority of the Bert architecture transformer models with a token-level predictive head
    for token classification from Hugging Face.

    Usage:
    ```python
    >>> processor = TransformersTokenTaggingProcessor(model_name_or_path="dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> processor.process(text=["string"])

    **Parameters:**

    * **model_name_or_path** - String defining HF token tagging model/tokenizer's name
    * **label_strings** - List of strings that specify label strings with index as key for this specific processor
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        label_strings: List[str] = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",  # Beginning of a person's name right after another person's name
            "I-PER",  # Person's name
            "B-ORG",  # Beginning of an organisation right after another organisation
            "I-ORG",  # Organisation
            "B-LOC",  # Beginning of a location right after another location
            "I-LOC",  # Location
        ],
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.label_strings = label_strings

    def process(
        self,
        text: List[str],
        max_seq_length: int = 512,
    ) -> Tuple:
        """Generate torch `Dataset` object from query/context string pairs using specified tokenizer from HF.
        This provides clear tensor input representations for compatible models.

        Returns a `Dataset`

        * **text** - List of text strings
        * **max_seq_length** - Maximum context token length. Check model configs to see max sequence length the model was trained with
        """

        tokens = self.tokenizer(
            text=text, return_tensors="pt", padding="max_length", truncation=True
        )
        dataset = TensorDataset(
            tokens["input_ids"],
            tokens["attention_mask"],
        )

        return dataset

    def process_batch(
        self,
        text: List,
        mini_batch_size: int = 8,
        max_seq_length: int = 512,
        use_gpu: bool = False,
    ) -> DataLoader:
        """Generate torch `DataLoader` object from text strings using specified tokenizer from HF.
        This provides clear tensor input representations for compatible models in an easy to use batch

        Returns a `DataLoader`

        * **text** - List of text strings
        * **mini_batch_size** - Batch size for inference
        * **max_seq_length** - Maximum context token length. Check model configs to see max sequence length the model was trained with
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

        dataset = self.process(text=text)

        dataloader = DataLoader(
            dataset,
            batch_size=mini_batch_size,
            collate_fn=lambda x: [t.to(device) for t in self._collate_fn(x)],
        )

        return dataloader

    def process_output(
        self,
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        pass

    def process_output_batch(self, outputs: List) -> List[List[Tuple[str, str]]]:
        """Process output of Transformers NER model

        * **outputs** - List of batch output tensors from a model's forward pass
        """
        results = []
        for logits, input_ids in outputs:
            tokens_batch = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids]
            argmax_batch = [torch.argmax(o, dim=1) for o in logits]
            for i in range(len(tokens_batch)):
                # Filter out padding
                results.append(
                    [
                        (token, self.label_strings[prediction])
                        for token, prediction in zip(
                            tokens_batch[i], argmax_batch[i].cpu().numpy()
                        )
                        if token != "[PAD]"
                    ]
                )
        return results

    def _collate_fn(self, data):
        batch = default_collate(data)
        return batch
