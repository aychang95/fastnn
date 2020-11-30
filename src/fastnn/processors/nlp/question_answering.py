import logging
from typing import List, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from transformers import (
    SquadExample,
    squad_convert_examples_to_features,
    AutoTokenizer,
)

from transformers.data.processors.squad import SquadFeatures, SquadResult

from fastnn.utils.qa.squad_metrics import (
    compute_predictions_logits,
)

from fastnn.processors.base_processing import Processor


logger = logging.getLogger(__name__)


class TransformersQAProcessor(Processor):
    """Question Answering Data Processor. Use this class to generate tensor inputs from human legible text/string data.
    This class can be used with a majority of the Bert architecture transformer models with the span-based extractive,
    Question Answering predictive head from Hugging Face.

    Usage:
    ```python
    >>> processor = TRansformersQAProcessor(model_name_or_path="distilbert-base-cased-distilled-squad")
    >>> processor.process(query=["string"], context["string"])

    **Parameters:**

    * **model_name_or_path** - String defining HF question answering model/tokenizer's name
    ```
    """

    def __init__(
        self, model_name_or_path: str = "distilbert-base-cased-distilled-squad"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )  # Can't use fast tokenizer yet for QA `use_fast=True`

    def process(
        self,
        query: List[str],
        context: List[str],
        max_seq_length: int = 512,
        doc_stride: int = 128,
        max_query_length: int = 64,
    ) -> Tuple[List[SquadExample], List[SquadFeatures], Dataset]:
        """Generate torch `Dataset` object from query/context string pairs using specified tokenizer from HF.
        This provides clear tensor input representations for compatible models.

        Returns a tuple `Dataset` and matching `SquadFeatures`

        * **query** - List of query strings, must be same length as `context`
        * **context** - List of context strings, must be same length as `query`
        * **max_seq_length** - Maximum context token length. Check model configs to see max sequence length the model was trained with
        * **doc_stride** - Number of token strides to take when splitting up context into chunks of size `max_seq_length`
        * **max_query_length** - Maximum token length for queries
        """
        examples = self._generate_squad_examples(query=query, context=context)
        features, dataset = squad_convert_examples_to_features(
            examples,
            self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )

        return examples, features, dataset

    def process_batch(
        self,
        query: List,
        context: List,
        mini_batch_size: int = 8,
        max_seq_length: int = 512,
        doc_stride: int = 128,
        max_query_length: int = 64,
        use_gpu: bool = False,
    ) -> Tuple[List[SquadExample], List[SquadFeatures], DataLoader]:
        """Generate torch `DataLoader` object from query/context string pairs using specified tokenizer from HF.
        This provides clear tensor input representations for compatible models in an easy to use batch

        Returns a tuple of (List[`SquadExample`], List[`SquadFeatures`], `DataLoader`)

        * **query** - List of query strings, must be same length as `context`
        * **context** - List of context strings, must be same length as `query`
        * **mini_batch_size** - Batch size for inference
        * **max_seq_length** - Maximum context token length. Check model configs to see max sequence length the model was trained with
        * **doc_stride** - Number of token strides to take when splitting up context into chunks of size `max_seq_length`
        * **max_query_length** - Maximum token length for queries
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

        examples, features, dataset = self.process(
            query=query,
            context=context,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=mini_batch_size,
            collate_fn=lambda x: [t.to(device) for t in self._qa_collate_fn(x)],
        )

        return examples, features, dataloader

    def process_output(
        self,
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        examples: List[SquadExample],
        features: List[SquadFeatures],
        n_best_size: int = 5,
        max_answer_length: int = 10,
        do_lower_case: bool = False,
        verbose_logging: bool = False,
        version_2_with_negative: bool = False,
        null_score_diff_threshold: float = 0.0,
    ):
        pass

    def process_output_batch(
        self,
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        examples: List[SquadExample],
        features: List[SquadFeatures],
        n_best_size: int = 5,
        max_answer_length: int = 64,
        do_lower_case: bool = False,
        verbose_logging: bool = False,
        version_2_with_negative: bool = False,
        null_score_diff_threshold: float = 0.0,
    ) -> Tuple[OrderedDict, OrderedDict]:
        """Process output of Transformers QA model into human legible results.

        * **outputs** - List of batch output tensors from a model's forward pass
        * **examples** - List of `SquadExample` objects for each original context/query pair used as input. This is returned from the built-in `process()` or `process_batch()` methods
        * **features** - List of `SquadFeature` objects for each context/query pair over the original doc_stride lengths. This is also returned from the built-in `process()` or `process_batch()` methods
        * **n_best_size** - Number of top n results you want
        * **max_answer_length** - Maximum token length for answers that are returned
        * **do_lower_case** - Set as `True` if using uncased QA models
        * **verbose_logging** - Set True if you want prediction verbose loggings
        * **version_2_with_negative** - Set as True if using QA model with SQUAD2.0
        * **null_score_diff_threshold** - Threshold for predicting null(no answer) in Squad 2.0 Model.  Default is 0.0.  Raise this if you want fewer null answers
        """

        # Generate results per example query
        all_results: List[SquadResult] = []
        for output in outputs:
            example_indices = output[2]
            for i, example_index in enumerate(example_indices):
                start_logits = self._to_list(output[0][i])
                end_logits = self._to_list(output[1][i])
                eval_feature = features[example_index[0].item()]
                unique_id = int(eval_feature.unique_id)
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        # Compute predictions based off logits on a per example basis
        answers, n_best = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            do_lower_case=do_lower_case,
            verbose_logging=verbose_logging,
            version_2_with_negative=version_2_with_negative,
            null_score_diff_threshold=null_score_diff_threshold,
            tokenizer=self.tokenizer,
        )

        return answers, n_best

    def _qa_collate_fn(self, data):
        batch = default_collate(data)
        # Generate same batch dims for scalars to address future batch inferencing
        batch[3].unsqueeze_(1)
        batch[4].unsqueeze_(1)
        return batch

    def _to_list(self, tensor: torch.Tensor):
        return tensor.detach().cpu().tolist()

    def _generate_squad_examples(
        self, query: List[str], context: List[str]
    ) -> List[SquadExample]:
        """Generate HF Squad Example objects with query/context pairs"""
        assert len(query) == len(context)
        examples = []
        title = "qa"
        is_impossible = False
        answer_text = None
        start_position_character = None
        answers = ["answer"]
        for idx, (q, c) in enumerate(zip(query, context)):
            example = SquadExample(
                qas_id=str(idx),
                question_text=q,
                context_text=c,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )
            examples.append(example)
        return examples
