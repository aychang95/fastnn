from typing import Tuple
import logging

from tritonclient.grpc import InferResult  # Purely for typing
from tritonclient import grpc
from tritonclient import http

import torch

logger = logging.getLogger(__name__)


class FastNNClient:
    """FastNN client class for pushing requests to the Triton Inference Server


    Usage:
    ```python
    >>> client = FastNNClient(url="127.0.0.1:8000", model_name="distilbert-squad", model_version="1")
    >>> client.request(batch=batch)
    ```

    **Parameters:**

    * **url** - String url of Triton Inference Server. Defaults to 127.0.0.1:8000
    * **model_name** - String name of model in `model_repository` directory
    * **model_version** - String model version name
    * **client_type** - A string for choosing between http and grpc protocol ["grpc", "http"] (Defaults to "grpc")
    * **verbose** - Bool for log verbosity
    """

    def __init__(
        self,
        url: str = "127.0.0.1:8000",
        model_name: str = "distilbert-squad",
        model_version: str = "1",
        client_type: str = "grpc",
        verbose: bool = False,
    ):

        if client_type == "grpc":
            self.client_pkg = grpc
        elif client_type == "http":
            self.client_pkg = http
        else:
            ValueError("Paramater 'client_type' must be either grpc or http")

        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.client_type = client_type

        self.triton_client = self.client_pkg.InferenceServerClient(
            url=url, verbose=verbose
        )
        self.model_metadata = self.triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )
        self.model_config = self.triton_client.get_model_config(
            model_name=model_name, model_version=model_version
        )

    def request(
        self, batch: Tuple[torch.Tensor], binary_data: bool = False
    ) -> InferResult:
        """Runs a request with the `batch` input that can be generated from a FastNN `Processor`

        * **batch** - Tuple of torch tensors, typically batch inputs from a dataloader
        """
        if self.client_type == "grpc":
            return self.request_grpc(batch=batch)
        elif self.client_type == "http":
            return self.request_http(batch=batch)
        else:
            ValueError("Paramater 'client_type' must be either grpc or http")

    def request_http(
        self, batch: Tuple[torch.Tensor], binary_data: bool = False
    ) -> InferResult:
        """Runs an http request with the `batch` input that can be generated from a FastNN `Processor`

        * **batch** - Tuple of torch tensors, typically batch inputs from a dataloader
        """
        inputs_metadata = self.model_metadata["inputs"]
        outputs_metadata = self.model_metadata["outputs"]

        # Assert batch input matches triton model metadata
        assert len(batch) == len(inputs_metadata)

        inputs = []
        for i, metadata in enumerate(inputs_metadata):
            inp = self.client_pkg.InferInput(
                metadata["name"], tuple(batch[i].shape), metadata["datatype"]
            )
            inp.set_data_from_numpy(batch[i].cpu().numpy(), binary_data=binary_data)
            inputs.append(inp)

        outputs = []
        for i, metadata in enumerate(outputs_metadata):
            out = self.client_pkg.InferRequestedOutput(
                metadata["name"], binary_data=binary_data
            )
            outputs.append(out)

        response = self.triton_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )

        return response

    def request_grpc(
        self, batch: Tuple[torch.Tensor], binary_data: bool = False
    ) -> InferResult:
        """Runs a grpc request with the `batch` input that can be generated from a FastNN `Processor`

        * **batch** - Tuple of torch tensors, typically batch inputs from a dataloader
        """
        inputs_metadata = self.model_metadata.inputs
        outputs_metadata = self.model_metadata.outputs

        # Assert batch input matches triton model metadata
        assert len(batch) == len(inputs_metadata)

        inputs = []
        for i, metadata in enumerate(inputs_metadata):
            inp = self.client_pkg.InferInput(
                metadata.name, tuple(batch[i].shape), metadata.datatype
            )
            inp.set_data_from_numpy(batch[i].cpu().numpy())
            inputs.append(inp)

        outputs = []
        for i, metadata in enumerate(outputs_metadata):
            out = self.client_pkg.InferRequestedOutput(metadata.name)
            outputs.append(out)

        response = self.triton_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )

        return response
