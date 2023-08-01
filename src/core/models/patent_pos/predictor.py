import logging
import sys
from typing import Sequence, cast
import torch

from clients.patents import patent_client
from common.utils.tensor import pad_or_truncate_to_size
from core.models.patent_pos.core import CombinedModel
from typings.patents import ApprovedPatentApplication as PatentApplication

from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    GNN_CATEGORICAL_FIELDS,
    TEXT_FIELDS,
    TRUE_THRESHOLD,
)
from .types import AllInput
from .utils import prepare_inputs


class ModelPredictor:
    """
    Class for model prediction

    Example:
    ```
    from core.models.patent_pos import ModelPredictor; from clients.patents import patent_client
    patents = patent_client.search(["asthma"], True, 0, "medium", max_results=1000)
    predictor = ModelPredictor()
    preds = predictor(patents)
    ```
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint_95.pt",
        dnn_input_dim: int = 5040,
        gnn_input_dim: int = 480,
    ):
        self.dnn_input_dim = dnn_input_dim
        self.gnn_input_dim = gnn_input_dim
        model = CombinedModel(dnn_input_dim, gnn_input_dim)
        checkpoint_obj = torch.load(
            f"{CHECKPOINT_PATH}/{checkpoint}",
            map_location=torch.device("mps"),
        )
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict_tensor(self, input_dict: AllInput) -> list[float]:
        """
        Predict probability of success for a given tensor input

        Args:
            input_dict (AllInput): Dictionary of input tensors
        """
        x1_padded = pad_or_truncate_to_size(
            input_dict["x1"], (input_dict["x1"].size(0), BATCH_SIZE, self.dnn_input_dim)
        )
        num_batches = x1_padded.size(0)

        output = torch.flatten(
            torch.cat([self.model(x1_padded[i]) for i in range(num_batches)])
        )

        return [output[i].item() for i in range(output.size(0))]

    def predict(self, patents: list[PatentApplication]) -> list[float]:
        """
        Predict probability of success for a given input

        Args:
            patents (list[PatentApplication]): List of patent applications

        Returns:
            list[float]: Probabilities of success
        """
        input_dict = prepare_inputs(
            patents, BATCH_SIZE, CATEGORICAL_FIELDS, TEXT_FIELDS, GNN_CATEGORICAL_FIELDS
        )

        output = self.predict_tensor(input_dict)

        for i, patent in enumerate(patents):
            logging.info(
                "Patent %s (%s): %s (%s)",
                patent["publication_number"],
                (output[i] > TRUE_THRESHOLD),
                patent["title"],
                output[i],
            )

        return output


def main(terms: list[str]):
    patents = cast(
        list[PatentApplication],
        patent_client.search(terms, True, 0, "medium", max_results=1000),
    )
    predictor = ModelPredictor()
    predictor.predict(patents)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m core.models.patent_pos.predictor [patent search term(s)]
            Trains patent PoS (probability of success) model
            """
        )
        sys.exit()

    if not sys.argv[1:]:
        raise ValueError("Please provide a patent search term")

    main(sys.argv[1:])
