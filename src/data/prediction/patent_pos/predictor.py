import logging
import sys
from typing import cast
import torch
from ignite.metrics import Precision, Recall

from clients.patents import search_client
from data.types import ModelMetrics
from utils.tensor import pad_or_truncate_to_size
from typings.patents import PatentApplication

from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    GNN_CATEGORICAL_FIELDS,
    TEXT_FIELDS,
    TRUE_THRESHOLD,
)
from .core import CombinedModel
from .types import AllInput
from .utils import prepare_inputs


class ModelPredictor:
    """
    Class for model prediction

    Example:
    ```
    from core.models.patent_pos import ModelPredictor; from clients.patents import patent_client
    patents = patent_client.search(["asthma"], None, True, 0, "medium", limit=1000)
    predictor = ModelPredictor()
    preds = predictor(patents)
    ```
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint_95.pt",
        dnn_input_dim: int = 5860,
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
        self.precision = Precision(average=True)
        self.recall = Recall(average=True)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def evaluate(self, y_input: torch.Tensor, prediction: torch.Tensor) -> ModelMetrics:
        """
        Evaluate model performance

        Args:
            input_dict (AllInput): Dictionary of input tensors
            prediction (torch.Tensor): Tensor of predictions (1d)
        """
        y_true = y_input > TRUE_THRESHOLD
        y_pred = prediction > TRUE_THRESHOLD
        self.precision.update((y_pred, y_true))
        self.recall.update((y_pred, y_true))

        metrics: ModelMetrics = {
            "precision": float(self.precision.compute()),
            "recall": float(self.recall.compute()),
            "f1": self.f1.compute(),
        }

        self.precision.reset()
        self.recall.reset()

        return metrics

    def predict_tensor(self, input_dict: AllInput) -> torch.Tensor:
        """
        Predict probability of success for a given tensor input

        Args:
            input_dict (AllInput): Dictionary of input tensors

        Returns:
            torch.Tensor: Tensor of predictions (1d)
        """
        x1_padded = pad_or_truncate_to_size(
            input_dict["x1"], (input_dict["x1"].size(0), BATCH_SIZE, self.dnn_input_dim)
        )
        num_batches = x1_padded.size(0)

        output = torch.flatten(
            torch.cat([self.model(x1_padded[i]) for i in range(num_batches)])
        )

        return output

    def predict(
        self, patents: list[PatentApplication]
    ) -> tuple[torch.Tensor, ModelMetrics]:
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

        predictions = self.predict_tensor(input_dict)
        actuals = torch.flatten(input_dict["y"])

        for i, patent in enumerate(patents):
            logging.info(
                "Patent %s (pred: %s, act: %s): %s (%s)",
                patent["publication_number"],
                (predictions[i] > TRUE_THRESHOLD).item(),
                (actuals[i] > TRUE_THRESHOLD).item(),
                patent["title"],
                predictions[i].item(),
            )

        metrics = self.evaluate(actuals, predictions)

        return (predictions, metrics)


def main(terms: list[str]):
    patents = cast(
        list[PatentApplication],
        search_client.search(terms, min_patent_years=0, limit=1000),
    )
    predictor = ModelPredictor()
    preds, metrics = predictor.predict(patents)
    logging.info(
        "Prediction mean: %s, min: %s and max: %s",
        preds.mean().item(),
        preds.min().item(),
        preds.max().item(),
    )
    logging.info("Metrics: %s", metrics)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m core.models.patent_pos.predictor [patent search term(s)]
            Predicts with patent PoS (probability of success) model

            Example:
                >>> python3 -m core.models.patent_pos.predictor asthma
            """
        )
        sys.exit()

    if not sys.argv[1:]:
        raise ValueError("Please provide a patent search term")

    main(sys.argv[1:])
