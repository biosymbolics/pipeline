import logging
import sys
import torch
from ignite.metrics import Precision, Recall
from clients.trials import fetch_trials

from data.prediction.constants import DEFAULT_BATCH_SIZE, DEFAULT_TRUE_THRESHOLD
from data.types import ModelMetrics
from typings.trials import TrialSummary
from utils.tensor import pad_or_truncate_to_size

from .constants import (
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    TEXT_FIELDS,
)
from .core import DNN
from .types import DnnInput
from .utils import prepare_inputs


class ModelPredictor:
    """
    Class for model prediction

    Example:
    ```
    from core.models.clindev import ModelPredictor; from clients.trials import fetch_trials
    trials = fetch_trials("COMPLETED")
    predictor = ModelPredictor()
    preds = predictor(trials)
    ```
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint_95.pt",
        dnn_input_dim: int = 5860,
    ):
        self.dnn_input_dim = dnn_input_dim
        model = DNN(dnn_input_dim, round(dnn_input_dim / 2))
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
        y_true = y_input > DEFAULT_TRUE_THRESHOLD
        y_pred = prediction > DEFAULT_TRUE_THRESHOLD
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

    def predict_tensor(self, input_dict: DnnInput) -> torch.Tensor:
        """
        Predict probability of success for a given tensor input

        Args:
            input_dict (DnnInput): Dictionary of input tensors

        Returns:
            torch.Tensor: Tensor of predictions (1d)
        """
        x1_padded = pad_or_truncate_to_size(
            input_dict["x1"],
            (input_dict["x1"].size(0), DEFAULT_BATCH_SIZE, self.dnn_input_dim),
        )
        num_batches = x1_padded.size(0)

        output = torch.flatten(
            torch.cat([self.model(x1_padded[i]) for i in range(num_batches)])
        )

        return output

    def predict(self, trials: list[TrialSummary]) -> tuple[torch.Tensor, ModelMetrics]:
        """
        Predict trial stuff

        Args:
            trials (list[TrialSummary]): List of trials

        Returns:
            list[float]: output predictions
        """
        input_dict = prepare_inputs(
            trials, DEFAULT_BATCH_SIZE, CATEGORICAL_FIELDS, TEXT_FIELDS
        )

        predictions = self.predict_tensor(input_dict)
        actuals = torch.flatten(input_dict["y"])

        for i, trial in enumerate(trials):
            logging.info(
                "Patent %s (pred: %s, act: %s): %s (%s)",
                trial["nct_id"],
                (predictions[i] > DEFAULT_TRUE_THRESHOLD).item(),
                (actuals[i] > DEFAULT_TRUE_THRESHOLD).item(),
                trial["title"],
                predictions[i].item(),
            )

        metrics = self.evaluate(actuals, predictions)

        return (predictions, metrics)


def main():
    trials = fetch_trials("COMPLETED")
    predictor = ModelPredictor()
    preds, metrics = predictor.predict(trials)
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
            Usage: python3 -m core.models.clindev.predictor
            """
        )
        sys.exit()

    main()
