import argparse
import logging
import sys
import torch

import system

system.initialize()

from data.prediction.utils import ModelInput

from .constants import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    InputRecord,
    input_field_lists,
)
from .model import TwoStageModel
from .utils import prepare_input_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelPredictor:
    """
    Model for prediction
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint_95.pt",
    ):
        """
        Initialize model

        Args:
            sizes (TwoStageModelSizes): Model sizes
        """
        torch.device(DEVICE)
        self.device = DEVICE
        self.model = self.__load_model(checkpoint, self.device)

    def __call__(self, *args, **kwargs):
        """
        Alias for self.predict
        """
        self.predict(*args, **kwargs)

    @staticmethod
    def __load_model(checkpoint: str, device: str):
        # sizes: TwoStageModelSizes
        sizes = None
        logger.info("Model sizes: %s", sizes)
        checkpoint_obj = torch.load(
            f"{CHECKPOINT_PATH}/{checkpoint}",
            map_location=torch.device(device),
        )
        model = TwoStageModel(checkpoint_obj["sizes"])
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()
        return model

    @staticmethod
    def __get_batch(i: int, input_dict: ModelInput) -> ModelInput:
        """
        Get input_dict for batch i
        """
        return ModelInput(
            **{
                f: v[i] if len(v) > i else torch.Tensor()
                for f, v in input_dict._asdict().items()
                if v is not None
            }
        )

    def predict(
        self,
        records: list[InputRecord],
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE,
    ):
        inputs = prepare_input_data(
            records,
            input_field_lists=input_field_lists,
            batch_size=batch_size,
            device=device,
        )

        def predict_batch(i: int):
            batch = ModelPredictor.__get_batch(i, inputs)

            y1_probs, _, y2_preds = self.model(
                torch.split(batch.multi_select, 1, dim=1),
                torch.split(batch.single_select, 1, dim=1),
                batch.text,
                batch.quantitative,
            )
            return [y1_probs, y2_preds]

        predictions = [predict_batch(i) for i in range(len(inputs.multi_select))]
        return predictions


def predict(records: list[InputRecord]):
    predictor = ModelPredictor()
    return predictor.predict(records)


def predict_single(record: InputRecord):
    predictor = ModelPredictor()
    return predictor.predict([record])


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.prediction.clindev.predictor --field: value
            (fields )
            """
        )
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", nargs="*")
    args = parser.parse_args()
    fields = dict(x.split("=") for x in args.fields)

    record = InputRecord(*fields.items())
    predict_single(record)
