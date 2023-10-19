import argparse
import logging
import sys
from pydash import flatten
import torch

import system

system.initialize()

from data.prediction.utils import ModelInput, decode_output

from .constants import (
    BASE_ENCODER_DIRECTORY,
    BATCH_SIZE,
    DEVICE,
    InputRecord,
    input_field_lists,
    output_field_lists,
)
from .model import ClindevPredictionModel
from .utils import (
    prepare_input_data,
    preprocess_inputs,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelPredictor:
    """
    Model for prediction
    """

    def __init__(
        self,
        checkpoint_epoch: int = 230,
        device: str = DEVICE,
    ):
        """
        Initialize model

        Args:
            sizes (ClinDevModelSizes): Model sizes
        """
        torch.device(device)
        self.device = device
        self.model = ClindevPredictionModel(checkpoint_epoch, device)

    def __call__(self, *args, **kwargs):
        """
        Alias for self.predict
        """
        self.predict(*args, **kwargs)

    @staticmethod
    def __get_batch(i: int, input_dict: ModelInput) -> ModelInput:
        """
        Get input_dict for batch i
        """
        return ModelInput(
            **{
                f: v[i] if len(v) > i else torch.Tensor()
                for f, v in input_dict.__dict__.items()
                if v is not None
            }
        )

    def predict(
        self,
        records: list[InputRecord],
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE,
    ):
        inputs, _ = prepare_input_data(
            preprocess_inputs(records, ["enrollment"]),
            field_lists=input_field_lists,
            batch_size=batch_size,
            device=device,
        )

        def predict_batch(i: int):
            batch = ModelPredictor.__get_batch(i, inputs)

            _, _, y2_preds, y1_probs_list = self.model(
                torch.split(batch.multi_select, 1, dim=1),
                torch.split(batch.single_select, 1, dim=1),
                batch.text,
                batch.quantitative,
            )
            return decode_output(
                y1_probs_list,
                y2_preds,
                output_field_lists,
                directory=BASE_ENCODER_DIRECTORY,
            )

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

            Example:
            python3 -m data.prediction.clindev.predictor --interventions pf07264660 --conditions 'Clinically Significant Portal Hypertension' --mesh_conditions Hypertension
            """
        )
        sys.exit()

    standard_fields = {
        # "conditions": ["covid-19"],
        # "mesh_conditions": ["covid-19"],
        # "interventions": ["hydroxychloroquine"],
        "sponsor_type": "INDUSTRY",
        "phase": "PHASE_3",
        "enrollment": 10000,
        "start_date": 2024,
    }

    parser = argparse.ArgumentParser()
    for field in flatten(input_field_lists.__dict__.values()):
        parser.add_argument(f"--{field}", nargs="*", default=standard_fields.get(field))

    record = InputRecord(*parser.parse_args().__dict__.values())
    res = predict_single(record)
    print("RESULT:", res)
