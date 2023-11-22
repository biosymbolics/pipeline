import argparse
from datetime import date
from enum import Enum
import json
import logging
import math
import sys
from typing import Any, Sequence, cast
from pydash import flatten, omit_by
import torch
import polars as pl


import system
from typings.trials import SponsorType, TrialPhase
from utils.encoding.json_encoder import DataclassJSONEncoder

system.initialize()

from data.prediction.clindev.types import PatentTrialPrediction
from data.prediction.utils import ModelInput, decode_output

from .constants import (
    BASE_ENCODER_DIRECTORY,
    BATCH_SIZE,
    DEVICE,
    InputRecord,
    input_field_lists,
    ALL_INPUT_FIELD_LISTS,
    output_field_lists,
)
from .model import ClindevPredictionModel
from .utils import (
    get_batch,
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
        checkpoint_epoch: int = 240,
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

    def predict(
        self,
        records: list[InputRecord],
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE,
    ) -> list[dict[str, Any]]:
        inputs, _ = prepare_input_data(
            preprocess_inputs(records, []),  # ["enrollment"]
            field_lists=input_field_lists,
            batch_size=batch_size,
            device=device,
        )

        def predict_batch(i: int) -> Sequence[dict[str, Any]]:
            batch = get_batch(i, inputs)

            if batch is None:
                return []

            _, _, y2_preds, y1_probs_list = self.model(
                ModelInput.get_instance(**batch.__dict__)
            )
            decoded = decode_output(
                y1_probs_list,
                y2_preds,
                output_field_lists,
                directory=BASE_ENCODER_DIRECTORY,
                actual_length=min(len(records) - i * batch_size, batch_size),
            )
            as_records = pl.DataFrame(decoded).to_dicts()
            return as_records

        predictions = [
            predict_batch(i) for i in range(math.ceil(len(records) / BATCH_SIZE))
        ]
        return flatten(predictions)


PHASES = [TrialPhase.PHASE_1, TrialPhase.PHASE_2, TrialPhase.PHASE_3]


def predict(inputs: list[dict]) -> list[PatentTrialPrediction]:
    """
    Output: dict[publication_number: str, trials: list[dict]]

    from data.prediction.clindev.predictor import predict
    input = [{ "publication_number": 'abcd123', "starting_phase": None, "conditions": ["heart disease"], "interventions": ["gpr86 antagonist", "antagonist"] }, { "publication_number": 'bb3311', "starting_phase": None, "conditions": ["asthma"], "interventions": ["advair"] }]
    predict(input)
    """
    predictor = ModelPredictor()

    def predict_phase(
        phase: TrialPhase, start_dates: list[int]
    ) -> list[PatentTrialPrediction]:
        records = [
            InputRecord(
                **{
                    **{k: v for k, v in input.items() if k in ALL_INPUT_FIELD_LISTS},
                    "phase": phase,
                    "start_date": sd,
                    "sponsor_type": SponsorType.INDUSTRY,
                }
            )
            for input, sd in zip(inputs, start_dates)
        ]
        predictions = predictor.predict(records)
        return [
            PatentTrialPrediction(
                **{**rec._asdict(), **input, **pred, "publication_number": input["publication_number"]}  # type: ignore
            )
            for input, rec, pred in zip(inputs, records, predictions)
        ]

    predictions: list[list[PatentTrialPrediction]] = []
    for phase in PHASES:
        start_dates = (
            [
                round(max(1, round(sum(d) / 365)) + date.today().year)
                for d in zip(*[[r.duration_exact for r in p] for p in predictions])
            ]
            if len(predictions) > 0
            else [date.today().year] * len(inputs)
        )
        predictions.append(predict_phase(phase, start_dates))

    df = pl.DataFrame(flatten(predictions), infer_schema_length=None)
    return [PatentTrialPrediction(**p) for p in df.to_dicts()]


def predict_single(record: dict):
    predictor = ModelPredictor()

    def predict_phase(phase: TrialPhase):
        input = InputRecord(
            **{
                "sponsor": "Janssen",
                **omit_by(record, lambda x: x is None),
                "phase": phase,
                "max_timeframe": phase._order,
            }
        )
        pred = predictor.predict([input])[0]
        return PatentTrialPrediction(**{**record, **input._asdict(), **pred})

    return {str(phase): predict_phase(phase) for phase in PHASES}


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.prediction.clindev.predictor --field: value

            Example:
            python3 -m data.prediction.clindev.predictor --interventions pf07264660 --conditions Hypertension
            python3 -m data.prediction.clindev.predictor --interventions lianhuaqingwen --conditions 'Coronavirus Infections'
            python3 -m data.prediction.clindev.predictor --interventions 'apatinib mesylate' --conditions Neoplasms
            """
        )
        sys.exit()

    standard_fields = {
        "sponsor_type": SponsorType.INDUSTRY,
        "start_date": 2024,
    }

    parser = argparse.ArgumentParser()
    for field in flatten(input_field_lists.__dict__.values()):
        parser.add_argument(f"--{field}", nargs="*", default=standard_fields.get(field))

    res = predict_single(parser.parse_args().__dict__)
    print("RESULT:", json.dumps(res, indent=2, cls=DataclassJSONEncoder))
