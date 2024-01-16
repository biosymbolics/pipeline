from dataclasses import dataclass
import math
from typing import Any, Optional

from data.prediction.types import AllCategorySizes
from typings.core import Dataclass


@dataclass(frozen=True)
class ClinDevModelInputSizes:
    multi_select_input: int
    quantitative_input: int
    single_select_input: int
    text_input: int

    categories_by_field: AllCategorySizes
    embedding_dim: int

    @property
    def text_output(self):
        return round(self.text_input / 40)

    @property
    def multi_select_output(self):
        if self.categories_by_field.multi_select is None:
            return 0

        return round(
            (self.multi_select_input / 5)  # shrink multis, many of which are zero
            * self.embedding_dim
            * math.log2(sum(self.categories_by_field.multi_select.values()))
        )

    @property
    def single_select_output(self):
        if self.categories_by_field.single_select is None:
            return 0

        return round(
            self.single_select_input
            * self.embedding_dim
            * math.log2(sum(self.categories_by_field.single_select.values()))
        )


@dataclass(frozen=True)
class ClinDevModelSizes(ClinDevModelInputSizes):
    """
    Sizes of inputs and outputs for two-stage model
    """

    stage1_output: int
    stage1_output_map: dict[str, int]
    stage2_output: int

    @property
    def stage1_input(self):
        return (
            self.multi_select_output
            + self.quantitative_input
            + self.single_select_output
            + self.text_output
        )

    @property
    def stage1_hidden(self):
        return round(self.stage1_input * 2.5)

    @property
    def stage2_input(self):
        return self.stage1_output + self.stage1_input

    @property
    def stage2_hidden(self):
        return round(self.stage2_input * 2.5)

    def __str__(self):
        return (
            " ".join([f"{f}: {v}" for f, v in self.__dict__.items()])
            + " "
            + f"multi_select_output: {self.multi_select_output} "
            f"single_select_output: {self.single_select_output} "
            f"stage1_input: {self.stage1_input} "
            f"stage1_hidden: {self.stage1_hidden} "
            f"stage2_input: {self.stage2_input} "
            f"stage2_hidden: {self.stage2_hidden} "
        )


@dataclass(frozen=True)
class StageSizes:
    input: int
    hidden: int
    output: int


@dataclass(frozen=True)
class PatentTrialPrediction(Dataclass):
    publication_number: Optional[str]
    blinding: str
    comparison_type: str
    conditions: list[str]
    design: str
    duration: str
    duration_exact: float
    enrollment: str
    interventions: list[str]
    # masking: Optional[str]
    phase: str
    randomization: str
    # sponsor_type: str
    start_date: str
    starting_phase: Optional[str]
    max_timeframe: Optional[float]
    sponsor: Optional[str]
