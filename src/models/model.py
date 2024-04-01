"""
General saveable torch model
"""

import os
from typing import Type, TypeVar, cast
import torch
import torch.nn as nn
import logging

from constants.core import DEFAULT_DEVICE

from .constants import DEFAULT_CHECKPOINT_PATH


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


T = TypeVar("T", bound="SaveableModel")


class SaveableModel(nn.Module):
    """
    General saveable torch model
    """

    key: str | None = None
    device = DEFAULT_DEVICE  # TODO can this be overwritten?
    checkpoint_path = DEFAULT_CHECKPOINT_PATH

    def __init__(self):
        super().__init__()

    @classmethod
    def get_checkpoint_name(cls, epoch: int):
        return f"{cls.key}-checkpoint_{epoch}.pt"

    @classmethod
    def get_checkpoint_path(cls, epoch: int):
        return os.path.join(cls.checkpoint_path, cls.get_checkpoint_name(epoch))

    def save(self, epoch: int):
        """
        Save model checkpoint

        Args:
            epoch (int): Epoch number
        """
        checkpoint_name = f"{self.key}-checkpoint_{epoch}.pt"

        try:
            torch.save(self, self.get_checkpoint_path(epoch))
            logger.info("Saved checkpoint %s", checkpoint_name)
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_name, e)
            raise e

    @classmethod
    def load(cls: Type[T], epoch: int) -> T:
        model = cast(
            T,
            torch.load(
                cls.get_checkpoint_path(epoch),
                map_location=torch.device(cls.device),
            ),
        )
        model.eval()
        return model
