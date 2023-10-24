import logging
import os
from typing import Sequence, TypeAlias, TypeVar, cast
from pydash import flatten
import polars as pl
from sklearn.calibration import LabelEncoder
import polars as pl
from joblib import dump, load
import numpy.typing as npt

from utils.list import is_sequence


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# for decoding
V = TypeVar("V", bound=Sequence | npt.NDArray | int | float)

EncodeableData: TypeAlias = pl.Series | Sequence | npt.NDArray
EncoderData: TypeAlias = pl.DataFrame | EncodeableData

EncodedData = int | float | list[int | float]


class Encoder:
    def __init__(
        self,
        impl,
        field: str | None = None,
        directory: str | None = None,
        *args,
        **kwargs,
    ):
        self._directory = directory
        self._field = field or "nf"
        self._impl = impl
        self.encoder_type = impl.__name__

        self.is_saveable = field is not None and directory is not None
        self._name = f"{self._field}-{self.encoder_type}"

        if self.is_saveable:
            self._file = f"{self._directory}/{self._name}.joblib"
        self._encoder, self._is_fit = self.load(*args, **kwargs)

        if directory is not None and not os.path.exists(directory):
            os.makedirs(directory)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Encoder instance
        return getattr(self._encoder, name)

    def load(self, *args, **kwargs):
        """
        Load encoder from file for a field
        """
        try:
            encoder = load(self._file)
            logger.debug("Using EXISTING encoder for %s", self._name)
            return encoder, True
        except:
            logging.info("Creating NEW instance of encoder for %s", self._name)
            return self._impl(*args, **kwargs), False

    def save(self):
        """
        Save encoder to file for a field
        """
        if not self.is_saveable:
            logger.warning("Cannot save encoder without field and directory")
            return
        logging.info("Saving encoder for %s to %s", self._field, self._file)
        dump(self._encoder, self._file)

    def _df_to_values(self, df: pl.DataFrame) -> list[EncodedData]:
        """
        Convert a dataframe to a list of values
        """
        if not self._field:
            raise ValueError("Cannot encode dataframe without field")

        values = df.select(pl.col(self._field)).to_series().to_list()
        return values

    def _encode_df(self, df: pl.DataFrame) -> list[EncodedData]:
        return self._encode(self._df_to_values(df))

    def _encode(self, values: EncodeableData) -> list[EncodedData]:
        """
        Encode values

        Args:
            values (EncodeableData): Values to encode
        """

        # fit if not already fit
        if not self._is_fit:
            self.fit(values)

        # if nested (list o' lists, at least one len > 1), encode each list separately
        if all([is_sequence(v) for v in values]) and any([len(v) > 1 for v in values]):
            encoded = [self._encoder.transform(v).tolist() for v in values]
        else:
            encoded = self._encoder.transform(values).tolist()

        logger.info(
            "Encoded field %s (e.g. %s vs %s)", self._field, encoded[0:6], values[0:6]
        )
        return encoded

    def fit(self, values: EncoderData):
        """
        Fit an encoder
        """
        if self._is_fit:
            logger.warning("Encoder already fit, skipping")
            return

        if isinstance(values, pl.DataFrame):
            values = self._df_to_values(values)

        self._encoder.fit(flatten(values))
        self.save()

    def fit_transform(self, data: EncoderData) -> list[EncodedData]:
        """
        Fit and transform a dataframe
        """
        if isinstance(data, pl.DataFrame):
            encoded_values = self._encode_df(data)
        else:
            encoded_values = self._encode(data)

        return encoded_values

    def inverse_transform(self, val: V) -> V:
        """
        Inverse transform a list of encoded value(s)
        """
        if not self._is_fit:
            raise ValueError("Cannot inverse transform before fitting")

        try:
            return self._encoder.inverse_transform(val)
        except ValueError as e:
            logger.error(
                "Could not inverse transform %s, impl %s: %s", val, self._name, e
            )
            return val


class LabelCategoryEncoder(Encoder):
    """
    Encoder for categorical fields

    Usage (examining saved encoder):
    ```
    from utils.encoding.saveable_encoder import LabelCategoryEncoder
    encoder = LabelCategoryEncoder("design", "clindev_model_checkpoints/encoders")
    print(encoder.classes_)
    ```
    """

    def __init__(self, *args, **kargs):
        super().__init__(LabelEncoder, *args, **kargs)

    def fit(self, *args, **kargs):
        super().fit(*args, **kargs)
        logger.info(
            "Finished fitting field %s (classes: %s)",
            self._field,
            self._encoder.classes_,
        )
