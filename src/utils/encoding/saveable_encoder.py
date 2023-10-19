import logging
import os
from pydash import flatten
import polars as pl
from sklearn.calibration import LabelEncoder
import polars as pl
from joblib import dump, load
import numpy.typing as npt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Encoder:
    def __init__(self, impl, field: str, directory: str, *args, **kwargs):
        self._directory = directory
        self._field = field
        self._impl = impl
        self.encoder_type = impl.__name__
        self._file = f"{self._directory}/{self._field}-{self.encoder_type}.joblib"
        self._encoder = self.load(*args, **kwargs)

        if not os.path.exists(directory):
            os.makedirs(directory)

    def load(self, *args, **kwargs):
        """
        Load encoder from file for a field
        """
        try:
            encoder = load(self._file)
            logger.info(
                "Using EXISTING encoder for %s (%s)", self._field, self.encoder_type
            )
            return encoder
        except:
            logging.info(
                "Creating NEW instance of encoder for %s (%s)",
                self._field,
                self.encoder_type,
            )
            return self._impl(*args, **kwargs)

    def save(self):
        """
        Save encoder to file for a field
        """
        logging.info("Saving encoder for %s to %s", self._field, self._file)
        dump(self._encoder, self._file)

    def _encode_df(self, df: pl.DataFrame) -> list:
        values = df.select(pl.col(self._field)).to_series().to_list()
        return self._encode(values)  # .reshape(-1, 1))

    def _encode(self, values: list | npt.NDArray) -> list:
        """
        Encode a categorical field from a dataframe

        Args:
            field (str): Field to encode
            df (pl.DataFrame): Dataframe to encode from
        """
        is_nested = isinstance(values[0], list)

        logger.info(
            "Encoding field %s (e.g. %s) len: %s", self._field, values[0:5], len(values)
        )
        self._encoder.fit(flatten(values))

        _values = values if is_nested else [values]
        encoded_values = [self._encoder.transform(v) for v in _values]
        encoded_values = encoded_values[0] if not is_nested else encoded_values

        logger.info(
            "Finished encoding field %s (classes: %s, e.g. %s)",
            self._field,
            self._encoder.classes_,
            encoded_values[0:5],
        )

        return encoded_values

    def fit(self):
        """
        Fit an encoder
        """
        raise NotImplementedError()

    def fit_transform(self, data: pl.DataFrame | list | npt.NDArray) -> list:
        """
        Fit and transform a dataframe
        """
        if isinstance(data, pl.DataFrame):
            encoded_values = self._encode_df(data)
        else:
            encoded_values = self._encode(data)

        self.save()
        return encoded_values


class LabelCategoryEncoder(Encoder):
    def __init__(self, *args, **kargs):
        super().__init__(LabelEncoder, *args, **kargs)
