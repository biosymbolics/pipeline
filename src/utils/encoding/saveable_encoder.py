import logging
from typing import cast
from pydash import flatten
import polars as pl
from sklearn.calibration import LabelEncoder
import polars as pl
from joblib import dump, load

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Encoder:
    def __init__(self, impl, field: str, directory: str, *args, **kwargs):
        self._directory = directory
        self._field = field
        self._file = f"{self._directory}/{self._field}.joblib"
        self._impl = impl
        self._encoder = self.load(*args, **kwargs)

    def load(self, *args, **kwargs):
        """
        Load encoder from file for a field
        """
        try:
            encoder = load(self._file)
            if not isinstance(encoder, self._impl):
                raise ValueError(
                    f"Encoder for field {self._field} is not of type {self._impl}"
                )
            logger.warn("Using existing encoder for %s", self._field)
            return encoder
        except:
            logging.info("Creating instance of encoder for %s", self._field)
            return self._impl(*args, **kwargs)

    def save(self):
        """
        Save encoder to file for a field
        """
        logging.info("Saving encoder for %s to %s", self._field, self._file)
        dump(self._encoder, self._file)

    def encode(self, field: str, df: pl.DataFrame) -> list[list[int]]:
        """
        Encode a categorical field from a dataframe

        Args:
            field (str): Field to encode
            df (pl.DataFrame): Dataframe to encode from
        """
        values = df.select(pl.col(field)).to_series().to_list()
        logger.info(
            "Encoding field %s (e.g. %s) length: %s", field, values[0:5], len(values)
        )
        self._encoder.fit(flatten(values))
        encoded_values = [
            self._encoder.transform([v] if isinstance(v, str) else v) for v in values
        ]

        if df.shape[0] != len(encoded_values):
            raise ValueError(
                "Encoded values length does not match dataframe length: %s != %s",
                len(encoded_values),
                df.shape[0],
            )

        logger.info("Finished encoding field %s (e.g. %s)", field, encoded_values[0:5])
        return cast(list[list[int]], encoded_values)

    def fit(self):
        """
        Fit an encoder
        """
        raise NotImplementedError()

    @staticmethod
    def fit_transform(df: pl.DataFrame, field: str, directory: str) -> list[list[int]]:
        """
        Fit and transform a dataframe
        """
        instance = Encoder(LabelEncoder, field, directory)
        encoded_values = instance.encode(field, df)
        instance.save()
        return encoded_values


class LabelCategoryEncoder(Encoder):
    def __init__(self, *args, **kargs):
        super().__init__(LabelEncoder, *args, **kargs)
