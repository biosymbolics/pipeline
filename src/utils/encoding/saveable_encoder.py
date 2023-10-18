import logging
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

    def _encode_df(self, field, df: pl.DataFrame) -> list:
        values = df.select(pl.col(field)).to_series().to_list()
        return self._encode(field, values)  # .reshape(-1, 1))

    def _encode(self, field: str, values: list | npt.NDArray) -> list:
        """
        Encode a categorical field from a dataframe

        Args:
            field (str): Field to encode
            df (pl.DataFrame): Dataframe to encode from
        """
        is_nested = isinstance(values[0], list)

        logger.info(
            "Encoding field %s (vals %s) len: %s", field, values[0:5], len(values)
        )
        self._encoder.fit(flatten(values))

        _values = values if is_nested else [values]
        encoded_values = [self._encoder.transform(v) for v in _values]

        logger.debug("Finished encoding field %s (e.g. %s)", field, encoded_values[0:5])

        return encoded_values[0] if not is_nested else encoded_values

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
            encoded_values = self._encode_df(self._field, data)
        else:
            encoded_values = self._encode(self._field, data)

        self.save()
        return encoded_values


class LabelCategoryEncoder(Encoder):
    def __init__(self, *args, **kargs):
        super().__init__(LabelEncoder, *args, **kargs)


class QuantitativeEncoder(Encoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _encode(self, field: str, values: npt.NDArray) -> list:
        """
        Encode a categorical field from a dataframe

        Args:
            field (str): Field to encode
            values (npt.NDArray): Values to encode
        """
        logger.info(
            "Encoding field %s (e.g. %s) length: %s", field, values[0:5], len(values)
        )

        self._encoder.fit(values)
        encoded_values = self._encoder.transform(values)

        return encoded_values.tolist()
