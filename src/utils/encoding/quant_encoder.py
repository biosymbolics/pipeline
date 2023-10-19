from functools import reduce
from typing import Literal, Optional, Sequence
import logging
import numpy as np
import polars as pl
from sklearn.preprocessing import KBinsDiscretizer
from kneed import KneeLocator
import polars as pl
import numpy.typing as npt


from .saveable_encoder import Encoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KbinsStrategy = Literal["uniform", "quantile", "kmeans"]


class BinEncoder(Encoder):
    def __init__(
        self,
        field: str,
        directory: str,
        # if None, will estimate (TODO: must adjust size of stage2 output accordingly!)
        n_bins: Optional[int] = 5,
        strategy: KbinsStrategy = "kmeans",
        encode: Literal["ordinal", "onehot", "onehot-dense"] = "ordinal",
        *args,
        **kwargs
    ):
        self.field = field
        self.strategy: KbinsStrategy = strategy
        self.n_bins = n_bins
        super().__init__(
            KBinsDiscretizer,
            field,
            directory,
            encode=encode,
            strategy=strategy,
            subsample=None,
            *args,
            **kwargs
        )

    @staticmethod
    def estimate_n_bins(
        data: list[str] | list[int] | list[float],
        kbins_strategy: Literal["uniform", "quantile", "kmeans"],
        bins_to_test=range(3, 20),
    ):
        """
        Estimate the optimal number of bins
        for use with bin_quantitative_values

        Calculates gini_impurity and uses elbow method to find optimal number of bins

        Args:
            data (list[str] | list[int] | list[float]): List of values
            bins_to_test (range): Range of bins to test
            kbins_strategy (str): Strategy to use for KBinsDiscretizer
        """

        def elbow(
            values: Sequence[float],
            bins: Sequence[int],
            strategy: Literal["first", "default"] = "first",
        ) -> int:
            # https://arvkevi-kneed.streamlit.app/
            # https://github.com/arvkevi/kneed
            kneedle = KneeLocator(
                bins, values, direction="decreasing", curve="concave", online=True
            )

            if kneedle.elbow is None:
                logger.warning("No elbow found for bins %s, using last index", bins)
                return len(bins) - 1

            if strategy == "first":
                return kneedle.all_elbows.pop()

            return int(kneedle.elbow)

        def gini_impurity(original, binned):
            hist, _ = np.histogram(original)
            gini_before = 1 - np.sum([p**2 for p in hist / len(original)])

            def bin_gini(bin):
                p = np.mean(binned == bin)
                return p * (1 - p)

            gini_after = reduce(lambda acc, bin: acc + bin_gini(bin), np.unique(binned))

            return gini_before - gini_after

        def score_bin(n_bins):
            est = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=kbins_strategy, subsample=None
            )
            bin_data = est.fit_transform(np.array(data).reshape(-1, 1))
            return gini_impurity(data, bin_data)

        scores = [score_bin(n_bins) for n_bins in bins_to_test]
        winner = elbow(scores, bins_to_test)

        return winner

    def _encode(self, values: npt.NDArray) -> list:
        """
        Encode a quant field

        Args:
            values (npt.NDArray): Values to encode
        """
        logger.info(
            "Encoding field %s (e.g. %s) length: %s",
            self._field,
            values[0:5],
            len(values),
        )

        self._encoder.fit(values)
        encoded_values = self._encoder.transform(values)

        if self._encoder.n_bins_ != self.n_bins:
            logger.error(
                "Actual bins != n_bins: %s vs %s", self._encoder.n_bins_, self.n_bins
            )

        logger.info(
            "Finished encoding field %s (%s bins)", self._field, self._encoder.n_bins_
        )

        self.save()

        return encoded_values.tolist()

    def bin(
        self,
        values: Sequence[float | int] | pl.Series,
    ) -> Sequence[list[int]]:
        """
        Bins quantiative values, turning them into categorical
        @see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html


        Args:
            values (Sequence[float | int]): List of values
        Returns:
            Sequence[list[int]]: List of lists of binned values (e.g. [[0.0], [2.0], [5.0], [0.0], [0.0]])
                (a list of lists because that matches our other categorical vars)
        """
        if self.n_bins is None:
            self.n_bins = self.estimate_n_bins(
                list(values), kbins_strategy=self.strategy
            )
            logger.info(
                "Using estimated optimal n_bins value of %s for field %s",
                self.n_bins,
                self.field,
            )
            logger.error(
                "THIS WILL PROBABLY BREAK YOUR MODEL unless stage2 output size (etc) are sized properly."
            )

        X = np.array(values).reshape(-1, 1)
        Xt = self._encode(X)
        return Xt

    def fit_transform(self, values, *args, **kwargs):
        return self.bin(values, *args, **kwargs)
