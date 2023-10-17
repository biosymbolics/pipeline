from functools import reduce
from typing import Literal, Optional, Sequence
import logging
import numpy as np
import polars as pl
from sklearn.preprocessing import KBinsDiscretizer
from kneed import KneeLocator
import polars as pl


from .saveable_encoder import Encoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinEncoder(Encoder):
    def __init__(self, field: str, directory: str, n_bins: int, *args, **kwargs):
        super().__init__(KBinsDiscretizer, field, directory, n_bins, *args, **kwargs)

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

    @staticmethod
    def bin(
        values: Sequence[float | int] | pl.Series,
        field: str,
        directory: str,
        n_bins: int | None = 5,
        kbins_strategy: Literal["uniform", "quantile", "kmeans"] = "kmeans",
    ) -> Sequence[list[int]]:
        """
        Bins quantiative values, turning them into categorical
        @see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html

        NOTE: specify n_bins when doing inference; i.e. ensure it matches with how the model was trained.

        Args:
            values (Sequence[float | int]): List of values
            field (str): Field name (used for logging)
            n_bins (int): Number of bins
            kbins_strategy (str): Strategy to use for KBinsDiscretizer

        Returns:
            Sequence[list[int]]: List of lists of binned values (e.g. [[0.0], [2.0], [5.0], [0.0], [0.0]])
                (a list of lists because that matches our other categorical vars)
        """
        if n_bins is None:
            n_bins = BinEncoder.estimate_n_bins(
                list(values), kbins_strategy=kbins_strategy
            )
            logger.info(
                "Using estimated optimal n_bins value of %s for field %s", n_bins, field
            )

        binner = Encoder(
            KBinsDiscretizer,
            field,
            directory,
            n_bins=n_bins,
            encode="ordinal",
            strategy=kbins_strategy,
            subsample=None,
        )

        X = np.array(values).reshape(-1, 1)
        Xt = binner.fit_transform(pl.DataFrame(X, schema=[field]))
        return Xt

    def fit_transform(self, values, field, directory, n_bins, *args, **kwargs):
        return self.bin(values, field, directory, n_bins, *args, **kwargs)
