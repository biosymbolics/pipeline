"""
Viz of dimensional reductions
"""
import matplotlib.pyplot as plt
import umap
import umap.plot


def render_umap(df):
    """
    Render a UMAP plot of a dataframe
    """
    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="correlation").fit(df)
    mapper = umap.UMAP().fit(df.data)
    umap.plot.points(mapper, labels=df.target)
