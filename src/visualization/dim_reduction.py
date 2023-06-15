"""
Viz of dimensional reductions
"""
import matplotlib.pyplot as plt
import umap


def render_umap(df):
    """
    Render a UMAP plot of a dataframe
    """
    embedding = umap.UMAP(
        n_neighbors=5, min_dist=0.3, metric="correlation"
    ).fit_transform(df)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=embedding, s=100)
    plt.show()
