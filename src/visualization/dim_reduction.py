"""
Viz of dimensional reductions
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def render_tsne():
    """
    Create tSNE projection; visualize it.
    """
    # Load the embeddings
    embeddings = np.load("embeddings.npy")

    tsne = TSNE(n_components=2)

    # Project the embeddings
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot the embeddings
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=embeddings, s=100)
    plt.show()
