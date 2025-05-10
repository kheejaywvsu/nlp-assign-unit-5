import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.visualize import plot_similarity_matrix

def compute_similarity(matrix, titles):
    """Calculate and visualize cosine similarities"""
    # Compute similarity matrix
    cosine_sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(
        cosine_sim,
        index=titles,
        columns=titles
    )

    # Find most similar pair
    np.fill_diagonal(cosine_sim, -1)  # Ignore self-similarity
    max_idx = np.argmax(cosine_sim)
    row, col = np.unravel_index(max_idx, cosine_sim.shape)

    # Visualize similarity matrix
    plot_similarity_matrix(sim_df, "Document Similarity Analysis")

    return sim_df, (titles[row], titles[col], cosine_sim[row, col])
