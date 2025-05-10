import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# Configure global plot settings
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def plot_term_matrix(matrix_df, title, top_terms=10, figsize=(12, 6)):
    # Select top terms across all documents
    top_terms = matrix_df.sum().sort_values(ascending=False).head(top_terms).index
    matrix_subset = matrix_df[top_terms]

    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix_subset,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Weight'}
    )
    plt.title(f"Top {len(top_terms)} Terms - {title}", size=14)
    plt.xlabel("Terms")
    plt.ylabel("Documents")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_similarity_matrix(sim_df, title, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(sim_df, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        sim_df,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f"Document Similarity - {title}", size=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# def plot_embeddings(vectors, labels, titles, figsize=(10, 8)):
#     # Reduce dimensionality
#     pca = PCA(n_components=2)
#     vectors_2d = pca.fit_transform(vectors)

#     # Create plot
#     plt.figure(figsize=figsize)
#     scatter = plt.scatter(
#         vectors_2d[:, 0],
#         vectors_2d[:, 1],
#         c=labels,
#         cmap='viridis',
#         s=150,
#         edgecolor='w',
#         alpha=0.8
#     )

#     # Add annotations
#     for i, (x, y) in enumerate(vectors_2d):
#         plt.text(
#             x + 0.05,
#             y + 0.05,
#             titles[i],
#             fontsize=8,
#             ha='left',
#             va='bottom',
#             alpha=0.7
#         )

#     # Format plot
#     plt.title("Document Embeddings Visualization (PCA)", pad=20)
#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
#     plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.show()

# def plot_classification_results(y_true, y_pred, labels, figsize=(8, 6)):
#     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

#     plt.figure(figsize=figsize)
#     disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
#     plt.title("Classification Results - Confusion Matrix")
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()
