from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from utils.visualize import plot_term_matrix

def build_matrices(documents, titles):

    # Raw frequency matrix
    count_vectorizer = CountVectorizer()
    raw_matrix = count_vectorizer.fit_transform(documents)
    raw_df = pd.DataFrame(
        raw_matrix.toarray(),
        columns=count_vectorizer.get_feature_names_out(),
        index=titles
    )

    # TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out(),
        index=titles
    )

    # Visualize matrices
    plot_term_matrix(raw_df, "Raw Frequency Matrix", top_terms=15)
    plot_term_matrix(tfidf_df, "TF-IDF Weighted Matrix", top_terms=15)

    return raw_df, tfidf_df
