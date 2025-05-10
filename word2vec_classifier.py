import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_classifier(documents, titles):

    # Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]

    # Train Word2Vec model
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    # Create document vectors
    doc_vectors = []
    for doc in tokenized_docs:
        vectors = [model.wv[word] for word in doc if word in model.wv]
        doc_vectors.append(np.mean(vectors, axis=0) if vectors else np.zeros(100))
    X = np.array(doc_vectors)
    y = np.array([0, 0, 1, 1, 1])  # Class labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        stratify=y,
        random_state=42
    )

    # Train and evaluate classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf.score(X_test, y_test)
